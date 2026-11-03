"""Read-only decision traces: the evidence, signals, and lineage behind one decision."""

import json
import uuid

from pydantic import (
    ValidationError as PydanticValidationError,
)
from sqlalchemy import and_, or_, select

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.errors import ValidationError
from poseidon.core.timeutil import iso_time, stored_timestamp
from poseidon.decision_loop.decisions import DecisionService, _decision_response
from poseidon.decision_loop.evaluation import verify_complete_run
from poseidon.decision_loop.trace_stages import _trace_row, build_trace_stages
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import (
    CampaignEvent,
    CampaignReview,
    ExperimentCampaign,
    HoldoutUse,
)
from poseidon.models.fill_allocation import FillAllocation
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.outcome import OutcomeRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.research_revision import ResearchRevision
from poseidon.models.signal import SignalRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.signals.schemas import SignalTraceMetadata


def project_trace_signal(signal, metadata=None):
    try:
        metadata = metadata or SignalTraceMetadata.model_validate(signal.metadata_)
    except PydanticValidationError as error:
        raise ValidationError("signal trace metadata is invalid") from error
    values = {
        "symbol": signal.symbol,
        "market": signal.market,
        "action": signal.action,
        "status": signal.status,
        "metadata": metadata.model_dump(mode="json", exclude_none=True),
    }
    if metadata.narrative_confidence is not None:
        values["Narrative confidence"] = metadata.narrative_confidence
        values["confidence_note"] = "Not a probability of profit"
    return _trace_row(
        "Signal",
        "Available",
        timestamp_value=iso_time(stored_timestamp(signal.signal_time, "signal.signal_time"), "signal.signal_time"),
        identities={"signal_id": str(signal.id)},
        values=values,
    )


def load_trace_signals(session, *, account_scope, version, snapshots, decision=None):
    snapshot_by_id = {snapshot.id: snapshot for snapshot in snapshots}
    snapshot_ids = list(snapshot_by_id)
    predicates = [SignalRecord.metadata_["evaluation_snapshot_id"].as_string() == str(value) for value in snapshot_ids]
    if decision is not None:
        predicates.append(SignalRecord.metadata_["decision_id"].as_string() == str(decision.id))
    if not predicates:
        return []
    signals = session.scalars(
        select(SignalRecord).where(or_(*predicates)).order_by(SignalRecord.signal_time, SignalRecord.id)
    ).all()
    rows = []
    for signal in signals:
        try:
            metadata = SignalTraceMetadata.model_validate(signal.metadata_)
        except PydanticValidationError as error:
            raise ValidationError("signal trace metadata is invalid") from error
        if decision is not None and metadata.decision_id is not None and metadata.decision_id != decision.id:
            continue
        if metadata.account_scope != account_scope:
            raise ValidationError("signal trace account scope mismatch")
        if signal.strategy_id != version.strategy_id:
            raise ValidationError("signal trace strategy identity mismatch")
        linked_snapshot = None
        if metadata.evaluation_snapshot_id is not None:
            linked_snapshot = snapshot_by_id.get(metadata.evaluation_snapshot_id)
            if linked_snapshot is None:
                raise ValidationError("signal trace snapshot identity mismatch")
        if decision is None and metadata.decision_id is not None:
            linked_decision = session.get(DecisionRecord, metadata.decision_id)
            selected_ids = (
                set()
                if linked_decision is None
                else set(linked_decision.original_json.get("selected_evaluation_ids", []))
            )
            if (
                linked_decision is None
                or linked_decision.account_scope != account_scope
                or linked_decision.strategy_version_id != version.id
                or metadata.evaluation_snapshot_id is None
                or str(metadata.evaluation_snapshot_id) not in selected_ids
            ):
                raise ValidationError("signal trace dual identity mismatch")
        matching_snapshots = [linked_snapshot] if linked_snapshot is not None else snapshots
        if not any(
            row is not None
            and (signal.symbol, signal.market, signal.instrument) == (row.symbol, row.market, row.instrument)
            for row in matching_snapshots
        ):
            raise ValidationError("signal trace snapshot identity mismatch")
        rows.append(project_trace_signal(signal, metadata))
    return rows


def project_outcome_rows(outcomes):
    rows = []
    state_by_status = {"available": "Available", "provisional": "Provisional", "unavailable": "Unavailable"}
    for outcome in outcomes:
        state = state_by_status.get(outcome.status, "Unavailable")
        base = {
            "kind": outcome.kind,
            "horizon_key": outcome.horizon_key,
            "reason_code": outcome.reason_code,
        }
        actual = json.loads(canonical_json((outcome.metrics_json or {}).get("actual", {})))
        confidence = actual.pop("narrative_confidence", None)
        if confidence is not None:
            actual["Narrative confidence"] = confidence
            actual["confidence_note"] = "Not a probability of profit"
        rows.append(
            _trace_row(
                "Actual",
                state,
                timestamp_value=outcome.maturity_at,
                identities={"outcome_id": str(outcome.id), "content_sha256": outcome.content_sha256},
                values={**base, **actual},
            )
        )
        counterfactuals = (outcome.metrics_json or {}).get("counterfactual", [])
        if isinstance(counterfactuals, list):
            for item in sorted(
                (item for item in counterfactuals if isinstance(item, dict)),
                key=lambda value: str(value.get("assumption_version", "")),
            ):
                assumption_version = item.get("assumption_version")
                if not isinstance(assumption_version, str) or not assumption_version:
                    continue
                values = json.loads(canonical_json(item))
                values.pop("assumption_version", None)
                rows.append(
                    _trace_row(
                        f"Counterfactual · {assumption_version}",
                        state,
                        timestamp_value=outcome.maturity_at,
                        identities={"outcome_id": str(outcome.id), "content_sha256": outcome.content_sha256},
                        values={**base, "assumption_version": assumption_version, **values},
                    )
                )
    return rows


def load_trace_evidence(session, *, version, snapshots, account_scope, decision_id=None):
    snapshot_ids = [snapshot.id for snapshot in snapshots]
    outcome_scope = OutcomeRecord.evaluation_snapshot_id.in_(snapshot_ids)
    if decision_id is not None:
        outcome_scope = and_(
            outcome_scope,
            or_(OutcomeRecord.kind != "trade", OutcomeRecord.decision_id == decision_id),
        )
    outcomes = session.scalars(
        select(OutcomeRecord)
        .where(outcome_scope)
        .order_by(
            OutcomeRecord.maturity_at,
            OutcomeRecord.kind,
            OutcomeRecord.horizon_key,
            OutcomeRecord.revision_no,
            OutcomeRecord.id,
        )
    ).all()
    campaigns = session.scalars(
        select(ExperimentCampaign)
        .where(
            or_(
                ExperimentCampaign.incumbent_strategy_version_id == version.id,
                ExperimentCampaign.candidate_strategy_version_id == version.id,
            )
        )
        .order_by(ExperimentCampaign.created_at, ExperimentCampaign.id)
    ).all()
    campaign_ids = [campaign.id for campaign in campaigns]
    trials = (
        session.scalars(
            select(ExperimentRecord)
            .where(ExperimentRecord.campaign_id.in_(campaign_ids))
            .order_by(
                ExperimentRecord.campaign_id,
                ExperimentRecord.original_trial_id,
                ExperimentRecord.strategy_version_id,
                ExperimentRecord.ablation_arm,
                ExperimentRecord.id,
            )
        ).all()
        if campaign_ids
        else []
    )
    campaign_events = (
        session.scalars(
            select(CampaignEvent)
            .where(CampaignEvent.campaign_id.in_(campaign_ids))
            .order_by(CampaignEvent.campaign_id, CampaignEvent.created_at, CampaignEvent.id)
        ).all()
        if campaign_ids
        else []
    )
    holdouts = (
        session.scalars(
            select(HoldoutUse)
            .where(HoldoutUse.campaign_id.in_(campaign_ids))
            .order_by(HoldoutUse.campaign_id, HoldoutUse.consumed_at, HoldoutUse.id)
        ).all()
        if campaign_ids
        else []
    )
    reviews = (
        session.scalars(
            select(CampaignReview)
            .where(CampaignReview.campaign_id.in_(campaign_ids))
            .order_by(CampaignReview.campaign_id, CampaignReview.created_at, CampaignReview.id)
        ).all()
        if campaign_ids
        else []
    )
    release_events = session.scalars(
        select(StrategyReleaseEvent)
        .where(
            StrategyReleaseEvent.strategy_id == version.strategy_id,
            StrategyReleaseEvent.account_scope == account_scope,
            or_(
                StrategyReleaseEvent.candidate_version_id == version.id,
                StrategyReleaseEvent.from_version_id == version.id,
                StrategyReleaseEvent.to_version_id == version.id,
            ),
        )
        .order_by(StrategyReleaseEvent.resulting_revision, StrategyReleaseEvent.id)
    ).all()
    return {
        "outcome_rows": project_outcome_rows(outcomes),
        "campaigns": [
            {
                "id": str(campaign.id),
                "contract_sha256": campaign.contract_sha256,
                "incumbent_strategy_version_id": str(campaign.incumbent_strategy_version_id),
                "candidate_strategy_version_id": str(campaign.candidate_strategy_version_id),
                "hypothesis": campaign.hypothesis,
                "declared_difference_json": json.loads(canonical_json(campaign.declared_difference_json)),
                "contract_json": json.loads(canonical_json(campaign.contract_json)),
                "created_by": campaign.created_by,
                "created_at": iso_time(
                    stored_timestamp(campaign.created_at, "campaign.created_at"), "campaign.created_at"
                ),
            }
            for campaign in campaigns
        ],
        "campaign_trials": [
            {
                "id": str(trial.id),
                "campaign_id": str(trial.campaign_id),
                "strategy_version_id": str(trial.strategy_version_id),
                "original_trial_id": trial.original_trial_id,
                "trial_role": trial.trial_role,
                "ablation_arm": trial.ablation_arm,
                "paired_sample_key_sha256": trial.paired_sample_key_sha256,
                "input_sha256": trial.input_sha256,
                "result_sha256": trial.result_sha256,
                "terminal_state": trial.terminal_state,
                "terminal_reason_json": json.loads(canonical_json(trial.terminal_reason_json or {})),
                "metrics_json": json.loads(canonical_json(trial.metrics_json or {})),
                "completed_at": (
                    iso_time(stored_timestamp(trial.completed_at, "trial.completed_at"), "trial.completed_at")
                    if trial.completed_at is not None
                    else None
                ),
            }
            for trial in trials
        ],
        "campaign_events": [
            {
                "id": str(event.id),
                "campaign_id": str(event.campaign_id),
                "event_type": event.event_type,
                "payload_json": json.loads(canonical_json(event.payload_json)),
                "idempotency_sha256": event.idempotency_sha256,
                "created_at": iso_time(
                    stored_timestamp(event.created_at, "campaign_event.created_at"), "campaign_event.created_at"
                ),
            }
            for event in campaign_events
        ],
        "holdout_uses": [
            {
                "id": str(holdout.id),
                "campaign_id": str(holdout.campaign_id),
                "holdout_identity_sha256": holdout.holdout_identity_sha256,
                "campaign_contract_sha256": holdout.campaign_contract_sha256,
                "audit_json": json.loads(canonical_json(holdout.audit_json)),
                "consumed_at": iso_time(
                    stored_timestamp(holdout.consumed_at, "holdout.consumed_at"), "holdout.consumed_at"
                ),
            }
            for holdout in holdouts
        ],
        "campaign_reviews": [
            {
                "id": str(review.id),
                "campaign_id": str(review.campaign_id),
                "status": review.status,
                "input_sha256": review.input_sha256,
                "result_sha256": review.result_sha256,
                "holdout_use_id": None if review.holdout_use_id is None else str(review.holdout_use_id),
                "input_identity_json": json.loads(canonical_json(review.input_identity_json or {})),
                "result_json": json.loads(canonical_json(review.result_json)),
                "created_at": iso_time(stored_timestamp(review.created_at, "review.created_at"), "review.created_at"),
            }
            for review in reviews
        ],
        "release_events": [
            {
                "id": str(event.id),
                "event_type": event.event_type,
                "candidate_version_id": (
                    None if event.candidate_version_id is None else str(event.candidate_version_id)
                ),
                "from_version_id": None if event.from_version_id is None else str(event.from_version_id),
                "to_version_id": None if event.to_version_id is None else str(event.to_version_id),
                "source_release_event_id": (
                    None if event.source_release_event_id is None else str(event.source_release_event_id)
                ),
                "campaign_id": None if event.campaign_id is None else str(event.campaign_id),
                "review_id": None if event.review_id is None else str(event.review_id),
                "holdout_use_id": None if event.holdout_use_id is None else str(event.holdout_use_id),
                "reconciliation_id": (None if event.reconciliation_id is None else str(event.reconciliation_id)),
                "actor_id": event.actor_id,
                "request_sha256": event.request_sha256,
                "reason": event.reason,
                "gate_json": json.loads(canonical_json(event.gate_json)),
                "resulting_revision": event.resulting_revision,
                "created_at": iso_time(stored_timestamp(event.created_at, "release.created_at"), "release.created_at"),
            }
            for event in release_events
        ],
    }


def decision_trace(session, decision_id, *, principal):
    """One decision's trace: its run, evaluations, orders, fills, lots, reconciliations, events, signals, and stages."""
    decision = DecisionService(session).get(decision_id, principal=principal)
    run, version, manifest, snapshots = verify_complete_run(session, decision.evaluation_run_id)
    selected_ids = {uuid.UUID(value) for value in decision.original_json["selected_evaluation_ids"]}
    evaluations = []
    for snapshot in snapshots:
        revision_ids = list(snapshot.research_revision_ids)
        if revision_ids:
            revisions = [session.get(ResearchRevision, uuid.UUID(value)) for value in revision_ids]
            research = {
                "revision_ids": revision_ids,
                "revisions": [
                    {"id": str(revision.id), "content_sha256": revision.content_sha256} for revision in revisions
                ],
            }
        else:
            research = {"research_status": snapshot.recommendation_json["research_status"]}
        evaluations.append(
            {
                "id": str(snapshot.id),
                "content_sha256": snapshot.content_sha256,
                "symbol": snapshot.symbol,
                "market": snapshot.market,
                "instrument": snapshot.instrument,
                "status": snapshot.status,
                "selected": snapshot.id in selected_ids,
                "recommendation_json": json.loads(canonical_json(snapshot.recommendation_json)),
                "technical_json": json.loads(canonical_json(snapshot.technical_json)),
                "reason_codes": list(snapshot.reason_codes),
                "valid_until": (
                    iso_time(stored_timestamp(snapshot.valid_until, "snapshot.valid_until"), "snapshot.valid_until")
                    if snapshot.valid_until is not None
                    else None
                ),
                "research": research,
            }
        )
    scoped_orders = session.scalars(
        select(OrderRecord)
        .where(OrderRecord.account_scope == decision.account_scope)
        .order_by(OrderRecord.created_at, OrderRecord.id)
    ).all()
    orders = [
        order
        for order in scoped_orders
        if order.decision_id == decision.id
        or str(decision.id) in (order.protective_context_json or {}).get("source_decision_ids", [])
    ]
    linked_event_filters = []
    for order in orders:
        context = order.protective_context_json
        digest = context.get("dedupe_sha256") if isinstance(context, dict) else None
        if (
            order.decision_id is not None
            and isinstance(digest, str)
            and len(digest) == 64
            and context.get("origin") == order.order_origin
        ):
            linked_event_filters.append(
                and_(
                    DecisionEvent.decision_id == order.decision_id,
                    DecisionEvent.event_type == f"protective_{digest[:13]}",
                )
            )
    event_filter = DecisionEvent.decision_id == decision.id
    if linked_event_filters:
        event_filter = or_(event_filter, *linked_event_filters)
    events = session.scalars(
        select(DecisionEvent).where(event_filter).order_by(DecisionEvent.created_at, DecisionEvent.id)
    ).all()
    order_ids = [order.id for order in orders]
    fills = (
        session.scalars(
            select(OrderFillRecord)
            .where(OrderFillRecord.order_id.in_(order_ids))
            .order_by(OrderFillRecord.fill_time, OrderFillRecord.id)
        ).all()
        if order_ids
        else []
    )
    fill_ids = [fill.id for fill in fills]
    allocations = (
        session.scalars(
            select(FillAllocation)
            .where(FillAllocation.closing_fill_id.in_(fill_ids))
            .order_by(FillAllocation.created_at, FillAllocation.id)
        ).all()
        if fill_ids
        else []
    )
    lot_ids = {allocation.position_lot_id for allocation in allocations}
    for order in orders:
        lot_ids.update(uuid.UUID(value) for value in (order.protective_context_json or {}).get("source_lot_ids", []))
    lots = session.scalars(
        select(PositionLot)
        .where((PositionLot.opening_decision_id == decision.id) | (PositionLot.id.in_(lot_ids)))
        .order_by(PositionLot.opened_at, PositionLot.id)
    ).all()
    account_generations = sorted({order.account_generation for order in orders if order.account_generation})
    reconciliations = (
        session.scalars(
            select(AccountReconciliation)
            .where(
                AccountReconciliation.account_scope == decision.account_scope,
                AccountReconciliation.account_generation.in_(account_generations),
            )
            .order_by(AccountReconciliation.as_of, AccountReconciliation.id)
        ).all()
        if account_generations
        else []
    )
    selected_snapshots = [snapshot for snapshot in snapshots if snapshot.id in selected_ids]
    selected_snapshot_ids = [snapshot.id for snapshot in selected_snapshots]
    signals = load_trace_signals(
        session,
        account_scope=decision.account_scope,
        version=version,
        snapshots=selected_snapshots,
        decision=decision,
    )
    evidence = load_trace_evidence(
        session,
        version=version,
        snapshots=selected_snapshots,
        account_scope=decision.account_scope,
        decision_id=decision.id,
    )
    trace = {
        "decision": {
            **_decision_response(decision),
            "evaluation_run_id": str(decision.evaluation_run_id),
            "strategy_version_id": str(decision.strategy_version_id),
            "account_scope": decision.account_scope,
            "decision_as_of": iso_time(stored_timestamp(decision.decision_as_of, "decision_as_of"), "decision_as_of"),
            "original_json": json.loads(canonical_json(decision.original_json)),
            "portfolio_snapshot_json": json.loads(canonical_json(decision.portfolio_snapshot_json)),
            "risk_snapshot_json": json.loads(canonical_json(decision.risk_snapshot_json)),
        },
        "decisions": [
            {
                **_decision_response(decision),
                "evaluation_run_id": str(decision.evaluation_run_id),
                "strategy_version_id": str(decision.strategy_version_id),
                "account_scope": decision.account_scope,
                "decision_as_of": iso_time(
                    stored_timestamp(decision.decision_as_of, "decision_as_of"), "decision_as_of"
                ),
            }
        ],
        "strategy_version": {
            "id": str(version.id),
            "content_sha256": version.content_sha256,
            "policy_sha256": content_sha256(version.policy_json),
        },
        "evaluation_run": {
            "id": str(run.id),
            "input_sha256": run.input_sha256,
            "coverage_json": json.loads(canonical_json(run.coverage_json)),
            "decision_as_of": iso_time(stored_timestamp(run.decision_as_of, "decision_as_of"), "decision_as_of"),
        },
        "manifest": {
            "id": str(manifest.id),
            "content_sha256": manifest.content_sha256,
            "created_at": (
                iso_time(stored_timestamp(manifest.created_at, "manifest.created_at"), "manifest.created_at")
                if manifest.created_at is not None
                else None
            ),
        },
        "evaluations": evaluations,
        "trace_evaluation_ids": [str(snapshot_id) for snapshot_id in selected_snapshot_ids],
        "execution_key": str(decision.execution_key) if decision.execution_key is not None else None,
        "execution_keys": sorted(
            {
                str(value)
                for value in (decision.execution_key, *(order.execution_key for order in orders))
                if value is not None
            }
        ),
        "orders": [
            {
                "id": str(order.id),
                "decision_id": str(order.decision_id) if order.decision_id is not None else None,
                "execution_key": str(order.execution_key) if order.execution_key is not None else None,
                "client_order_ref": order.client_order_ref,
                "intent_sha256": order.intent_sha256,
                "order_origin": order.order_origin,
                "market": order.market,
                "symbol": order.symbol,
                "instrument": order.instrument,
                "side": order.side,
                "action": order.action,
                "quantity": order.quantity,
                "status": order.status,
                "reconciliation_status": order.reconciliation_status,
                "reservation_status": order.reservation_status,
                "reserved_quantity": order.reserved_quantity,
                "submit_attempted_at": (
                    iso_time(stored_timestamp(order.submit_attempted_at, "submit_attempted_at"), "submit_attempted_at")
                    if order.submit_attempted_at is not None
                    else None
                ),
                "protective_context_json": (
                    json.loads(canonical_json(order.protective_context_json))
                    if order.protective_context_json is not None
                    else None
                ),
            }
            for order in orders
        ],
        "fills": [
            {
                "id": str(fill.id),
                "order_id": str(fill.order_id),
                "broker_fill_id": fill.broker_fill_id,
                "fill_quantity": fill.fill_quantity,
                "fill_price": fill.fill_price,
                "fill_time": iso_time(stored_timestamp(fill.fill_time, "fill_time"), "fill_time"),
                "projection_status": fill.projection_status,
            }
            for fill in fills
        ],
        "lots": [
            {
                "id": str(lot.id),
                "opening_fill_id": str(lot.opening_fill_id),
                "opening_decision_id": str(lot.opening_decision_id),
                "original_quantity": lot.original_quantity,
                "open_quantity": lot.open_quantity,
                "reserved_close_quantity": lot.reserved_close_quantity,
                "cost_basis_sha256": content_sha256(lot.cost_basis_json),
                "opened_at": iso_time(stored_timestamp(lot.opened_at, "lot.opened_at"), "lot.opened_at"),
            }
            for lot in lots
        ],
        "allocations": [
            {
                "id": str(allocation.id),
                "closing_fill_id": str(allocation.closing_fill_id),
                "position_lot_id": str(allocation.position_lot_id),
                "closing_decision_id": str(allocation.closing_decision_id),
                "quantity": allocation.quantity,
                "realized_cost_sha256": content_sha256(allocation.realized_cost_json),
                "created_at": iso_time(
                    stored_timestamp(allocation.created_at, "allocation.created_at"),
                    "allocation.created_at",
                ),
            }
            for allocation in allocations
        ],
        "reconciliations": [
            {
                "id": str(row.id),
                "status": row.status,
                "policy_sha256": row.policy_sha256,
                "broker_state_watermark": row.broker_state_watermark,
                "internal_state_watermark": row.internal_state_watermark,
                "broker_snapshot_sha256": row.broker_snapshot_sha256,
                "as_of": iso_time(stored_timestamp(row.as_of, "reconciliation.as_of"), "reconciliation.as_of"),
            }
            for row in reconciliations
        ],
        "events": [
            {
                "id": str(event.id),
                "event_type": event.event_type,
                "actor_id": event.actor_id,
                "expected_revision": event.expected_revision,
                "idempotency_key": event.idempotency_key,
                "request_sha256": event.request_sha256,
                "payload_json": json.loads(canonical_json(event.payload_json)),
                "created_at": (
                    iso_time(stored_timestamp(event.created_at, "event.created_at"), "event.created_at")
                    if event.created_at is not None
                    else None
                ),
            }
            for event in events
        ],
        "signals": signals,
        **evidence,
    }
    trace["stages"] = build_trace_stages(trace)
    return trace
