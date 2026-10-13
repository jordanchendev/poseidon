"""Frozen portfolio decisions and human-only terminal transitions."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)
from sqlalchemy import and_, or_, select, text

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.evaluation import TERMINAL_STATUSES, verify_complete_run
from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    iso_time,
    required_text,
    timestamp,
    validate_manifest,
)
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
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
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion
from poseidon.signals.schemas import SignalTraceMetadata

ACTION_VALUES = frozenset({"watch", "hold", "enter", "add", "reduce", "exit", "reject"})
EXECUTABLE_ACTIONS = frozenset({"enter", "add", "reduce", "exit"})
LIVE_STATUSES = frozenset({"pending_approval", "approved"})
SYSTEM_ACTOR = "system:decision-service"
CLAIM_SCAN_LIMIT = 100
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveStrictFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeStrictFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
TRACE_STAGE_NAMES = (
    "signal_or_no_trade",
    "manifest",
    "research_revision",
    "evaluation",
    "decision_events",
    "execution",
    "outcomes",
    "campaign_review",
    "release",
)


def _trace_row(label, state, *, timestamp_value=None, identities=None, values=None):
    return {
        "label": label,
        "state": state,
        "timestamp": timestamp_value,
        "identities": identities or {},
        "values": values or {},
    }


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
        timestamp_value=iso_time(_stored_time(signal.signal_time, "signal.signal_time"), "signal.signal_time"),
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
        if metadata.account_scope != account_scope:
            raise ValidationError("signal trace account scope mismatch")
        if signal.strategy_id != version.strategy_id:
            raise ValidationError("signal trace strategy identity mismatch")
        linked_snapshot = None
        if metadata.evaluation_snapshot_id is not None:
            linked_snapshot = snapshot_by_id.get(metadata.evaluation_snapshot_id)
            if linked_snapshot is None:
                raise ValidationError("signal trace snapshot identity mismatch")
        if decision is not None:
            if metadata.decision_id is not None and metadata.decision_id != decision.id:
                raise ValidationError("signal trace decision identity mismatch")
        elif metadata.decision_id is not None:
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


def load_trace_evidence(session, *, version, snapshots, account_scope):
    snapshot_ids = [snapshot.id for snapshot in snapshots]
    outcomes = session.scalars(
        select(OutcomeRecord)
        .where(OutcomeRecord.evaluation_snapshot_id.in_(snapshot_ids))
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
                "created_at": iso_time(_stored_time(campaign.created_at, "campaign.created_at"), "campaign.created_at"),
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
                    iso_time(_stored_time(trial.completed_at, "trial.completed_at"), "trial.completed_at")
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
                    _stored_time(event.created_at, "campaign_event.created_at"), "campaign_event.created_at"
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
                    _stored_time(holdout.consumed_at, "holdout.consumed_at"), "holdout.consumed_at"
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
                "created_at": iso_time(_stored_time(review.created_at, "review.created_at"), "review.created_at"),
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
                "created_at": iso_time(_stored_time(event.created_at, "release.created_at"), "release.created_at"),
            }
            for event in release_events
        ],
    }


def build_trace_stages(trace):
    evaluations = trace["evaluations"]
    signals = trace.get("signals", [])
    decisions = trace.get("decisions", [])
    if not decisions and trace.get("decision") is not None:
        decisions = [trace["decision"]]
    relevant_ids = set(trace.get("trace_evaluation_ids", []))
    if not relevant_ids:
        relevant_ids = {row["id"] for row in evaluations if row.get("selected")}
    relevant_evaluations = [row for row in evaluations if not relevant_ids or row["id"] in relevant_ids]
    no_trade_evaluations = [row for row in relevant_evaluations if row["status"] == "no_trade"]
    no_trade = bool(no_trade_evaluations)
    if signals:
        signal_state, signal_reason = "Available", "Stored signal carries exact trace identity"
    elif no_trade:
        signal_state, signal_reason = "No trade", "Evaluation completed without an executable trade"
    elif not decisions and relevant_evaluations:
        signal_state, signal_reason = "Unavailable", "No stored signal is linked to the evaluated evidence"
    else:
        signal_state, signal_reason = "Legacy unresolved", "No stored signal or no-trade identity links this trace"
    no_trade_rows = [
        _trace_row(
            "No trade",
            "No trade",
            timestamp_value=trace["evaluation_run"]["decision_as_of"],
            identities={
                "evaluation_snapshot_id": evaluation["id"],
                "content_sha256": evaluation["content_sha256"],
            },
            values={
                "status": evaluation["status"],
                "reason_codes": evaluation["reason_codes"],
            },
        )
        for evaluation in no_trade_evaluations
    ]
    if not decisions:
        decision_state = "Not required" if no_trade else "Unavailable"
        decision_reason = (
            "No decision was required for no-trade evidence"
            if no_trade
            else "No decision is linked to the evaluated evidence"
        )
    elif any(row["status"] == "pending_approval" for row in decisions):
        decision_state = "Provisional"
        decision_reason = "Decision approval is pending"
    elif all(row["status"] in {"rejected", "expired", "superseded"} for row in decisions):
        decision_state = "Blocked"
        decision_reason = "Decision terminal state prevents execution"
    else:
        decision_state = "Available"
        decision_reason = "Decisions and ordered events are available"
    executable_decisions = [
        row for row in decisions if row.get("final_json", {}).get("final_action") in EXECUTABLE_ACTIONS
    ]
    if trace["orders"]:
        execution_state = "Filled" if trace["fills"] else "Available"
        execution_reason = "Orders, fills, and reconciliation remain separate"
    elif not decisions:
        execution_state = "Not required" if no_trade else "Unavailable"
        execution_reason = (
            "No order was required for no-trade evidence" if no_trade else "No decision is linked to execution evidence"
        )
    elif not executable_decisions:
        execution_state = "Not required"
        execution_reason = "The final decision action is not executable"
    elif any(row["status"] == "pending_approval" for row in executable_decisions):
        execution_state = "Provisional"
        execution_reason = "Executable decision is awaiting approval"
    elif any(row["status"] in {"rejected", "expired", "superseded"} for row in executable_decisions):
        execution_state = "Blocked"
        execution_reason = "Executable decision cannot proceed"
    else:
        execution_state = "Unavailable"
        execution_reason = "Approved executable decision has no stored order"
    research_rows = [
        _trace_row(
            "Research revision",
            "Available",
            identities={"research_revision_id": revision["id"], "content_sha256": revision["content_sha256"]},
        )
        for evaluation in evaluations
        for revision in evaluation["research"].get("revisions", [])
    ]
    if research_rows:
        research_state, research_reason = "Available", "Exact stored research revisions are linked"
    elif all(evaluation["research"].get("research_status") == "not_required" for evaluation in evaluations):
        research_state, research_reason = "Not required", "Research was explicitly not required"
    else:
        research_state, research_reason = "Unavailable", "Research evidence is unavailable"
    outcome_rows = trace.get("outcome_rows", [])
    outcome_state = (
        "Unavailable"
        if not outcome_rows
        else "Provisional"
        if any(row["state"] == "Provisional" for row in outcome_rows)
        else "Unavailable"
        if all(row["state"] == "Unavailable" for row in outcome_rows)
        else "Available"
    )
    reviews = trace.get("campaign_reviews", [])
    campaign_events = trace.get("campaign_events", [])
    terminal_states = {row["terminal_state"] for row in trace.get("campaign_trials", [])}
    campaign_state = (
        "Unavailable"
        if not trace.get("campaigns")
        else "Blocked"
        if any(row["status"] == "failed" for row in reviews)
        or any(row["event_type"] == "holdout_reuse_denied" for row in campaign_events)
        or bool(
            terminal_states
            - {
                "succeeded",
                "statistically_inconclusive",
                "capability_unavailable",
            }
        )
        else "Inconclusive"
        if any(row["status"] == "inconclusive" for row in reviews) or "statistically_inconclusive" in terminal_states
        else "Unavailable"
        if not reviews
        or any(row["status"] == "unavailable" for row in reviews)
        or "capability_unavailable" in terminal_states
        else "Available"
    )
    release_events = trace.get("release_events", [])
    last_release = release_events[-1] if release_events else None
    release_state = {
        "release": "Released",
        "pause": "Paused",
        "rollback": "Rolled back",
        "reject": "Rejected",
    }.get(None if last_release is None else last_release["event_type"], "Unavailable")
    stages = [
        {
            "position": 1,
            "name": "signal_or_no_trade",
            "state": signal_state,
            "reason": signal_reason,
            "timestamp": signals[0]["timestamp"]
            if signals
            else no_trade_rows[0]["timestamp"]
            if no_trade_rows
            else None,
            "identities": (
                {} if len(no_trade_evaluations) != 1 else {"evaluation_snapshot_id": no_trade_evaluations[0]["id"]}
            ),
            "rows": signals if signals else no_trade_rows,
            "raw_payload": None,
        },
        {
            "position": 2,
            "name": "manifest",
            "state": "Available",
            "reason": "Frozen manifest identity is available",
            "timestamp": trace["manifest"].get("created_at"),
            "identities": {
                "manifest_id": trace["manifest"]["id"],
                "content_sha256": trace["manifest"]["content_sha256"],
            },
            "rows": [],
            "raw_payload": None,
        },
        {
            "position": 3,
            "name": "research_revision",
            "state": research_state,
            "reason": research_reason,
            "timestamp": None,
            "identities": {},
            "rows": research_rows,
            "raw_payload": None,
        },
        {
            "position": 4,
            "name": "evaluation",
            "state": "Available",
            "reason": "Immutable evaluation run and snapshots are available",
            "timestamp": trace["evaluation_run"]["decision_as_of"],
            "identities": {
                "evaluation_run_id": trace["evaluation_run"]["id"],
                "input_sha256": trace["evaluation_run"]["input_sha256"],
            },
            "rows": [],
            "raw_payload": evaluations,
        },
        {
            "position": 5,
            "name": "decision_events",
            "state": decision_state,
            "reason": decision_reason,
            "timestamp": None if not decisions else decisions[0]["decision_as_of"],
            "identities": {} if len(decisions) != 1 else {"decision_id": decisions[0]["id"]},
            "rows": [
                *[
                    _trace_row(
                        "Decision",
                        "Available",
                        timestamp_value=row["decision_as_of"],
                        identities={"decision_id": row["id"]},
                        values={
                            "status": row["status"],
                            "revision": row["revision"],
                            "account_scope": row["account_scope"],
                        },
                    )
                    for row in decisions
                ],
                *[
                    _trace_row(
                        "Decision event",
                        "Available",
                        timestamp_value=row["created_at"],
                        identities={"decision_event_id": row["id"]},
                        values={key: value for key, value in row.items() if key not in {"id", "created_at"}},
                    )
                    for row in trace["events"]
                ],
            ],
            "raw_payload": None,
        },
        {
            "position": 6,
            "name": "execution",
            "state": execution_state,
            "reason": execution_reason,
            "timestamp": None,
            "identities": {},
            "rows": [
                *[
                    _trace_row(
                        "Order",
                        "Filled" if row["status"] == "filled" else "Available",
                        identities={"order_id": row["id"]},
                        values={key: value for key, value in row.items() if key != "id"},
                    )
                    for row in trace["orders"]
                ],
                *[
                    _trace_row(
                        "Fill",
                        "Filled",
                        timestamp_value=row["fill_time"],
                        identities={"fill_id": row["id"], "order_id": row["order_id"]},
                        values={key: value for key, value in row.items() if key not in {"id", "order_id", "fill_time"}},
                    )
                    for row in trace["fills"]
                ],
                *[
                    _trace_row(
                        "Reconciliation",
                        "Available" if row["status"] == "matched" else "Provisional",
                        timestamp_value=row["as_of"],
                        identities={"reconciliation_id": row["id"]},
                        values={key: value for key, value in row.items() if key not in {"id", "as_of"}},
                    )
                    for row in trace["reconciliations"]
                ],
            ],
            "raw_payload": None,
        },
        {
            "position": 7,
            "name": "outcomes",
            "state": outcome_state,
            "reason": "Outcome facts are unavailable"
            if not outcome_rows
            else "Actual and counterfactual facts remain separate",
            "timestamp": None,
            "identities": {},
            "rows": outcome_rows,
            "raw_payload": None,
        },
        {
            "position": 8,
            "name": "campaign_review",
            "state": campaign_state,
            "reason": "No frozen campaign is linked"
            if not trace.get("campaigns")
            else "Frozen campaigns and reviews are ordered",
            "timestamp": None,
            "identities": {},
            "rows": [
                *[
                    _trace_row(
                        "Campaign",
                        "Available",
                        timestamp_value=row["created_at"],
                        identities={"campaign_id": row["id"], "contract_sha256": row["contract_sha256"]},
                        values={key: value for key, value in row.items() if key not in {"id", "created_at"}},
                    )
                    for row in trace.get("campaigns", [])
                ],
                *[
                    _trace_row(
                        "Paired cell",
                        {
                            "succeeded": "Available",
                            "statistically_inconclusive": "Inconclusive",
                            "capability_unavailable": "Unavailable",
                        }.get(row["terminal_state"], "Blocked"),
                        timestamp_value=row["completed_at"],
                        identities={
                            "experiment_id": row["id"],
                            "campaign_id": row["campaign_id"],
                            "strategy_version_id": row["strategy_version_id"],
                            "paired_sample_key_sha256": row["paired_sample_key_sha256"],
                            "input_sha256": row["input_sha256"],
                            "result_sha256": row["result_sha256"],
                        },
                        values={
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "id",
                                "campaign_id",
                                "strategy_version_id",
                                "paired_sample_key_sha256",
                                "input_sha256",
                                "result_sha256",
                                "completed_at",
                            }
                        },
                    )
                    for row in trace.get("campaign_trials", [])
                ],
                *[
                    _trace_row(
                        "Campaign event",
                        "Blocked" if row["event_type"] == "holdout_reuse_denied" else "Available",
                        timestamp_value=row["created_at"],
                        identities={
                            "campaign_event_id": row["id"],
                            "campaign_id": row["campaign_id"],
                            "idempotency_sha256": row["idempotency_sha256"],
                        },
                        values={"event_type": row["event_type"], "payload_json": row["payload_json"]},
                    )
                    for row in trace.get("campaign_events", [])
                ],
                *[
                    _trace_row(
                        "Holdout use",
                        "Available",
                        timestamp_value=row["consumed_at"],
                        identities={
                            "holdout_use_id": row["id"],
                            "campaign_id": row["campaign_id"],
                            "holdout_identity_sha256": row["holdout_identity_sha256"],
                            "campaign_contract_sha256": row["campaign_contract_sha256"],
                        },
                        values={"audit_json": row["audit_json"]},
                    )
                    for row in trace.get("holdout_uses", [])
                ],
                *[
                    _trace_row(
                        "Campaign review",
                        "Inconclusive"
                        if row["status"] == "inconclusive"
                        else "Blocked"
                        if row["status"] == "failed"
                        else "Unavailable"
                        if row["status"] == "unavailable"
                        else "Available",
                        timestamp_value=row["created_at"],
                        identities={
                            key: value
                            for key, value in {
                                "review_id": row["id"],
                                "campaign_id": row["campaign_id"],
                                "input_sha256": row["input_sha256"],
                                "result_sha256": row["result_sha256"],
                                "holdout_use_id": row["holdout_use_id"],
                            }.items()
                            if value is not None
                        },
                        values={
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "id",
                                "campaign_id",
                                "input_sha256",
                                "result_sha256",
                                "holdout_use_id",
                                "created_at",
                            }
                        },
                    )
                    for row in reviews
                ],
            ],
            "raw_payload": None,
        },
        {
            "position": 9,
            "name": "release",
            "state": release_state,
            "reason": "No release event is linked" if last_release is None else "Immutable release history is ordered",
            "timestamp": None if last_release is None else last_release["created_at"],
            "identities": {
                "strategy_version_id": trace["strategy_version"]["id"],
                "content_sha256": trace["strategy_version"]["content_sha256"],
            },
            "rows": [
                _trace_row(
                    "Release event",
                    {
                        "release": "Released",
                        "pause": "Paused",
                        "rollback": "Rolled back",
                        "reject": "Rejected",
                    }[row["event_type"]],
                    timestamp_value=row["created_at"],
                    identities={
                        key: value
                        for key, value in {
                            "release_event_id": row["id"],
                            "candidate_version_id": row["candidate_version_id"],
                            "from_version_id": row["from_version_id"],
                            "to_version_id": row["to_version_id"],
                            "source_release_event_id": row["source_release_event_id"],
                            "campaign_id": row["campaign_id"],
                            "review_id": row["review_id"],
                            "holdout_use_id": row["holdout_use_id"],
                            "reconciliation_id": row["reconciliation_id"],
                            "request_sha256": row["request_sha256"],
                        }.items()
                        if value is not None
                    },
                    values={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "id",
                            "candidate_version_id",
                            "from_version_id",
                            "to_version_id",
                            "source_release_event_id",
                            "campaign_id",
                            "review_id",
                            "holdout_use_id",
                            "reconciliation_id",
                            "request_sha256",
                            "created_at",
                        }
                    },
                )
                for row in release_events
            ],
            "raw_payload": None,
        },
    ]
    assert tuple(stage["name"] for stage in stages) == TRACE_STAGE_NAMES
    return stages


class DecisionNotFoundError(LookupError):
    pass


class DecisionConflictError(RuntimeError):
    pass


class HardLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_gross_exposure: PositiveFiniteFloat
    max_position_weight: PositiveFiniteFloat

    @field_validator("max_gross_exposure", "max_position_weight", mode="before")
    @classmethod
    def reject_boolean_limits(cls, value):
        if isinstance(value, bool):
            raise ValueError("hard limits must be numeric")
        return value


class ProtectiveExit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_without_approval: StrictBool


class ReleaseGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_mature_samples: PositiveStrictInt
    requires_human_release: StrictBool


class FreshnessRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_age_seconds: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class PerpInstrumentRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity_step: PositiveStrictFiniteFloat
    contract_multiplier: PositiveStrictFiniteFloat
    margin_semantics: Annotated[str, Field(min_length=1)]
    funding_semantics: Annotated[str, Field(min_length=1)]

    @field_validator("margin_semantics", "funding_semantics")
    @classmethod
    def require_trimmed_semantics(cls, value):
        if value != value.strip():
            raise ValueError("semantics must be non-empty trimmed text")
        return value


class ReconciliationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_generation: Annotated[str, Field(min_length=1)]
    opening_cash: NonNegativeStrictFiniteFloat
    currency: Annotated[str, Field(min_length=3, max_length=10, pattern=r"^[A-Z][A-Z0-9]+$")]
    max_reconciliation_age_seconds: PositiveStrictInt
    cash_tolerance: NonNegativeStrictFiniteFloat
    position_tolerance: NonNegativeStrictFiniteFloat
    fill_tolerance: NonNegativeStrictFiniteFloat
    tw_stock_quantity_rounding: Literal["whole_share_floor"]
    perp_instrument_rules: dict[str, PerpInstrumentRule]

    @field_validator("account_generation")
    @classmethod
    def require_trimmed_generation(cls, value):
        if value != value.strip():
            raise ValueError("account_generation must be non-empty trimmed text")
        return value

    @field_validator("perp_instrument_rules")
    @classmethod
    def require_canonical_instrument_keys(cls, rules):
        if any(not instrument.strip() or instrument != instrument.strip() for instrument in rules):
            raise ValueError("perp instrument keys must be non-empty trimmed text")
        return rules


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: Annotated[str, Field(min_length=1)]
    mode: Literal["human", "assisted", "automatic"]
    market: Literal["tw_stock", "crypto_perp"]
    account_scope: Annotated[str, Field(min_length=1)]
    universe_id: Annotated[str, Field(min_length=1)]
    decision_ttl_seconds: PositiveStrictInt
    required_data: Annotated[dict[str, FreshnessRule], Field(min_length=1)]
    hard_limits: HardLimits
    approval_roles: Annotated[list[str], Field(min_length=1)]
    protective_exit: ProtectiveExit
    release_gate: ReleaseGate
    reconciliation: ReconciliationPolicy | None = None

    @field_validator("policy_id", "account_scope", "universe_id")
    @classmethod
    def require_trimmed_text(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("must be non-empty trimmed text")
        return value

    @field_validator("required_data")
    @classmethod
    def require_data_kinds(cls, rules):
        if any(not kind.strip() or kind != kind.strip() for kind in rules):
            raise ValueError("required data kinds must be non-empty trimmed text")
        return rules

    @field_validator("approval_roles")
    @classmethod
    def require_portfolio_manager(cls, roles):
        if len(set(roles)) != len(roles) or any(not role.strip() or role != role.strip() for role in roles):
            raise ValueError("approval roles must be unique non-empty text")
        if "portfolio_manager" not in roles:
            raise ValueError("portfolio_manager approval is required")
        return roles

    @model_validator(mode="after")
    def require_perp_rules(self):
        if self.market == "crypto_perp" and (
            self.reconciliation is None or not self.reconciliation.perp_instrument_rules
        ):
            raise ValueError("crypto_perp requires perp_instrument_rules")
        return self


class ExecutableIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_snapshot_id: uuid.UUID
    symbol: Annotated[str, Field(min_length=1)]
    market: Literal["tw_stock", "crypto_perp"]
    instrument: Annotated[str, Field(min_length=1)]
    action: Literal["enter", "add", "reduce", "exit"]
    side: Literal["long", "short"]
    target_weight: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    order_type: Literal["market"] = "market"

    @field_validator("symbol", "instrument")
    @classmethod
    def require_trimmed_identity(cls, value):
        if value != value.strip():
            raise ValueError("identity must be non-empty trimmed text")
        return value


class ApprovalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: PositiveStrictInt
    final_action: Literal["watch", "hold", "enter", "add", "reduce", "exit", "reject"] | None = None
    override_reason: str | None = None


class RejectionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: PositiveStrictInt
    reason: Annotated[str, Field(min_length=1)]

    @field_validator("reason")
    @classmethod
    def require_reason(cls, value):
        if not value.strip():
            raise ValueError("rejection reason must be non-empty")
        return value


def _json_object(value, field):
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be a JSON object")
    return json.loads(canonical_json(value))


def _validate_selection(snapshots, selected_values, final_action):
    if not isinstance(selected_values, list) or not selected_values:
        raise ValidationError("selected_evaluation_ids must be a non-empty list")
    try:
        selected_ids = [uuid.UUID(value) for value in selected_values]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError("selected_evaluation_ids must contain UUID strings") from error
    if len(set(selected_ids)) != len(selected_ids):
        raise ValidationError("selected_evaluation_ids must be unique")
    if not isinstance(final_action, str) or final_action not in ACTION_VALUES:
        raise ValidationError("final_action is invalid")
    snapshot_by_id = {row.id: row for row in snapshots}
    try:
        selected = [snapshot_by_id[selected_id] for selected_id in selected_ids]
    except KeyError as error:
        raise ValidationError("selected evaluation does not belong to the run") from error
    if any(row.status not in TERMINAL_STATUSES for row in selected):
        raise ValidationError("selected evaluation is not terminal")
    if final_action in {"enter", "add"} and any(row.status != "evaluated" for row in selected):
        raise ValidationError("exposure increase requires evaluated selections")
    return selected


def _validate_order_intents(selected, final_json, policy):
    final_action = final_json.get("final_action")
    raw_intents = final_json.get("order_intents")
    if final_action not in EXECUTABLE_ACTIONS:
        if raw_intents not in (None, []):
            raise ValidationError("non-executable action must not contain order_intents")
        return []
    if not isinstance(raw_intents, list) or not raw_intents:
        raise ValidationError("executable decision requires order_intents")
    try:
        intents = [ExecutableIntent.model_validate(value) for value in raw_intents]
    except PydanticValidationError as error:
        raise ValidationError("order_intents are invalid") from error

    intent_ids = [intent.evaluation_snapshot_id for intent in intents]
    selected_by_id = {row.id: row for row in selected}
    if len(set(intent_ids)) != len(intent_ids):
        raise ValidationError("order_intents require one unique selected snapshot")
    if set(intent_ids) != set(selected_by_id):
        raise ValidationError("order_intents must reference every selected snapshot exactly once")
    for intent in intents:
        snapshot = selected_by_id[intent.evaluation_snapshot_id]
        if (intent.symbol, intent.market, intent.instrument) != (
            snapshot.symbol,
            snapshot.market,
            snapshot.instrument,
        ):
            raise ValidationError("order intent does not match snapshot identity")
        if intent.market == "crypto_perp" and (
            policy.reconciliation is None or intent.instrument not in policy.reconciliation.perp_instrument_rules
        ):
            raise ValidationError("order intent requires an owner-frozen perp instrument rule")
        if intent.side != snapshot.recommendation_json.get("side"):
            raise ValidationError("order intent does not match snapshot side")
        if intent.action != final_action:
            raise ValidationError("order intent action does not match final_action")
        if intent.action in {"enter", "add"} and intent.target_weight <= 0:
            raise ValidationError("enter/add target_weight must be positive")
        if intent.action == "exit" and intent.target_weight != 0:
            raise ValidationError("exit target_weight must be zero")
    return sorted(
        intents,
        key=lambda intent: (
            intent.symbol,
            intent.market,
            intent.instrument,
            intent.side,
            str(intent.evaluation_snapshot_id),
        ),
    )


def _stored_time(value, field):
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return timestamp(value, field)


def _decision_response(decision):
    return {
        "id": str(decision.id),
        "status": decision.status,
        "revision": decision.revision,
        "creation_sha256": decision.creation_sha256,
        "policy_sha256": decision.policy_sha256,
        "valid_until": iso_time(_stored_time(decision.valid_until, "valid_until"), "valid_until"),
        "final_json": json.loads(canonical_json(decision.final_json)),
    }


class DecisionService:
    def __init__(self, session):
        self.session = session

    def _lock_scope(self, account_scope, strategy_version_id):
        if self.session.get_bind().dialect.name == "postgresql":
            lock_key = int(
                content_sha256({"account_scope": account_scope, "strategy_version_id": str(strategy_version_id)})[:16],
                16,
            )
            if lock_key >= 2**63:
                lock_key -= 2**64
            self.session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})

    def _append_event(
        self,
        decision,
        event_type,
        actor_id,
        expected_revision,
        *,
        idempotency_key=None,
        request_sha256=None,
        payload_json=None,
    ):
        event = DecisionEvent(
            decision_id=decision.id,
            event_type=event_type,
            actor_id=actor_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            payload_json=payload_json or {},
        )
        self.session.add(event)
        return event

    def _policy_is_current(self, decision):
        version = self.session.get(StrategyVersion, decision.strategy_version_id)
        if version is None:
            return False
        try:
            version.verify_content()
            DecisionPolicy.model_validate(version.policy_json)
        except (TypeError, ValueError, PydanticValidationError):
            return False
        return content_sha256(version.policy_json) == decision.policy_sha256

    def _authorized_decision(self, decision_id, principal, *, role=None):
        if role is not None:
            principal.require_role(role)
        decision = self.session.get(DecisionRecord, decision_id)
        if decision is None:
            raise DecisionNotFoundError("decision does not exist")
        principal.require_account_scope(decision.account_scope)
        return decision

    def _idempotency_event(self, decision_id, idempotency_key):
        return (
            self.session.query(DecisionEvent)
            .filter_by(decision_id=decision_id, idempotency_key=idempotency_key)
            .one_or_none()
        )

    @staticmethod
    def _replay_event(event, request_sha256):
        if event.request_sha256 != request_sha256:
            raise DecisionConflictError("idempotency key was used for a different request")
        response = event.payload_json.get("response")
        if not isinstance(response, dict):
            raise DecisionConflictError("stored idempotency response is invalid")
        return json.loads(canonical_json(response))

    def _transition(
        self,
        operation,
        decision_id,
        raw_body,
        body_type,
        *,
        principal,
        idempotency_key,
        now=None,
    ):
        self._authorized_decision(decision_id, principal, role="portfolio_manager")
        required_text(idempotency_key, "Idempotency-Key")
        if len(idempotency_key) > 256:
            raise ValidationError("Idempotency-Key is too long")
        body = body_type.model_validate(raw_body)
        body_json = body.model_dump(mode="json")
        request_sha256 = content_sha256(
            {
                "operation": operation,
                "decision_id": str(decision_id),
                "expected_revision": body.expected_revision,
                "actor_id": principal.actor_id,
                "body": body_json,
            }
        )
        event = self._idempotency_event(decision_id, idempotency_key)
        if event is not None:
            return self._replay_event(event, request_sha256)

        decision = (
            self.session.query(DecisionRecord)
            .filter(DecisionRecord.id == decision_id)
            .with_for_update()
            .populate_existing()
            .one_or_none()
        )
        if decision is None:
            raise DecisionNotFoundError("decision does not exist")
        event = self._idempotency_event(decision_id, idempotency_key)
        if event is not None:
            return self._replay_event(event, request_sha256)

        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        if decision.status != "pending_approval":
            raise DecisionConflictError("decision is not pending approval")
        if decision.revision != body.expected_revision:
            raise DecisionConflictError("decision revision changed")
        if current_time >= _stored_time(decision.valid_until, "valid_until"):
            raise DecisionConflictError("decision has expired")
        if not self._policy_is_current(decision):
            raise DecisionConflictError("decision policy changed")

        if operation == "approve":
            hard_failures = decision.risk_snapshot_json.get("hard_failures")
            allowed_actions = decision.risk_snapshot_json.get("allowed_actions")
            if not isinstance(hard_failures, list) or hard_failures:
                raise DecisionConflictError("hard risk failures prevent approval")
            final_json = json.loads(canonical_json(decision.original_json))
            final_action = body.final_action or final_json.get("final_action")
            if not isinstance(allowed_actions, list) or final_action not in allowed_actions:
                raise DecisionConflictError("override action is not allowed by frozen risk")
            if final_action != final_json.get("final_action"):
                if not isinstance(body.override_reason, str) or not body.override_reason.strip():
                    raise DecisionConflictError("action override requires a reason")
                final_json["final_action"] = final_action
                final_json["override_reason"] = body.override_reason
            try:
                _, _, _, snapshots = verify_complete_run(self.session, decision.evaluation_run_id)
                selected = _validate_selection(snapshots, final_json.get("selected_evaluation_ids"), final_action)
                version = self.session.get(StrategyVersion, decision.strategy_version_id)
                policy = DecisionPolicy.model_validate(version.policy_json)
                intents = _validate_order_intents(selected, final_json, policy)
            except ValidationError as error:
                raise DecisionConflictError(str(error)) from error
            if intents:
                final_json["order_intents"] = [intent.model_dump(mode="json") for intent in intents]
            decision.final_json = final_json

        event_type = "approved" if operation == "approve" else "rejected"
        expected_revision = decision.revision
        decision.status = event_type
        decision.revision += 1
        response = _decision_response(decision)
        self._append_event(
            decision,
            event_type,
            principal.actor_id,
            expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            payload_json={"request": body_json, "response": response},
        )
        self.session.flush()
        return response

    def approve(self, decision_id, body, *, principal, idempotency_key, now=None):
        return self._transition(
            "approve",
            decision_id,
            body,
            ApprovalBody,
            principal=principal,
            idempotency_key=idempotency_key,
            now=now,
        )

    def reject(self, decision_id, body, *, principal, idempotency_key, now=None):
        return self._transition(
            "reject",
            decision_id,
            body,
            RejectionBody,
            principal=principal,
            idempotency_key=idempotency_key,
            now=now,
        )

    def create_decision(
        self,
        evaluation_run_id,
        *,
        principal: AuthPrincipal,
        account_scope,
        original_json,
        portfolio_snapshot_json,
        risk_snapshot_json,
        now=None,
    ):
        principal.require_role("decision-worker")
        principal.require_account_scope(account_scope)
        if self.session.get(EvaluationRun, evaluation_run_id) is None:
            raise DecisionNotFoundError("evaluation run does not exist")
        run, version, manifest, snapshots = verify_complete_run(self.session, evaluation_run_id)
        policy = DecisionPolicy.model_validate(version.policy_json)
        manifest_payload = manifest.payload_json
        for field, actual, expected in (
            ("market", policy.market, manifest.market),
            ("account_scope", policy.account_scope, account_scope),
            ("manifest.account_scope", manifest_payload.get("account_scope"), account_scope),
            ("universe_id", policy.universe_id, manifest_payload.get("universe_id")),
        ):
            if actual != expected:
                raise ValidationError(f"policy {field} mismatch")
        validate_manifest(
            dict(
                manifest_payload,
                as_of=iso_time(_stored_time(run.decision_as_of, "decision_as_of"), "decision_as_of"),
                required_data={kind: rule.model_dump() for kind, rule in policy.required_data.items()},
            )
        )

        original = _json_object(original_json, "original_json")
        portfolio = _json_object(portfolio_snapshot_json, "portfolio_snapshot_json")
        risk = _json_object(risk_snapshot_json, "risk_snapshot_json")
        final_action = original.get("final_action")
        selected = _validate_selection(snapshots, original.get("selected_evaluation_ids"), final_action)
        original["selected_evaluation_ids"] = [str(row.id) for row in selected]
        intents = _validate_order_intents(selected, original, policy)
        if intents:
            original["order_intents"] = [intent.model_dump(mode="json") for intent in intents]
        hard_failures = risk.get("hard_failures")
        allowed_actions = risk.get("allowed_actions")
        if not isinstance(hard_failures, list):
            raise ValidationError("hard_failures must be a list")
        if (
            not isinstance(allowed_actions, list)
            or not allowed_actions
            or any(not isinstance(action, str) for action in allowed_actions)
            or len(set(allowed_actions)) != len(allowed_actions)
            or any(action not in ACTION_VALUES for action in allowed_actions)
        ):
            raise ValidationError("allowed_actions must be a unique non-empty action list")
        if final_action not in allowed_actions:
            raise ValidationError("final_action is not allowed by frozen risk")

        decision_as_of = _stored_time(run.decision_as_of, "decision_as_of")
        expiry_candidates = [decision_as_of + timedelta(seconds=policy.decision_ttl_seconds)]
        expiry_candidates.extend(
            _stored_time(row.valid_until, "snapshot.valid_until") for row in selected if row.valid_until is not None
        )
        valid_until = min(expiry_candidates)
        if valid_until <= decision_as_of:
            raise ValidationError("decision validity must extend beyond its cutoff")
        if now is not None:
            timestamp(now, "now")

        policy_sha256 = content_sha256(version.policy_json)
        creation_sha256 = content_sha256(
            {
                "evaluation_run_id": str(run.id),
                "evaluation_input_sha256": run.input_sha256,
                "strategy_version_id": str(version.id),
                "strategy_content_sha256": version.content_sha256,
                "account_scope": account_scope,
                "decision_as_of": iso_time(decision_as_of, "decision_as_of"),
                "valid_until": iso_time(valid_until, "valid_until"),
                "policy_sha256": policy_sha256,
                "original_json": original,
                "final_json": original,
                "portfolio_snapshot_json": portfolio,
                "risk_snapshot_json": risk,
            }
        )
        self._lock_scope(account_scope, version.id)
        replay = self.session.query(DecisionRecord).filter_by(creation_sha256=creation_sha256).one_or_none()
        if replay is not None:
            return replay

        decision = DecisionRecord(
            evaluation_run_id=run.id,
            strategy_version_id=version.id,
            account_scope=account_scope,
            decision_as_of=decision_as_of,
            valid_until=valid_until,
            status="pending_approval",
            revision=1,
            creation_sha256=creation_sha256,
            policy_sha256=policy_sha256,
            original_json=original,
            final_json=json.loads(canonical_json(original)),
            portfolio_snapshot_json=portfolio,
            risk_snapshot_json=risk,
        )
        self.session.add(decision)
        self.session.flush()
        self._append_event(
            decision,
            "created",
            principal.actor_id,
            0,
            payload_json={"response": _decision_response(decision)},
        )

        older = (
            self.session.query(DecisionRecord)
            .filter(
                DecisionRecord.id != decision.id,
                DecisionRecord.account_scope == account_scope,
                DecisionRecord.strategy_version_id == version.id,
                DecisionRecord.status.in_(LIVE_STATUSES),
            )
            .with_for_update()
            .populate_existing()
            .order_by(DecisionRecord.valid_until, DecisionRecord.id)
            .all()
        )
        for candidate in older:
            expected_revision = candidate.revision
            candidate.status = "superseded"
            candidate.revision += 1
            self._append_event(
                candidate,
                "superseded",
                SYSTEM_ACTOR,
                expected_revision,
                payload_json={"response": _decision_response(candidate)},
            )
        self.session.flush()
        return decision

    def expire_due(self, as_of):
        cutoff = timestamp(as_of, "as_of")
        due = (
            self.session.query(DecisionRecord)
            .filter(DecisionRecord.status.in_(LIVE_STATUSES), DecisionRecord.valid_until <= cutoff)
            .with_for_update()
            .populate_existing()
            .order_by(DecisionRecord.valid_until, DecisionRecord.id)
            .all()
        )
        for decision in due:
            expected_revision = decision.revision
            decision.status = "expired"
            decision.revision += 1
            self._append_event(
                decision,
                "expired",
                SYSTEM_ACTOR,
                expected_revision,
                payload_json={"response": _decision_response(decision)},
            )
        self.session.flush()
        return due

    def get(self, decision_id, *, principal):
        return self._authorized_decision(decision_id, principal)

    def list_pending(self, *, principal, now=None):
        if not principal.account_scopes:
            raise HTTPException(status_code=403, detail="Account scope not authorized")
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        return (
            self.session.query(DecisionRecord)
            .filter(
                DecisionRecord.status == "pending_approval",
                DecisionRecord.valid_until > current_time,
                DecisionRecord.account_scope.in_(principal.account_scopes),
            )
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
            .all()
        )

    def trace(self, decision_id, *, principal):
        decision = self.get(decision_id, principal=principal)
        run, version, manifest, snapshots = verify_complete_run(self.session, decision.evaluation_run_id)
        selected_ids = {uuid.UUID(value) for value in decision.original_json["selected_evaluation_ids"]}
        evaluations = []
        for snapshot in snapshots:
            revision_ids = list(snapshot.research_revision_ids)
            if revision_ids:
                revisions = [self.session.get(ResearchRevision, uuid.UUID(value)) for value in revision_ids]
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
                        iso_time(_stored_time(snapshot.valid_until, "snapshot.valid_until"), "snapshot.valid_until")
                        if snapshot.valid_until is not None
                        else None
                    ),
                    "research": research,
                }
            )
        scoped_orders = self.session.scalars(
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
        events = self.session.scalars(
            select(DecisionEvent).where(event_filter).order_by(DecisionEvent.created_at, DecisionEvent.id)
        ).all()
        order_ids = [order.id for order in orders]
        fills = (
            self.session.scalars(
                select(OrderFillRecord)
                .where(OrderFillRecord.order_id.in_(order_ids))
                .order_by(OrderFillRecord.fill_time, OrderFillRecord.id)
            ).all()
            if order_ids
            else []
        )
        fill_ids = [fill.id for fill in fills]
        allocations = (
            self.session.scalars(
                select(FillAllocation)
                .where(FillAllocation.closing_fill_id.in_(fill_ids))
                .order_by(FillAllocation.created_at, FillAllocation.id)
            ).all()
            if fill_ids
            else []
        )
        lot_ids = {allocation.position_lot_id for allocation in allocations}
        for order in orders:
            lot_ids.update(
                uuid.UUID(value) for value in (order.protective_context_json or {}).get("source_lot_ids", [])
            )
        lots = self.session.scalars(
            select(PositionLot)
            .where((PositionLot.opening_decision_id == decision.id) | (PositionLot.id.in_(lot_ids)))
            .order_by(PositionLot.opened_at, PositionLot.id)
        ).all()
        account_generations = sorted({order.account_generation for order in orders if order.account_generation})
        reconciliations = (
            self.session.scalars(
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
            self.session,
            account_scope=decision.account_scope,
            version=version,
            snapshots=selected_snapshots,
            decision=decision,
        )
        evidence = load_trace_evidence(
            self.session,
            version=version,
            snapshots=snapshots,
            account_scope=decision.account_scope,
        )
        trace = {
            "decision": {
                **_decision_response(decision),
                "evaluation_run_id": str(decision.evaluation_run_id),
                "strategy_version_id": str(decision.strategy_version_id),
                "account_scope": decision.account_scope,
                "decision_as_of": iso_time(_stored_time(decision.decision_as_of, "decision_as_of"), "decision_as_of"),
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
                        _stored_time(decision.decision_as_of, "decision_as_of"), "decision_as_of"
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
                "decision_as_of": iso_time(_stored_time(run.decision_as_of, "decision_as_of"), "decision_as_of"),
            },
            "manifest": {
                "id": str(manifest.id),
                "content_sha256": manifest.content_sha256,
                "created_at": (
                    iso_time(_stored_time(manifest.created_at, "manifest.created_at"), "manifest.created_at")
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
                        iso_time(_stored_time(order.submit_attempted_at, "submit_attempted_at"), "submit_attempted_at")
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
                    "fill_time": iso_time(_stored_time(fill.fill_time, "fill_time"), "fill_time"),
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
                    "as_of": iso_time(_stored_time(row.as_of, "reconciliation.as_of"), "reconciliation.as_of"),
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
                        iso_time(_stored_time(event.created_at, "event.created_at"), "event.created_at")
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

    def _validate_execution_record(self, decision, expected_revision, as_of):
        if decision.status != "approved":
            raise DecisionConflictError("decision is not approved")
        if decision.revision != expected_revision:
            raise DecisionConflictError("decision revision changed")
        if _stored_time(decision.valid_until, "valid_until") <= timestamp(as_of, "as_of"):
            raise DecisionConflictError("decision has expired")
        if not self._policy_is_current(decision):
            raise DecisionConflictError("decision policy changed")

        version = self.session.get(StrategyVersion, decision.strategy_version_id)
        policy = DecisionPolicy.model_validate(version.policy_json)
        if policy.reconciliation is None:
            raise DecisionConflictError("execution requires an owner reconciliation policy")
        hard_failures = decision.risk_snapshot_json.get("hard_failures")
        if not isinstance(hard_failures, list) or hard_failures:
            raise DecisionConflictError("hard risk failures prevent execution")
        try:
            _, _, _, snapshots = verify_complete_run(self.session, decision.evaluation_run_id)
            selected = _validate_selection(
                snapshots,
                decision.final_json.get("selected_evaluation_ids"),
                decision.final_json.get("final_action"),
            )
            return _validate_order_intents(selected, decision.final_json, policy)
        except ValidationError as error:
            raise DecisionConflictError(str(error)) from error

    def validate_execution(self, decision_id, expected_revision, as_of):
        decision = self.session.get(DecisionRecord, decision_id)
        if decision is None:
            raise DecisionNotFoundError("decision does not exist")
        return self._validate_execution_record(decision, expected_revision, as_of)

    @staticmethod
    def _execution_claim_response(decision):
        return {
            "decision_id": str(decision.id),
            "execution_key": str(decision.execution_key),
            "status": decision.status,
            "revision": decision.revision,
            "claimed_at": iso_time(_stored_time(decision.claimed_at, "claimed_at"), "claimed_at"),
        }

    def _claim_locked(self, decision, expected_revision, *, principal, as_of):
        principal.require_account_scope(decision.account_scope)
        intents = self._validate_execution_record(decision, expected_revision, as_of)
        if not intents:
            raise DecisionConflictError("execution claim requires order_intents")

        claimed_at = timestamp(as_of, "as_of")
        decision.execution_key = uuid.uuid4()
        decision.claimed_at = claimed_at
        decision.status = "execution_claimed"
        decision.revision += 1
        response = self._execution_claim_response(decision)
        self._append_event(
            decision,
            "execution_claimed",
            principal.actor_id,
            expected_revision,
            payload_json={"response": response},
        )
        self.session.flush()
        return response

    def claim_execution(self, decision_id, expected_revision, *, principal, now=None):
        from poseidon.decision_loop.releases import require_active_pointer

        principal.require_role("decision-worker")
        if type(expected_revision) is not int or expected_revision <= 0:
            raise DecisionConflictError("expected revision is invalid")
        candidate = self._authorized_decision(decision_id, principal)
        require_active_pointer(
            self.session,
            candidate,
            lock=True,
            error_type=DecisionConflictError,
        )
        decision = (
            self.session.query(DecisionRecord)
            .filter(DecisionRecord.id == decision_id)
            .with_for_update(skip_locked=True)
            .populate_existing()
            .one_or_none()
        )
        if decision is None:
            raise DecisionConflictError("decision is locked")
        require_active_pointer(
            self.session,
            decision,
            lock=False,
            error_type=DecisionConflictError,
        )
        return self._claim_locked(
            decision,
            expected_revision,
            principal=principal,
            as_of=now if now is not None else datetime.now(UTC),
        )

    def claim_next_approved(self, account_scope, *, principal, now=None):
        from poseidon.decision_loop.releases import require_active_pointer

        principal.require_role("decision-worker")
        principal.require_account_scope(account_scope)
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        skipped_ids = []
        # ponytail: keep claim transactions bounded; quarantine invalid rows if this ceiling is reached.
        for _ in range(CLAIM_SCAN_LIMIT):
            query = (
                self.session.query(DecisionRecord)
                .join(StrategyVersion, StrategyVersion.id == DecisionRecord.strategy_version_id)
                .join(StrategyRecord, StrategyRecord.id == StrategyVersion.strategy_id)
                .filter(
                    DecisionRecord.account_scope == account_scope,
                    DecisionRecord.status == "approved",
                    DecisionRecord.execution_key.is_(None),
                    DecisionRecord.claimed_at.is_(None),
                    DecisionRecord.valid_until > current_time,
                    StrategyRecord.active_version_id == DecisionRecord.strategy_version_id,
                )
            )
            if skipped_ids:
                query = query.filter(DecisionRecord.id.notin_(skipped_ids))
            candidate = (
                query.order_by(DecisionRecord.created_at, DecisionRecord.id)
                .with_for_update(of=StrategyRecord, skip_locked=True)
                .populate_existing()
                .first()
            )
            if candidate is None:
                return None
            decision = (
                self.session.query(DecisionRecord)
                .filter(DecisionRecord.id == candidate.id)
                .with_for_update(skip_locked=True)
                .populate_existing()
                .one_or_none()
            )
            if decision is None:
                skipped_ids.append(candidate.id)
                continue
            try:
                require_active_pointer(
                    self.session,
                    decision,
                    lock=False,
                    error_type=DecisionConflictError,
                )
                return self._claim_locked(
                    decision,
                    decision.revision,
                    principal=principal,
                    as_of=current_time,
                )
            except DecisionConflictError:
                skipped_ids.append(decision.id)
        raise DecisionConflictError("claim candidate scan limit exceeded")

    def is_execution_eligible(self, decision_id, expected_revision, as_of):
        try:
            self.validate_execution(decision_id, expected_revision, as_of)
        except (DecisionNotFoundError, DecisionConflictError, PydanticValidationError):
            return False
        return True
