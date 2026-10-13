"""Fail-closed release gate contracts."""

import copy
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import product

import pytest
from fastapi import HTTPException
from sqlalchemy import func, text

from poseidon.api.auth import AuthPrincipal
from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionPolicy
from poseidon.decision_loop.manifest import content_sha256
from poseidon.decision_loop.releases import ReleaseGateRejected, ReleaseGateResult, ReleasePredicate, ReleaseService
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.data_manifest import DataManifest
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import CampaignReview, HoldoutUse
from poseidon.models.order import OrderRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from poseidon.research.campaign import CampaignService, build_review_input_identity
from tests.test_campaign_contract import _complete_campaign_contract, _terminal_trial_kwargs
from tests.test_decision_service import synthetic_policy, synthetic_reconciliation_policy

NOW = datetime(2026, 10, 13, 2, 0, tzinfo=UTC)
GATE_CODES = (
    "candidate_identity",
    "policy_identity",
    "campaign_identity",
    "incumbent_pointer",
    "six_cell_evidence",
    "review_identity",
    "holdout_ownership",
    "reconciliation",
    "release_authority",
)
MEMBERSHIP = [
    {"date": "2026-10-01", "symbol": symbol, "horizon": "1_session", "regime": "risk_on"}
    for symbol in ("0050", "2317", "2330")
]


@dataclass(frozen=True)
class GateGraph:
    strategy_id: uuid.UUID
    candidate_id: uuid.UUID
    campaign_id: uuid.UUID
    review_id: uuid.UUID
    reconciliation_id: uuid.UUID
    holdout_identity: str


def _seed_gate_graph(
    session,
    *,
    terminal_state="succeeded",
    review_status="passed",
    reconciliation_status="matched",
    mature_samples=10,
    policy_changes=None,
    strategy_id=None,
    incumbent_id=None,
    candidate_config_json=None,
    strategy_changes=None,
    campaign_universe=None,
):
    marker = uuid.uuid4().hex
    if (strategy_id is None) != (incumbent_id is None):
        raise ValueError("strategy_id and incumbent_id must be provided together")
    if strategy_id is None:
        policy_changes = dict(policy_changes or {})
        reconciliation = policy_changes.pop("reconciliation", synthetic_reconciliation_policy())
        policy = synthetic_policy(
            reconciliation=reconciliation,
            **policy_changes,
        )
        DecisionPolicy.model_validate(policy)
        strategy_values = {
            "name": f"phase100-gate-{marker}",
            "strategy_type": "technical",
            "config": {},
            "symbol": "2330",
            "market": "tw_stock",
            "interval": "1d",
            "active": False,
        }
        strategy_values.update(strategy_changes or {})
        strategy = StrategyRecord(
            **strategy_values,
        )
        session.add(strategy)
        session.flush()
        incumbent = None
        next_version = 1
    else:
        strategy = session.get(StrategyRecord, strategy_id)
        incumbent = session.get(StrategyVersion, incumbent_id)
        if strategy is None or incumbent is None or incumbent.strategy_id != strategy.id:
            raise ValueError("follow-up campaign requires its strategy-owned incumbent")
        policy = copy.deepcopy(incumbent.policy_json)
        policy.update(copy.deepcopy(policy_changes or {}))
        DecisionPolicy.model_validate(policy)
        next_version = (
            session.query(func.max(StrategyVersion.version_no))
            .filter(StrategyVersion.strategy_id == strategy.id)
            .scalar()
            + 1
        )
    versions = [] if incumbent is None else [incumbent]
    roles = ("incumbent", "candidate") if incumbent is None else ("candidate",)
    for offset, role in enumerate(roles):
        config = {
            "strategy_type": "rule",
            "symbol": strategy.symbol,
            "market": strategy.market,
            "instrument": "perpetual" if strategy.market == "crypto_perp" else "spot",
            "interval": strategy.interval,
            "payload": {
                "name": f"phase100-{marker}-{role}",
                "description": "Phase 100 executable release fixture",
                "rules": [
                    {
                        "condition": {
                            "type": "feature_above",
                            "column": "close",
                            "threshold": 200.0,
                        },
                        "action": "long",
                        "quantity_pct": 0.1,
                        "confidence": 0.9,
                    }
                ],
            },
        }
        if role == "candidate" and candidate_config_json is not None:
            config = copy.deepcopy(candidate_config_json)
        artifact = {"uri": f"fixture://phase100/{marker}/{role}"}
        version = StrategyVersion(
            strategy_id=strategy.id,
            version_no=next_version + offset,
            status="draft",
            config_json=config,
            policy_json=policy,
            artifact_json=artifact,
            content_sha256=strategy_version_digest(config, policy, artifact),
        )
        session.add(version)
        session.flush()
        versions.append(version)
    incumbent, candidate = versions
    campaign_contract = _complete_campaign_contract(incumbent, candidate, marker=marker)
    if campaign_universe is not None:
        campaign_contract["contract_json"]["universe"] = copy.deepcopy(campaign_universe)
    campaign = CampaignService(session).create_frozen_campaign(campaign_contract)
    tracker = ExperimentTracker(session)
    trials = []
    for index, ((role, version), arm) in enumerate(
        product(
            (("incumbent", incumbent), ("candidate", candidate)),
            ("fundamental_only", "technical_only", "combined"),
        )
    ):
        kwargs = _terminal_trial_kwargs(campaign, version, arm=arm)
        kwargs.update(
            original_trial_id=f"{role}-{arm}",
            paired_sample_key_sha256="3" * 64,
            input_sha256=f"{100 + index:064x}",
            result_sha256=content_sha256({"marker": marker, "role": role, "arm": arm}),
            terminal_state=terminal_state if index == 5 else "succeeded",
            metrics_json={"sample_membership": MEMBERSHIP},
        )
        trials.append(tracker.append_campaign_terminal_trial(**kwargs))

    holdout_identity = content_sha256({"holdout": marker})
    holdout = HoldoutUse(
        holdout_identity_sha256=holdout_identity,
        campaign_id=campaign.id,
        campaign_contract_sha256=campaign.contract_sha256,
        audit_json={"campaign_id": str(campaign.id)},
        consumed_at=NOW,
    )
    session.add(holdout)
    session.flush()
    CampaignService(session).append_event(
        campaign.id,
        event_type="holdout_consumed",
        payload_json={
            "holdout_identity_sha256": holdout_identity,
            "holdout_use_id": str(holdout.id),
        },
    )
    input_identity = build_review_input_identity(
        campaign,
        trials,
        holdout_identity_sha256=holdout_identity,
        panel_sha256="e" * 64,
    )
    coverage = {
        f"{role}:{arm}": {"present": True, "terminal_state": "succeeded"}
        for role in ("incumbent", "candidate")
        for arm in ("fundamental_only", "technical_only", "combined")
    }
    review_cells = {cell: {"1_session": {"status": "available", "estimate": 0.1, "coverage": 1.0}} for cell in coverage}
    result = {
        "status": review_status,
        "reason_code": None if review_status == "passed" else f"fixture_{review_status}",
        "coverage_matrix": coverage,
        "paired_sample_digest": input_identity["paired_sample_sha256"],
        "cross_sectional_ic": review_cells,
        "time_series_ic": review_cells,
        "horizon_decay": {"1_session": {"effect": 0.01}},
        "slices": {"time": {"all": {"effect": 0.01}}},
        "hac": {"status": "available", "sample_count": mature_samples},
        "gross": {"incumbent_mean": 0.01, "candidate_mean": 0.02, "effect": 0.01},
        "net": {"minimum_effect": 0.009},
        "drawdown": {"maximum": 0.01},
        "turnover": {"incumbent_mean": 0.1, "candidate_mean": 0.1},
        "capacity": {"status": "available", "value": "1000000"},
        "cost_sensitivity": {"base": {"effect": 0.009}},
        "failed_gates": [],
    }
    review = CampaignReview(
        campaign_id=campaign.id,
        input_sha256=content_sha256(input_identity),
        result_sha256=content_sha256(result),
        status=review_status,
        result_json=result,
        holdout_use_id=holdout.id,
        input_identity_json=input_identity,
    )
    session.add(review)
    reconciliation = AccountReconciliation(
        account_scope=policy["account_scope"],
        account_generation=policy["reconciliation"]["account_generation"],
        as_of=NOW,
        broker_state_watermark=f"broker:{marker}",
        internal_state_watermark=f"internal:{marker}",
        broker_snapshot_sha256=content_sha256({"broker": marker}),
        broker_snapshot_json={},
        internal_snapshot_json={},
        difference_json={},
        policy_sha256=content_sha256(policy),
        status=reconciliation_status,
    )
    session.add(reconciliation)
    session.flush()
    return GateGraph(
        strategy.id,
        candidate.id,
        campaign.id,
        review.id,
        reconciliation.id,
        holdout_identity,
    )


def _release(service, graph, *, key="release-key", reason="release fixture", **changes):
    values = {
        "expected_revision": 0,
        "expected_active_version_id": None,
        "principal": release_manager(),
        "idempotency_key": key,
        "reason": reason,
        "now": NOW,
    }
    values.update(changes)
    return service.release(
        graph.strategy_id,
        graph.candidate_id,
        graph.campaign_id,
        graph.review_id,
        graph.reconciliation_id,
        **values,
    )


def _evaluate(session, graph, *, principal=None, now=NOW, **changes):
    values = {
        "strategy_id": graph.strategy_id,
        "candidate_version_id": graph.candidate_id,
        "campaign_id": graph.campaign_id,
        "review_id": graph.review_id,
        "reconciliation_id": graph.reconciliation_id,
    }
    values.update(changes)
    return ReleaseService(session).evaluate_gate(
        values["strategy_id"],
        values["candidate_version_id"],
        values["campaign_id"],
        values["review_id"],
        values["reconciliation_id"],
        principal=principal or release_manager(),
        now=now,
    )


def _by_code(result, code):
    return next(predicate for predicate in result.predicates if predicate.code == code)


def release_manager(scope="paper:pilot"):
    return AuthPrincipal(
        "human:phase100-release-manager",
        frozenset({"release_manager"}),
        frozenset({scope}),
    )


@pytest.mark.postgresql
def test_gate_retains_every_missing_predicate_in_stable_order(phase99_session_factory):
    missing = {name: uuid.uuid4() for name in ("strategy", "candidate", "campaign", "review", "reconciliation")}

    with phase99_session_factory() as session:
        result = ReleaseService(session).evaluate_gate(
            missing["strategy"],
            missing["candidate"],
            missing["campaign"],
            missing["review"],
            missing["reconciliation"],
            principal=release_manager(),
            now=NOW,
        )

    assert result.allowed is False
    assert tuple(predicate.code for predicate in result.predicates) == GATE_CODES
    assert all(predicate.state in {"blocked", "unavailable", "inconclusive"} for predicate in result.predicates)
    assert all(predicate.reason and predicate.evidence for predicate in result.predicates)


def test_gate_result_values_are_immutable_and_canonicalizable():
    predicate = ReleasePredicate(
        code="candidate_identity",
        state="passed",
        reason="candidate content and ownership match",
        evidence={"candidate_version_id": str(uuid.uuid4()), "content_sha256": "a" * 64},
    )
    result = ReleaseGateResult((predicate,))

    assert result.allowed is True
    assert content_sha256(result.as_json())
    with pytest.raises((AttributeError, TypeError)):
        predicate.state = "blocked"


@pytest.mark.postgresql
def test_gate_passes_candidate_policy_campaign_six_cell_review_and_own_committed_holdout(
    phase99_session_factory,
):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        result = _evaluate(session, graph)

        assert result.allowed is True, [
            predicate.as_json() for predicate in result.predicates if predicate.state != "passed"
        ]
        assert [predicate.state for predicate in result.predicates] == ["passed"] * len(GATE_CODES)
        assert _by_code(result, "holdout_ownership").reason == "same-campaign committed holdout use matches"
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_candidate_policy_market_outside_strategy_market(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(
            session,
            policy_changes={
                "market": "crypto_perp",
                "reconciliation": synthetic_reconciliation_policy(
                    currency="USDT",
                    perp_instrument_rules={
                        "BTC/USDT:USDT": {
                            "quantity_step": 0.001,
                            "contract_multiplier": 1.0,
                            "margin_semantics": "synthetic-isolated",
                            "funding_semantics": "synthetic-periodic-cashflow",
                        }
                    },
                ),
            },
        )

        result = _evaluate(session, graph)

        assert _by_code(result, "policy_identity").state == "blocked"
        assert "market" in _by_code(result, "policy_identity").reason
        session.rollback()


@pytest.mark.postgresql
@pytest.mark.parametrize(
    ("variant", "code", "state"),
    (
        ("stale", "reconciliation", "blocked"),
        ("unresolved", "reconciliation", "blocked"),
        ("unauthorized", "release_authority", "blocked"),
        ("wrong_scope", "release_authority", "blocked"),
        ("trial_unavailable", "six_cell_evidence", "unavailable"),
        ("trial_inconclusive", "six_cell_evidence", "inconclusive"),
        ("review_unavailable", "review_identity", "unavailable"),
        ("review_inconclusive", "review_identity", "inconclusive"),
        ("review_failed", "review_identity", "blocked"),
        ("immature", "review_identity", "inconclusive"),
        ("automatic_policy", "policy_identity", "blocked"),
    ),
)
def test_gate_blocks_stale_unavailable_inconclusive_policy_reconciliation_or_principal_evidence(
    phase99_session_factory,
    variant,
    code,
    state,
):
    options = {}
    if variant == "unresolved":
        options["reconciliation_status"] = "unresolved"
    elif variant == "trial_unavailable":
        options["terminal_state"] = "capability_unavailable"
    elif variant == "trial_inconclusive":
        options["terminal_state"] = "statistically_inconclusive"
    elif variant.startswith("review_"):
        options["review_status"] = variant.removeprefix("review_")
    elif variant == "immature":
        options["mature_samples"] = 2
    elif variant == "automatic_policy":
        options["policy_changes"] = {"mode": "automatic"}

    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session, **options)
        principal = None
        now = NOW
        if variant == "stale":
            now += timedelta(seconds=301)
        elif variant == "unauthorized":
            principal = AuthPrincipal("worker", frozenset({"decision-worker"}), frozenset({"paper:pilot"}))
        elif variant == "wrong_scope":
            principal = release_manager("paper:other")
        result = _evaluate(session, graph, principal=principal, now=now)

        predicate = _by_code(result, code)
        assert result.allowed is False
        assert predicate.state == state
        assert predicate.reason
        assert predicate.evidence
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_incumbent_pointer_drift(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        review = session.get(CampaignReview, graph.review_id)
        event = StrategyReleaseEvent(
            strategy_id=graph.strategy_id,
            event_type="release",
            candidate_version_id=graph.candidate_id,
            from_version_id=None,
            to_version_id=graph.candidate_id,
            campaign_id=graph.campaign_id,
            review_id=graph.review_id,
            holdout_use_id=review.holdout_use_id,
            reconciliation_id=graph.reconciliation_id,
            actor_id="human:fixture",
            account_scope="paper:pilot",
            expected_revision=0,
            resulting_revision=1,
            idempotency_key=f"fixture-{uuid.uuid4()}",
            request_sha256="9" * 64,
            reason="fixture active pointer",
            gate_json=[],
            response_json={},
        )
        session.add(event)
        session.flush()
        session.execute(text("SELECT set_config('poseidon.release_event_id', :id, true)"), {"id": str(event.id)})
        session.execute(
            text("UPDATE strategies SET active_version_id=:candidate,release_revision=1 WHERE id=:strategy"),
            {"candidate": graph.candidate_id, "strategy": graph.strategy_id},
        )
        session.expire_all()

        result = _evaluate(session, graph)

        assert _by_code(result, "incumbent_pointer").state == "blocked"
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_attempted_or_denied_holdout_reuse(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        CampaignService(session).append_event(
            graph.campaign_id,
            event_type="holdout_reuse_denied",
            payload_json={
                "holdout_identity_sha256": graph.holdout_identity,
                "owner_campaign_id": str(graph.campaign_id),
                "owner_campaign_contract_sha256": "0" * 64,
            },
        )

        result = _evaluate(session, graph)

        assert _by_code(result, "holdout_ownership").state == "blocked"
        assert "denied" in _by_code(result, "holdout_ownership").reason
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_newer_unresolved_reconciliation(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        current = session.get(AccountReconciliation, graph.reconciliation_id)
        session.add(
            AccountReconciliation(
                account_scope=current.account_scope,
                account_generation=current.account_generation,
                as_of=NOW + timedelta(seconds=1),
                broker_state_watermark="broker:newer",
                internal_state_watermark="internal:newer",
                broker_snapshot_sha256="8" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={"unresolved": ["fixture"]},
                policy_sha256=current.policy_sha256,
                status="unresolved",
            )
        )
        session.flush()

        result = _evaluate(session, graph, now=NOW + timedelta(seconds=2))

        assert _by_code(result, "reconciliation").state == "blocked"
        assert "newer" in _by_code(result, "reconciliation").reason
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_later_same_as_of_reconciliation(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        current = session.get(AccountReconciliation, graph.reconciliation_id)
        session.add(
            AccountReconciliation(
                account_scope=current.account_scope,
                account_generation=current.account_generation,
                as_of=current.as_of,
                broker_state_watermark="broker:same-as-of-later",
                internal_state_watermark="internal:same-as-of-later",
                broker_snapshot_sha256="7" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={"unresolved": ["fixture"]},
                policy_sha256=current.policy_sha256,
                status="unresolved",
                created_at=current.created_at + timedelta(microseconds=1),
            )
        )
        session.flush()

        result = _evaluate(session, graph, now=NOW + timedelta(seconds=1))

        assert _by_code(result, "reconciliation").state == "blocked"
        assert "newer" in _by_code(result, "reconciliation").reason
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_review_result_for_a_different_paired_sample(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        review = session.get(CampaignReview, graph.review_id)
        review.result_json = copy.deepcopy(review.result_json)
        review.result_json["paired_sample_digest"] = "0" * 64
        review.result_sha256 = content_sha256(review.result_json)

        with session.no_autoflush:
            result = _evaluate(session, graph)

        assert _by_code(result, "review_identity").state == "blocked"
        session.rollback()


@pytest.mark.postgresql
@pytest.mark.parametrize(
    ("field", "value"),
    (("cross_sectional_ic", {}), ("turnover", None)),
)
def test_gate_blocks_incomplete_passed_review_output(phase99_session_factory, field, value):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        review = session.get(CampaignReview, graph.review_id)
        review.result_json = copy.deepcopy(review.result_json)
        review.result_json[field] = value
        review.result_sha256 = content_sha256(review.result_json)

        with session.no_autoflush:
            result = _evaluate(session, graph)

        assert _by_code(result, "review_identity").state == "inconclusive"
        session.rollback()


@pytest.mark.postgresql
@pytest.mark.parametrize(
    ("field", "value"),
    (("original_trial_id", "undeclared-trial"), ("trial_role", "search")),
)
def test_gate_blocks_terminal_trial_identity_outside_frozen_declaration(
    phase99_session_factory,
    field,
    value,
):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        trial = (
            session.query(ExperimentRecord)
            .filter_by(campaign_id=graph.campaign_id)
            .order_by(ExperimentRecord.id)
            .first()
        )
        setattr(trial, field, value)

        with session.no_autoflush:
            result = _evaluate(session, graph)

        assert _by_code(result, "six_cell_evidence").state == "blocked"
        assert "declared" in _by_code(result, "six_cell_evidence").reason
        session.rollback()


@pytest.mark.postgresql
def test_gate_blocks_foreign_candidate_review_and_holdout_consumption(phase99_session_factory):
    with phase99_session_factory() as session:
        own = _seed_gate_graph(session)
        foreign = _seed_gate_graph(session)

        result = _evaluate(
            session,
            own,
            candidate_version_id=foreign.candidate_id,
            review_id=foreign.review_id,
        )

        assert _by_code(result, "candidate_identity").state == "blocked"
        assert _by_code(result, "review_identity").state == "blocked"
        assert _by_code(result, "holdout_ownership").state == "blocked"
        assert "another campaign" in _by_code(result, "holdout_ownership").reason
        session.rollback()


@pytest.mark.postgresql
def test_gate_keeps_legacy_unbound_review_readable_but_ineligible(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        bound = session.get(CampaignReview, graph.review_id)
        legacy_id = uuid.uuid4()
        session.execute(text("ALTER TABLE campaign_reviews DISABLE TRIGGER trg_campaign_reviews_phase100_binding"))
        try:
            session.execute(
                text(
                    "INSERT INTO campaign_reviews "
                    "(id,campaign_id,input_sha256,result_sha256,status,result_json) "
                    "VALUES (:id,:campaign,:input,:result,'passed',CAST(:result_json AS jsonb))"
                ),
                {
                    "id": legacy_id,
                    "campaign": graph.campaign_id,
                    "input": "1" * 64,
                    "result": bound.result_sha256,
                    "result_json": json.dumps(bound.result_json),
                },
            )
        finally:
            session.execute(text("ALTER TABLE campaign_reviews ENABLE TRIGGER trg_campaign_reviews_phase100_binding"))

        result = _evaluate(session, graph, review_id=legacy_id)

        assert _by_code(result, "review_identity").state == "unavailable"
        assert "legacy" in _by_code(result, "review_identity").reason
        assert _by_code(result, "holdout_ownership").state == "unavailable"
        session.rollback()


@pytest.mark.postgresql
def test_release_replays_exact_request_and_rejects_changed_digest(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)

        response = _release(service, graph)
        replay = _release(service, graph)

        strategy = session.get(StrategyRecord, graph.strategy_id)
        events = session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).all()
        assert replay == response
        assert (strategy.active_version_id, strategy.release_revision) == (graph.candidate_id, 1)
        assert len(events) == 1
        assert events[0].response_json == response
        with pytest.raises(DecisionConflictError, match="different request"):
            _release(service, graph, reason="changed payload")
        session.rollback()


@pytest.mark.postgresql
def test_release_replay_rechecks_retained_account_scope(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)
        _release(service, graph)

        with pytest.raises(HTTPException) as denied:
            _release(service, graph, principal=release_manager("paper:revoked"))

        assert denied.value.status_code == 403
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 1
        session.rollback()


def _seed_rollback_work(session, graph, work_kind):
    candidate = session.get(StrategyVersion, graph.candidate_id)
    if work_kind in {"standalone_order", "pending_order"}:
        row = OrderRecord(
            strategy_name="phase100-legacy-order",
            symbol="2330",
            market="tw_stock",
            action="buy",
            order_type="market",
            target_weight=0.1,
            quantity=1.0,
            side="long",
            status="pending" if work_kind == "pending_order" else "pending_submit",
            broker_mode="paper",
            order_origin="signal",
            account_scope="paper:pilot",
            account_generation=candidate.policy_json["reconciliation"]["account_generation"],
        )
        session.add(row)
        session.flush()
        return row.id

    manifest = DataManifest(
        content_sha256=content_sha256({"manifest": uuid.uuid4().hex}),
        market="tw_stock",
        interval="1d",
        as_of=NOW,
        capability_json={},
        sources_json=[],
        payload_json={},
    )
    session.add(manifest)
    session.flush()
    run = EvaluationRun(
        strategy_version_id=graph.candidate_id,
        manifest_id=manifest.id,
        decision_as_of=NOW,
        universe_json=[],
        input_sha256=content_sha256({"run": uuid.uuid4().hex}),
        status="succeeded",
        coverage_json={},
    )
    session.add(run)
    session.flush()
    row = DecisionRecord(
        evaluation_run_id=run.id,
        strategy_version_id=graph.candidate_id,
        account_scope="paper:pilot",
        decision_as_of=NOW,
        valid_until=NOW + timedelta(hours=1),
        status="pending_approval" if work_kind == "pending_approval_decision" else "approved",
        creation_sha256=content_sha256({"decision": uuid.uuid4().hex}),
        policy_sha256=candidate.content_sha256,
        original_json={"order_intents": [{"symbol": "2330"}]},
        final_json={"order_intents": [{"symbol": "2330"}]},
        portfolio_snapshot_json={},
        risk_snapshot_json={},
    )
    session.add(row)
    session.flush()
    return row.id


@pytest.mark.postgresql
@pytest.mark.parametrize(
    "work_kind",
    ("pending_approval_decision", "approved_decision", "pending_order", "standalone_order"),
)
def test_rollback_blocks_outstanding_ordinary_work(phase99_session_factory, work_kind):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)
        _release(service, graph)
        service.pause(
            graph.strategy_id,
            expected_revision=1,
            expected_active_version_id=graph.candidate_id,
            principal=release_manager(),
            idempotency_key="pause-before-blocked-rollback",
            reason="pause before rollback",
        )
        work_id = _seed_rollback_work(session, graph, work_kind)

        with pytest.raises(DecisionConflictError, match="rollback_outstanding_work"):
            service.rollback(
                graph.strategy_id,
                graph.candidate_id,
                graph.reconciliation_id,
                expected_revision=2,
                expected_active_version_id=None,
                principal=release_manager(),
                idempotency_key=f"blocked-rollback-{work_kind}",
                reason="must not strand ordinary work",
                now=NOW,
            )

        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (None, 2)
        model = DecisionRecord if work_kind.endswith("decision") else OrderRecord
        assert session.get(model, work_id) is not None
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 2
        session.rollback()


@pytest.mark.postgresql
@pytest.mark.parametrize("identity_change", ("account_scope", "account_generation"))
@pytest.mark.parametrize("pause_before_rollback", (False, True))
def test_rollback_rejects_active_version_with_different_account_identity(
    phase99_session_factory,
    identity_change,
    pause_before_rollback,
):
    with phase99_session_factory() as session:
        target = _seed_gate_graph(session)
        service = ReleaseService(session)
        _release(service, target)

        target_version = session.get(StrategyVersion, target.candidate_id)
        active_scope = target_version.policy_json["account_scope"]
        active_reconciliation = copy.deepcopy(target_version.policy_json["reconciliation"])
        policy_changes = {}
        if identity_change == "account_scope":
            active_scope = "paper:phase100-other"
            policy_changes["account_scope"] = active_scope
        else:
            active_reconciliation["account_generation"] = f"phase100-other-{uuid.uuid4().hex}"
            policy_changes["reconciliation"] = active_reconciliation

        active = _seed_gate_graph(
            session,
            strategy_id=target.strategy_id,
            incumbent_id=target.candidate_id,
            policy_changes=policy_changes,
        )
        principal = AuthPrincipal(
            "human:phase100-release-manager",
            frozenset({"release_manager"}),
            frozenset({target_version.policy_json["account_scope"], active_scope}),
        )
        _release(
            service,
            active,
            key=f"release-cross-account-{identity_change}",
            expected_revision=1,
            expected_active_version_id=target.candidate_id,
            principal=principal,
        )
        active_version = session.get(StrategyVersion, active.candidate_id)
        order = OrderRecord(
            strategy_name="phase100-cross-account-order",
            symbol="2330",
            market="tw_stock",
            action="buy",
            order_type="market",
            target_weight=0.1,
            quantity=1.0,
            side="long",
            status="pending",
            broker_mode="paper",
            order_origin="signal",
            account_scope=active_version.policy_json["account_scope"],
            account_generation=active_version.policy_json["reconciliation"]["account_generation"],
        )
        session.add(order)
        session.flush()

        expected_revision = 2
        expected_pointer = active.candidate_id
        if pause_before_rollback:
            service.pause(
                target.strategy_id,
                expected_revision=2,
                expected_active_version_id=active.candidate_id,
                principal=principal,
                idempotency_key=f"pause-cross-account-{identity_change}",
                reason="pause must not erase account identity",
            )
            expected_revision = 3
            expected_pointer = None

        with pytest.raises(DecisionConflictError, match="rollback_policy"):
            service.rollback(
                target.strategy_id,
                target.candidate_id,
                target.reconciliation_id,
                expected_revision=expected_revision,
                expected_active_version_id=expected_pointer,
                principal=principal,
                idempotency_key=f"rollback-cross-account-{identity_change}-{pause_before_rollback}",
                reason="do not abandon active account work",
                now=NOW,
            )

        strategy = session.get(StrategyRecord, target.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (expected_pointer, expected_revision)
        assert session.get(OrderRecord, order.id) is not None
        assert (
            session.query(StrategyReleaseEvent).filter_by(strategy_id=target.strategy_id).count() == expected_revision
        )
        session.rollback()


@pytest.mark.postgresql
def test_four_transitions_preserve_pointer_semantics_and_rollback_creates_no_work(
    phase99_session_factory,
):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)
        _release(service, graph)
        baseline = (
            session.query(DecisionRecord).count(),
            session.query(OrderRecord).count(),
            session.query(PositionLot).count(),
        )

        rejected = service.reject(
            graph.strategy_id,
            graph.candidate_id,
            expected_revision=1,
            expected_active_version_id=graph.candidate_id,
            principal=release_manager(),
            idempotency_key="reject-key",
            reason="candidate terminally reviewed",
        )
        paused = service.pause(
            graph.strategy_id,
            expected_revision=2,
            expected_active_version_id=graph.candidate_id,
            principal=release_manager(),
            idempotency_key="pause-key",
            reason="operator pause",
        )
        rolled_back = service.rollback(
            graph.strategy_id,
            graph.candidate_id,
            graph.reconciliation_id,
            expected_revision=3,
            expected_active_version_id=None,
            principal=release_manager(),
            idempotency_key="rollback-key",
            reason="restore prior released version",
            now=NOW,
        )

        strategy = session.get(StrategyRecord, graph.strategy_id)
        events = (
            session.query(StrategyReleaseEvent)
            .filter_by(strategy_id=graph.strategy_id)
            .order_by(StrategyReleaseEvent.resulting_revision)
            .all()
        )
        assert rejected["to_version_id"] == str(graph.candidate_id)
        assert paused["to_version_id"] is None
        assert rolled_back["to_version_id"] == str(graph.candidate_id)
        assert (strategy.active_version_id, strategy.release_revision) == (graph.candidate_id, 4)
        assert [(event.event_type, event.resulting_revision) for event in events] == [
            ("release", 1),
            ("reject", 2),
            ("pause", 3),
            ("rollback", 4),
        ]
        assert {predicate["code"] for predicate in events[-1].gate_json["predicates"]} == {
            "rollback_target",
            "rollback_policy",
            "rollback_holdings",
            "rollback_outstanding_work",
            "rollback_reconciliation",
            "release_authority",
        }
        assert (
            session.query(DecisionRecord).count(),
            session.query(OrderRecord).count(),
            session.query(PositionLot).count(),
        ) == baseline
        session.rollback()


@pytest.mark.postgresql
def test_release_rejects_executable_market_outside_policy_and_campaign_scope(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(
            session,
            candidate_config_json={
                "strategy_type": "rule",
                "symbol": "2330",
                "market": "crypto_spot",
                "instrument": "spot",
                "interval": "1d",
                "payload": {
                    "name": "wrong-market",
                    "rules": [
                        {
                            "condition": {"type": "feature_above", "column": "close", "threshold": 200.0},
                            "action": "long",
                        }
                    ],
                },
            },
        )

        with pytest.raises(ReleaseGateRejected) as rejected:
            _release(ReleaseService(session), graph, key="wrong-executable-market")

        predicates = {predicate.code: predicate for predicate in rejected.value.result.predicates}
        assert predicates["candidate_identity"].state == "blocked"
        assert "routing identity" in predicates["candidate_identity"].reason
        assert predicates["campaign_identity"].state == "blocked"
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 0
        session.rollback()


@pytest.mark.postgresql
@pytest.mark.parametrize(
    "candidate_config",
    [
        {
            "strategy_type": "rule",
            "symbol": "2330",
            "market": "tw_stock",
            "instrument": "spot",
            "interval": "1d",
            "payload": {
                "name": "unknown-condition",
                "rules": [{"condition": {"type": "not_registered"}, "action": "long"}],
            },
        },
        {
            "strategy_type": "rule",
            "symbol": "2330",
            "market": "tw_stock",
            "instrument": "spot",
            "interval": "1d",
            "payload": {
                "name": "unknown-action",
                "rules": [
                    {
                        "condition": {"type": "feature_above", "column": "close", "threshold": 200.0},
                        "action": "launch",
                    }
                ],
            },
        },
        {
            "strategy_type": "voting",
            "symbol": "2330",
            "market": "tw_stock",
            "instrument": "spot",
            "interval": "1d",
            "payload": {
                "name": "invalid-vote",
                "sub_signals": [{"type": "not_registered"}],
                "min_votes": 1,
            },
        },
    ],
)
def test_release_rejects_non_executable_nested_dsl_before_pointer_mutation(
    phase99_session_factory,
    candidate_config,
):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session, candidate_config_json=candidate_config)

        with pytest.raises(ReleaseGateRejected) as rejected:
            _release(ReleaseService(session), graph, key=f"invalid-dsl-{candidate_config['payload']['name']}")

        predicates = {predicate.code: predicate for predicate in rejected.value.result.predicates}
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert predicates["candidate_identity"].state == "blocked"
        assert "executable config" in predicates["candidate_identity"].reason
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=strategy.id).count() == 0
        session.rollback()


@pytest.mark.postgresql
def test_release_requires_exact_campaign_instrument_for_perp_candidate(phase99_session_factory):
    symbol = "BTC/USDT:USDT"
    reconciliation = synthetic_reconciliation_policy(
        currency="USDT",
        perp_instrument_rules={
            symbol: {
                "quantity_step": 0.001,
                "contract_multiplier": 1.0,
                "margin_semantics": "synthetic-isolated",
                "funding_semantics": "synthetic-periodic-cashflow",
            }
        },
    )
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(
            session,
            strategy_changes={"symbol": symbol, "market": "crypto_perp", "interval": "4h"},
            policy_changes={"market": "crypto_perp", "reconciliation": reconciliation},
            campaign_universe=[{"symbol": symbol, "market": "crypto_perp", "instrument": "spot"}],
        )

        with pytest.raises(ReleaseGateRejected) as rejected:
            _release(ReleaseService(session), graph, key="wrong-campaign-instrument")

        predicates = {predicate.code: predicate for predicate in rejected.value.result.predicates}
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert predicates["campaign_identity"].state == "blocked"
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=strategy.id).count() == 0
        session.rollback()


@pytest.mark.postgresql
def test_release_requires_exact_perp_contract_rule_for_executable_symbol(phase99_session_factory):
    symbol = "BTC/USDT:USDT"
    reconciliation = synthetic_reconciliation_policy(
        currency="USDT",
        perp_instrument_rules={
            "ETH/USDT:USDT": {
                "quantity_step": 0.001,
                "contract_multiplier": 1.0,
                "margin_semantics": "synthetic-isolated",
                "funding_semantics": "synthetic-periodic-cashflow",
            }
        },
    )
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(
            session,
            strategy_changes={"symbol": symbol, "market": "crypto_perp", "interval": "4h"},
            policy_changes={"market": "crypto_perp", "reconciliation": reconciliation},
            campaign_universe=[{"symbol": symbol, "market": "crypto_perp", "instrument": "perpetual"}],
        )

        with pytest.raises(ReleaseGateRejected) as rejected:
            _release(ReleaseService(session), graph, key="missing-perp-contract-rule")

        predicates = {predicate.code: predicate for predicate in rejected.value.result.predicates}
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert predicates["policy_identity"].state == "blocked"
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=strategy.id).count() == 0
        session.rollback()


@pytest.mark.postgresql
def test_rollback_rejects_target_whose_executable_interval_no_longer_matches_strategy(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)
        _release(service, graph)
        service.pause(
            graph.strategy_id,
            expected_revision=1,
            expected_active_version_id=graph.candidate_id,
            principal=release_manager(),
            idempotency_key="pause-before-routing-drift",
            reason="pause before validating rollback target",
        )
        strategy = session.get(StrategyRecord, graph.strategy_id)
        strategy.interval = "1h"
        session.flush()

        with pytest.raises(ReleaseGateRejected) as rejected:
            service.rollback(
                graph.strategy_id,
                graph.candidate_id,
                graph.reconciliation_id,
                expected_revision=2,
                expected_active_version_id=None,
                principal=release_manager(),
                idempotency_key="rollback-routing-drift",
                reason="must not restore mismatched executable routing",
                now=NOW,
            )

        target = next(
            predicate for predicate in rejected.value.result.predicates if predicate.code == "rollback_target"
        )
        assert target.state == "blocked"
        assert "routing identity" in target.reason
        assert (strategy.active_version_id, strategy.release_revision) == (None, 2)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 2
        session.rollback()


@pytest.mark.postgresql
def test_transition_rejects_stale_pointer_revision_and_unauthorized_principal(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        service = ReleaseService(session)
        worker = AuthPrincipal(
            "service:phase100-worker",
            frozenset({"decision-worker"}),
            frozenset({"paper:pilot"}),
        )

        with pytest.raises(HTTPException) as denied:
            _release(service, graph, principal=worker)
        assert denied.value.status_code == 403
        with pytest.raises(DecisionConflictError, match="revision"):
            _release(service, graph, expected_revision=1, key="stale-revision")
        with pytest.raises(DecisionConflictError, match="pointer"):
            _release(
                service,
                graph,
                expected_active_version_id=uuid.uuid4(),
                key="stale-pointer",
            )
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 0
        session.rollback()


@pytest.mark.postgresql
def test_caller_rollback_removes_release_event_pointer_and_revision(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        session.commit()

        _release(ReleaseService(session), graph)
        assert session.get(StrategyRecord, graph.strategy_id).release_revision == 1
        session.rollback()

        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 0


@pytest.mark.postgresql
def test_failure_after_event_flush_is_atomic(phase99_session_factory, monkeypatch):
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        session.commit()
        execute = session.execute

        def fail_guard(statement, *args, **kwargs):
            if "set_config('poseidon.release_event_id'" in str(statement):
                raise RuntimeError("injected pointer guard failure")
            return execute(statement, *args, **kwargs)

        monkeypatch.setattr(session, "execute", fail_guard)
        with pytest.raises(RuntimeError, match="injected pointer guard failure"):
            _release(ReleaseService(session), graph)
        session.rollback()

        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 0
