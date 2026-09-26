"""Decision service contracts; PostgreSQL race proof remains Plan 97-04."""

import copy
import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import create_engine, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import (
    DecisionConflictError,
    DecisionPolicy,
    DecisionService,
)
from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.manifest import ManifestService, ValidationError, content_sha256
from poseidon.models.base import Base
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    tables = [
        Base.metadata.tables[name]
        for name in (
            "data_manifests",
            "research_revisions",
            "strategy_versions",
            "evaluation_runs",
            "evaluation_snapshots",
            "decision_records",
            "decision_events",
        )
    ]
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as session:
        yield session
    engine.dispose()


def worker(scope="paper:pilot"):
    return AuthPrincipal("decision-worker:synthetic", frozenset({"decision-worker"}), frozenset({scope}))


def manager(actor="portfolio-manager:synthetic", scope="paper:pilot"):
    return AuthPrincipal(actor, frozenset({"portfolio_manager"}), frozenset({scope}))


def synthetic_policy(**changes):
    policy = {
        "policy_id": "synthetic-policy-v1",
        "mode": "human",
        "market": "tw_stock",
        "account_scope": "paper:pilot",
        "universe_id": "tw-stock-pilot-v1",
        "decision_ttl_seconds": 3600,
        "required_data": {"ohlcv": {"max_age_seconds": 86400}},
        "hard_limits": {"max_gross_exposure": 1.0, "max_position_weight": 0.2},
        "approval_roles": ["portfolio_manager"],
        "protective_exit": {"allowed_without_approval": True},
        "release_gate": {"minimum_mature_samples": 10, "requires_human_release": True},
    }
    policy.update(changes)
    return policy


def manifest_request():
    payload = {"close": 100.0, "symbol": "2330"}
    return {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": "2026-09-26T12:00:00Z",
        "account_scope": "paper:pilot",
        "universe_id": "tw-stock-pilot-v1",
        "capability_json": {"ohlcv": True},
        "required_data": {"ohlcv": {"max_age_seconds": 86400}},
        "evidence": [
            {
                "id": "ohlcv",
                "event_time": "2026-09-26T00:00:00Z",
                "available_at": "2026-09-26T01:00:00Z",
                "recorded_at": "2026-09-26T02:00:00Z",
                "source_uri": "fixture://ohlcv",
                "kind": "ohlcv",
                "payload": payload,
                "content_sha256": content_sha256(payload),
            }
        ],
    }


def decision_inputs(db, *, policy=None, non_evaluated_status="no_trade"):
    manifest = ManifestService(db).freeze(manifest_request())
    policy = synthetic_policy() if policy is None else policy
    version = StrategyVersion(
        strategy_id=uuid.uuid4(),
        version_no=1,
        config_json={"fixture": "synthetic"},
        policy_json=policy,
        artifact_json={},
        content_sha256=strategy_version_digest({"fixture": "synthetic"}, policy, {}),
        status="draft",
    )
    db.add(version)
    db.flush()
    universe = [
        {"symbol": "2330", "market": "tw_stock", "instrument": "spot"},
        {"symbol": "2317", "market": "tw_stock", "instrument": "spot"},
    ]
    snapshots = [
        {
            **universe[0],
            "status": "evaluated",
            "recommendation_json": {"research_status": "not_required"},
            "reason_codes": [],
            "valid_until": "2026-09-26T15:00:00Z",
        },
        {
            **universe[1],
            "status": non_evaluated_status,
            "recommendation_json": {"research_status": "not_required"},
            "reason_codes": ["synthetic_no_trade"],
            "valid_until": "2026-09-26T15:00:00Z",
        },
    ]
    run = EvaluationService(db).evaluate_run(version.id, manifest.id, universe, snapshots)
    rows = db.query(EvaluationSnapshot).filter_by(evaluation_run_id=run.id).all()
    snapshot_by_symbol = {row.symbol: row for row in rows}
    snapshot_ids = [str(snapshot_by_symbol[symbol].id) for symbol in ("2330", "2317")]
    return run, version, snapshot_ids


def create_decision(db, run_id, snapshot_ids, **changes):
    values = {
        "principal": worker(),
        "account_scope": "paper:pilot",
        "original_json": {"selected_evaluation_ids": [snapshot_ids[0]], "final_action": "hold"},
        "portfolio_snapshot_json": {"cash": 100000.0, "positions": []},
        "risk_snapshot_json": {"hard_failures": [], "allowed_actions": ["hold", "reduce", "exit"]},
        "now": datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
    }
    values.update(changes)
    return DecisionService(db).create_decision(run_id, **values)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("policy_id"),
        lambda value: value.update(extra="forbidden"),
        lambda value: value.update(mode="unknown"),
        lambda value: value.update(market="crypto"),
        lambda value: value.update(decision_ttl_seconds=True),
        lambda value: value.update(decision_ttl_seconds=0),
        lambda value: value.update(hard_limits={"max_gross_exposure": 1.0}),
        lambda value: value.update(hard_limits={"max_gross_exposure": float("inf"), "max_position_weight": 0.2}),
        lambda value: value.update(hard_limits={"max_gross_exposure": 1.0, "max_position_weight": 0.0}),
        lambda value: value.update(approval_roles=[]),
        lambda value: value.update(approval_roles=["viewer"]),
        lambda value: value.update(protective_exit={}),
        lambda value: value.update(release_gate={"minimum_mature_samples": 10}),
        lambda value: value.update(release_gate={"minimum_mature_samples": 0, "requires_human_release": True}),
    ],
)
def test_policy_validation_rejects_incomplete_or_unsafe_values(mutate):
    policy = synthetic_policy()
    mutate(policy)
    with pytest.raises(PydanticValidationError):
        DecisionPolicy.model_validate(policy)


@pytest.mark.parametrize(
    "rule",
    [
        {},
        None,
        {"max_age_seconds": -1},
        {"max_age_seconds": True},
        {"max_age_seconds": "86400"},
        {"max_age_seconds": float("inf")},
        {"max_age_seconds": float("nan")},
        {"max_age_seconds": 86400, "unknown": True},
    ],
)
def test_policy_rejects_invalid_freshness_rules(rule):
    with pytest.raises(PydanticValidationError):
        DecisionPolicy.model_validate(synthetic_policy(required_data={"ohlcv": rule}))


@pytest.mark.parametrize("kind", ["", " ohlcv", "ohlcv "])
def test_policy_rejects_invalid_required_data_kind(kind):
    with pytest.raises(PydanticValidationError):
        DecisionPolicy.model_validate(synthetic_policy(required_data={kind: {"max_age_seconds": 86400}}))


@pytest.mark.parametrize(
    "required_data,reason",
    [
        ({"ohlcv": {"max_age_seconds": 1}}, "stale"),
        ({"fundamental": {"max_age_seconds": 86400}}, "missing"),
    ],
)
def test_creation_enforces_policy_data_rules_over_manifest_rules(db, required_data, reason):
    run, _, snapshot_ids = decision_inputs(db, policy=synthetic_policy(required_data=required_data))
    with pytest.raises(ValidationError, match=reason):
        create_decision(db, run.id, snapshot_ids)
    assert db.query(DecisionRecord).count() == 0


def test_create_decision_is_replay_safe_and_appends_one_event(db):
    run, _, snapshot_ids = decision_inputs(db)
    first = create_decision(db, run.id, snapshot_ids)
    replay = create_decision(db, run.id, snapshot_ids)

    assert replay.id == first.id
    assert first.status == "pending_approval"
    assert first.revision == 1
    assert first.original_json == first.final_json
    assert db.query(DecisionEvent).filter_by(decision_id=first.id).count() == 1


@pytest.mark.parametrize("field", ["account_scope", "universe_id"])
def test_create_decision_rejects_policy_scope_mismatch(db, field):
    policy = synthetic_policy(**{field: "mismatch"})
    run, _, snapshot_ids = decision_inputs(db, policy=policy)
    with pytest.raises(ValidationError, match=field):
        create_decision(db, run.id, snapshot_ids)


@pytest.mark.parametrize("case", ["foreign", "duplicate", "nonterminal", "increase", "risk", "portfolio"])
def test_create_decision_rejects_invalid_selection_or_snapshot(db, case):
    run, _, snapshot_ids = decision_inputs(db)
    changes = {}
    original = {"selected_evaluation_ids": [snapshot_ids[0]], "final_action": "hold"}
    if case == "foreign":
        original["selected_evaluation_ids"] = [str(uuid.uuid4())]
    elif case == "duplicate":
        original["selected_evaluation_ids"] *= 2
    elif case == "nonterminal":
        row = db.get(EvaluationSnapshot, uuid.UUID(snapshot_ids[0]))
        row.status = "pending"
    elif case == "increase":
        original = {"selected_evaluation_ids": [snapshot_ids[1]], "final_action": "enter"}
        changes["risk_snapshot_json"] = {"hard_failures": [], "allowed_actions": ["enter"]}
    elif case == "risk":
        changes["risk_snapshot_json"] = {"hard_failures": "none", "allowed_actions": ["hold"]}
    else:
        changes["portfolio_snapshot_json"] = []
    changes["original_json"] = original

    with db.no_autoflush, pytest.raises(ValidationError):
        create_decision(db, run.id, snapshot_ids, **changes)
    assert db.query(DecisionRecord).count() == 0


def test_changed_frozen_input_supersedes_the_previous_candidate_once(db):
    run, _, snapshot_ids = decision_inputs(db)
    first = create_decision(db, run.id, snapshot_ids)
    second = create_decision(
        db,
        run.id,
        snapshot_ids,
        portfolio_snapshot_json={"cash": 90000.0, "positions": []},
    )

    assert second.id != first.id
    assert second.creation_sha256 != first.creation_sha256
    assert first.status == "superseded"
    assert first.revision == 2
    assert db.query(DecisionEvent).filter_by(decision_id=first.id, event_type="superseded").count() == 1


def test_expiry_is_one_transition_and_eligibility_is_fail_closed(db):
    run, version, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    service = DecisionService(db)
    decision.status = "approved"
    assert service.is_execution_eligible(decision.id, 1, datetime(2026, 9, 26, 12, 30, tzinfo=UTC))
    assert not service.is_execution_eligible(decision.id, 2, datetime(2026, 9, 26, 12, 30, tzinfo=UTC))

    expired = service.expire_due(datetime(2026, 9, 26, 13, 0, tzinfo=UTC))
    assert [row.id for row in expired] == [decision.id]
    assert decision.status == "expired"
    assert decision.revision == 2
    assert service.expire_due(datetime(2026, 9, 26, 14, 0, tzinfo=UTC)) == []
    assert not service.is_execution_eligible(decision.id, 2, datetime(2026, 9, 26, 12, 30, tzinfo=UTC))
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="expired").count() == 1

    version.policy_json = copy.deepcopy(version.policy_json) | {"decision_ttl_seconds": 7200}
    assert not service.is_execution_eligible(decision.id, 2, datetime(2026, 9, 26, 12, 30, tzinfo=UTC))


@pytest.mark.parametrize("operation", ["superseded", "expired"])
def test_system_transition_refreshes_revision_after_another_writer(db, operation):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    # Keep the identity map stale, as when another transaction approves the row.
    db.execute(
        update(DecisionRecord)
        .where(DecisionRecord.id == decision.id)
        .values(status="approved", revision=2)
        .execution_options(synchronize_session=False)
    )
    assert decision.revision == 1
    if operation == "superseded":
        create_decision(db, run.id, snapshot_ids, portfolio_snapshot_json={"cash": 90000.0})
    else:
        DecisionService(db).expire_due(datetime(2026, 9, 26, 13, 0, tzinfo=UTC))
    assert (decision.status, decision.revision) == (operation, 3)
    event = db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type=operation).one()
    assert event.expected_revision == 2


def test_creation_requires_worker_role_and_exact_account_scope(db):
    run, _, snapshot_ids = decision_inputs(db)
    for principal in (manager(), worker("paper:other")):
        with pytest.raises(HTTPException) as error:
            create_decision(db, run.id, snapshot_ids, principal=principal)
        assert error.value.status_code == 403


def test_approve_exact_double_click_replays_one_transition(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    service = DecisionService(db)
    body = {"expected_revision": 1}

    first = service.approve(
        decision.id,
        body,
        principal=manager(),
        idempotency_key="approve-once",
        now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
    )
    replay = service.approve(
        decision.id,
        body,
        principal=manager(),
        idempotency_key="approve-once",
        now=datetime(2026, 9, 26, 12, 31, tzinfo=UTC),
    )

    assert replay == first
    assert first["status"] == "approved"
    assert first["revision"] == 2
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="approved").count() == 1


def test_idempotency_conflicts_on_changed_body_operation_actor_or_new_key(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    service = DecisionService(db)
    now = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)
    service.approve(
        decision.id,
        {"expected_revision": 1},
        principal=manager(),
        idempotency_key="same-key",
        now=now,
    )

    attempts = [
        lambda: service.approve(
            decision.id,
            {"expected_revision": 1, "override_reason": "changed"},
            principal=manager(),
            idempotency_key="same-key",
            now=now,
        ),
        lambda: service.reject(
            decision.id,
            {"expected_revision": 1, "reason": "changed operation"},
            principal=manager(),
            idempotency_key="same-key",
            now=now,
        ),
        lambda: service.approve(
            decision.id,
            {"expected_revision": 1},
            principal=manager(actor="portfolio-manager:other"),
            idempotency_key="same-key",
            now=now,
        ),
        lambda: service.approve(
            decision.id,
            {"expected_revision": 1},
            principal=manager(),
            idempotency_key="different-key",
            now=now,
        ),
    ]
    for attempt in attempts:
        with pytest.raises(DecisionConflictError):
            attempt()
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="approved").count() == 1


@pytest.mark.parametrize("case", ["stale", "expired", "superseded", "policy", "hard_failure"])
def test_approval_conflicts_for_invalid_frozen_state(db, case):
    run, version, snapshot_ids = decision_inputs(db)
    risk = {"hard_failures": [], "allowed_actions": ["hold", "reduce", "exit"]}
    if case == "hard_failure":
        risk["hard_failures"] = ["synthetic_limit"]
    decision = create_decision(db, run.id, snapshot_ids, risk_snapshot_json=risk)
    body = {"expected_revision": 2 if case == "stale" else 1}
    now = datetime(2026, 9, 26, 13 if case == "expired" else 12, 30, tzinfo=UTC)
    if case == "superseded":
        create_decision(
            db,
            run.id,
            snapshot_ids,
            portfolio_snapshot_json={"cash": 90000.0, "positions": []},
        )
    elif case == "policy":
        version.policy_json = copy.deepcopy(version.policy_json) | {"decision_ttl_seconds": 7200}

    with db.no_autoflush, pytest.raises(DecisionConflictError):
        DecisionService(db).approve(
            decision.id,
            body,
            principal=manager(),
            idempotency_key=f"approve-{case}",
            now=now,
        )
    with db.no_autoflush:
        assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="approved").count() == 0


def test_authorization_runs_before_stored_idempotency_replay(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    service = DecisionService(db)
    body = {"expected_revision": 1}
    now = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)
    service.approve(decision.id, body, principal=manager(), idempotency_key="protected", now=now)

    for principal in (worker(), manager(scope="paper:other")):
        with pytest.raises(HTTPException) as error:
            service.approve(decision.id, body, principal=principal, idempotency_key="protected", now=now)
        assert error.value.status_code == 403


def test_soft_override_replaces_final_json_and_preserves_original(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    original = copy.deepcopy(decision.original_json)
    response = DecisionService(db).approve(
        decision.id,
        {"expected_revision": 1, "final_action": "reduce", "override_reason": "Synthetic review"},
        principal=manager(),
        idempotency_key="override",
        now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
    )

    assert decision.original_json == original
    assert decision.final_json == {
        **original,
        "final_action": "reduce",
        "override_reason": "Synthetic review",
    }
    assert response["final_json"] == decision.final_json
    db.expire(decision, ["final_json"])
    assert decision.final_json["final_action"] == "reduce"


@pytest.mark.parametrize("status", ["failed", "excluded", "no_trade"])
@pytest.mark.parametrize("action", ["enter", "add"])
def test_override_cannot_increase_exposure_for_non_evaluated_selection(db, status, action):
    run, _, snapshot_ids = decision_inputs(db, non_evaluated_status=status)
    decision = create_decision(
        db,
        run.id,
        snapshot_ids,
        original_json={"selected_evaluation_ids": [snapshot_ids[1]], "final_action": "hold"},
        risk_snapshot_json={"hard_failures": [], "allowed_actions": ["hold", action]},
    )
    with pytest.raises(DecisionConflictError, match="evaluated selections"):
        DecisionService(db).approve(
            decision.id,
            {"expected_revision": 1, "final_action": action, "override_reason": "Synthetic override"},
            principal=manager(),
            idempotency_key="invalid-exposure-override",
            now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )
    assert decision.status == "pending_approval"
    assert decision.revision == 1
    assert decision.final_json["final_action"] == "hold"
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="approved").count() == 0


@pytest.mark.parametrize(
    "body",
    [
        {"expected_revision": 1, "final_action": "reduce"},
        {"expected_revision": 1, "final_action": "enter", "override_reason": "Synthetic review"},
    ],
)
def test_soft_override_requires_reason_and_frozen_allowed_action(db, body):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    with pytest.raises(DecisionConflictError):
        DecisionService(db).approve(
            decision.id,
            body,
            principal=manager(),
            idempotency_key="invalid-override",
            now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )


def test_reject_requires_reason_and_replays_exact_response(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    service = DecisionService(db)
    body = {"expected_revision": 1, "reason": "Synthetic rejection"}
    now = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)
    first = service.reject(decision.id, body, principal=manager(), idempotency_key="reject-once", now=now)
    replay = service.reject(decision.id, body, principal=manager(), idempotency_key="reject-once", now=now)

    assert replay == first
    assert first["status"] == "rejected"
    assert db.query(DecisionEvent).filter_by(decision_id=decision.id, event_type="rejected").count() == 1


def test_get_pending_and_trace_are_account_scoped_and_frozen(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    service = DecisionService(db)
    principal = manager()

    assert service.get(decision.id, principal=principal).id == decision.id
    assert [
        row.id for row in service.list_pending(principal=principal, now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC))
    ] == [decision.id]
    trace = service.trace(decision.id, principal=principal)

    assert trace["decision"]["id"] == str(decision.id)
    assert trace["evaluation_run"]["id"] == str(run.id)
    assert trace["manifest"]["id"]
    assert trace["manifest"]["content_sha256"]
    assert len(trace["evaluations"]) == 2
    assert sum(item["selected"] for item in trace["evaluations"]) == 1
    assert {item["research"]["research_status"] for item in trace["evaluations"]} == {"not_required"}
    assert [item["event_type"] for item in trace["events"]] == ["created"]
    assert not {"orders", "claims", "fills"}.intersection(trace)

    with pytest.raises(HTTPException) as error:
        service.trace(decision.id, principal=manager(scope="paper:other"))
    assert error.value.status_code == 403


def test_transition_body_rejects_actor_identity(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    with pytest.raises(PydanticValidationError):
        DecisionService(db).approve(
            decision.id,
            {"expected_revision": 1, "actor_id": "forged"},
            principal=manager(),
            idempotency_key="forged",
            now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )
