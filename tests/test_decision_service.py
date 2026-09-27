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
from poseidon.orders.schemas import Fill, Order
from poseidon.orders.state_machine import OrderStatus, transition_order


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
            "orders",
            "order_fills",
            "position_lots",
            "fill_allocations",
            "account_reconciliations",
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


def synthetic_reconciliation_policy(**changes):
    policy = {
        "account_generation": "synthetic-generation-1",
        "opening_cash": 100000.0,
        "currency": "TWD",
        "max_reconciliation_age_seconds": 300,
        "cash_tolerance": 0.01,
        "position_tolerance": 0.0,
        "fill_tolerance": 0.0,
        "tw_stock_quantity_rounding": "whole_share_floor",
        "perp_instrument_rules": {},
    }
    policy.update(changes)
    return policy


def executable_intent(snapshot_id, symbol="2330", **changes):
    intent = {
        "evaluation_snapshot_id": snapshot_id,
        "symbol": symbol,
        "market": "tw_stock",
        "instrument": "spot",
        "action": "enter",
        "side": "long",
        "target_weight": 0.1,
        "order_type": "market",
    }
    intent.update(changes)
    return intent


def manifest_request(*, market="tw_stock", symbol="2330"):
    payload = {"close": 100.0, "symbol": symbol}
    return {
        "market": market,
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


def decision_inputs(db, *, policy=None, non_evaluated_status="no_trade", market="tw_stock", universe=None):
    universe = universe or [
        {"symbol": "2330", "market": "tw_stock", "instrument": "spot"},
        {"symbol": "2317", "market": "tw_stock", "instrument": "spot"},
    ]
    manifest = ManifestService(db).freeze(manifest_request(market=market, symbol=universe[0]["symbol"]))
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
    snapshots = [
        {
            **member,
            "status": "evaluated" if index == 0 else non_evaluated_status,
            "recommendation_json": {"research_status": "not_required", "side": "long"},
            "reason_codes": [] if index == 0 else ["synthetic_no_trade"],
            "valid_until": "2026-09-26T15:00:00Z",
        }
        for index, member in enumerate(universe)
    ]
    run = EvaluationService(db).evaluate_run(version.id, manifest.id, universe, snapshots)
    rows = db.query(EvaluationSnapshot).filter_by(evaluation_run_id=run.id).all()
    snapshot_by_symbol = {row.symbol: row for row in rows}
    snapshot_ids = [str(snapshot_by_symbol[member["symbol"]].id) for member in universe]
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


@pytest.mark.parametrize(
    "change",
    [
        {"cash_tolerance": -0.01},
        {"position_tolerance": float("inf")},
        {"fill_tolerance": True},
        {"max_reconciliation_age_seconds": 0},
        {"opening_cash": float("nan")},
        {"currency": " twd"},
        {"unknown": "forbidden"},
    ],
)
def test_reconciliation_policy_is_optional_but_strict_without_tolerance_defaults(change):
    historical = DecisionPolicy.model_validate(synthetic_policy())
    assert historical.reconciliation is None

    current = DecisionPolicy.model_validate(synthetic_policy(reconciliation=synthetic_reconciliation_policy()))
    assert current.reconciliation.cash_tolerance == 0.01
    assert current.reconciliation.position_tolerance == 0.0
    assert current.reconciliation.fill_tolerance == 0.0

    invalid = synthetic_reconciliation_policy(**change)
    with pytest.raises(PydanticValidationError):
        DecisionPolicy.model_validate(synthetic_policy(reconciliation=invalid))


def test_sizing_policy_requires_whole_share_floor_and_complete_perp_rules():
    with pytest.raises(PydanticValidationError):
        DecisionPolicy.model_validate(
            synthetic_policy(reconciliation=synthetic_reconciliation_policy(tw_stock_quantity_rounding="nearest_lot"))
        )

    crypto_policy = synthetic_policy(
        market="crypto_perp",
        reconciliation=synthetic_reconciliation_policy(currency="USDT"),
    )
    with pytest.raises(PydanticValidationError, match="perp_instrument_rules"):
        DecisionPolicy.model_validate(crypto_policy)

    crypto_policy["reconciliation"]["perp_instrument_rules"] = {
        "BTC/USDT:USDT": {
            "quantity_step": 0.001,
            "contract_multiplier": 1.0,
            "margin_semantics": "synthetic-isolated",
            "funding_semantics": "synthetic-periodic-cashflow",
        }
    }
    parsed = DecisionPolicy.model_validate(crypto_policy)
    assert parsed.reconciliation.perp_instrument_rules["BTC/USDT:USDT"].quantity_step == 0.001


def test_sizing_policy_requires_a_rule_for_each_perp_intent_instrument(db):
    crypto_policy = synthetic_policy(
        market="crypto_perp",
        reconciliation=synthetic_reconciliation_policy(
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
    )
    universe = [{"symbol": "ETH/USDT", "market": "crypto_perp", "instrument": "ETH/USDT:USDT"}]
    run, _, snapshot_ids = decision_inputs(
        db,
        policy=crypto_policy,
        market="crypto_perp",
        universe=universe,
    )
    with pytest.raises(ValidationError, match="perp instrument rule"):
        create_decision(
            db,
            run.id,
            snapshot_ids,
            original_json={
                "selected_evaluation_ids": snapshot_ids,
                "final_action": "enter",
                "order_intents": [
                    executable_intent(
                        snapshot_ids[0],
                        symbol="ETH/USDT",
                        market="crypto_perp",
                        instrument="ETH/USDT:USDT",
                    )
                ],
            },
            risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
        )


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


@pytest.mark.parametrize(
    "order_intents",
    [
        None,
        [],
        [{"extra": "forbidden"}],
    ],
)
def test_order_intent_is_required_and_extra_forbidden_for_executable_actions(db, order_intents):
    run, _, snapshot_ids = decision_inputs(db)
    original = {"selected_evaluation_ids": [snapshot_ids[0]], "final_action": "enter"}
    if order_intents is not None:
        original["order_intents"] = (
            [executable_intent(snapshot_ids[0], extra="forbidden")] if order_intents else order_intents
        )
    with pytest.raises(ValidationError, match="order_intents"):
        create_decision(
            db,
            run.id,
            snapshot_ids,
            original_json=original,
            risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
        )


@pytest.mark.parametrize("action", ["watch", "hold"])
def test_order_intent_is_absent_for_non_executable_action(db, action):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(
        db,
        run.id,
        snapshot_ids,
        original_json={"selected_evaluation_ids": [snapshot_ids[0]], "final_action": action},
        risk_snapshot_json={"hard_failures": [], "allowed_actions": [action]},
    )
    assert "order_intents" not in decision.final_json

    with pytest.raises(ValidationError, match="must not contain order_intents"):
        create_decision(
            db,
            run.id,
            snapshot_ids,
            original_json={
                "selected_evaluation_ids": [snapshot_ids[0]],
                "final_action": action,
                "order_intents": [executable_intent(snapshot_ids[0])],
            },
            risk_snapshot_json={"hard_failures": [], "allowed_actions": [action]},
        )


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda intent: intent.update(evaluation_snapshot_id=str(uuid.uuid4())), "selected snapshot"),
        (lambda intent: intent.update(symbol="2317"), "snapshot identity"),
        (lambda intent: intent.update(market="crypto_perp"), "snapshot identity"),
        (lambda intent: intent.update(instrument="perpetual"), "snapshot identity"),
        (lambda intent: intent.update(side="short"), "snapshot side"),
        (lambda intent: intent.update(target_weight=float("inf")), "finite JSON"),
        (lambda intent: intent.update(unknown="forbidden"), "order_intents"),
    ],
)
def test_order_intent_rejects_foreign_mismatched_nonfinite_or_extra_values(db, mutate, reason):
    run, _, snapshot_ids = decision_inputs(db)
    intent = executable_intent(snapshot_ids[0])
    mutate(intent)
    with pytest.raises(ValidationError, match=reason):
        create_decision(
            db,
            run.id,
            snapshot_ids,
            original_json={
                "selected_evaluation_ids": [snapshot_ids[0]],
                "final_action": "enter",
                "order_intents": [intent],
            },
            risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
        )


def test_order_intent_is_unique_per_snapshot_and_canonical_across_two_symbols(db):
    run, _, snapshot_ids = decision_inputs(db, non_evaluated_status="evaluated")
    original = {
        "selected_evaluation_ids": snapshot_ids,
        "final_action": "enter",
        "order_intents": [
            executable_intent(snapshot_ids[1], symbol="2317", target_weight=0.2),
            executable_intent(snapshot_ids[0], symbol="2330", target_weight=0.1),
        ],
    }
    decision = create_decision(
        db,
        run.id,
        snapshot_ids,
        original_json=original,
        risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
    )
    assert [intent["symbol"] for intent in decision.final_json["order_intents"]] == ["2317", "2330"]
    assert len({intent["evaluation_snapshot_id"] for intent in decision.final_json["order_intents"]}) == 2

    duplicate = copy.deepcopy(original)
    duplicate["order_intents"][1] = copy.deepcopy(duplicate["order_intents"][0])
    with pytest.raises(ValidationError, match="unique selected snapshot"):
        create_decision(
            db,
            run.id,
            snapshot_ids,
            original_json=duplicate,
            risk_snapshot_json={"hard_failures": [], "allowed_actions": ["enter"]},
        )


@pytest.mark.parametrize(
    "action,intent_action,target_weight",
    [("enter", "add", 0.1), ("add", "add", 0.0), ("exit", "exit", 0.1)],
)
def test_action_consistency_and_target_weight_fail_closed(db, action, intent_action, target_weight):
    run, _, snapshot_ids = decision_inputs(db)
    with pytest.raises(ValidationError):
        create_decision(
            db,
            run.id,
            snapshot_ids,
            original_json={
                "selected_evaluation_ids": [snapshot_ids[0]],
                "final_action": action,
                "order_intents": [
                    executable_intent(snapshot_ids[0], action=intent_action, target_weight=target_weight)
                ],
            },
            risk_snapshot_json={"hard_failures": [], "allowed_actions": [action]},
        )


def test_order_intent_state_machine_preserves_unknown_attempt_and_dto_compatibility():
    assert transition_order(OrderStatus.PENDING, OrderStatus.PENDING_SUBMIT) == OrderStatus.PENDING_SUBMIT
    assert (
        transition_order(OrderStatus.PENDING_SUBMIT, OrderStatus.RECONCILIATION_REQUIRED)
        == OrderStatus.RECONCILIATION_REQUIRED
    )
    with pytest.raises(ValueError):
        transition_order(OrderStatus.PENDING_SUBMIT, OrderStatus.REJECTED)
    with pytest.raises(ValueError):
        transition_order(OrderStatus.RECONCILIATION_REQUIRED, OrderStatus.PENDING_SUBMIT)

    legacy_order = Order("2330", "tw_stock", "buy", "market", 0.1, 1.0, "legacy", "paper")
    legacy_fill = Fill(legacy_order.id, 100.0, 1.0, datetime(2026, 9, 26, tzinfo=UTC))
    assert legacy_order.decision_id is None
    assert legacy_order.client_order_ref is None
    assert legacy_fill.projection_status is None

    fully_positional = Order(
        "2330",
        "tw_stock",
        "buy",
        "market",
        0.1,
        1.0,
        "legacy",
        "paper",
        "long",
        None,
        OrderStatus.PENDING,
        None,
        None,
        None,
        "manual",
        "legacy-order-id",
    )
    assert fully_positional.id == "legacy-order-id"
    assert fully_positional.decision_id is None


@pytest.mark.parametrize("uuid_format", ["uppercase", "hex"])
def test_selected_uuid_formats_share_identity_and_trace_selection(db, uuid_format):
    run, _, snapshot_ids = decision_inputs(db)
    selected_id = snapshot_ids[0].upper() if uuid_format == "uppercase" else uuid.UUID(snapshot_ids[0]).hex
    original = {"selected_evaluation_ids": [selected_id], "final_action": "hold"}
    first = create_decision(db, run.id, snapshot_ids, original_json=original)
    replay = create_decision(db, run.id, snapshot_ids)
    assert first.id == replay.id
    assert first.original_json["selected_evaluation_ids"] == [snapshot_ids[0]]
    assert original["selected_evaluation_ids"] == [selected_id]
    trace = DecisionService(db).trace(first.id, principal=manager())
    assert [row["id"] for row in trace["evaluations"] if row["selected"]] == [snapshot_ids[0]]
    assert db.query(DecisionRecord).count() == 1
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
    run, version, snapshot_ids = decision_inputs(
        db,
        policy=synthetic_policy(reconciliation=synthetic_reconciliation_policy()),
    )
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


def test_legacy_executable_decision_remains_traceable_but_validator_reports_missing_intents(db):
    run, _, snapshot_ids = decision_inputs(
        db,
        policy=synthetic_policy(reconciliation=synthetic_reconciliation_policy()),
    )
    decision = create_decision(db, run.id, snapshot_ids)
    decision.status = "approved"
    decision.revision = 2
    decision.final_json = {"selected_evaluation_ids": [snapshot_ids[0]], "final_action": "enter"}

    trace = DecisionService(db).trace(decision.id, principal=manager())
    assert trace["decision"]["final_json"]["final_action"] == "enter"
    with pytest.raises(DecisionConflictError, match="requires order_intents"):
        DecisionService(db).validate_execution(
            decision.id,
            2,
            datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )


def test_legacy_executable_validator_reports_missing_reconciliation_policy(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    decision.status = "approved"
    decision.revision = 2

    with pytest.raises(DecisionConflictError, match="reconciliation policy"):
        DecisionService(db).validate_execution(
            decision.id,
            2,
            datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )


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


def test_action_consistency_rejects_override_without_matching_frozen_intents(db):
    run, _, snapshot_ids = decision_inputs(db)
    decision = create_decision(db, run.id, snapshot_ids)
    original = copy.deepcopy(decision.original_json)
    with pytest.raises(DecisionConflictError, match="order_intents"):
        DecisionService(db).approve(
            decision.id,
            {"expected_revision": 1, "final_action": "reduce", "override_reason": "Synthetic review"},
            principal=manager(),
            idempotency_key="override",
            now=datetime(2026, 9, 26, 12, 30, tzinfo=UTC),
        )

    assert decision.original_json == original
    assert decision.final_json == original
    assert decision.status == "pending_approval"


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
    assert "claims" not in trace
    assert trace["orders"] == []
    assert trace["fills"] == []
    assert trace["lots"] == []
    assert trace["allocations"] == []
    assert trace["reconciliations"] == []

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
