"""Deterministic manifest -> evaluation run -> pending decision producer (Phase 100-07)."""

import ast
import copy
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionPolicy
from poseidon.decision_loop.evaluation import EvaluationService, snapshot_payload
from poseidon.decision_loop.manifest import ManifestService, content_sha256, validate_manifest
from poseidon.decision_loop.producer import (
    CANDIDATE_STATUS,
    ProducerService,
    build_run_inputs,
    research_scope_key,
    resolve_worker_principal,
)
from poseidon.models.base import Base
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.evaluation_snapshot import EvaluationSnapshot
from poseidon.models.position_lot import PositionLot
from poseidon.models.signal import SignalRecord
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import synthetic_policy, synthetic_reconciliation_policy

POSEIDON_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "paper:pilot"
AS_OF = "2026-09-26T12:00:00Z"
LAST_BAR = datetime(2026, 9, 26, tzinfo=UTC)
RISING = [100.0, 100.5, 101.0, 101.2, 106.5]
FLAT = [100.0, 100.0, 100.0, 100.0, 100.0]


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db():
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def register_postgres_text_functions(dbapi_connection, _connection_record):
        dbapi_connection.create_function("char_length", 1, len)
        dbapi_connection.create_function("btrim", 1, lambda value: value.strip())

    tables = [
        Base.metadata.tables[name]
        for name in (
            "strategies",
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
            "signals",
        )
    ]
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as session:
        yield session
    engine.dispose()


def worker(scope=ACCOUNT):
    return AuthPrincipal("decision-worker:producer", frozenset({"decision-worker"}), frozenset({scope}))


def rule_config(symbol="2330", strategy_type="rule"):
    payload = (
        {}
        if strategy_type == "liquidity_sweep"
        else {
            "name": f"producer-{symbol}",
            "rules": [
                {
                    "condition": {"type": "price_change_pct", "threshold": 0.01},
                    "action": "long",
                    "quantity_pct": 0.1,
                    "confidence": 0.8,
                }
            ],
        }
    )
    return {
        "strategy_type": strategy_type,
        "symbol": symbol,
        "market": "tw_stock",
        "instrument": "spot",
        "interval": "1d",
        "payload": payload,
    }


def ohlcv_evidence(symbol, closes=RISING, last_bar=LAST_BAR):
    start = last_bar - timedelta(days=len(closes) - 1)
    bars = [
        {
            "time": (start + timedelta(days=index)).isoformat().replace("+00:00", "Z"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
        }
        for index, close in enumerate(closes)
    ]
    payload = {"symbol": symbol, "market": "tw_stock", "interval": "1d", "bars": bars}
    event_time = bars[-1]["time"]
    return {
        "id": f"ohlcv:{symbol}",
        "kind": "ohlcv",
        "source_uri": f"fixture://ohlcv/{symbol}",
        "event_time": event_time,
        "available_at": event_time,
        "recorded_at": AS_OF,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def manifest_request(*evidence):
    return {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": AS_OF,
        "account_scope": ACCOUNT,
        "universe_id": "tw-stock-pilot-v1",
        "capability_json": {"ohlcv": True},
        "required_data": {"ohlcv": {"max_age_seconds": 86400}},
        "evidence": list(evidence) or [ohlcv_evidence("2330")],
    }


def make_version(db, config=None, *, policy=None, artifact=None, strategy_id=None):
    config = config or rule_config()
    policy = synthetic_policy() if policy is None else policy
    artifact = artifact or {}
    strategy_id = strategy_id or uuid.uuid4()
    version = StrategyVersion(
        strategy_id=strategy_id,
        version_no=1,
        config_json=config,
        policy_json=policy,
        artifact_json=artifact,
        content_sha256=strategy_version_digest(config, policy, artifact),
        status="draft",
    )
    db.add(version)
    db.flush()
    return version


def long_signal(_executable, _bars):
    return [{"action": "long", "confidence": 0.9, "quantity_pct": 0.1}]


def no_signal(_executable, _bars):
    return []


def close_signal(_executable, _bars):
    return [{"action": "close", "confidence": 1.0, "quantity_pct": None}]


def broken_signal(_executable, _bars):
    raise RuntimeError("synthetic strategy failure")


def _build(request, version, *, evaluator=long_signal, lookup=lambda _scope: None, open_positions=None):
    kwargs = {} if open_positions is None else {"open_positions": open_positions}
    return build_run_inputs(
        validate_manifest(request),
        version,
        DecisionPolicy.model_validate(version.policy_json),
        research_lookup=lookup,
        signal_evaluator=evaluator,
        **kwargs,
    )


def _counts(db):
    return tuple(db.query(model).count() for model in (EvaluationRun, EvaluationSnapshot, DecisionRecord))


def test_build_run_inputs_is_pure_and_deterministic(db):
    version = make_version(db)
    request = manifest_request(ohlcv_evidence("2330"))
    first_universe, first = _build(copy.deepcopy(request), version)
    second_universe, second = _build(copy.deepcopy(request), version)
    assert first_universe == second_universe == [{"symbol": "2330", "market": "tw_stock", "instrument": "spot"}]
    assert [content_sha256(snapshot_payload(item)) for item in first] == [
        content_sha256(snapshot_payload(item)) for item in second
    ]
    assert first[0]["status"] == CANDIDATE_STATUS
    assert first[0]["valid_until"] == "2026-09-26T13:00:00Z"
    assert first[0]["recommendation_json"]["intent"] == {
        "symbol": "2330",
        "market": "tw_stock",
        "instrument": "spot",
        "action": "enter",
        "side": "long",
        "target_weight": 0.1,
        "order_type": "market",
    }


def test_producer_module_has_no_io_imports():
    tree = ast.parse((POSEIDON_ROOT / "src/poseidon/decision_loop/producer.py").read_text(encoding="utf-8"))
    modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    forbidden = ("requests", "httpx", "redis", "poseidon.data", "poseidon.workers", "poseidon.core.redis")
    assert not [module for module in modules if module.startswith(forbidden)]
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_run_inputs")
    names = {node.id for node in ast.walk(builder) if isinstance(node, ast.Name)}
    assert not names & {"datetime", "SessionLocal", "session", "select"}


@pytest.mark.parametrize(
    ("evidence", "evaluator", "status", "reasons"),
    [
        ([ohlcv_evidence("2330")], long_signal, CANDIDATE_STATUS, []),
        ([ohlcv_evidence("2330")], no_signal, "no_trade", ["no_entry_signal"]),
        ([ohlcv_evidence("2330")], close_signal, "no_trade", ["signal_close"]),
        ([ohlcv_evidence("2330")], broken_signal, "failed", ["signal_evaluation_failed"]),
        ([ohlcv_evidence("2317")], long_signal, "excluded", ["ohlcv_missing"]),
        (
            [ohlcv_evidence("2330", last_bar=LAST_BAR - timedelta(days=6)), ohlcv_evidence("2317")],
            long_signal,
            "excluded",
            ["ohlcv_stale"],
        ),
    ],
)
def test_every_member_gets_exactly_one_terminal_snapshot(db, evidence, evaluator, status, reasons):
    version = make_version(db)
    request = manifest_request(*evidence)
    universe, snapshots = _build(request, version, evaluator=evaluator)
    assert len(snapshots) == len(universe) == 1
    assert snapshots[0]["status"] == status
    assert snapshots[0]["reason_codes"] == reasons
    manifest = ManifestService(db).freeze(request)
    run = EvaluationService(db).evaluate_run(version.id, manifest.id, universe, snapshots)
    assert run.coverage_json == {"total": 1, "terminal": 1, "by_status": {status: 1}}


def test_research_status_is_explicit_and_never_invented(db):
    request = manifest_request(ohlcv_evidence("2330"))
    technical = make_version(db)
    _universe, [snapshot] = _build(request, technical)
    assert snapshot["recommendation_json"]["research_status"] == "not_required"
    assert snapshot["research_revision_ids"] == []

    research_version = make_version(db, artifact={"research_required": True})
    revision_id = str(uuid.uuid4())
    seen = []
    _universe, [backed] = _build(request, research_version, lookup=lambda scope: seen.append(scope) or revision_id)
    assert seen == [research_scope_key({"symbol": "2330", "market": "tw_stock", "instrument": "spot"})]
    assert backed["research_revision_ids"] == [revision_id]
    assert backed["recommendation_json"]["research_status"] == "completed"

    _universe, [missing] = _build(request, research_version, lookup=lambda _scope: None)
    assert missing["status"] == "excluded"
    assert missing["reason_codes"] == ["research_missing"]
    assert missing["research_revision_ids"] == []
    assert missing["recommendation_json"]["research_status"] == "not_required"


def test_liquidity_sweep_members_are_excluded_as_not_live(db):
    version = make_version(db, rule_config(strategy_type="liquidity_sweep"))
    calls = []
    _universe, [snapshot] = _build(
        manifest_request(ohlcv_evidence("2330")),
        version,
        evaluator=lambda *args: calls.append(args) or [],
    )
    assert snapshot["status"] == "excluded"
    assert snapshot["reason_codes"] == ["strategy_not_live"]
    assert calls == []


@pytest.mark.parametrize(("closes", "status"), [(RISING, CANDIDATE_STATUS), (FLAT, "no_trade")])
def test_default_evaluator_reuses_the_live_rule_strategy(db, closes, status):
    from poseidon.decision_loop.strategy_signals import evaluate_strategy_signals

    version = make_version(db)
    request = manifest_request(ohlcv_evidence("2330", closes=closes))
    _universe, [first] = _build(request, version, evaluator=evaluate_strategy_signals)
    _universe, [second] = _build(request, version, evaluator=evaluate_strategy_signals)
    assert first["status"] == status
    assert content_sha256(snapshot_payload(first)) == content_sha256(snapshot_payload(second))


@pytest.mark.parametrize("mode", ["legacy", "halted"])
def test_legacy_and_halted_modes_write_nothing(db, mode):
    version = make_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    before = _counts(db)
    result = ProducerService(db, signal_evaluator=long_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode=mode
    )
    assert result == {"skipped": mode}
    assert _counts(db) == before == (0, 0, 0)


def test_shadow_mode_writes_the_run_and_parity_but_no_decision(db):
    strategy_id = uuid.uuid4()
    version = make_version(db, strategy_id=strategy_id)
    manifest = ManifestService(db).freeze(manifest_request())
    db.add(
        SignalRecord(
            id=uuid.uuid4(),
            strategy_id=strategy_id,
            symbol="2330",
            market="tw_stock",
            instrument="spot",
            action="long",
            confidence=0.9,
            signal_time=LAST_BAR,
            interval="1d",
            params={},
            status="passed",
            metadata_={},
        )
    )
    db.flush()
    result = ProducerService(db, signal_evaluator=long_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="shadow"
    )
    assert result["mode"] == "shadow"
    assert _counts(db) == (1, 1, 0)
    parity = result["shadow_parity"]
    assert parity["actor_id"] == "system:shadow-parity"
    assert parity["frozen_targets"] == [["2330", "long"]]
    assert parity["legacy_targets"] == [["2330", "long"]]
    assert parity["frozen_exits"] == []
    assert parity["matched"] is True


def test_decision_mode_creates_one_pending_decision_and_replays(db):
    version = make_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    service = ProducerService(db, signal_evaluator=long_signal)
    first = service.produce(version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision")
    decision = db.get(DecisionRecord, uuid.UUID(first["decision_id"]))
    candidate_ids = [
        str(row.id)
        for row in db.query(EvaluationSnapshot).filter_by(evaluation_run_id=uuid.UUID(first["evaluation_run_id"]))
        if row.status == CANDIDATE_STATUS
    ]
    assert decision.status == "pending_approval"
    assert decision.original_json["selected_evaluation_ids"] == candidate_ids
    assert decision.original_json["final_action"] == "enter"
    assert [intent["evaluation_snapshot_id"] for intent in decision.original_json["order_intents"]] == candidate_ids
    assert decision.risk_snapshot_json["hard_failures"] == []
    assert decision.risk_snapshot_json["allowed_actions"] == ["enter", "exit", "hold", "watch"]
    assert decision.portfolio_snapshot_json == {"account_scope": ACCOUNT, "account_generation": None, "positions": []}
    created = db.query(DecisionEvent).filter_by(decision_id=decision.id).one()
    assert created.actor_id == "decision-worker:producer"

    replay = service.produce(version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision")
    assert replay["decision_id"] == first["decision_id"]
    assert replay["evaluation_run_id"] == first["evaluation_run_id"]
    assert _counts(db) == (1, 1, 1)


def test_decision_mode_without_candidates_creates_no_decision(db):
    version = make_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    result = ProducerService(db, signal_evaluator=no_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision"
    )
    assert result["decision_id"] is None
    assert result["decision_skipped"] == "no_candidates"
    assert _counts(db) == (1, 1, 0)


def test_producer_rejects_human_or_out_of_scope_principals(db):
    version = make_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    service = ProducerService(db, signal_evaluator=long_signal)
    researcher = AuthPrincipal("human:researcher", frozenset({"researcher"}), frozenset({ACCOUNT}))
    with pytest.raises(HTTPException):
        service.produce(version.id, manifest.id, decision_as_of=AS_OF, principal=researcher, mode="decision")
    with pytest.raises(HTTPException):
        service.produce(version.id, manifest.id, decision_as_of=AS_OF, principal=worker("paper:other"), mode="decision")
    assert _counts(db) == (0, 0, 0)


@pytest.mark.parametrize(
    ("actor_id", "principals"),
    [
        ("", [worker()]),
        ("decision-worker:producer", []),
        ("decision-worker:producer", [worker(), worker()]),
        (
            "decision-worker:producer",
            [AuthPrincipal("decision-worker:producer", frozenset({"researcher"}), frozenset({ACCOUNT}))],
        ),
        (
            "decision-worker:producer",
            [AuthPrincipal("decision-worker:producer", frozenset({"decision-worker", "viewer"}), frozenset({ACCOUNT}))],
        ),
        (
            "decision-worker:producer",
            [AuthPrincipal("decision-worker:producer", frozenset({"decision-worker"}), frozenset())],
        ),
    ],
)
def test_worker_principal_must_resolve_to_a_pure_decision_worker(actor_id, principals):
    with pytest.raises(ValueError):
        resolve_worker_principal(actor_id, principals)


def test_worker_principal_resolves_from_configured_identity():
    assert resolve_worker_principal("decision-worker:producer", [worker()]) == worker()


def test_settings_require_worker_principal_for_shadow_and_decision():
    from poseidon.core.config import Settings

    identity = {
        "decision_loop_approved_account_scope": ACCOUNT,
        "decision_loop_approved_market": "tw_stock",
        "decision_loop_approved_account_generation": "generation-1",
    }
    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None, decision_loop_execution_mode="shadow")
    with pytest.raises(PydanticValidationError):
        Settings(
            _env_file=None,
            decision_loop_execution_mode="decision",
            decision_loop_execution_enabled=True,
            **identity,
        )
    shadow = Settings(
        _env_file=None,
        decision_loop_execution_mode="shadow",
        decision_loop_worker_principal_id="decision-worker:producer",
    )
    assert shadow.decision_loop_worker_principal_id == "decision-worker:producer"
    assert Settings(_env_file=None).decision_loop_worker_principal_id == ""


def test_active_versions_read_only_match_market_interval_and_scope(db):
    matching = make_version(db)
    other_scope = make_version(db, policy=synthetic_policy(account_scope="paper:other"))
    other_interval = make_version(db, dict(rule_config(), interval="1h"))
    for version in (matching, other_scope, other_interval):
        db.add(
            StrategyRecord(
                id=version.strategy_id,
                name=f"producer-{version.id}",
                strategy_type="rule",
                symbol="2330",
                market="tw_stock",
                interval="1d",
                config={},
                active=True,
                active_version_id=version.id,
                release_revision=1,
            )
        )
    db.flush()
    pointers = {row.id: row.active_version_id for row in db.query(StrategyRecord)}
    assert [version.id for version in ProducerService(db).active_versions("tw_stock", "1d", ACCOUNT)] == [matching.id]
    assert {row.id: row.active_version_id for row in db.query(StrategyRecord)} == pointers


@pytest.mark.parametrize("mode", ["legacy", "halted"])
@pytest.mark.parametrize("task_name", ["freeze_market_manifest", "produce_decision_runs_for_market"])
def test_market_tasks_are_no_ops_outside_shadow_and_decision(monkeypatch, mode, task_name):
    from types import SimpleNamespace

    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        SimpleNamespace(
            decision_loop_execution_mode=mode,
            decision_loop_approved_market="tw_stock",
            decision_loop_approved_account_scope=ACCOUNT,
        ),
    )
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail("legacy mode opened a session"))
    assert getattr(cpu_tasks, task_name).run("tw_stock", "1d", AS_OF) == {"skipped": mode}


@pytest.mark.parametrize("mode", ["legacy", "halted"])
def test_produce_task_is_a_no_op_outside_shadow_and_decision(monkeypatch, mode):
    from types import SimpleNamespace

    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "settings", SimpleNamespace(decision_loop_execution_mode=mode))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail("legacy mode opened a session"))
    assert cpu_tasks.produce_decision_run.run(str(uuid.uuid4()), str(uuid.uuid4()), AS_OF) == {"skipped": mode}


def test_market_task_skips_unapproved_market(monkeypatch):
    from types import SimpleNamespace

    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        SimpleNamespace(
            decision_loop_execution_mode="shadow",
            decision_loop_approved_market="crypto_perp",
            decision_loop_approved_account_scope=ACCOUNT,
        ),
    )
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail("unapproved market opened a session"))
    assert cpu_tasks.produce_decision_runs_for_market.run("tw_stock", "1d", AS_OF) == {"skipped": "market_not_approved"}


def test_produce_task_resolves_worker_from_configured_principals(monkeypatch, db):
    import hashlib
    import json
    from types import SimpleNamespace

    from poseidon.core.config import settings as global_settings
    from poseidon.workers import cpu_tasks

    version = make_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    db.commit()
    monkeypatch.setattr(
        global_settings,
        "api_principals_json",
        json.dumps(
            {
                hashlib.sha256(b"producer-worker").hexdigest(): {
                    "actor_id": "decision-worker:producer",
                    "roles": ["decision-worker"],
                    "account_scopes": [ACCOUNT],
                }
            }
        ),
    )
    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        SimpleNamespace(
            decision_loop_execution_mode="shadow",
            decision_loop_worker_principal_id="decision-worker:producer",
            decision_loop_approved_account_scope=ACCOUNT,
        ),
    )

    class _Factory:
        def __call__(self):
            return Session(db.get_bind())

    monkeypatch.setattr(cpu_tasks, "SessionLocal", _Factory())
    monkeypatch.setattr("poseidon.decision_loop.strategy_signals.evaluate_strategy_signals", long_signal, raising=True)
    result = cpu_tasks.produce_decision_run.run(str(version.id), str(manifest.id), AS_OF)
    assert result["mode"] == "shadow"
    assert db.query(EvaluationRun).count() == 1
    assert db.query(DecisionRecord).count() == 0

    monkeypatch.setattr(
        cpu_tasks,
        "settings",
        SimpleNamespace(
            decision_loop_execution_mode="shadow",
            decision_loop_worker_principal_id="human:unknown",
            decision_loop_approved_account_scope=ACCOUNT,
        ),
    )
    with pytest.raises(ValueError, match="exactly one configured principal"):
        cpu_tasks.produce_decision_run.run(str(version.id), str(manifest.id), AS_OF)


# --- Phase 101-02: ordinary exit path for rule close signals (D-23.11) ---

HELD_LONG = frozenset({("tw_stock", "2330", "spot", "long")})
GENERATION = "synthetic-generation-1"


def close_and_long_signal(_executable, _bars):
    return [
        {"action": "close", "confidence": 1.0, "quantity_pct": None},
        {"action": "long", "confidence": 0.9, "quantity_pct": 0.1},
    ]


def weaker_and_stronger_close(_executable, _bars):
    return [
        {"action": "close", "confidence": 0.4, "quantity_pct": None},
        {"action": "close", "confidence": 0.7, "quantity_pct": 1.0},
    ]


def reconciled_version(db, **kwargs):
    return make_version(db, policy=synthetic_policy(reconciliation=synthetic_reconciliation_policy()), **kwargs)


def seed_open_lot(db, *, side="long", quantity=1000.0, symbol="2330"):
    lot = PositionLot(
        account_scope=ACCOUNT,
        account_generation=GENERATION,
        market="tw_stock",
        symbol=symbol,
        instrument="spot",
        side=side,
        opening_fill_id=uuid.uuid4(),
        opening_decision_id=uuid.uuid4(),
        original_quantity=quantity,
        open_quantity=quantity,
        reserved_close_quantity=0.0,
        cost_basis_json={"price": 100.0},
        opened_at=LAST_BAR - timedelta(days=3),
    )
    db.add(lot)
    db.flush()
    return lot


def test_close_signal_on_held_position_emits_exit_intent(db):
    version = make_version(db)
    _universe, [snapshot] = _build(
        manifest_request(), version, evaluator=weaker_and_stronger_close, open_positions=HELD_LONG
    )
    assert snapshot["status"] == CANDIDATE_STATUS
    assert snapshot["reason_codes"] == []
    assert snapshot["recommendation_json"]["side"] == "long"
    assert snapshot["recommendation_json"]["confidence"] == 0.7
    assert snapshot["recommendation_json"]["intent"] == {
        "symbol": "2330",
        "market": "tw_stock",
        "instrument": "spot",
        "action": "exit",
        "side": "long",
        "target_weight": 0.0,
        "order_type": "market",
    }


def test_close_signal_exits_the_held_side_including_short(db):
    version = make_version(db)
    _universe, [snapshot] = _build(
        manifest_request(),
        version,
        evaluator=close_signal,
        open_positions=frozenset({("tw_stock", "2330", "spot", "short")}),
    )
    assert snapshot["recommendation_json"]["intent"]["action"] == "exit"
    assert snapshot["recommendation_json"]["intent"]["side"] == "short"


@pytest.mark.parametrize(
    "open_positions",
    [
        None,
        frozenset(),
        frozenset({("tw_stock", "2317", "spot", "long")}),
        frozenset({("crypto_perp", "2330", "spot", "long")}),
    ],
)
def test_close_signal_without_held_identity_stays_no_trade(db, open_positions):
    version = make_version(db)
    _universe, [snapshot] = _build(manifest_request(), version, evaluator=close_signal, open_positions=open_positions)
    assert snapshot["status"] == "no_trade"
    assert snapshot["reason_codes"] == ["signal_close"]
    assert "intent" not in snapshot["recommendation_json"]


def test_close_and_entry_on_the_same_bar_is_a_deterministic_no_trade(db):
    version = make_version(db)
    for open_positions in (frozenset(), HELD_LONG):
        _universe, [snapshot] = _build(
            manifest_request(), version, evaluator=close_and_long_signal, open_positions=open_positions
        )
        assert snapshot["status"] == "no_trade"
        assert snapshot["reason_codes"] == ["conflicting_close_and_entry_signals"]


def test_close_with_two_open_sides_is_ambiguous(db):
    version = make_version(db)
    both = HELD_LONG | {("tw_stock", "2330", "spot", "short")}
    _universe, [snapshot] = _build(manifest_request(), version, evaluator=close_signal, open_positions=both)
    assert snapshot["status"] == "no_trade"
    assert snapshot["reason_codes"] == ["ambiguous_open_position"]


def test_exit_snapshots_are_deterministic(db):
    version = make_version(db)
    request = manifest_request()
    _universe, [first] = _build(copy.deepcopy(request), version, evaluator=close_signal, open_positions=HELD_LONG)
    _universe, [second] = _build(copy.deepcopy(request), version, evaluator=close_signal, open_positions=HELD_LONG)
    assert content_sha256(snapshot_payload(first)) == content_sha256(snapshot_payload(second))


def test_decision_mode_close_on_open_lot_creates_exit_decision(db):
    version = reconciled_version(db)
    seed_open_lot(db)
    manifest = ManifestService(db).freeze(manifest_request())
    result = ProducerService(db, signal_evaluator=close_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision"
    )
    decision = db.get(DecisionRecord, uuid.UUID(result["decision_id"]))
    assert decision.status == "pending_approval"
    assert decision.original_json["final_action"] == "exit"
    [intent] = decision.original_json["order_intents"]
    assert intent["action"] == "exit"
    assert intent["side"] == "long"
    assert intent["target_weight"] == 0.0
    assert decision.risk_snapshot_json["allowed_actions"] == ["enter", "exit", "hold", "watch"]
    assert decision.risk_snapshot_json["hard_failures"] == []
    assert decision.portfolio_snapshot_json["positions"] == [
        {"market": "tw_stock", "symbol": "2330", "instrument": "spot", "side": "long", "open_quantity": 1000.0}
    ]


def test_decision_mode_entry_on_already_held_side_creates_no_decision(db):
    version = reconciled_version(db)
    seed_open_lot(db)
    manifest = ManifestService(db).freeze(manifest_request())
    result = ProducerService(db, signal_evaluator=long_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision"
    )
    assert result["decision_id"] is None
    assert result["decision_skipped"] == "no_candidates"
    assert db.query(DecisionRecord).count() == 0


def test_decision_mode_close_without_open_lot_creates_no_decision(db):
    version = reconciled_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    result = ProducerService(db, signal_evaluator=close_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision"
    )
    assert result["decision_skipped"] == "no_candidates"
    snapshot = db.query(EvaluationSnapshot).one()
    assert (snapshot.status, snapshot.reason_codes) == ("no_trade", ["signal_close"])


def test_decision_mode_fails_closed_on_mixed_actions(db, monkeypatch):
    from poseidon.decision_loop import producer

    real_build = producer.build_run_inputs

    def two_member_build(*args, **kwargs):
        universe, [exit_snapshot] = real_build(*args, **kwargs)
        assert exit_snapshot["recommendation_json"]["intent"]["action"] == "exit"
        entry_member = {"symbol": "2317", "market": "tw_stock", "instrument": "spot"}
        entry_snapshot = copy.deepcopy(exit_snapshot) | entry_member
        entry_snapshot["recommendation_json"]["intent"] = {
            **entry_member,
            "action": "enter",
            "side": "long",
            "target_weight": 0.1,
            "order_type": "market",
        }
        return [*universe, entry_member], [exit_snapshot, entry_snapshot]

    monkeypatch.setattr(producer, "build_run_inputs", two_member_build)
    version = reconciled_version(db)
    seed_open_lot(db)
    manifest = ManifestService(db).freeze(manifest_request())
    result = ProducerService(db, signal_evaluator=close_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="decision"
    )
    assert result["decision_id"] is None
    assert result["decision_skipped"] == "mixed_actions"
    assert db.query(DecisionRecord).count() == 0


def test_shadow_parity_reports_exits_separately_from_entry_targets(db):
    strategy_id = uuid.uuid4()
    version = reconciled_version(db, strategy_id=strategy_id)
    seed_open_lot(db)
    manifest = ManifestService(db).freeze(manifest_request())
    db.add(
        SignalRecord(
            id=uuid.uuid4(),
            strategy_id=strategy_id,
            symbol="2330",
            market="tw_stock",
            instrument="spot",
            action="close",
            confidence=1.0,
            signal_time=LAST_BAR,
            interval="1d",
            params={},
            status="passed",
            metadata_={},
        )
    )
    db.flush()
    result = ProducerService(db, signal_evaluator=close_signal).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="shadow"
    )
    parity = result["shadow_parity"]
    assert parity["frozen_targets"] == []
    assert parity["frozen_exits"] == [["2330", "long"]]
    assert parity["legacy_targets"] == []
    assert parity["matched"] is True
    assert db.query(DecisionRecord).count() == 0


def _shadow_market_settings():
    from types import SimpleNamespace

    return SimpleNamespace(
        decision_loop_execution_mode="shadow",
        decision_loop_approved_market="tw_stock",
        decision_loop_approved_account_scope=ACCOUNT,
        thalassa_base_url="http://thalassa.test",
    )


def _activate(db, version, name="producer-active"):
    strategy = StrategyRecord(
        id=version.strategy_id,
        name=name,
        strategy_type="rule",
        symbol="2330",
        market="tw_stock",
        interval="1d",
        active_version_id=version.id,
    )
    db.add(strategy)
    db.commit()
    return strategy


def test_freeze_task_skips_stale_required_evidence_instead_of_raising(monkeypatch, db):
    """Phase 101 D-04: a non-trading day leaves the newest bar past max_age; the beat task skips."""
    from poseidon.models.data_manifest import DataManifest
    from poseidon.workers import cpu_tasks

    _activate(db, make_version(db))
    monkeypatch.setattr(cpu_tasks, "settings", _shadow_market_settings())
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: Session(db.get_bind()))
    monkeypatch.setattr(cpu_tasks, "_decision_worker_principal", lambda: worker())
    stale_bar = LAST_BAR - timedelta(days=2)
    monkeypatch.setattr(
        cpu_tasks,
        "_frozen_ohlcv_evidence",
        lambda market, interval, symbol, as_of: ohlcv_evidence(symbol, last_bar=stale_bar),
    )
    monkeypatch.setattr(
        cpu_tasks.produce_decision_run, "delay", lambda *args, **kwargs: pytest.fail("producer queued on stale data")
    )
    expected = {"skipped": "evidence_stale", "stale": {"ohlcv": "2026-09-24T00:00:00Z"}, "as_of": AS_OF}
    assert cpu_tasks.freeze_market_manifest.run("tw_stock", "1d", AS_OF) == expected
    assert cpu_tasks.produce_decision_runs_for_market.run("tw_stock", "1d", AS_OF) == expected
    assert db.query(DataManifest).count() == 0


def test_freeze_task_freezes_fresh_required_evidence(monkeypatch, db):
    from poseidon.models.data_manifest import DataManifest
    from poseidon.workers import cpu_tasks

    _activate(db, make_version(db))
    monkeypatch.setattr(cpu_tasks, "settings", _shadow_market_settings())
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: Session(db.get_bind()))
    monkeypatch.setattr(
        cpu_tasks, "_frozen_ohlcv_evidence", lambda market, interval, symbol, as_of: ohlcv_evidence(symbol)
    )
    result = cpu_tasks.freeze_market_manifest.run("tw_stock", "1d", AS_OF)
    assert set(result) == {"manifest_id", "content_sha256", "as_of"}
    assert db.query(DataManifest).count() == 1


def test_freeze_task_still_fails_closed_without_any_evidence(monkeypatch, db):
    from poseidon.decision_loop.manifest import ValidationError
    from poseidon.workers import cpu_tasks

    _activate(db, make_version(db))
    monkeypatch.setattr(cpu_tasks, "settings", _shadow_market_settings())
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: Session(db.get_bind()))
    monkeypatch.setattr(cpu_tasks, "_frozen_ohlcv_evidence", lambda market, interval, symbol, as_of: None)
    # No bars at all is not "stale": the manifest service still refuses an empty evidence list.
    with pytest.raises(ValidationError, match="manifest evidence must be a non-empty list"):
        cpu_tasks.freeze_market_manifest.run("tw_stock", "1d", AS_OF)


def test_stale_required_evidence_reports_only_kinds_past_max_age():
    from poseidon.workers.cpu_tasks import _stale_required_evidence

    fresh = ohlcv_evidence("2330")
    stale = ohlcv_evidence("0050", last_bar=LAST_BAR - timedelta(days=3))
    stale["kind"] = "benchmark"
    required = {
        "ohlcv": {"max_age_seconds": 86400},
        "benchmark": {"max_age_seconds": 86400},
        "news": {"max_age_seconds": 1},
    }
    assert _stale_required_evidence(required, [fresh, stale], AS_OF) == {"benchmark": "2026-09-23T00:00:00Z"}
    assert _stale_required_evidence({"ohlcv": {"max_age_seconds": 0}}, [], AS_OF) == {}


def _producer_task_env(monkeypatch, db, evidence):
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "settings", _shadow_market_settings())
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: Session(db.get_bind()))
    monkeypatch.setattr(cpu_tasks, "_decision_worker_principal", lambda: worker())
    monkeypatch.setattr(cpu_tasks, "_frozen_ohlcv_evidence", evidence)
    enqueued = []
    monkeypatch.setattr(
        cpu_tasks.produce_decision_run, "delay", lambda *args, **kwargs: enqueued.append((args, kwargs))
    )
    return cpu_tasks, enqueued


def test_produce_task_skips_when_the_day_already_has_a_complete_run(monkeypatch, db):
    """Phase 101-17.2: a retry slot after a successful freeze must not freeze a second manifest."""
    from poseidon.models.data_manifest import DataManifest

    version = make_version(db)
    _activate(db, version)
    manifest = ManifestService(db).freeze(manifest_request())
    result = ProducerService(db).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="shadow"
    )
    db.commit()
    run_id = result["evaluation_run_id"]

    def evidence(market, interval, symbol, as_of):
        pytest.fail("retry slot froze evidence although the day was already produced")

    cpu_tasks, enqueued = _producer_task_env(monkeypatch, db, evidence)
    retry_as_of = "2026-09-26T12:15:00Z"
    assert cpu_tasks.produce_decision_runs_for_market.run("tw_stock", "1d", retry_as_of) == {
        "skipped": "already_produced",
        "as_of": retry_as_of,
        "runs": [run_id],
    }
    assert enqueued == []
    assert db.query(DataManifest).count() == 1


def test_produce_task_retry_slot_freezes_once_evidence_turns_fresh(monkeypatch, db):
    """Phase 101-17.2: 06:40Z stale -> skip; the 06:55Z slot finds the new bar and produces."""
    from poseidon.models.data_manifest import DataManifest

    version = make_version(db)
    _activate(db, version)
    bars = {"last_bar": LAST_BAR - timedelta(days=3)}
    cpu_tasks, enqueued = _producer_task_env(
        monkeypatch, db, lambda market, interval, symbol, as_of: ohlcv_evidence(symbol, last_bar=bars["last_bar"])
    )
    first = cpu_tasks.produce_decision_runs_for_market.run("tw_stock", "1d", AS_OF)
    assert first["skipped"] == "evidence_stale"
    assert enqueued == [] and db.query(DataManifest).count() == 0

    bars["last_bar"] = LAST_BAR
    retry_as_of = "2026-09-26T12:15:00Z"
    second = cpu_tasks.produce_decision_runs_for_market.run("tw_stock", "1d", retry_as_of)
    assert second["as_of"] == retry_as_of and second["enqueued"] == 1
    assert db.query(DataManifest).count() == 1
    assert enqueued == [((str(version.id), second["manifest_id"], retry_as_of), {})]


def test_produce_task_guard_ignores_runs_from_other_days(monkeypatch, db):
    from poseidon.models.data_manifest import DataManifest

    version = make_version(db)
    _activate(db, version)
    manifest = ManifestService(db).freeze(manifest_request())
    ProducerService(db).produce(version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="shadow")
    db.commit()
    next_day_bar = LAST_BAR + timedelta(days=1)
    cpu_tasks, enqueued = _producer_task_env(
        monkeypatch, db, lambda market, interval, symbol, as_of: ohlcv_evidence(symbol, last_bar=next_day_bar)
    )
    next_as_of = "2026-09-27T01:00:00Z"  # 3600 s after the 09-27 bar: fresh, unlike the 09-26 run
    result = cpu_tasks.produce_decision_runs_for_market.run("tw_stock", "1d", next_as_of)
    assert result["enqueued"] == 1 and result["as_of"] == next_as_of
    assert len(enqueued) == 1
    assert db.query(DataManifest).count() == 2


def test_complete_runs_on_day_matches_only_complete_runs_of_the_versions(db):
    from poseidon.workers.cpu_tasks import _complete_runs_on_day

    version = make_version(db)
    manifest = ManifestService(db).freeze(manifest_request())
    run_id = ProducerService(db).produce(
        version.id, manifest.id, decision_as_of=AS_OF, principal=worker(), mode="shadow"
    )["evaluation_run_id"]
    db.flush()
    assert _complete_runs_on_day(db, [version.id], AS_OF) == [run_id]
    assert _complete_runs_on_day(db, [version.id], "2026-09-27T00:00:00Z") == []
    assert _complete_runs_on_day(db, [uuid.uuid4()], AS_OF) == []
    assert _complete_runs_on_day(db, [], AS_OF) == []
    db.query(EvaluationRun).filter(EvaluationRun.id == uuid.UUID(run_id)).update({"status": "failed"})
    db.flush()
    assert _complete_runs_on_day(db, [version.id], AS_OF) == []
