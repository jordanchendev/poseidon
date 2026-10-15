"""Phase 101-08: daily outcome manifests that the 07:30Z labeler can consume (D-23.4)."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from poseidon.decision_loop.manifest import ValidationError, content_sha256, validate_manifest
from poseidon.decision_loop.outcome_manifest import (
    OUTCOME_PRICE_MAX_AGE_SECONDS,
    build_outcome_manifest_request,
    load_session_calendar,
    validate_session_calendar,
)
from poseidon.decision_loop.outcomes import _manifest_facts, validate_outcome_label_contract

POSEIDON_ROOT = Path(__file__).resolve().parents[1]
CALENDAR_PATH = POSEIDON_ROOT / "config" / "calendars" / "xtai_1d_sessions.json"
IDENTITY = "XTAI:exchange_calendars-4.13.2:2026-01-01..2027-12-31:close-05:30Z"
ACCOUNT = "paper:pilot"
UNIVERSE = "tw-stock-pilot-v1"
BASE_URL = "http://thalassa.invalid:8001"
SESSION_S = datetime(2026, 10, 7, 5, 30, tzinfo=UTC)  # Wednesday
SESSION_S1 = datetime(2026, 10, 8, 5, 30, tzinfo=UTC)  # Thursday
ANCHOR = datetime(2026, 10, 7, 6, 40, tzinfo=UTC)  # 06:40Z producer cutoff on S
AS_OF = datetime(2026, 10, 8, 7, 20, tzinfo=UTC)  # 07:20Z outcome freeze on S+1
FRIDAY = datetime(2026, 10, 2, 5, 30, tzinfo=UTC)
SATURDAY_AS_OF = datetime(2026, 10, 3, 7, 20, tzinfo=UTC)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def contract_json(marker="fixture", **changes):
    contract = {
        "calendar": {"identity": IDENTITY, "evidence_id": "calendar:XTAI"},
        "horizons": {
            "signal": [{"key": "signal-1-session", "anchor": "decision_as_of", "session_offset": 1}],
            "trade": [{"key": "trade-1-session", "anchor": "decision_as_of", "session_offset": 1}],
            "research": [{"key": "research-5-session", "anchor": "decision_as_of", "session_offset": 5}],
        },
        "benchmark": {"symbol": "0050", "evidence_id": "benchmark:0050"},
        "reporting_currency": "TWD",
        "required_cost_components": ["commission", "tax"],
        "cost_model_version": f"tw-paper-cost-v1:{marker}",
        "fx_model_version": "twd-identity-v1",
        "pnl_tolerance": "0.01",
        "research_assessment": {
            "expiry_horizon_key": "research-5-session",
            "statuses": ["confirmed", "not_confirmed", "unavailable"],
        },
        "counterfactual_assumption_versions": ["paper-execution-v1"],
    }
    contract.update(changes)
    return contract


def calendar():
    return load_session_calendar(CALENDAR_PATH)


def daily_bars(last_session, *, start=datetime(2026, 8, 3, tzinfo=UTC), base=100.0):
    """One 05:30Z bar per calendar session in [start, last_session], close rising by 1 per bar."""
    sessions = [
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in calendar()["sessions"]
        if start <= datetime.fromisoformat(value.replace("Z", "+00:00")) <= last_session
    ]
    return [{"time": _iso(session), "close": base + index} for index, session in enumerate(sessions)]


def build(**changes):
    arguments = {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": _iso(AS_OF),
        "account_scope": ACCOUNT,
        "universe_id": UNIVERSE,
        "symbol": "2330",
        "contract": contract_json(),
        "calendar": calendar(),
        "bars": daily_bars(SESSION_S1),
        "benchmark_bars": daily_bars(SESSION_S1, base=200.0),
        "anchors": [_iso(ANCHOR)],
        "source_base_url": BASE_URL,
    }
    arguments.update(changes)
    return build_outcome_manifest_request(**arguments)


def _by_kind(request):
    return {item["kind"]: item for item in request["evidence"]}


class _FrozenManifest:
    """The verify_manifest view of a frozen row, without a database."""

    def __init__(self, request):
        payload = validate_manifest(request)
        self.payload_json = payload
        self.content_sha256 = content_sha256(payload)
        self.market = payload["market"]
        self.interval = payload["interval"]
        self.as_of = datetime.fromisoformat(payload["as_of"].replace("Z", "+00:00"))
        self.capability_json = payload["capability_json"]
        self.sources_json = payload.get("sources_json", [])


# ---------------------------------------------------------------------------
# Task 1: committed XTAI session file and the pure builder
# ---------------------------------------------------------------------------


def test_session_file_is_the_frozen_xtai_calendar():
    raw = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    sessions = raw["sessions"]
    assert raw["identity"] == IDENTITY
    assert raw["bar_time_utc"] == "05:30:00"
    assert raw["generator"]["exchange_calendars"] == "4.13.2"
    assert "get_calendar('XTAI'" in raw["generator"]["command"]
    assert sessions == sorted(set(sessions))
    assert all(value.endswith("T05:30:00Z") for value in sessions)
    assert all(datetime.fromisoformat(value.replace("Z", "+00:00")).weekday() < 5 for value in sessions)
    assert "2026-10-02T05:30:00Z" in sessions
    assert "2026-10-10T05:30:00Z" not in sessions  # National Day
    assert sessions[0].startswith("2026-") and sessions[-1].startswith("2027-")
    assert calendar() == {"identity": IDENTITY, "sessions": sessions}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(sessions=list(reversed(value["sessions"]))),
        lambda value: value.update(sessions=[*value["sessions"], value["sessions"][-1]]),
        lambda value: value.update(sessions=["2026-10-02T06:00:00Z"]),
        lambda value: value.update(identity=" "),
        lambda value: value.update(sessions=[]),
    ],
)
def test_session_calendar_validation_refuses_malformed_files(mutation):
    value = copy.deepcopy(calendar())
    mutation(value)
    with pytest.raises(ValidationError):
        validate_session_calendar(value)


def test_builder_request_passes_manifest_and_label_contract_checks():
    request = build()
    payload = validate_manifest(request)
    assert payload["capability_json"] == {kind: True for kind in ("price", "benchmark", "fx", "calendar", "research")}
    assert payload["required_data"] == {"price": {"max_age_seconds": 14400}}
    assert OUTCOME_PRICE_MAX_AGE_SECONDS == 14_400
    contract = validate_outcome_label_contract(contract_json())
    _payload, by_id, sessions, counterfactuals = _manifest_facts(_FrozenManifest(request), contract, AS_OF)
    assert set(by_id) == {"price:2330", "benchmark:0050", "fx:TWD:TWD", "calendar:XTAI", "research:not_required:2330"}
    assert SESSION_S1 in sessions and sessions == sorted(set(sessions))
    assert counterfactuals == []


def test_price_and_benchmark_values_cover_sessions_and_anchors():
    evidence = _by_kind(build())
    price, benchmark = evidence["price"], evidence["benchmark"]
    assert price["id"] == "price:2330"
    assert price["payload"]["symbol"] == "2330"
    values = price["payload"]["values"]
    bars = {bar["time"]: bar["close"] for bar in daily_bars(SESSION_S1)}
    assert values[_iso(SESSION_S1)] == str(bars[_iso(SESSION_S1)])
    # The 06:40Z anchor maps to the close of the latest bar at or before it (session S).
    assert values[_iso(ANCHOR)] == values[_iso(SESSION_S)] == str(bars[_iso(SESSION_S)])
    assert set(values) == set(bars) | {_iso(ANCHOR)}
    assert all(isinstance(value, str) for value in values.values())
    assert benchmark["id"] == "benchmark:0050"
    assert benchmark["payload"]["symbol"] == "0050"
    assert benchmark["payload"]["values"][_iso(ANCHOR)] == str(200.0 + list(bars).index(_iso(SESSION_S)))
    assert price["event_time"] == price["available_at"] == _iso(SESSION_S1)
    assert price["source_uri"].startswith(f"{BASE_URL}/api/v1/ohlcv?symbol=2330")
    assert "key" not in price["source_uri"].lower()


def test_bars_after_as_of_never_enter_the_manifest():
    future = [*daily_bars(SESSION_S1), {"time": "2026-10-12T05:30:00Z", "close": 999}]
    values = _by_kind(build(bars=future))["price"]["payload"]["values"]
    assert "2026-10-12T05:30:00Z" not in values
    assert all(item["event_time"] <= _iso(AS_OF) for item in build(bars=future)["evidence"])


def test_fx_research_and_calendar_evidence_shapes():
    evidence = _by_kind(build())
    assert evidence["fx"]["id"] == "fx:TWD:TWD"
    assert evidence["fx"]["payload"] == {"base": "TWD", "quote": "TWD", "rate": "1", "model_version": "twd-identity-v1"}
    assert evidence["research"]["id"] == "research:not_required:2330"
    assert evidence["research"]["payload"] == {"symbol": "2330", "status": "not_required"}
    calendar_item = evidence["calendar"]
    assert calendar_item["id"] == "calendar:XTAI"
    assert calendar_item["payload"]["identity"] == IDENTITY
    sessions = calendar_item["payload"]["sessions"]
    # Future sessions stay in so the 5-session research horizon resolves.
    assert "2026-10-15T05:30:00Z" in sessions
    first_bar = daily_bars(SESSION_S1)[0]["time"]
    assert sessions[0] == first_bar
    for item in evidence.values():
        assert item["recorded_at"] == _iso(AS_OF)
        assert item["event_time"] <= _iso(AS_OF) and item["available_at"] <= _iso(AS_OF)
        assert item["source_uri"]
        assert item["content_sha256"] == content_sha256(item["payload"])


def test_contract_drift_is_refused_never_defaulted():
    with pytest.raises(ValidationError, match="calendar identity drift"):
        build(contract=contract_json(calendar={"identity": "XTAI:other", "evidence_id": "calendar:XTAI"}))
    with pytest.raises(ValidationError, match="tw_stock outcome manifests require TWD reporting"):
        build(contract=contract_json(reporting_currency="USD"))
    incomplete = contract_json()
    del incomplete["benchmark"]
    with pytest.raises(ValidationError):
        build(contract=incomplete)
    with pytest.raises(ValidationError, match="benchmark bars are missing"):
        build(benchmark_bars=[])
    with pytest.raises(ValidationError, match="price bars are missing"):
        build(bars=[])


def test_identical_inputs_produce_an_identical_request_hash():
    first, second = build(), build(anchors=[_iso(ANCHOR), _iso(ANCHOR)])
    assert content_sha256(first) == content_sha256(second)
    assert content_sha256(validate_manifest(first)) == content_sha256(validate_manifest(build()))
    assert content_sha256(first) != content_sha256(build(symbol="2317"))


def test_weekend_cutoff_makes_price_evidence_stale():
    request = build(
        as_of=_iso(SATURDAY_AS_OF),
        bars=daily_bars(FRIDAY),
        benchmark_bars=daily_bars(FRIDAY, base=200.0),
        anchors=[],
    )
    with pytest.raises(ValidationError, match="required evidence is stale: price"):
        validate_manifest(request)


def test_builder_module_is_pure():
    source = (POSEIDON_ROOT / "src" / "poseidon" / "decision_loop" / "outcome_manifest.py").read_text()
    for forbidden in ("poseidon.core.database", "poseidon.data", "requests", "httpx", "datetime.now", "SessionLocal"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Task 2: freeze_outcome_manifests task, beat entry, and the label path
# ---------------------------------------------------------------------------


def _frame(bars):
    import pandas as pd

    columns = ["open", "high", "low", "close", "volume"]
    if not bars:
        return pd.DataFrame(columns=columns)
    closes = [bar["close"] for bar in bars]
    index = pd.DatetimeIndex([pd.Timestamp(bar["time"]) for bar in bars])
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000.0] * len(bars)},
        index=index,
    )


class FakeRepository:
    def __init__(self, bars_by_symbol):
        self.bars_by_symbol = bars_by_symbol
        self.calls = []

    def read_ohlcv(self, symbol, market, interval, start, end):
        self.calls.append((symbol, market, interval, start, end))
        return _frame(self.bars_by_symbol.get(symbol, []))


def task_settings(mode="shadow", **changes):
    values = {
        "decision_loop_execution_mode": mode,
        "decision_loop_approved_market": "tw_stock",
        "decision_loop_approved_account_scope": ACCOUNT,
        "decision_loop_label_definition_version": "label-v1",
        "thalassa_base_url": BASE_URL,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _patch_task(monkeypatch, session_factory, repository, **settings_changes):
    from poseidon.data.remote_repository import RemoteDataRepository
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "settings", task_settings(**settings_changes))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", session_factory)
    monkeypatch.setattr(cpu_tasks, "OUTCOME_CALENDAR_PATH", str(CALENDAR_PATH))
    monkeypatch.setattr(RemoteDataRepository, "from_settings", staticmethod(lambda: repository))
    return cpu_tasks


@pytest.fixture
def sqlite_factory():
    from sqlalchemy import create_engine, event
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from poseidon.models.base import Base

    @compiles(JSONB, "sqlite")
    def compile_jsonb_sqlite(type_, compiler, **kw):
        return "JSON"

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def register_postgres_text_functions(dbapi_connection, _connection_record):
        dbapi_connection.create_function("char_length", 1, len)
        dbapi_connection.create_function("btrim", 1, lambda value: value.strip())

    names = ("strategies", "strategy_versions", "data_manifests", "evaluation_runs", "outcome_label_contracts")
    Base.metadata.create_all(engine, tables=[Base.metadata.tables[name] for name in names])
    yield sessionmaker(bind=engine, autoflush=False, future=True)
    engine.dispose()


def _seed_pilot(session, *, symbol="2330", contract_version="label-v1", marker="fixture", active=True):
    """Active pilot version, its contract row, and evaluation runs; returns the version id."""
    import uuid

    from poseidon.decision_loop.manifest import ManifestService
    from poseidon.models.evaluation_run import EvaluationRun
    from poseidon.models.outcome import OutcomeLabelContract
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
    from tests.test_decision_run_producer import manifest_request, rule_config
    from tests.test_decision_service import synthetic_policy

    config, policy = rule_config(symbol), synthetic_policy()
    strategy_id = uuid.uuid4()
    version = StrategyVersion(
        strategy_id=strategy_id,
        version_no=1,
        config_json=config,
        policy_json=policy,
        artifact_json={},
        content_sha256=strategy_version_digest(config, policy, {}),
        status="draft",
    )
    session.add(version)
    session.flush()
    session.add(
        StrategyRecord(
            id=strategy_id,
            name=f"outcome-{marker}-{symbol}",
            strategy_type="rule",
            symbol=symbol,
            market="tw_stock",
            interval="1d",
            config={},
            active=True,
            active_version_id=version.id if active else None,
            release_revision=1,
        )
    )
    evaluation_manifest = ManifestService(session).freeze(manifest_request())
    for decision_as_of, status in (
        (ANCHOR, "complete"),
        (ANCHOR - timedelta(days=1), "pending"),
        (AS_OF - timedelta(days=200), "complete"),
    ):
        session.add(
            EvaluationRun(
                strategy_version_id=version.id,
                manifest_id=evaluation_manifest.id,
                decision_as_of=decision_as_of,
                universe_json=[{"symbol": symbol, "market": "tw_stock", "instrument": "spot"}],
                input_sha256=content_sha256({"run": marker, "at": _iso(decision_as_of), "status": status}),
                status=status,
                coverage_json={"total": 1, "terminal": 1, "by_status": {"evaluated": 1}},
            )
        )
    contract = contract_json(marker)
    session.add(
        OutcomeLabelContract(version=contract_version, contract_json=contract, contract_sha256=content_sha256(contract))
    )
    session.commit()
    return version.id


def _bars_through(last_session):
    return {"2330": daily_bars(last_session), "0050": daily_bars(last_session, base=200.0)}


@pytest.mark.parametrize("mode", ["legacy", "halted"])
def test_task_is_a_write_free_no_op_outside_shadow_and_decision(monkeypatch, mode):
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "settings", task_settings(mode))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail("no-op mode opened a session"))
    assert cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF)) == {"skipped": mode}


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"decision_loop_approved_account_scope": ""}, "market_not_approved"),
        ({"decision_loop_approved_market": "crypto_perp"}, "market_not_supported"),
        ({"decision_loop_label_definition_version": ""}, "label_definition_version_unset"),
    ],
)
def test_task_gates_before_any_session(monkeypatch, changes, expected):
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "settings", task_settings("decision", **changes))
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: pytest.fail(f"{expected} opened a session"))
    assert cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF)) == {"skipped": expected}


def test_task_never_retries_and_is_registered():
    from poseidon.workers import cpu_tasks

    task = cpu_tasks.freeze_outcome_manifests
    assert task.name == "poseidon.workers.cpu_tasks.freeze_outcome_manifests"
    assert task.max_retries == 0
    # The worker resolves the calendar relative to /app (Dockerfile: COPY config/ config/).
    assert (POSEIDON_ROOT / cpu_tasks.OUTCOME_CALENDAR_PATH).resolve() == CALENDAR_PATH.resolve()
    assert "COPY config/ config/" in (POSEIDON_ROOT / "Dockerfile").read_text()


@pytest.mark.parametrize("rows", [0, 2])
def test_contract_version_must_resolve_exactly_once(monkeypatch, sqlite_factory, rows):
    from poseidon.models.outcome import OutcomeLabelContract

    with sqlite_factory() as session:
        _seed_pilot(session, contract_version="other-version")
        for index in range(rows):
            contract = contract_json(f"dup-{index}")
            session.add(
                OutcomeLabelContract(
                    version="label-v1", contract_json=contract, contract_sha256=content_sha256(contract)
                )
            )
        session.commit()
    repository = FakeRepository(_bars_through(SESSION_S1))
    cpu_tasks = _patch_task(monkeypatch, sqlite_factory, repository)
    with pytest.raises(ValueError, match="must resolve exactly once"):
        cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF))
    assert repository.calls == []


def test_task_freezes_one_manifest_per_symbol_insert_once(monkeypatch, sqlite_factory):
    from poseidon.decision_loop.manifest import verify_manifest
    from poseidon.decision_loop.outcomes import _manifest_facts
    from poseidon.models.data_manifest import DataManifest

    with sqlite_factory() as session:
        _seed_pilot(session)
    repository = FakeRepository(_bars_through(SESSION_S1))
    cpu_tasks = _patch_task(monkeypatch, sqlite_factory, repository, mode="decision")
    first = cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF))
    assert set(first) == {"as_of", "manifests", "skipped_symbols"}
    assert first["as_of"] == _iso(AS_OF)
    assert list(first["manifests"]) == ["2330"]
    assert first["skipped_symbols"] == {}
    assert {call[0] for call in repository.calls} == {"2330", "0050"}
    assert all(call[1:3] == ("tw_stock", "1d") and call[4] == AS_OF for call in repository.calls)
    second = cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF))
    assert second == first
    with sqlite_factory() as session:
        outcome_rows = [row for row in session.query(DataManifest) if cpu_tasks._is_outcome_manifest(row)]
        assert [str(row.id) for row in outcome_rows] == [first["manifests"]["2330"]]
        payload = verify_manifest(outcome_rows[0])
        values = {item["kind"]: item for item in payload["evidence"]}["price"]["payload"]["values"]
        # Only the complete run inside the 120-day window becomes an anchor.
        assert _iso(ANCHOR) in values
        assert _iso(ANCHOR - timedelta(days=1)) not in values
        assert payload["account_scope"] == ACCOUNT and payload["universe_id"] == UNIVERSE
        _manifest_facts(outcome_rows[0], validate_outcome_label_contract(contract_json()), AS_OF)


def test_weekend_cutoff_skips_symbol_as_price_stale(monkeypatch, sqlite_factory):
    from poseidon.models.data_manifest import DataManifest

    with sqlite_factory() as session:
        _seed_pilot(session)
    cpu_tasks = _patch_task(monkeypatch, sqlite_factory, FakeRepository(_bars_through(FRIDAY)))
    result = cpu_tasks.freeze_outcome_manifests.run(_iso(SATURDAY_AS_OF))
    assert result == {"as_of": _iso(SATURDAY_AS_OF), "manifests": {}, "skipped_symbols": {"2330": "price_stale"}}
    with sqlite_factory() as session:
        assert not [row for row in session.query(DataManifest) if cpu_tasks._is_outcome_manifest(row)]


def test_missing_bars_are_skipped_with_a_reason(monkeypatch, sqlite_factory):
    with sqlite_factory() as session:
        _seed_pilot(session)
    cpu_tasks = _patch_task(monkeypatch, sqlite_factory, FakeRepository({"2330": daily_bars(SESSION_S1)}))
    assert cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF))["skipped_symbols"] == {"2330": "benchmark_missing"}
    cpu_tasks = _patch_task(monkeypatch, sqlite_factory, FakeRepository({"0050": daily_bars(SESSION_S1)}))
    assert cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF))["skipped_symbols"] == {"2330": "price_missing"}


def test_no_active_pilot_freezes_nothing(monkeypatch, sqlite_factory):
    with sqlite_factory() as session:
        _seed_pilot(session, active=False)
    repository = FakeRepository(_bars_through(SESSION_S1))
    cpu_tasks = _patch_task(monkeypatch, sqlite_factory, repository)
    assert cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF)) == {"skipped": "no_active_strategies"}
    assert repository.calls == []


def test_beat_runs_the_freeze_between_producer_and_labeler():
    from celery.schedules import crontab

    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert schedule["decision-loop-freeze-outcome-manifests"] == {
        "task": "poseidon.workers.cpu_tasks.freeze_outcome_manifests",
        "schedule": crontab(hour=7, minute=20),
    }
    assert schedule["decision-loop-produce-tw-daily"]["schedule"] == crontab(hour=6, minute=40)
    assert schedule["decision-loop-label-outcomes"]["schedule"] == crontab(hour=7, minute=30)
    assert len(schedule) == 23


@pytest.mark.postgresql
def test_frozen_outcome_manifest_feeds_the_labeler(monkeypatch, phase99_session_factory):
    """S 06:40Z run + decision, S+1 07:20Z freeze, S+1 07:30Z label: signal and trade at offset 1."""
    import uuid

    from poseidon.decision_loop.evaluation import snapshot_payload
    from poseidon.decision_loop.manifest import ManifestService
    from poseidon.decision_loop.producer import ProducerService
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.evaluation_run import EvaluationRun
    from poseidon.models.evaluation_snapshot import EvaluationSnapshot
    from poseidon.models.outcome import OutcomeLabelContract, OutcomeRecord
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
    from tests.test_decision_run_producer import manifest_request, rule_config
    from tests.test_decision_service import synthetic_policy

    marker = uuid.uuid4().hex
    symbol = f"O{marker[:7]}"
    label_version = f"label-{marker}"
    config, policy = rule_config(symbol), synthetic_policy()
    with phase99_session_factory() as session:
        strategy = StrategyRecord(
            name=f"phase101-outcome-{marker}", strategy_type="rule", symbol=symbol, market="tw_stock", interval="1d"
        )
        session.add(strategy)
        session.flush()
        version = StrategyVersion(
            strategy_id=strategy.id,
            version_no=1,
            config_json=config,
            policy_json=policy,
            artifact_json={},
            content_sha256=strategy_version_digest(config, policy, {}),
        )
        session.add(version)
        session.flush()
        evaluation_request = manifest_request()
        evaluation_request["evidence"][0]["payload"]["marker"] = marker
        evaluation_request["evidence"][0]["content_sha256"] = content_sha256(
            evaluation_request["evidence"][0]["payload"]
        )
        evaluation_manifest = ManifestService(session).freeze(evaluation_request)
        contract = contract_json(marker)
        session.add(
            OutcomeLabelContract(
                version=label_version, contract_json=contract, contract_sha256=content_sha256(contract)
            )
        )
        run = EvaluationRun(
            strategy_version_id=version.id,
            manifest_id=evaluation_manifest.id,
            decision_as_of=ANCHOR,
            universe_json=[{"symbol": symbol, "market": "tw_stock", "instrument": "spot"}],
            input_sha256=content_sha256({"phase101-outcome": marker}),
            status="complete",
            coverage_json={"total": 1, "terminal": 1, "by_status": {"evaluated": 1}},
        )
        session.add(run)
        session.flush()
        body = {
            "symbol": symbol,
            "market": "tw_stock",
            "instrument": "spot",
            "status": "evaluated",
            "recommendation_json": {"research_status": "not_required"},
            "technical_json": {},
            "research_revision_ids": [],
            "reason_codes": [],
            "valid_until": None,
        }
        snapshot = EvaluationSnapshot(
            evaluation_run_id=run.id, content_sha256=content_sha256(snapshot_payload(body)), **body
        )
        session.add(snapshot)
        session.flush()
        intent = {
            "evaluation_snapshot_id": str(snapshot.id),
            "symbol": symbol,
            "market": "tw_stock",
            "instrument": "spot",
            "action": "enter",
            "side": "long",
            "target_weight": 0.1,
            "order_type": "market",
        }
        session.add(
            DecisionRecord(
                evaluation_run_id=run.id,
                strategy_version_id=version.id,
                account_scope=ACCOUNT,
                decision_as_of=ANCHOR,
                valid_until=ANCHOR + timedelta(seconds=68_700),
                status="approved",
                revision=1,
                creation_sha256=content_sha256({"phase101-outcome-decision": marker}),
                policy_sha256=content_sha256(policy),
                original_json={"selected_evaluation_ids": [str(snapshot.id)]},
                final_json={"selected_evaluation_ids": [str(snapshot.id)], "order_intents": [intent]},
                portfolio_snapshot_json={},
                risk_snapshot_json={},
            )
        )
        session.commit()
        version_id, snapshot_id = version.id, snapshot.id

    # The PostgreSQL release-authority trigger guards active_version_id; the pointer read is
    # covered by the SQLite tests above, so here only the active-version lookup is substituted.
    monkeypatch.setattr(
        ProducerService,
        "active_versions",
        lambda self, market, interval, scope: [self.session.get(StrategyVersion, version_id)],
    )
    repository = FakeRepository({symbol: daily_bars(SESSION_S1), "0050": daily_bars(SESSION_S1, base=200.0)})
    cpu_tasks = _patch_task(
        monkeypatch,
        phase99_session_factory,
        repository,
        mode="decision",
        decision_loop_label_definition_version=label_version,
    )
    frozen = cpu_tasks.freeze_outcome_manifests.run(_iso(AS_OF))
    assert frozen["skipped_symbols"] == {}
    manifest_id = frozen["manifests"][symbol]

    with phase99_session_factory() as session:
        assert cpu_tasks._latest_outcome_as_of(session, AS_OF + timedelta(minutes=10)) == _iso(AS_OF)
        batches, unlabeled = cpu_tasks._mature_outcome_batches(session, _iso(AS_OF), label_version)
    assert snapshot_id in batches[uuid.UUID(manifest_id)]
    assert str(snapshot_id) not in unlabeled

    # The 07:30Z labeler labels against the 07:20Z cutoff it finds (as_of must equal the manifest cutoff).
    labeled = cpu_tasks.label_decision_outcomes.run(_iso(AS_OF))
    assert labeled["label_definition_version"] == label_version
    with phase99_session_factory() as session:
        rows = session.query(OutcomeRecord).filter_by(evaluation_snapshot_id=snapshot_id).all()
        by_kind = {row.kind: row for row in rows}
    assert set(by_kind) == {"signal", "trade"}
    signal, trade = by_kind["signal"], by_kind["trade"]
    assert (signal.status, signal.horizon_key) == ("available", "signal-1-session")
    assert signal.maturity_at.astimezone(UTC) == SESSION_S1
    assert signal.metrics_json["actual"]["status"] == "available"
    assert trade.horizon_key == "trade-1-session"
    assert trade.reason_code
    assert str(trade.manifest_id) == manifest_id
