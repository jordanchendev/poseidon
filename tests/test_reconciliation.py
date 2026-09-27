"""Reconcile durable decision intents without blind broker retries."""

from __future__ import annotations

import contextlib
import copy
import inspect
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from threading import Event
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from poseidon.broker.base import (
    BrokerAccountSnapshot,
    BrokerAdapter,
    BrokerCapabilities,
    BrokerFillSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
)
from poseidon.broker.paper_adapter import PAPER_RECONCILIATION_CAPABILITIES, PaperBrokerAdapter
from poseidon.decision_loop.decisions import DecisionService
from poseidon.decision_loop.execution import ExecutionConflictError, internal_state_watermark
from poseidon.decision_loop.reconciliation import (
    ReconciliationConflictError,
    ReconciliationService,
    submit_or_reconcile_order,
)
from poseidon.decision_loop.recovery import RecoverySelector
from poseidon.decision_loop.transactions import materialize_order_intents
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.base import Base
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion
from poseidon.orders.schemas import Order
from tests.test_decision_order_wiring import NOW, PRICE_2330, _claimed
from tests.test_decision_service import synthetic_reconciliation_policy, worker
from tests.test_execution_concurrency_postgres import CLAIM_TIME, ENGINE, claim_seed  # noqa: F401


@pytest.fixture
def decision_task_settings(monkeypatch):
    from poseidon.core.config import Settings
    from poseidon.workers import cpu_tasks

    config = Settings(
        _env_file=None,
        decision_loop_execution_mode="decision",
        decision_loop_execution_enabled=True,
        decision_loop_approved_account_scope="paper:pilot",
        decision_loop_approved_market="tw_stock",
        decision_loop_approved_account_generation="generation-1",
    )
    monkeypatch.setattr(cpu_tasks, "settings", config)
    return config


@pytest.mark.parametrize("mode", ["legacy", "shadow", "halted"])
def test_uuid_submit_boundary_blocks_never_attempted_ordinary(sessions, monkeypatch, decision_task_settings, mode):
    from poseidon.workers import cpu_tasks

    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)
    decision_task_settings.decision_loop_execution_mode = mode
    decision_task_settings.decision_loop_execution_enabled = False
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: adapter)
    result = cpu_tasks.submit_decision_order.run(str(order_id))
    assert result["skipped"] == "ordinary_execution_not_authorized"
    assert adapter.place_calls == adapter.lookup_calls == 0
    with sessions() as session:
        assert session.get(OrderRecord, order_id).submit_attempted_at is None


@pytest.mark.parametrize("mode", ["legacy", "shadow", "halted"])
def test_uuid_materialize_boundary_blocks_ordinary(sessions, monkeypatch, decision_task_settings, mode):
    from poseidon.workers import cpu_tasks

    with sessions() as session:
        decision_id, _ = _claimed(session)
    decision_task_settings.decision_loop_execution_mode = mode
    decision_task_settings.decision_loop_execution_enabled = False
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: pytest.fail("broker snapshot IO"))
    assert cpu_tasks.materialize_execution_claim.run(str(decision_id))["skipped"] == "ordinary_execution_not_authorized"
    with sessions() as session:
        assert session.query(OrderRecord).count() == 0


@pytest.mark.parametrize("mode", ["legacy", "shadow", "halted"])
def test_uuid_attempted_boundary_remains_lookup_only_in_every_mode(sessions, monkeypatch, decision_task_settings, mode):
    from poseidon.workers import cpu_tasks

    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions, timeout=True, no_order=True)
    with pytest.raises(TimeoutError):
        submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)
    decision_task_settings.decision_loop_execution_mode = mode
    decision_task_settings.decision_loop_execution_enabled = False
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: adapter)
    cpu_tasks.submit_decision_order.run(str(order_id))
    assert (adapter.place_calls, adapter.lookup_calls) == (1, 1)


@pytest.mark.parametrize(
    "task_name",
    [
        "materialize_execution_claim",
        "submit_decision_order",
        "reconcile_execution_claim",
        "project_decision_fill",
        "reconcile_paper_account",
    ],
)
def test_internal_task_uuid_only_rejects_malformed_before_io(monkeypatch, task_name):
    from poseidon.workers import cpu_tasks

    task = getattr(cpu_tasks, task_name)
    assert len(inspect.signature(task.run).parameters) == 1

    def forbidden(*args, **kwargs):
        pytest.fail("malformed task payload performed I/O")

    monkeypatch.setattr(cpu_tasks, "SessionLocal", forbidden)
    with pytest.raises(ValueError):
        task.run("not-a-uuid")


def test_submit_task_redelivery_preserves_unknown_marker(sessions, monkeypatch, decision_task_settings):
    from poseidon.workers import cpu_tasks

    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions, timeout=True, no_order=True)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: adapter)
    with pytest.raises(TimeoutError):
        cpu_tasks.submit_decision_order.run(str(order_id))
    result = cpu_tasks.submit_decision_order.run(str(order_id))
    assert result["reconciliation_status"] == "required"
    assert adapter.place_calls == 1
    assert adapter.lookup_calls == 1
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        assert order.submit_attempted_at is not None
        assert order.reconciliation_status == "required"


def test_recovery_sweep_uuid_routes_reconcile_to_decision(sessions, monkeypatch):
    from poseidon.workers import cpu_tasks

    decision_id, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions, timeout=True, no_order=True)
    with pytest.raises(TimeoutError):
        submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    queued = []
    for name in (
        "materialize_execution_claim",
        "submit_decision_order",
        "reconcile_execution_claim",
        "project_decision_fill",
        "reconcile_paper_account",
    ):
        monkeypatch.setattr(getattr(cpu_tasks, name), "delay", lambda value, name=name: queued.append((name, value)))
    assert not inspect.signature(cpu_tasks.recover_decision_execution.run).parameters
    cpu_tasks.recover_decision_execution.run()
    assert ("reconcile_execution_claim", str(decision_id)) in queued
    assert ("submit_decision_order", str(order_id)) not in queued
    assert all(isinstance(value, str) and str(uuid.UUID(value)) == value for _, value in queued)


def test_reconcile_task_decision_uuid_never_submits_unattempted(sessions, monkeypatch):
    from poseidon.workers import cpu_tasks

    decision_id, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: adapter)
    cpu_tasks.reconcile_execution_claim.run(str(decision_id))
    assert adapter.place_calls == 0
    assert adapter.lookup_calls == 0
    with sessions() as session:
        assert session.get(OrderRecord, order_id).submit_attempted_at is None


def test_materialization_short_liquidation_nav():
    from poseidon.workers.cpu_tasks import _decision_liquidation_nav

    policy = SimpleNamespace(
        perp_instrument_rules={
            "BTC-USDT": SimpleNamespace(
                contract_multiplier=2, margin_semantics="full_notional", funding_semantics="excluded"
            )
        }
    )
    position = BrokerPositionSnapshot("crypto_perp", "BTC", "BTC-USDT", "short", 1)
    snapshot = SimpleNamespace(cash=800, positions=(position,))
    order = SimpleNamespace(market="crypto_perp", symbol="BTC", instrument="BTC-USDT", side="short", action="sell")
    fill = SimpleNamespace(fill_quantity=1, fill_price=100)
    assert (
        _decision_liquidation_nav(snapshot, policy, {("crypto_perp", "BTC", "BTC-USDT"): 120}, [(order, fill)]) == 960
    )


def test_materialization_stock_short_liquidation_nav_partial_close():
    from poseidon.workers.cpu_tasks import _decision_liquidation_nav

    position = BrokerPositionSnapshot("tw_stock", "2330", "spot", "short", 1)
    snapshot = SimpleNamespace(cash=950, positions=(position,))
    opening = SimpleNamespace(market="tw_stock", symbol="2330", instrument="spot", side="short", action="sell")
    closing = SimpleNamespace(**{**vars(opening), "action": "buy"})
    ledger = [
        (opening, SimpleNamespace(fill_quantity=2, fill_price=100)),
        (closing, SimpleNamespace(fill_quantity=1, fill_price=150)),
    ]
    assert _decision_liquidation_nav(snapshot, None, PRICE_2330, ledger) == 1050


@pytest.mark.parametrize(
    "corruption",
    ["side", "action", "snapshot_duplicate", "negative", "missing_mark", "infinite_mark", "multiplier", "semantics"],
)
def test_materialization_liquidation_nav_fails_closed(corruption):
    from poseidon.workers.cpu_tasks import _decision_liquidation_nav

    rule = SimpleNamespace(contract_multiplier=2, margin_semantics="full_notional", funding_semantics="excluded")
    policy = SimpleNamespace(perp_instrument_rules={"BTC-USDT": rule})
    position = BrokerPositionSnapshot("crypto_perp", "BTC", "BTC-USDT", "short", 1)
    snapshot = SimpleNamespace(cash=800, positions=(position,))
    order = SimpleNamespace(market="crypto_perp", symbol="BTC", instrument="BTC-USDT", side="short", action="sell")
    fill = SimpleNamespace(fill_quantity=1, fill_price=100)
    prices = {("crypto_perp", "BTC", "BTC-USDT"): 120}
    if corruption == "side":
        order.side = "garbage"
        snapshot.positions = (replace(position, side="garbage"),)
    elif corruption == "action":
        order.action = "garbage"
        snapshot.positions = (replace(position, quantity=-1),)
    elif corruption == "snapshot_duplicate":
        snapshot.positions = (position, position)
    elif corruption == "negative":
        order.action = "buy"
    elif corruption == "missing_mark":
        prices.clear()
    elif corruption == "infinite_mark":
        prices[("crypto_perp", "BTC", "BTC-USDT")] = float("inf")
    elif corruption == "multiplier":
        rule.contract_multiplier = -2
    else:
        rule.margin_semantics = "leveraged"
    with pytest.raises(ValueError):
        _decision_liquidation_nav(snapshot, policy, prices, [(order, fill)])


def test_materialize_task_loads_current_prices_outside_db_session(sessions, monkeypatch, decision_task_settings):
    from poseidon.workers import cpu_tasks

    with sessions() as session:
        decision_id, _ = _claimed(session)
    active = []

    @contextlib.contextmanager
    def tracked_sessions():
        with sessions() as session:
            active.append(session)
            try:
                yield session
            finally:
                active.remove(session)

    def prices(symbols):
        assert not active, "price I/O retained a DB session"
        assert symbols == ["2330"]
        return {"2330": 200.0}

    monkeypatch.setattr(cpu_tasks, "SessionLocal", tracked_sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: PaperBrokerAdapter(sessions))
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", prices)
    # The durable seed has a bounded validity window; freeze the service's clock.
    monkeypatch.setattr("poseidon.decision_loop.execution.timestamp", lambda value, field: NOW)
    result = cpu_tasks.materialize_execution_claim.run(str(decision_id))
    with sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(result["order_ids"][0]))
        assert order.price == 200
        assert order.quantity == 50


def test_materialize_task_rejects_account_changed_during_price_io(sessions, monkeypatch, decision_task_settings):
    from poseidon.workers import cpu_tasks

    with sessions() as session:
        decision_id, _ = _claimed(session)

    def prices(symbols):
        with sessions() as session:
            account = session.query(PaperBrokerAccount).one()
            account.state_version += 1
            account.updated_at = NOW - timedelta(seconds=10)
            reconciliation = session.query(AccountReconciliation).one()
            reconciliation.broker_state_watermark = "broker:1"
            reconciliation.as_of = NOW
            session.commit()
        return {"2330": 100.0}

    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", prices)
    monkeypatch.setattr("poseidon.decision_loop.execution.timestamp", lambda value, field: NOW)
    with pytest.raises(ValueError, match="changed"):
        cpu_tasks.materialize_execution_claim.run(str(decision_id))
    with sessions() as session:
        assert session.query(OrderRecord).count() == 0


def test_recovery_sweep_repeated_execution_keeps_one_effective_order_fill(
    sessions, monkeypatch, decision_task_settings
):
    from poseidon.workers import cpu_tasks

    decision_id, order_id = _intent(sessions)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )
    queued = []
    for name in (
        "materialize_execution_claim",
        "submit_decision_order",
        "reconcile_execution_claim",
        "project_decision_fill",
        "reconcile_paper_account",
    ):
        monkeypatch.setattr(getattr(cpu_tasks, name), "delay", lambda value, name=name: queued.append((name, value)))
    # Two sweeps before delivery enqueue the same submit UUID twice.
    cpu_tasks.recover_decision_execution.run()
    cpu_tasks.recover_decision_execution.run()
    for task_name, value in queued:
        if task_name == "submit_decision_order":
            getattr(cpu_tasks, task_name).run(value)
    assert queued.count(("submit_decision_order", str(order_id))) == 2
    queued.clear()
    cpu_tasks.recover_decision_execution.run()
    cpu_tasks.reconcile_execution_claim.run(str(decision_id))
    cpu_tasks.submit_decision_order.run(str(order_id))
    with sessions() as session:
        assert session.query(PaperBrokerOrder).count() == 1
        assert session.query(PaperBrokerFill).count() == 1
        assert session.query(OrderFillRecord).count() == 1


def test_future_task_seams_fail_observably(sessions, monkeypatch):
    from poseidon.positions.lots import FillProjectionConflictError
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    with pytest.raises(FillProjectionConflictError, match="fill does not exist"):
        cpu_tasks.project_decision_fill.run(str(uuid.uuid4()))
    with pytest.raises(ValueError, match="paper account does not exist"):
        cpu_tasks.reconcile_paper_account.run(str(uuid.uuid4()))


def test_recovery_sweep_deduplicates_orders_of_one_decision(sessions, monkeypatch):
    from poseidon.workers import cpu_tasks

    with sessions() as session:
        decision_id, _ = _claimed(session, symbols=("2330", "2317"))
    result = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000,
        prices={**PRICE_2330, ("tw_stock", "2317", "spot"): 100},
        now=NOW,
    )
    for order_id in result["order_ids"]:
        with pytest.raises(TimeoutError):
            submit_or_reconcile_order(sessions, uuid.UUID(order_id), SnapshotAdapter(sessions, timeout=True), now=NOW)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    queued = []
    monkeypatch.setattr(cpu_tasks.reconcile_execution_claim, "delay", queued.append)
    monkeypatch.setattr(cpu_tasks.reconcile_paper_account, "delay", lambda value: None)
    cpu_tasks.recover_decision_execution.run()
    assert queued == [str(decision_id)]


def test_materialize_task_loads_pending_reservation_marks(sessions, monkeypatch, decision_task_settings):
    from poseidon.workers import cpu_tasks

    with sessions() as session:
        decision_id, _ = _claimed(session)
    response = materialize_order_intents(
        sessions, decision_id, principal=worker(), account_nav=100_000, prices=PRICE_2330, now=NOW
    )
    with sessions() as session:
        # This outstanding reservation has no broker inventory yet.
        session.get(OrderRecord, uuid.UUID(response["order_ids"][0])).symbol = "2317"
        session.commit()
    observed = []

    def prices(symbols):
        observed.extend(symbols)
        return {symbol: 100.0 for symbol in symbols}

    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", prices)
    monkeypatch.setattr("poseidon.decision_loop.execution.timestamp", lambda value, field: NOW)
    # Replay rejects our isolated reservation fault, after collecting marks.
    with pytest.raises(ExecutionConflictError, match="content changed"):
        cpu_tasks.materialize_execution_claim.run(str(decision_id))
    assert observed == ["2317", "2330"]


def test_reconcile_task_processes_siblings_after_observable_failure(sessions, monkeypatch):
    from poseidon.workers import cpu_tasks

    with sessions() as session:
        decision_id, _ = _claimed(session, symbols=("2330", "2317"))
    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000,
        prices={**PRICE_2330, ("tw_stock", "2317", "spot"): 100},
        now=NOW,
    )
    bad_id, good_id = sorted(uuid.UUID(value) for value in response["order_ids"])
    adapter = SnapshotAdapter(sessions)
    with sessions() as session, session.begin():
        for order_id in (bad_id, good_id):
            ReconciliationService(session).prepare_attempt(order_id, adapter, now=NOW)
        bad_ref = session.get(OrderRecord, bad_id).client_order_ref
        good = session.get(OrderRecord, good_id)
        adapter.snapshot = _snapshot(good)
        adapter.fills = [_fill(good, adapter.snapshot)]
    lookup = adapter.find_order_by_client_ref

    def failing_lookup(client_order_ref, **kwargs):
        if client_order_ref == bad_ref:
            raise TimeoutError("unavailable first order")
        return lookup(client_order_ref, **kwargs)

    monkeypatch.setattr(adapter, "find_order_by_client_ref", failing_lookup)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", sessions)
    monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda market: adapter)
    with pytest.raises(ExceptionGroup) as error:
        cpu_tasks.reconcile_execution_claim.run(str(decision_id))
    assert len(error.value.exceptions) == 1
    assert isinstance(error.value.exceptions[0], TimeoutError)
    assert str(error.value.exceptions[0]) == "unavailable first order"
    with sessions() as session:
        assert session.get(OrderRecord, bad_id).reconciliation_status == "required"
        assert session.get(OrderRecord, good_id).status == "filled"
        assert session.query(OrderFillRecord).filter_by(order_id=good_id).count() == 1


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reconciliation.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def _intent(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session)
    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    return decision_id, uuid.UUID(response["order_ids"][0])


@pytest.fixture
def postgres_intent(claim_seed):  # noqa: F811
    sessions = sessionmaker(ENGINE, expire_on_commit=False)
    decision_id = claim_seed.decision_ids[0]
    with sessions() as session:
        DecisionService(session).claim_execution(
            decision_id,
            2,
            principal=claim_seed.worker,
            now=CLAIM_TIME,
        )
        decision = session.get(DecisionRecord, decision_id)
        version = session.get(StrategyVersion, decision.strategy_version_id)
        reconciliation = version.policy_json["reconciliation"]
        session.add(
            PaperBrokerAccount(
                account_scope=claim_seed.account,
                account_generation=reconciliation["account_generation"],
                opening_cash=reconciliation["opening_cash"],
                currency=reconciliation["currency"],
                created_at=CLAIM_TIME - timedelta(minutes=5),
                updated_at=CLAIM_TIME - timedelta(minutes=5),
            )
        )
        session.flush()
        session.add(
            AccountReconciliation(
                account_scope=claim_seed.account,
                account_generation=reconciliation["account_generation"],
                as_of=CLAIM_TIME - timedelta(minutes=1),
                broker_state_watermark="broker:0",
                internal_state_watermark=internal_state_watermark(
                    session,
                    claim_seed.account,
                    reconciliation["account_generation"],
                ),
                broker_snapshot_sha256="a" * 64,
                broker_snapshot_json={},
                internal_snapshot_json={},
                difference_json={},
                policy_sha256=decision.policy_sha256,
                status="matched",
            )
        )
        session.commit()
    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=claim_seed.worker,
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=CLAIM_TIME,
    )
    order_id = uuid.UUID(response["order_ids"][0])
    yield sessions, order_id
    with Session(ENGINE) as session:
        session.execute(delete(OrderFillRecord).where(OrderFillRecord.order_id == order_id))
        session.execute(delete(OrderRecord).where(OrderRecord.id == order_id))
        session.execute(delete(AccountReconciliation).where(AccountReconciliation.account_scope == claim_seed.account))
        session.execute(delete(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == claim_seed.account))
        session.commit()


def _snapshot(order, *, status="filled", broker_order_id="paper-order-1"):
    return BrokerOrderSnapshot(
        broker_order_id=broker_order_id,
        client_order_ref=order.client_order_ref,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
        market=order.market,
        symbol=order.symbol,
        instrument=order.instrument,
        action=order.action,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price,
        status=status,
        accepted_at=NOW,
        state_version=1,
    )


def _fill(order, snapshot, *, broker_fill_id="paper-fill-1", quantity=None, price=None):
    return BrokerFillSnapshot(
        broker_order_id=snapshot.broker_order_id,
        broker_fill_id=broker_fill_id,
        account_scope=order.account_scope,
        account_generation=order.account_generation,
        market=order.market,
        symbol=order.symbol,
        instrument=order.instrument,
        side=order.side,
        fill_price=order.price if price is None else price,
        fill_quantity=order.quantity if quantity is None else quantity,
        fill_time=NOW,
        state_version=1,
    )


class SnapshotAdapter(BrokerAdapter):
    capabilities = PAPER_RECONCILIATION_CAPABILITIES

    def __init__(self, sessions, *, status="filled", no_order=False, timeout=False):
        self.sessions = sessions
        self.status = status
        self.no_order = no_order
        self.timeout = timeout
        self.place_calls = 0
        self.lookup_calls = 0
        self.snapshot = None
        self.fills = []

    def login(self):
        return True

    def place_order(self, order, *, client_order_ref=None):
        self.place_calls += 1
        with self.sessions() as session:
            stored = session.get(OrderRecord, uuid.UUID(order.id))
            assert stored.submit_attempted_at is not None
            assert stored.status == "reconciliation_required"
            assert stored.reconciliation_status == "required"
        if self.timeout:
            raise TimeoutError("broker response unknown")
        self.snapshot = _snapshot(order, status=self.status)
        self.fills = [] if self.status in {"submitted", "rejected", "cancelled"} else [_fill(order, self.snapshot)]
        return self.snapshot

    def find_order_by_client_ref(self, client_order_ref, *, account_scope, account_generation):
        self.lookup_calls += 1
        return None if self.no_order else self.snapshot

    def query_order(self, broker_order_id, *, account_scope, account_generation):
        return self.snapshot

    def query_fills(self, broker_order_id, *, account_scope=None, account_generation=None):
        return self.fills

    def query_account_snapshot(self, account_scope, account_generation):
        return BrokerAccountSnapshot(account_scope, account_generation, "TWD", 90_000.0, 1, NOW, ())

    def query_positions(self):
        return []

    def logout(self):
        return None


def test_runner_commits_attempt_before_io_and_service_flushes_only(sessions, monkeypatch):
    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)

    with sessions() as session:
        monkeypatch.setattr(session, "commit", lambda: (_ for _ in ()).throw(AssertionError("service committed")))
        prepared = ReconciliationService(session).prepare_attempt(order_id, adapter, now=NOW)
        assert prepared.should_submit is True
        session.rollback()

    result = submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)

    assert result["status"] == "filled"
    assert adapter.place_calls == 1
    assert adapter.lookup_calls == 0
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        assert order.status == "filled"
        assert order.broker_order_id == "paper-order-1"
        assert session.query(OrderFillRecord).filter_by(order_id=order_id).one().projection_status == (
            "projection_pending"
        )


def test_postgres_attempt_lock_commits_before_io_and_concurrent_redelivery_only_looks_up(postgres_intent):
    sessions, order_id = postgres_intent
    started = Event()
    release = Event()

    class BlockingAdapter(SnapshotAdapter):
        def place_order(self, order, *, client_order_ref=None):
            started.set()
            assert release.wait(timeout=10)
            return super().place_order(order, client_order_ref=client_order_ref)

    adapter = BlockingAdapter(sessions)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(submit_or_reconcile_order, sessions, order_id, adapter, now=CLAIM_TIME)
        assert started.wait(timeout=10)
        redelivery = submit_or_reconcile_order(sessions, order_id, adapter, now=CLAIM_TIME)
        release.set()
        accepted = first.result(timeout=10)

    assert redelivery["status"] == "reconciliation_required"
    assert accepted["status"] == "filled"
    assert (adapter.place_calls, adapter.lookup_calls) == (1, 1)


def test_attempted_redelivery_only_looks_up_and_unknown_stays_required(sessions):
    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions, no_order=True, timeout=True)

    with pytest.raises(TimeoutError, match="unknown"):
        submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)
    assert adapter.place_calls == 1

    adapter.timeout = False
    result = submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)

    assert result["status"] == "reconciliation_required"
    assert result["operation"] == "reconcile_order"
    assert adapter.place_calls == 1
    assert adapter.lookup_calls == 1
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        assert order.status == "reconciliation_required"
        assert order.reconciliation_status == "required"


def test_accept_then_timeout_restarts_by_reference_without_duplicate_broker_truth(sessions, monkeypatch):
    _, order_id = _intent(sessions)
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: type("Repo", (), {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [100.0]})})(),
    )

    class LoseResponse(PaperBrokerAdapter):
        calls = 0

        def place_decision_order(self, order, *, client_order_ref):
            type(self).calls += 1
            super().place_decision_order(order, client_order_ref=client_order_ref)
            raise TimeoutError("response lost after acceptance")

    with pytest.raises(TimeoutError, match="after acceptance"):
        submit_or_reconcile_order(sessions, order_id, LoseResponse(sessions), now=NOW)

    result = submit_or_reconcile_order(sessions, order_id, PaperBrokerAdapter(sessions), now=NOW)

    assert result["status"] == "filled"
    assert LoseResponse.calls == 1
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PaperBrokerOrder)) == 1
        assert session.scalar(select(func.count()).select_from(PaperBrokerFill)) == 1
        assert session.query(OrderFillRecord).filter_by(order_id=order_id).count() == 1


@pytest.mark.parametrize(
    ("broker_status", "internal_status", "fill_quantity", "reconciliation_status"),
    [
        ("open", "submitted", None, "required"),
        ("submitted", "submitted", None, "required"),
        ("rejected", "rejected", None, "resolved"),
        ("cancelled", "cancelled", None, "resolved"),
        ("partially_filled", "partially_filled", 40.0, "required"),
        ("filled", "filled", 100.0, "resolved"),
    ],
)
def test_known_broker_statuses_normalize_legally(
    sessions,
    broker_status,
    internal_status,
    fill_quantity,
    reconciliation_status,
):
    _, order_id = _intent(sessions)
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        snapshot = _snapshot(order, status=broker_status)
        fills = [] if fill_quantity is None else [_fill(order, snapshot, quantity=fill_quantity)]
        service.import_broker_state(order_id, snapshot, fills, now=NOW)
        assert order.status == internal_status
        assert order.reconciliation_status == reconciliation_status
        session.commit()


def test_fill_replay_is_exact_and_changed_economics_fail_closed(sessions):
    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)
    submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)

    with sessions() as session:
        service = ReconciliationService(session)
        service.import_broker_state(order_id, adapter.snapshot, adapter.fills, now=NOW)
        assert session.query(OrderFillRecord).filter_by(order_id=order_id).count() == 1
        changed = replace(adapter.fills[0], fill_price=101.0)
        with pytest.raises(ReconciliationConflictError, match="fill identity changed economics"):
            service.import_broker_state(order_id, adapter.snapshot, [changed], now=NOW)


def test_partial_replay_cannot_drop_an_imported_fill(sessions):
    _, order_id = _intent(sessions)
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        snapshot = _snapshot(order, status="partially_filled")
        first = _fill(order, snapshot, broker_fill_id="fill-1", quantity=40.0)
        second = _fill(order, snapshot, broker_fill_id="fill-2", quantity=20.0)
        service.import_broker_state(order_id, snapshot, [first], now=NOW)
        service.import_broker_state(order_id, snapshot, [first, second], now=NOW)
        with pytest.raises(ReconciliationConflictError, match="broker replay omitted an imported fill"):
            service.import_broker_state(order_id, snapshot, [first], now=NOW)


def test_broker_snapshot_price_must_match_fill_economics(sessions):
    _, order_id = _intent(sessions)
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        snapshot = _snapshot(order, status="filled")
        with pytest.raises(ReconciliationConflictError, match="snapshot price disagrees"):
            service.import_broker_state(order_id, snapshot, [_fill(order, snapshot, price=101.0)], now=NOW)


def test_multi_price_fill_average_uses_canonical_float_representation(sessions):
    _, order_id = _intent(sessions)
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        average = float((100.0 + 2 * 101.0) / 3)
        snapshot = replace(_snapshot(order, status="partially_filled"), price=average)
        fills = [
            _fill(order, snapshot, broker_fill_id="fill-1", quantity=1.0, price=100.0),
            _fill(order, snapshot, broker_fill_id="fill-2", quantity=2.0, price=101.0),
        ]
        service.import_broker_state(order_id, snapshot, fills, now=NOW)
        assert order.status == "partially_filled"


def test_terminal_runner_redelivery_only_looks_up_and_replays(sessions):
    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)

    first = submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)
    with sessions() as session:
        first_updated_at = session.get(OrderRecord, order_id).updated_at
    replay = submit_or_reconcile_order(sessions, order_id, adapter, now=NOW + timedelta(seconds=1))

    assert replay == first
    assert (adapter.place_calls, adapter.lookup_calls) == (1, 1)
    with sessions() as session:
        assert session.query(OrderFillRecord).filter_by(order_id=order_id).count() == 1
        assert session.get(OrderRecord, order_id).updated_at == first_updated_at


def test_terminal_lookup_missing_persists_required_without_illegal_status_reset(sessions):
    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)
    submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)
    adapter.no_order = True

    result = submit_or_reconcile_order(sessions, order_id, adapter, now=NOW + timedelta(seconds=1))

    assert result["status"] == "filled"
    assert result["reconciliation_status"] == "required"
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        assert (order.status, order.reconciliation_status) == ("filled", "required")
        assert ("reconcile_order", order_id) in {
            (action.operation, action.persisted_id) for action in RecoverySelector(session).select(now=NOW)
        }


def test_cancelled_partial_fill_stays_reserved_until_projection(sessions):
    _, order_id = _intent(sessions)
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        snapshot = _snapshot(order, status="cancelled")
        service.import_broker_state(order_id, snapshot, [_fill(order, snapshot, quantity=40.0)], now=NOW)
        session.commit()

    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        fill = session.query(OrderFillRecord).filter_by(order_id=order_id).one()
        assert order.status == "cancelled"
        assert order.reservation_status == "reserved"
        assert fill.projection_status == "projection_pending"


@pytest.mark.parametrize(
    ("cancelled", "expected_fifo"),
    [("first", (40.0, 0.0)), ("second", (60.0, 0.0))],
)
def test_zero_fill_cancel_rebuilds_other_close_reservations_fifo(sessions, cancelled, expected_fifo):
    with sessions() as session:
        decision_id, _ = _claimed(session, action="reduce", target_weight=0.04)
        lot_ids = (uuid.uuid4(), uuid.uuid4())
        for index, (lot_id, quantity) in enumerate(zip(lot_ids, (60.0, 40.0), strict=True)):
            session.add(
                PositionLot(
                    id=lot_id,
                    account_scope="paper:pilot",
                    account_generation="generation-1",
                    market="tw_stock",
                    symbol="2330",
                    instrument="spot",
                    side="long",
                    opening_fill_id=uuid.uuid4(),
                    opening_decision_id=decision_id,
                    original_quantity=quantity,
                    open_quantity=quantity,
                    reserved_close_quantity=0.0,
                    cost_basis_json={"price": 80.0},
                    opened_at=NOW - timedelta(days=2 - index),
                    created_at=NOW - timedelta(days=2 - index),
                    updated_at=NOW - timedelta(days=2 - index),
                )
            )
        session.commit()
    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    first_order_id = uuid.UUID(response["order_ids"][0])
    second_order_id = uuid.uuid4()

    with sessions() as session:
        first = session.get(OrderRecord, first_order_id)
        assert first.quantity == 60.0
        if cancelled == "first":
            ReconciliationService(session).prepare_attempt(first_order_id, SnapshotAdapter(sessions), now=NOW)
        values = {column.name: copy.deepcopy(getattr(first, column.name)) for column in OrderRecord.__table__.columns}
        values.update(
            id=second_order_id,
            quantity=40.0,
            target_weight=0.0,
            status="reconciliation_required" if cancelled == "second" else "pending_submit",
            broker_order_id=None,
            execution_key=uuid.uuid4(),
            client_order_ref=f"{first.client_order_ref}-later",
            reserved_quantity=40.0,
            reservation_status="reserved",
            reconciliation_status="required" if cancelled == "second" else "pending",
            submit_attempted_at=NOW if cancelled == "second" else None,
        )
        session.add(OrderRecord(**values))
        session.get(PositionLot, lot_ids[1]).reserved_close_quantity = 40.0
        session.commit()

    with sessions() as session:
        order_id = first_order_id if cancelled == "first" else second_order_id
        order = session.get(OrderRecord, order_id)
        ReconciliationService(session).import_broker_state(
            order_id,
            _snapshot(order, status="cancelled"),
            [],
            now=NOW,
        )
        session.commit()

    with sessions() as session:
        assert session.get(OrderRecord, order_id).reservation_status == "released"
        assert tuple(session.get(PositionLot, lot_id).reserved_close_quantity for lot_id in lot_ids) == expected_fifo
        order = session.get(OrderRecord, order_id)
        ReconciliationService(session).import_broker_state(
            order_id,
            _snapshot(order, status="cancelled"),
            [],
            now=NOW + timedelta(seconds=1),
        )
        session.commit()

    with sessions() as session:
        assert tuple(session.get(PositionLot, lot_id).reserved_close_quantity for lot_id in lot_ids) == expected_fifo


def test_cancel_after_applied_partial_fill_releases_only_unfilled_close_reservation(sessions):
    with sessions() as session:
        decision_id, _ = _claimed(session, action="exit", target_weight=0.0)
        lot_id = uuid.uuid4()
        session.add(
            PositionLot(
                id=lot_id,
                account_scope="paper:pilot",
                account_generation="generation-1",
                market="tw_stock",
                symbol="2330",
                instrument="spot",
                side="long",
                opening_fill_id=uuid.uuid4(),
                opening_decision_id=decision_id,
                original_quantity=100.0,
                open_quantity=100.0,
                reserved_close_quantity=0.0,
                cost_basis_json={"price": 80.0},
                opened_at=NOW - timedelta(days=1),
                created_at=NOW - timedelta(days=1),
                updated_at=NOW - timedelta(days=1),
            )
        )
        session.commit()
    response = materialize_order_intents(
        sessions,
        decision_id,
        principal=worker(),
        account_nav=100_000.0,
        prices=PRICE_2330,
        now=NOW,
    )
    order_id = uuid.UUID(response["order_ids"][0])
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        partial = _snapshot(order, status="partially_filled")
        fill = _fill(order, partial, quantity=40.0)
        service.import_broker_state(order_id, partial, [fill], now=NOW)
        session.commit()
    with sessions() as session:
        internal_fill = session.query(OrderFillRecord).filter_by(order_id=order_id).one()
        internal_fill.projection_status = "corrupt"
        session.commit()
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        with pytest.raises(ReconciliationConflictError, match="projection status"):
            ReconciliationService(session).import_broker_state(
                order_id,
                _snapshot(order, status="cancelled"),
                [_fill(order, partial, quantity=40.0)],
                now=NOW + timedelta(seconds=1),
            )
    with sessions() as session:
        assert session.get(OrderRecord, order_id).reservation_status == "reserved"
        assert session.get(PositionLot, lot_id).reserved_close_quantity == 100.0
        internal_fill = session.query(OrderFillRecord).filter_by(order_id=order_id).one()
        internal_fill.projection_status = "applied"
        lot = session.get(PositionLot, lot_id)
        lot.open_quantity = 60.0
        lot.reserved_close_quantity = 60.0
        session.commit()
    with sessions() as session:
        assert ("reconcile_order", order_id) in {
            (action.operation, action.persisted_id) for action in RecoverySelector(session).select(now=NOW)
        }
        order = session.get(OrderRecord, order_id)
        ReconciliationService(session).import_broker_state(
            order_id,
            _snapshot(order, status="cancelled"),
            [_fill(order, partial, quantity=40.0)],
            now=NOW + timedelta(seconds=1),
        )
        session.commit()
    with sessions() as session:
        assert session.get(OrderRecord, order_id).reservation_status == "released"
        assert session.get(PositionLot, lot_id).reserved_close_quantity == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantity", 200.0),
        ("symbol", "2317"),
        ("account_scope", "paper:other"),
        ("client_order_ref", "DL-corrupt-9999"),
    ],
)
def test_pre_submit_rejects_columns_changed_from_frozen_intent(sessions, field, value):
    _, order_id = _intent(sessions)
    with sessions() as session:
        setattr(session.get(OrderRecord, order_id), field, value)
        session.commit()
    adapter = SnapshotAdapter(sessions)

    with pytest.raises(ReconciliationConflictError, match="durable order intent changed"):
        submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)

    assert adapter.place_calls == 0
    with sessions() as session:
        assert session.get(OrderRecord, order_id).submit_attempted_at is None


def test_market_execution_price_change_is_not_reconciliation_drift(sessions):
    _, order_id = _intent(sessions)
    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        service = ReconciliationService(session)
        service.prepare_attempt(order_id, SnapshotAdapter(sessions), now=NOW)
        snapshot = replace(_snapshot(order), price=101.0)
        fills = [_fill(order, snapshot, price=101.0)]
        service.import_broker_state(order_id, snapshot, fills, now=NOW)
        session.commit()

    account = BrokerAccountSnapshot("paper:pilot", "generation-1", "TWD", 89_900.0, 1, NOW, ())
    with sessions() as session:
        assert ReconciliationService(session).compare_execution_state(order_id, snapshot, fills, account) == []


def test_corrupt_internal_economics_report_drift_without_mutating_broker_truth(sessions):
    _, order_id = _intent(sessions)
    adapter = SnapshotAdapter(sessions)
    submit_or_reconcile_order(sessions, order_id, adapter, now=NOW)
    account = adapter.query_account_snapshot("paper:pilot", "generation-1")

    with sessions() as session:
        order = session.get(OrderRecord, order_id)
        order.quantity = 99.0
        order.reserved_cash_json = {"currency": "TWD", "amount": 123.0}
        session.query(OrderFillRecord).filter_by(order_id=order_id).one().fill_price = 88.0
        session.commit()

    with sessions() as session:
        drift = ReconciliationService(session).compare_execution_state(
            order_id,
            adapter.snapshot,
            adapter.fills,
            account,
        )
    assert {item["field"] for item in drift} >= {
        "order.quantity",
        "fill.paper-fill-1.fill_price",
        "cash.reserved_amount",
    }
    assert adapter.snapshot.quantity == 100.0
    assert adapter.fills[0].fill_price == 100.0


def test_recovery_selector_classifies_gaps_without_broker_calls(sessions):
    pending_decision, pending_order = _intent(sessions)
    with sessions() as session:
        source_decision = session.get(DecisionRecord, pending_decision)
        claimed_without_intent = uuid.uuid4()
        session.add(
            DecisionRecord(
                id=claimed_without_intent,
                evaluation_run_id=source_decision.evaluation_run_id,
                strategy_version_id=source_decision.strategy_version_id,
                account_scope=source_decision.account_scope,
                decision_as_of=source_decision.decision_as_of,
                valid_until=source_decision.valid_until,
                status="execution_claimed",
                revision=3,
                creation_sha256=uuid.uuid4().hex,
                policy_sha256=source_decision.policy_sha256,
                original_json=copy.deepcopy(source_decision.original_json),
                final_json=copy.deepcopy(source_decision.final_json),
                portfolio_snapshot_json=copy.deepcopy(source_decision.portfolio_snapshot_json),
                risk_snapshot_json=copy.deepcopy(source_decision.risk_snapshot_json),
                execution_key=uuid.uuid4(),
                claimed_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        pending = session.get(OrderRecord, pending_order)
        attempted_order = uuid.uuid4()
        session.add(
            OrderRecord(
                id=attempted_order,
                strategy_name=pending.strategy_name,
                symbol=pending.symbol,
                market=pending.market,
                action=pending.action,
                order_type=pending.order_type,
                target_weight=pending.target_weight,
                quantity=pending.quantity,
                price=pending.price,
                side=pending.side,
                status="reconciliation_required",
                broker_mode="paper",
                order_origin="decision",
                decision_id=pending.decision_id,
                account_scope=pending.account_scope,
                account_generation=pending.account_generation,
                execution_key=pending.execution_key,
                client_order_ref=f"{pending.client_order_ref}-recovery",
                instrument=pending.instrument,
                intent_json=copy.deepcopy(pending.intent_json),
                intent_sha256=pending.intent_sha256,
                reserved_cash_json=copy.deepcopy(pending.reserved_cash_json),
                reserved_quantity=pending.reserved_quantity,
                reservation_status="reserved",
                reconciliation_status="required",
                submit_attempted_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            OrderFillRecord(
                id=uuid.uuid4(),
                order_id=attempted_order,
                fill_price=100.0,
                fill_quantity=1.0,
                fill_time=NOW,
                broker_fill_id="selector-fill",
                projection_status="projection_pending",
                created_at=NOW,
            )
        )
        session.commit()

    with sessions() as session:
        actions = RecoverySelector(session).select()
        assert actions == RecoverySelector(session).select()

    keys = {(action.operation, action.persisted_id) for action in actions}
    assert ("materialize_decision", claimed_without_intent) in keys
    assert ("submit_order", pending_order) in keys
    assert ("reconcile_order", attempted_order) in keys
    assert any(action.operation == "project_fill" for action in actions)
    assert any(action.operation == "reconcile_account" for action in actions)


def test_recovery_selector_uses_owner_age_for_stale_account(sessions):
    with sessions() as session:
        _claimed(session)

    with sessions() as session:
        actions = RecoverySelector(session).select(now=NOW + timedelta(seconds=301))

    assert any(action.operation == "reconcile_account" for action in actions)


def test_recovery_selector_matches_policy_to_account_generation(sessions):
    generation_1 = synthetic_reconciliation_policy(account_generation="generation-1", max_reconciliation_age_seconds=10)
    generation_2 = synthetic_reconciliation_policy(
        account_generation="generation-2",
        max_reconciliation_age_seconds=1_000,
    )
    with sessions() as session:
        first_id, _ = _claimed(session, reconciliation=generation_1)
        second_id, _ = _claimed(session, reconciliation=generation_2)
        session.get(DecisionRecord, first_id).created_at = NOW - timedelta(minutes=2)
        session.get(DecisionRecord, second_id).created_at = NOW - timedelta(minutes=1)
        accounts = {row.account_generation: row.id for row in session.scalars(select(PaperBrokerAccount))}
        session.commit()

    with sessions() as session:
        account_actions = {
            action.persisted_id
            for action in RecoverySelector(session).select(now=NOW)
            if action.operation == "reconcile_account"
        }

    assert accounts["generation-1"] in account_actions
    assert accounts["generation-2"] not in account_actions


def test_recovery_selector_routes_policy_digest_drift(sessions):
    with sessions() as session:
        _claimed(session)
        reconciliation = session.scalar(select(AccountReconciliation))
        reconciliation.policy_sha256 = "f" * 64
        account_id = session.scalar(select(PaperBrokerAccount.id))
        session.commit()

    with sessions() as session:
        actions = RecoverySelector(session).select(now=NOW)

    assert ("reconcile_account", account_id) in {(action.operation, action.persisted_id) for action in actions}


def test_unsupported_adapter_fails_before_attempt_marker(sessions):
    _, order_id = _intent(sessions)

    class Unsupported(SnapshotAdapter):
        capabilities = BrokerCapabilities()

        def place_order(self, order: Order, *, client_order_ref=None):
            raise AssertionError("must not submit")

    with pytest.raises(ReconciliationConflictError, match="missing reconciliation contract"):
        submit_or_reconcile_order(sessions, order_id, Unsupported(sessions), now=NOW)
    with sessions() as session:
        assert session.get(OrderRecord, order_id).submit_attempted_at is None
