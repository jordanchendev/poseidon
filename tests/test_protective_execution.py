"""Durable protective materialization, reservation, routing, and trace proofs."""

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from threading import Barrier

import pandas as pd
import pytest
from sqlalchemy import delete, event, func, select, update

from poseidon.broker.base import BrokerAdapter, BrokerCapabilities, BrokerCapabilityError, BrokerOrderSnapshot
from poseidon.broker.paper_adapter import PaperBrokerAdapter
from poseidon.core.config import settings
from poseidon.decision_loop.decisions import DecisionService
from poseidon.decision_loop.execution import ExecutionConflictError, ProtectiveExecutionService
from poseidon.decision_loop.reconciliation import (
    ReconciliationConflictError,
    ReconciliationService,
    submit_or_reconcile_order,
)
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.paper_cash_movement import PaperCashMovement
from poseidon.models.portfolio_holding import PortfolioHoldingRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from poseidon.positions.lots import FillProjectionConflictError
from poseidon.strategies.portfolio.position_tracker import PositionTracker
from poseidon.strategies.portfolio.schemas import RebalanceOrder
from poseidon.workers import cpu_tasks
from poseidon.workers.cpu_tasks import _protective_execution_route
from tests.test_decision_service import worker
from tests.test_position_lot_allocation import project
from tests.test_position_lot_allocation import seed as _position_lot_seed


@pytest.fixture(name="seed")
def protective_seed():
    """Keep the shared lot fixture available regardless of collection order."""
    fixture = _position_lot_seed.__wrapped__()
    value = next(fixture)
    try:
        yield value
    finally:
        with value.sessions() as session, session.begin():
            account_id = session.scalar(
                select(PaperBrokerAccount.id).where(
                    PaperBrokerAccount.account_scope == value.account,
                    PaperBrokerAccount.account_generation == "generation-1",
                )
            )
            session.execute(
                delete(PortfolioHoldingRecord).where(
                    PortfolioHoldingRecord.strategy_name.startswith("phase98-protective:")
                )
            )
            legacy_ids = getattr(value, "phase98_legacy_holding_ids", ())
            if legacy_ids:
                session.execute(delete(PortfolioHoldingRecord).where(PortfolioHoldingRecord.id.in_(legacy_ids)))
            if account_id is not None:
                session.execute(
                    delete(PortfolioHoldingRecord).where(
                        PortfolioHoldingRecord.strategy_name == f"decision-lots:{account_id}"
                    )
                )
            broker_order_ids = select(PaperBrokerOrder.id).where(PaperBrokerOrder.account_scope == value.account)
            session.execute(delete(PaperBrokerFill).where(PaperBrokerFill.paper_broker_order_id.in_(broker_order_ids)))
            session.execute(delete(PaperCashMovement).where(PaperCashMovement.account_scope == value.account))
            session.execute(delete(PaperBrokerOrder).where(PaperBrokerOrder.account_scope == value.account))
        next(fixture, None)


NOW = datetime(2026, 9, 27, 9, 0, tzinfo=UTC)


class _MarkerAdapter(BrokerAdapter):
    capabilities = BrokerCapabilities(True, True, True, True, True)

    def __init__(self, sessions, order_id, *, lose_response=False):
        self.sessions = sessions
        self.order_id = order_id
        self.lose_response = lose_response
        self.place_calls = 0
        self.lookup_calls = 0
        self.accepted = None

    def login(self):
        return True

    def place_order(self, order, *, client_order_ref=None):
        self.place_calls += 1
        with self.sessions() as session:
            persisted = session.get(OrderRecord, self.order_id)
            assert persisted.submit_attempted_at is not None
            assert persisted.status == "reconciliation_required"
        self.accepted = BrokerOrderSnapshot(
            broker_order_id=f"PAPER-{uuid.uuid4().hex}",
            client_order_ref=client_order_ref,
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
            status="submitted",
            accepted_at=NOW,
            state_version=1,
        )
        if self.lose_response:
            self.lose_response = False
            raise RuntimeError("response lost after independent acceptance")
        return self.accepted

    def query_fills(self, *_args, **_kwargs):
        return []

    def query_positions(self):
        return []

    def logout(self):
        return None

    def find_order_by_client_ref(self, *_args, **_kwargs):
        self.lookup_calls += 1
        return self.accepted

    def query_order(self, *_args, **_kwargs):
        return self.accepted

    def query_account_snapshot(self, *_args, **_kwargs):
        raise AssertionError("not used")


def _opening(seed, *, quantity=5, price=100, side="long", time=NOW):
    fill_id = seed.fill(quantity=quantity, price=price, side=side, time=time)
    project(seed, fill_id)
    with seed.sessions() as session:
        fill = session.get(OrderFillRecord, fill_id)
        lot = session.scalar(select(PositionLot).where(PositionLot.opening_fill_id == fill.id))
        return lot.id, lot.opening_decision_id


def _legacy_holding(
    seed,
    *,
    market="tw_stock",
    symbol="2330",
    shares=5,
    entry_price=100,
    side="long",
    stop_loss_pct=0.10,
    projected=False,
    holding_id=None,
):
    with seed.sessions() as session, session.begin():
        account = session.scalar(
            select(PaperBrokerAccount).where(
                PaperBrokerAccount.account_scope == seed.account,
                PaperBrokerAccount.account_generation == "generation-1",
            )
        )
        if account is None:
            account = PaperBrokerAccount(
                account_scope=seed.account,
                account_generation="generation-1",
                opening_cash=100_000,
                currency="TWD" if market == "tw_stock" else "USDT",
            )
            session.add(account)
            session.flush()
        holding = PortfolioHoldingRecord(
            id=holding_id,
            strategy_name=(f"decision-lots:{account.id}" if projected else f"phase98-protective:{uuid.uuid4().hex}"),
            symbol=symbol,
            market=market,
            weight=0.1,
            shares=shares,
            entry_price=entry_price,
            side=side,
            entry_date=NOW,
            stop_loss_pct=stop_loss_pct,
            closed=False,
        )
        session.add(holding)
        session.flush()
        seed.phase98_legacy_holding_ids = [
            *getattr(seed, "phase98_legacy_holding_ids", ()),
            holding.id,
        ]
        return holding.id, account.id


def _materialize(seed, origin="stop_loss", trigger="price:80", **changes):
    values = {
        "account_scope": seed.account,
        "account_generation": "generation-1",
        "market": "tw_stock",
        "symbol": "2330",
        "instrument": "spot",
        "side": "long",
        "origin": origin,
        "trigger_generation": trigger,
        "price": 80.0,
        "principal": worker(seed.account),
        "now": NOW,
    }
    values.update(changes)
    with seed.sessions() as session, session.begin():
        return ProtectiveExecutionService(session).materialize(**values)


def _materialize_legacy(seed, holding_id, origin="stop_loss", trigger="legacy:price:80", **changes):
    return _materialize(
        seed,
        origin=origin,
        trigger=trigger,
        source_holding_ids=[holding_id],
        allow_legacy_holdings=True,
        **changes,
    )


def _paper_price(monkeypatch, price=79.0):
    repo = type(
        "Repo",
        (),
        {"read_ohlcv": lambda _self, *_args: pd.DataFrame({"close": [price]})},
    )()
    monkeypatch.setattr(
        "poseidon.data.remote_repository.RemoteDataRepository.from_settings",
        lambda: repo,
    )


def _partial_legacy_close(monkeypatch, seed, holding_id, *, side="long", entry_price=100, filled=2, price=150):
    result = _materialize_legacy(
        seed,
        holding_id,
        trigger=f"legacy:partial:{holding_id}",
        side=side,
        price=price,
    )
    order_id = uuid.UUID(result["order_ids"][0])
    adapter = PaperBrokerAdapter(seed.sessions)
    _paper_price(monkeypatch, price)
    with seed.sessions() as session, session.begin():
        prepared = ReconciliationService(session).prepare_attempt(order_id, adapter, now=NOW)
    accepted = adapter.place_order(prepared.order, client_order_ref=prepared.order.client_order_ref)
    with seed.sessions() as session, session.begin():
        broker_order = session.scalar(
            select(PaperBrokerOrder).where(PaperBrokerOrder.broker_order_id == accepted.broker_order_id)
        )
        broker_order.status = "cancelled"
        session.scalar(
            select(PaperBrokerFill).where(PaperBrokerFill.paper_broker_order_id == broker_order.id)
        ).fill_quantity = filled
        cash = price * filled if side == "long" else (2 * entry_price - price) * filled
        session.scalar(
            select(PaperCashMovement).where(
                PaperCashMovement.account_scope == seed.account,
                PaperCashMovement.state_version == broker_order.state_version,
            )
        ).amount = cash
    snapshot = adapter.find_order_by_client_ref(
        prepared.order.client_order_ref,
        account_scope=seed.account,
        account_generation="generation-1",
    )
    fills = adapter.query_fills(
        snapshot.broker_order_id,
        account_scope=seed.account,
        account_generation="generation-1",
    )
    with seed.sessions() as session, session.begin():
        ReconciliationService(session).import_broker_state(order_id, snapshot, fills, now=NOW)
    with seed.sessions() as session:
        fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == order_id))
    project(seed, fill_id)
    return adapter, order_id, fill_id


def _set_owner_protection(seed, decision_id, **values):
    with seed.sessions() as session, session.begin():
        decision = session.get(DecisionRecord, decision_id)
        version = session.get(StrategyVersion, decision.strategy_version_id)
        config = {**version.config_json, "protective_exit": values}
        digest = strategy_version_digest(
            config,
            version.policy_json,
            version.artifact_json,
        )
        session.execute(
            update(StrategyVersion)
            .where(StrategyVersion.id == version.id)
            .values(config_json=config, content_sha256=digest)
        )


@pytest.mark.parametrize("origin", ["stop_loss", "liquidation", "manual_emergency"])
def test_exact_protective_origins_create_reduction_only_durable_intents(seed, origin):
    lot_id, decision_id = _opening(seed)

    result = _materialize(seed, origin=origin)

    with seed.sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(result["order_ids"][0]))
        event = session.scalar(
            select(DecisionEvent).where(
                DecisionEvent.decision_id == decision_id,
                DecisionEvent.event_type.like("protective_%"),
            )
        )
        assert order.order_origin == origin
        assert order.intent_json["frozen_intent"]["action"] == "exit"
        assert order.action == "sell"
        assert order.signal_id is None
        assert order.protective_context_json["source_lot_ids"] == [str(lot_id)]
        payload = event.payload_json
        assert payload["protective_context_sha256"] == order.protective_context_json["dedupe_sha256"]
        assert payload["origin"] == origin
        assert payload["trigger_generation"] == "price:80"
        assert payload["current_price"] == 80.0
        assert payload["identity"] == {
            "market": "tw_stock",
            "symbol": "2330",
            "instrument": "spot",
            "side": "long",
        }
        assert payload["action"] == "exit"
        assert payload["quantity"] == 5.0


@pytest.mark.parametrize("origin", ["manual", "decision", "signal", "stop_loss_extra", ""])
def test_non_exact_protective_origins_fail_before_mutation(seed, origin):
    _opening(seed)

    with pytest.raises(ExecutionConflictError, match="protective origin"):
        _materialize(seed, origin=origin)

    with seed.sessions() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(OrderRecord)
                .where(
                    OrderRecord.account_scope == seed.account,
                    OrderRecord.order_origin != "decision",
                )
            )
            == 0
        )


@pytest.mark.parametrize("action", ["enter", "add"])
def test_protective_intent_cannot_increase_exposure(seed, action):
    _opening(seed)

    with pytest.raises(ExecutionConflictError, match="reduction-only"):
        _materialize(seed, action=action)


def test_dedupe_maps_all_source_lots_and_replays_one_order(seed):
    first_lot, first_decision = _opening(seed, quantity=3)
    second_lot, second_decision = _opening(seed, quantity=2, time=NOW.replace(minute=1))

    first = _materialize(seed)
    replay = _materialize(seed)

    assert replay == first
    with seed.sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(first["order_ids"][0]))
        context = order.protective_context_json
        assert context["source_lot_ids"] == [str(first_lot), str(second_lot)]
        assert context["source_decision_ids"] == sorted([str(first_decision), str(second_decision)])
        assert order.quantity == order.reserved_quantity == 5
        assert (
            session.scalar(
                select(func.count())
                .select_from(OrderRecord)
                .where(
                    OrderRecord.account_scope == seed.account,
                    OrderRecord.order_origin == "stop_loss",
                )
            )
            == 1
        )


def test_concurrent_stop_loss_and_liquidation_reserve_one_net_close(seed):
    _opening(seed, quantity=5)
    barrier = Barrier(2)

    def materialize(origin):
        barrier.wait(timeout=10)
        return _materialize(seed, origin=origin, trigger=f"same-risk:{origin}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(timeout=20)
            for future in (pool.submit(materialize, "stop_loss"), pool.submit(materialize, "liquidation"))
        ]

    assert sorted(len(result["order_ids"]) for result in results) == [0, 1]
    with seed.sessions() as session:
        lots = session.scalars(select(PositionLot).where(PositionLot.account_scope == seed.account)).all()
        protective_orders = session.scalars(
            select(OrderRecord).where(OrderRecord.account_scope == seed.account, OrderRecord.order_origin != "decision")
        ).all()
        assert len(protective_orders) == 1
        assert sum(lot.reserved_close_quantity for lot in lots) == 5


@pytest.mark.parametrize("mode", ["decision", "halted"])
def test_tw_approved_legacy_holding_creates_durable_null_decision_order(monkeypatch, seed, mode):
    holding_id, _account_id = _legacy_holding(seed)
    _exact_settings(monkeypatch, mode=mode, market="tw_stock", scope=seed.account)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", (seed.account,))
    monkeypatch.setattr(cpu_tasks, "datetime", _TradingTime)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 80.0})
    queued = []
    monkeypatch.setattr(cpu_tasks.submit_decision_order, "delay", lambda order_id: queued.append(order_id))

    result = cpu_tasks.portfolio_stop_loss_monitor.run()

    assert len(result["durable_order_ids"]) == 1
    assert queued == result["durable_order_ids"]
    with seed.sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(result["durable_order_ids"][0]))
        assert order.decision_id is None
        assert order.action == "sell"
        assert order.protective_context_json["legacy_exception"] is True
        assert order.protective_context_json["source_holding_ids"] == [str(holding_id)]
        assert order.protective_context_json["source_lot_ids"] == []
        assert order.protective_context_json["source_decision_ids"] == []
        assert order.protective_context_json["source_holding_risk"] == {
            str(holding_id): {"entry_price": 100.0, "stop_loss_pct": 0.1},
        }


def test_legacy_holding_materialization_requires_full_available_close(seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)

    with pytest.raises(ExecutionConflictError, match="full available close"):
        _materialize(
            seed,
            source_holding_ids=[holding_id],
            allow_legacy_holdings=True,
            quantity=2,
        )


def test_concurrent_legacy_holding_origins_reserve_one_net_close(seed):
    holding_id, _account_id = _legacy_holding(seed)
    barrier = Barrier(2)

    def materialize(origin):
        barrier.wait(timeout=10)
        return _materialize_legacy(seed, holding_id, origin=origin, trigger=f"legacy-risk:{origin}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(timeout=20)
            for future in (pool.submit(materialize, "stop_loss"), pool.submit(materialize, "liquidation"))
        ]

    assert sorted(len(result["order_ids"]) for result in results) == [0, 1]
    with seed.sessions() as session:
        orders = session.scalars(
            select(OrderRecord).where(
                OrderRecord.account_scope == seed.account,
                OrderRecord.order_origin != "decision",
            )
        ).all()
        assert len(orders) == 1
        assert orders[0].decision_id is None
        assert orders[0].reserved_quantity == 5


def test_overlapping_legacy_reservation_omits_fully_reserved_source(seed):
    first_id, _account_id = _legacy_holding(seed, shares=5, holding_id=uuid.UUID(int=1))
    second_id, _account_id = _legacy_holding(seed, shares=5, holding_id=uuid.UUID(int=2))
    _materialize_legacy(seed, first_id, trigger="legacy:first-reservation")

    result = _materialize(
        seed,
        trigger="legacy:overlap",
        source_holding_ids=[first_id, second_id],
        allow_legacy_holdings=True,
    )

    with seed.sessions() as session:
        order = session.get(OrderRecord, uuid.UUID(result["order_ids"][0]))
        assert order.quantity == 5
        assert order.protective_context_json["source_holding_ids"] == [str(second_id)]
        assert order.protective_context_json["source_holding_quantities"] == {str(second_id): 5.0}
        ProtectiveExecutionService(session).validate_order(order)


def test_legacy_holding_partial_projection_and_replay_are_deterministic(seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "partially_filled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        order.reconciliation_status = "resolved"
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=2,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    first_projection = project(seed, fill_id)
    replay_projection = project(seed, fill_id)
    replay_order = _materialize_legacy(seed, holding_id)

    assert replay_projection == first_projection
    assert replay_order["order_ids"] == result["order_ids"]
    with seed.sessions() as session:
        holding = session.get(PortfolioHoldingRecord, holding_id)
        assert (holding.shares, holding.closed) == (3, False)
        assert session.get(OrderFillRecord, fill_id).projection_status == "applied"


def test_legacy_holding_partial_cancel_releases_unfilled_remainder(seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "cancelled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        order.reconciliation_status = "resolved"
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=2,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    project(seed, fill_id)

    with seed.sessions() as session:
        order = session.get(OrderRecord, order_id)
        ReconciliationService(session)._validate_durable_intent(order)
        holding = session.get(PortfolioHoldingRecord, holding_id)
        assert (order.reservation_status, holding.shares, holding.closed) == ("released", 3, False)


def test_durable_lot_and_legacy_holding_fills_project_without_cross_reservation(seed):
    lot_id, _decision_id = _opening(seed, quantity=5)
    holding_id, _account_id = _legacy_holding(seed, shares=4)
    durable = _materialize(seed, trigger="coexist:durable")
    legacy = _materialize_legacy(seed, holding_id, trigger="coexist:legacy")
    durable_order_id = uuid.UUID(durable["order_ids"][0])
    legacy_order_id = uuid.UUID(legacy["order_ids"][0])
    durable_fill_id = uuid.uuid4()
    legacy_fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        for order_id, fill_id, quantity in (
            (durable_order_id, durable_fill_id, 5),
            (legacy_order_id, legacy_fill_id, 4),
        ):
            order = session.get(OrderRecord, order_id)
            order.status = "filled"
            order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
            order.submit_attempted_at = NOW
            order.reconciliation_status = "resolved"
            session.add(
                OrderFillRecord(
                    id=fill_id,
                    order_id=order.id,
                    broker_fill_id=f"fill-{uuid.uuid4().hex}",
                    fill_price=79,
                    fill_quantity=quantity,
                    fill_time=NOW,
                    projection_status="projection_pending",
                    created_at=NOW,
                )
            )

    project(seed, durable_fill_id)
    project(seed, legacy_fill_id)

    with seed.sessions() as session:
        lot = session.get(PositionLot, lot_id)
        holding = session.get(PortfolioHoldingRecord, holding_id)
        assert (lot.open_quantity, lot.reserved_close_quantity) == (0, 0)
        assert (holding.shares, holding.closed) == (0, True)


def test_legacy_holding_projection_rejects_overfill(seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "filled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=6,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    with pytest.raises(FillProjectionConflictError):
        project(seed, fill_id)

    with seed.sessions() as session:
        assert session.get(PortfolioHoldingRecord, holding_id).shares == 5


def test_legacy_holding_true_paper_adapter_full_close_uses_one_durable_baseline(monkeypatch, seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    _paper_price(monkeypatch)

    submitted = submit_or_reconcile_order(seed.sessions, order_id, PaperBrokerAdapter(seed.sessions), now=NOW)

    assert submitted["status"] == "filled"
    snapshot = PaperBrokerAdapter(seed.sessions).query_account_snapshot(seed.account, "generation-1")
    assert snapshot.positions == ()
    with seed.sessions() as session:
        rows = session.scalars(
            select(PaperBrokerOrder)
            .where(PaperBrokerOrder.account_scope == seed.account)
            .order_by(PaperBrokerOrder.client_order_ref)
        ).all()
        assert [(row.action, row.quantity) for row in rows] == [
            ("position_alloc", 5),
            ("position_base", 5),
            ("sell", 5),
        ]
        attribution = rows[0]
        assert attribution.status == "position_attribution"
        assert attribution.state_version == rows[2].state_version
        assert attribution.accepted_at == rows[2].accepted_at
        assert (
            session.scalar(select(PaperBrokerFill.id).where(PaperBrokerFill.paper_broker_order_id == attribution.id))
            is None
        )


@pytest.mark.parametrize(("side", "expected_cash"), [("long", 750.0), ("short", 1250.0)])
def test_later_legacy_holding_uses_its_own_chronological_cost_basis(monkeypatch, seed, side, expected_cash):
    first_id, _account_id = _legacy_holding(
        seed,
        shares=5,
        entry_price=100,
        side=side,
        holding_id=uuid.UUID(int=1),
    )
    adapter, _first_order_id, _first_fill_id = _partial_legacy_close(
        monkeypatch,
        seed,
        first_id,
        side=side,
        entry_price=100,
        filled=2,
        price=150,
    )
    second_id, _account_id = _legacy_holding(
        seed,
        shares=5,
        entry_price=200,
        side=side,
        holding_id=uuid.UUID(int=2),
    )
    second = _materialize_legacy(
        seed,
        second_id,
        trigger="legacy:later-basis",
        side=side,
        price=150,
    )
    _paper_price(monkeypatch, 150)

    submit_or_reconcile_order(
        seed.sessions,
        uuid.UUID(second["order_ids"][0]),
        adapter,
        now=NOW.replace(minute=1),
    )

    with seed.sessions() as session:
        latest = session.scalar(
            select(PaperCashMovement)
            .where(PaperCashMovement.account_scope == seed.account)
            .order_by(PaperCashMovement.state_version.desc())
        )
        assert latest.amount == expected_cash


def test_legacy_broker_attribution_survives_a_partial_b_full_then_a_remainder(monkeypatch, seed):
    first_id, _account_id = _legacy_holding(seed, shares=5, entry_price=100, holding_id=uuid.UUID(int=1))
    adapter, _first_order_id, _first_fill_id = _partial_legacy_close(
        monkeypatch,
        seed,
        first_id,
        filled=2,
        price=150,
    )
    second_id, _account_id = _legacy_holding(seed, shares=5, entry_price=200, holding_id=uuid.UUID(int=2))
    second = _materialize_legacy(seed, second_id, trigger="legacy:b-full", price=150)
    _paper_price(monkeypatch, 150)
    submit_or_reconcile_order(
        seed.sessions,
        uuid.UUID(second["order_ids"][0]),
        adapter,
        now=NOW.replace(minute=1),
    )
    with seed.sessions() as session:
        second_fill_id = session.scalar(
            select(OrderFillRecord.id).where(OrderFillRecord.order_id == uuid.UUID(second["order_ids"][0]))
        )
    project(seed, second_fill_id)
    remainder = _materialize_legacy(seed, first_id, trigger="legacy:a-remainder", price=150)

    submit_or_reconcile_order(
        seed.sessions,
        uuid.UUID(remainder["order_ids"][0]),
        adapter,
        now=NOW.replace(minute=2),
    )

    assert adapter.query_account_snapshot(seed.account, "generation-1").positions == ()


def test_legacy_broker_attribution_validation_is_scoped_to_position_identity(monkeypatch, seed):
    first_id, _account_id = _legacy_holding(seed, symbol="2330", shares=2)
    second_id, _account_id = _legacy_holding(seed, symbol="0050", shares=3)
    adapter = PaperBrokerAdapter(seed.sessions)
    _paper_price(monkeypatch, 80)

    for minute, (holding_id, symbol) in enumerate(((first_id, "2330"), (second_id, "0050"))):
        result = _materialize_legacy(
            seed,
            holding_id,
            trigger=f"legacy:multi-identity:{symbol}",
            symbol=symbol,
            price=80,
        )
        submit_or_reconcile_order(
            seed.sessions,
            uuid.UUID(result["order_ids"][0]),
            adapter,
            now=NOW.replace(minute=minute),
        )

    assert adapter.query_account_snapshot(seed.account, "generation-1").positions == ()
    with seed.sessions() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(PaperBrokerOrder)
                .where(
                    PaperBrokerOrder.account_scope == seed.account,
                    PaperBrokerOrder.action == "position_alloc",
                )
            )
            == 2
        )


def test_stored_legacy_risk_mutation_is_rejected_before_adapter_io(seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        context = copy.deepcopy(order.protective_context_json)
        context["source_holding_risk"][str(holding_id)]["entry_price"] = 101.0
        order.protective_context_json = context
    adapter = _MarkerAdapter(seed.sessions, order_id)

    with pytest.raises(ReconciliationConflictError, match="durable order intent"):
        submit_or_reconcile_order(seed.sessions, order_id, adapter, now=NOW)

    assert adapter.place_calls == adapter.lookup_calls == 0


@pytest.mark.parametrize(
    "corruption",
    ["partial_quantity", "zero_source", "extra_quantity", "nonfinite_risk", "extra_risk_field"],
)
def test_legacy_holding_adapter_revalidates_full_close_context(monkeypatch, seed, corruption):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    adapter = PaperBrokerAdapter(seed.sessions)
    _paper_price(monkeypatch)
    with seed.sessions() as session, session.begin():
        prepared = ReconciliationService(session).prepare_attempt(order_id, adapter, now=NOW)
    order = replace(prepared.order, protective_context_json=copy.deepcopy(prepared.order.protective_context_json))
    holding_key = str(holding_id)
    if corruption == "partial_quantity":
        order.quantity = 2
    elif corruption == "zero_source":
        order.protective_context_json["source_holding_quantities"][holding_key] = 0
    elif corruption == "extra_quantity":
        order.protective_context_json["source_holding_quantities"][str(uuid.uuid4())] = 1
    elif corruption == "nonfinite_risk":
        order.protective_context_json["source_holding_risk"][holding_key]["entry_price"] = float("nan")
    else:
        order.protective_context_json["source_holding_risk"][holding_key]["mutable"] = True

    with pytest.raises(BrokerCapabilityError, match="legacy protective context"):
        adapter.place_order(order, client_order_ref=order.client_order_ref)


def test_legacy_holding_partial_cancel_then_new_trigger_reuses_baseline(monkeypatch, seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    first = _materialize_legacy(seed, holding_id, trigger="legacy:first")
    first_order_id = uuid.UUID(first["order_ids"][0])
    adapter = PaperBrokerAdapter(seed.sessions)
    _paper_price(monkeypatch)
    with seed.sessions() as session, session.begin():
        prepared = ReconciliationService(session).prepare_attempt(first_order_id, adapter, now=NOW)
    accepted = adapter.place_order(prepared.order, client_order_ref=prepared.order.client_order_ref)
    with seed.sessions() as session, session.begin():
        broker_order = session.scalar(
            select(PaperBrokerOrder).where(PaperBrokerOrder.broker_order_id == accepted.broker_order_id)
        )
        broker_order.status = "cancelled"
        session.scalar(
            select(PaperBrokerFill).where(PaperBrokerFill.paper_broker_order_id == broker_order.id)
        ).fill_quantity = 2
        session.scalar(
            select(PaperCashMovement).where(
                PaperCashMovement.account_scope == seed.account,
                PaperCashMovement.state_version == broker_order.state_version,
            )
        ).amount = 158
    partial_positions = adapter.query_account_snapshot(seed.account, "generation-1").positions
    assert [(position.symbol, position.quantity) for position in partial_positions] == [("2330", 3.0)]
    snapshot = adapter.find_order_by_client_ref(
        prepared.order.client_order_ref,
        account_scope=seed.account,
        account_generation="generation-1",
    )
    fills = adapter.query_fills(
        snapshot.broker_order_id,
        account_scope=seed.account,
        account_generation="generation-1",
    )
    with seed.sessions() as session, session.begin():
        ReconciliationService(session).import_broker_state(first_order_id, snapshot, fills, now=NOW)
    with seed.sessions() as session:
        fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == first_order_id))
    project(seed, fill_id)

    second = _materialize_legacy(seed, holding_id, trigger="legacy:second")
    second_order_id = uuid.UUID(second["order_ids"][0])
    submit_or_reconcile_order(
        seed.sessions,
        second_order_id,
        adapter,
        now=NOW.replace(minute=1),
    )
    with seed.sessions() as session:
        second_fill_id = session.scalar(select(OrderFillRecord.id).where(OrderFillRecord.order_id == second_order_id))
    project(seed, second_fill_id)
    project(seed, fill_id)

    assert adapter.query_account_snapshot(seed.account, "generation-1").positions == ()
    with seed.sessions() as session:
        assert (
            session.get(PortfolioHoldingRecord, holding_id).shares,
            session.get(PortfolioHoldingRecord, holding_id).closed,
        ) == (0, True)
        assert (
            session.scalar(
                select(func.count())
                .select_from(PaperBrokerOrder)
                .where(
                    PaperBrokerOrder.account_scope == seed.account,
                    PaperBrokerOrder.action == "position_base",
                )
            )
            == 1
        )


def test_legacy_holding_baseline_replay_detects_corruption(monkeypatch, seed):
    holding_id, _account_id = _legacy_holding(seed, shares=5)
    result = _materialize_legacy(seed, holding_id)
    order_id = uuid.UUID(result["order_ids"][0])
    adapter = PaperBrokerAdapter(seed.sessions)
    _paper_price(monkeypatch)
    with seed.sessions() as session, session.begin():
        prepared = ReconciliationService(session).prepare_attempt(order_id, adapter, now=NOW)
    adapter.place_order(prepared.order, client_order_ref=prepared.order.client_order_ref)
    with seed.sessions() as session, session.begin():
        baseline = session.scalar(
            select(PaperBrokerOrder).where(
                PaperBrokerOrder.account_scope == seed.account,
                PaperBrokerOrder.action == "position_base",
            )
        )
        baseline.price = 999

    with pytest.raises(BrokerCapabilityError, match="legacy position baseline"):
        adapter.place_order(prepared.order, client_order_ref=prepared.order.client_order_ref)


def test_projected_holding_is_not_a_legacy_protective_source(monkeypatch, seed):
    _holding_id, _account_id = _legacy_holding(seed, projected=True)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", (seed.account,))
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 1.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {}
    assert result.get("legacy_sources", {}) == {}


def test_other_account_projected_marker_is_not_a_legacy_source(monkeypatch, seed):
    holding_id, _account_id = _legacy_holding(seed)
    with seed.sessions() as session, session.begin():
        session.get(PortfolioHoldingRecord, holding_id).strategy_name = f"decision-lots:{uuid.uuid4()}"
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", (seed.account,))
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 1.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {}
    assert result.get("legacy_sources", {}) == {}


def test_invalid_legacy_risk_does_not_suppress_valid_durable_lot_trigger(monkeypatch, seed):
    _lot_id, decision_id = _opening(seed)
    _legacy_holding(seed, entry_price=None)
    _set_owner_protection(seed, decision_id, stop_loss_pct=0.10)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", (seed.account,))
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 80.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {("tw_stock", "2330", "spot", "long"): 80.0}
    assert result["legacy_sources"] == {}
    assert result["skipped"] == "legacy_protective_policy_unresolved"


def test_tw_monitor_splits_durable_lot_and_allowlisted_legacy_holding_sources(monkeypatch, seed):
    _lot_id, decision_id = _opening(seed, quantity=5)
    holding_id, _account_id = _legacy_holding(seed, shares=4)
    _set_owner_protection(seed, decision_id, stop_loss_pct=0.10)
    _exact_settings(monkeypatch, mode="halted", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", (seed.account,))
    monkeypatch.setattr(cpu_tasks, "datetime", _TradingTime)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 80.0})
    queued = []
    monkeypatch.setattr(cpu_tasks.submit_decision_order, "delay", lambda order_id: queued.append(order_id))

    result = cpu_tasks.portfolio_stop_loss_monitor.run()

    assert len(result["durable_order_ids"]) == 2
    assert sorted(queued) == sorted(result["durable_order_ids"])
    with seed.sessions() as session:
        orders = session.scalars(
            select(OrderRecord)
            .where(OrderRecord.id.in_([uuid.UUID(value) for value in result["durable_order_ids"]]))
            .order_by(OrderRecord.decision_id.nulls_first())
        ).all()
        legacy, durable = orders
        assert legacy.decision_id is None
        assert legacy.quantity == 4
        assert legacy.protective_context_json["source_holding_ids"] == [str(holding_id)]
        assert durable.decision_id == decision_id
        assert durable.quantity == 5
        assert "source_holding_ids" not in durable.protective_context_json


def test_partial_fill_retains_remainder_and_replay_does_not_duplicate(seed):
    _opening(seed, quantity=5)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "partially_filled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        order.reconciliation_status = "resolved"
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=2,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    project(seed, fill_id)
    replay = _materialize(seed)

    assert replay["order_ids"] == result["order_ids"]
    assert replay["execution_key"] == result["execution_key"]
    assert replay["status"] == "partially_filled"
    with seed.sessions() as session:
        lot = session.scalar(select(PositionLot).where(PositionLot.account_scope == seed.account))
        assert (lot.open_quantity, lot.reserved_close_quantity) == (3, 3)
        assert (
            session.scalar(
                select(func.count())
                .select_from(OrderRecord)
                .where(
                    OrderRecord.account_scope == seed.account,
                    OrderRecord.order_origin == "stop_loss",
                )
            )
            == 1
        )


def test_partial_cancel_releases_only_the_unfilled_remainder_after_projection(seed):
    _opening(seed, quantity=5)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "cancelled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        order.reconciliation_status = "resolved"
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=2,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    project(seed, fill_id)

    with seed.sessions() as session:
        order = session.get(OrderRecord, order_id)
        ReconciliationService(session)._validate_durable_intent(order)
        lot = session.scalar(select(PositionLot).where(PositionLot.account_scope == seed.account))
        assert (order.reservation_status, lot.open_quantity, lot.reserved_close_quantity) == ("released", 3, 0)


def test_rejected_zero_fill_releases_full_protective_reservation(seed):
    _opening(seed, quantity=5)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "rejected"
        order.reservation_status = "released"
        order.reconciliation_status = "resolved"
        lot = session.scalar(select(PositionLot).where(PositionLot.account_scope == seed.account))
        lot.reserved_close_quantity = 0

    with seed.sessions() as session:
        ReconciliationService(session)._validate_durable_intent(session.get(OrderRecord, order_id))


def test_terminal_release_with_pending_projection_still_requires_reservation(seed):
    _opening(seed, quantity=5)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.status = "cancelled"
        order.reservation_status = "released"
        lot = session.scalar(select(PositionLot).where(PositionLot.account_scope == seed.account))
        lot.reserved_close_quantity = 0
        session.add(
            OrderFillRecord(
                id=uuid.uuid4(),
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=2,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    with seed.sessions() as session, pytest.raises(ReconciliationConflictError, match="durable order intent"):
        ReconciliationService(session)._validate_durable_intent(session.get(OrderRecord, order_id))


def test_same_trigger_with_a_new_lot_does_not_replay_a_closed_order(seed):
    _opening(seed, quantity=5)
    first = _materialize(seed)
    first_order_id = uuid.UUID(first["order_ids"][0])
    closing_fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, first_order_id)
        order.status = "filled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        order.reconciliation_status = "resolved"
        session.add(
            OrderFillRecord(
                id=closing_fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=5,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )
    project(seed, closing_fill_id)
    _opening(seed, quantity=4, time=NOW.replace(minute=2))

    second = _materialize(seed)

    assert second["order_ids"] != first["order_ids"]
    with seed.sessions() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(OrderRecord)
                .where(
                    OrderRecord.account_scope == seed.account,
                    OrderRecord.order_origin == "stop_loss",
                )
            )
            == 2
        )


def test_attempt_marker_commits_before_io_and_response_loss_replays_by_lookup(seed):
    _opening(seed)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    adapter = _MarkerAdapter(seed.sessions, order_id, lose_response=True)

    with pytest.raises(RuntimeError, match="response lost"):
        submit_or_reconcile_order(seed.sessions, order_id, adapter, now=NOW)
    replay = submit_or_reconcile_order(seed.sessions, order_id, adapter, now=NOW)

    assert adapter.place_calls == 1
    assert adapter.lookup_calls == 1
    assert replay["status"] == "submitted"


def test_prepare_attempt_takes_account_lock_before_order_row_lock(monkeypatch, seed):
    _opening(seed)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    events = []
    with seed.sessions() as session, session.begin():
        service = ReconciliationService(session)
        original_order = service._locked_order
        original_account = service._locked_order_and_account

        def locked_order(value):
            events.append("order")
            return original_order(value)

        def locked_account(value):
            events.append("account")
            return original_account(value)

        monkeypatch.setattr(service, "_locked_order", locked_order)
        monkeypatch.setattr(service, "_locked_order_and_account", locked_account)
        service.prepare_attempt(order_id, _MarkerAdapter(seed.sessions, order_id), now=NOW)

    assert events[0] == "account"


def test_prepare_attempt_does_not_lock_fills_before_account_advisory(seed):
    _opening(seed)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    lock_events = []

    def observe_lock_order(_connection, _cursor, statement, _parameters, _context, _many):
        normalized = " ".join(statement.lower().split())
        if "pg_advisory_xact_lock" in normalized:
            lock_events.append("account")
        elif normalized.startswith("select") and "from order_fills" in normalized and "for update" in normalized:
            lock_events.append("fills")

    with seed.sessions() as session, session.begin():
        engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", observe_lock_order)
        try:
            ReconciliationService(session).prepare_attempt(
                order_id,
                _MarkerAdapter(seed.sessions, order_id),
                now=NOW,
            )
        finally:
            event.remove(engine, "before_cursor_execute", observe_lock_order)

    assert lock_events[0] == "account"
    assert lock_events.count("fills") == 1


@pytest.mark.parametrize("origin", ["manual", "unknown"])
def test_submission_rejects_non_exact_durable_origin_before_io(seed, origin):
    _opening(seed)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    with seed.sessions() as session, session.begin():
        session.get(OrderRecord, order_id).order_origin = origin
    adapter = _MarkerAdapter(seed.sessions, order_id)

    with pytest.raises(ReconciliationConflictError, match="durable order intent"):
        submit_or_reconcile_order(seed.sessions, order_id, adapter, now=NOW)

    assert adapter.place_calls == adapter.lookup_calls == 0


@pytest.mark.parametrize("origin", ["manual", "unknown"])
def test_projection_rejects_non_exact_durable_origin(seed, origin):
    _opening(seed)
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        order.order_origin = origin
        order.status = "partially_filled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=1,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )

    with pytest.raises(FillProjectionConflictError, match="canonical decision"):
        project(seed, fill_id)


@pytest.mark.parametrize(
    ("mode", "enabled", "exact", "legacy_allowed", "expected"),
    [
        ("legacy", False, False, False, "legacy"),
        ("shadow", False, True, False, "legacy"),
        ("decision", True, True, False, "durable"),
        ("decision", False, True, False, "skip"),
        ("halted", False, True, False, "durable"),
        ("decision", True, False, True, "legacy"),
        ("halted", False, False, False, "skip"),
    ],
)
def test_protective_mode_scope_matrix(mode, enabled, exact, legacy_allowed, expected):
    assert (
        _protective_execution_route(
            mode,
            enabled=enabled,
            approved_identity=exact,
            legacy_allowed=legacy_allowed,
        )
        == expected
    )


def test_trace_adds_fill_allocation_lots_and_reconciliation_without_snapshots(seed):
    first_lot, first_decision = _opening(seed, quantity=3)
    _second_lot, second_decision = _opening(seed, quantity=2, time=NOW.replace(minute=1))
    result = _materialize(seed)
    order_id = uuid.UUID(result["order_ids"][0])
    fill_id = uuid.uuid4()
    with seed.sessions() as session, session.begin():
        order = session.get(OrderRecord, order_id)
        decision = session.get(DecisionRecord, first_decision)
        order.status = "partially_filled"
        order.broker_order_id = f"PAPER-{uuid.uuid4().hex}"
        order.submit_attempted_at = NOW
        order.reconciliation_status = "resolved"
        session.add(
            OrderFillRecord(
                id=fill_id,
                order_id=order.id,
                broker_fill_id=f"fill-{uuid.uuid4().hex}",
                fill_price=79,
                fill_quantity=2,
                fill_time=NOW,
                projection_status="projection_pending",
                created_at=NOW,
            )
        )
        session.add(
            AccountReconciliation(
                account_scope=seed.account,
                account_generation="generation-1",
                as_of=NOW,
                broker_state_watermark="broker:7",
                internal_state_watermark="internal:abc",
                broker_snapshot_sha256="a" * 64,
                broker_snapshot_json={"secret": "must-not-leak"},
                internal_snapshot_json={"cash": 1},
                difference_json={"cash": {"difference": 1}},
                policy_sha256=decision.policy_sha256,
                status="mismatch",
            )
        )
    project(seed, fill_id)

    with seed.sessions() as session:
        anchor_decision = session.get(OrderRecord, order_id).decision_id
        non_anchor_decision = second_decision if anchor_decision == first_decision else first_decision
        trace = DecisionService(session).trace(non_anchor_decision, principal=worker(seed.account))

    assert trace["execution_key"] is not None
    assert set(trace["execution_keys"]) == {trace["execution_key"], result["execution_key"]}
    protective = next(row for row in trace["orders"] if row["id"] == result["order_ids"][0])
    assert protective["execution_key"] == result["execution_key"]
    assert {
        "market",
        "symbol",
        "instrument",
        "side",
        "action",
        "quantity",
    }.issubset(protective)
    assert len(trace["lots"]) == 2
    assert any(row["id"] == str(fill_id) for row in trace["fills"])
    assert any(row["position_lot_id"] == str(first_lot) for row in trace["allocations"])
    assert trace["reconciliations"] == [
        {
            "id": trace["reconciliations"][0]["id"],
            "status": "mismatch",
            "policy_sha256": trace["decision"]["policy_sha256"],
            "broker_state_watermark": "broker:7",
            "internal_state_watermark": "internal:abc",
            "broker_snapshot_sha256": "a" * 64,
            "as_of": "2026-09-27T09:00:00Z",
        }
    ]
    assert any(event["event_type"].startswith("protective_") for event in trace["events"])
    assert "broker_snapshot_json" not in repr(trace)
    assert "internal_snapshot_json" not in repr(trace)
    assert "must-not-leak" not in repr(trace)


def test_durable_trigger_missing_owner_risk_values_fails_closed(monkeypatch, seed):
    _opening(seed)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 1.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result == {"checked": 1, "triggered": {}, "skipped": "protective_policy_unresolved"}


def test_tw_mixed_lots_use_earliest_owner_frozen_stop(monkeypatch, seed):
    _first_lot, first_decision = _opening(seed, quantity=3)
    _second_lot, second_decision = _opening(seed, quantity=2, time=NOW.replace(minute=1))
    _set_owner_protection(seed, first_decision, stop_loss_pct=0.20)
    _set_owner_protection(seed, second_decision, stop_loss_pct=0.05)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 94.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {("tw_stock", "2330", "spot", "long"): 94.0}


@pytest.mark.parametrize(
    ("side", "first_price", "second_price", "mark"),
    [
        ("long", 100.0, 200.0, 150.0),
        ("short", 200.0, 100.0, 110.0),
    ],
)
def test_mixed_cost_lots_evaluate_each_owner_stop_before_identity_merge(
    monkeypatch,
    seed,
    side,
    first_price,
    second_price,
    mark,
):
    _first_lot, first_decision = _opening(seed, quantity=1, price=first_price, side=side)
    _second_lot, second_decision = _opening(
        seed,
        quantity=1,
        price=second_price,
        side=side,
        time=NOW.replace(minute=1),
    )
    _set_owner_protection(seed, first_decision, stop_loss_pct=0.20)
    _set_owner_protection(seed, second_decision, stop_loss_pct=0.05)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": mark})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {("tw_stock", "2330", "spot", side): mark}


def test_trigger_identity_does_not_collapse_same_symbol_opposite_side(monkeypatch, seed):
    _long_lot, long_decision = _opening(seed)
    short_fill = seed.fill(quantity=2, price=100, side="short", time=NOW.replace(minute=1))
    project(seed, short_fill)
    with seed.sessions() as session:
        short_lot = session.scalar(select(PositionLot).where(PositionLot.opening_fill_id == short_fill))
        short_decision = short_lot.opening_decision_id
    _set_owner_protection(seed, long_decision, stop_loss_pct=0.10)
    _set_owner_protection(seed, short_decision, stop_loss_pct=0.10)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 80.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {("tw_stock", "2330", "spot", "long"): 80.0}


class _Holding:
    market = "tw_stock"
    entry_price = 100.0
    shares = 5.0
    weight = 0.1
    stop_loss_pct = 0.1
    entry_date = NOW
    side = "long"


class _TradingTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 28, 2, 0, tzinfo=UTC)


def _exact_settings(monkeypatch, *, mode, market, scope="paper:pilot"):
    monkeypatch.setattr(settings, "decision_loop_execution_mode", mode)
    monkeypatch.setattr(settings, "decision_loop_execution_enabled", mode == "decision")
    monkeypatch.setattr(settings, "decision_loop_approved_account_scope", scope)
    monkeypatch.setattr(settings, "decision_loop_approved_account_generation", "generation-1")
    monkeypatch.setattr(settings, "decision_loop_approved_market", market)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", ())


def test_tw_decision_monitor_uses_durable_lot_without_legacy_holding(monkeypatch, seed):
    _lot, decision_id = _opening(seed)
    _set_owner_protection(seed, decision_id, stop_loss_pct=0.10)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "datetime", _TradingTime)
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: pytest.fail("legacy tracker called"))
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 80.0})
    monkeypatch.setattr(cpu_tasks, "_build_tw_stock_broker", lambda *_args: pytest.fail("legacy broker called"))
    calls = []
    monkeypatch.setattr(cpu_tasks, "_run_durable_protective", lambda **kwargs: calls.append(kwargs) or ["order-1"])

    result = cpu_tasks.portfolio_stop_loss_monitor.run()

    assert result["durable_order_ids"] == ["order-1"]
    assert calls[0]["origin"] == "stop_loss"
    assert calls[0]["triggered"] == {("tw_stock", "2330", "spot", "long"): 80.0}


def test_perp_halted_monitor_uses_durable_lot_without_legacy_adapter(monkeypatch, seed):
    fill_id = seed.fill(
        quantity=0.3,
        price=100,
        market="crypto_perp",
        symbol="BTCUSDT",
        instrument="BTCUSDT-PERP",
    )
    project(seed, fill_id)
    with seed.sessions() as session:
        lot = session.scalar(select(PositionLot).where(PositionLot.opening_fill_id == fill_id))
        decision_id = lot.opening_decision_id
    _set_owner_protection(seed, decision_id, leverage=3.0, margin_ratio_threshold=0.15)
    _exact_settings(monkeypatch, mode="halted", market="crypto_perp", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "_build_perp_adapter_from_db", lambda: pytest.fail("legacy adapter called"))
    monkeypatch.setattr(cpu_tasks, "_get_perp_mark_prices", lambda _symbols: {"BTCUSDT": 10.0})
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: pytest.fail("legacy tracker called"))
    calls = []
    monkeypatch.setattr(cpu_tasks, "_run_durable_protective", lambda **kwargs: calls.append(kwargs) or ["order-2"])

    result = cpu_tasks.perp_liquidation_monitor.run()

    assert result["durable_order_ids"] == ["order-2"]
    assert calls[0]["origin"] == "liquidation"
    assert calls[0]["triggered"] == {("crypto_perp", "BTCUSDT", "BTCUSDT-PERP", "long"): 10.0}


def test_tw_legacy_monitor_does_not_call_durable_service(monkeypatch):
    _exact_settings(monkeypatch, mode="legacy", market="tw_stock")
    monkeypatch.setattr(cpu_tasks, "datetime", _TradingTime)
    monkeypatch.setattr(cpu_tasks, "_run_durable_protective", lambda **_kwargs: pytest.fail("durable called"))
    tracker = type("EmptyTracker", (), {"current_holdings": lambda _self: {}})()
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: tracker)

    assert cpu_tasks.portfolio_stop_loss_monitor.run() == {"checked": 0, "stopped_out": []}


def test_perp_shadow_monitor_does_not_call_durable_service(monkeypatch):
    _exact_settings(monkeypatch, mode="shadow", market="crypto_perp")
    monkeypatch.setattr(cpu_tasks, "_run_durable_protective", lambda **_kwargs: pytest.fail("durable called"))
    adapter = type("EmptyPerpAdapter", (), {"_positions": {}})()
    monkeypatch.setattr(cpu_tasks, "_build_perp_adapter_from_db", lambda: adapter)

    assert cpu_tasks.perp_liquidation_monitor.run() == {"checked": 0, "closed": []}


@pytest.mark.parametrize("allowlisted", [False, True])
def test_perp_legacy_fallback_requires_explicit_scope_allowlist(monkeypatch, seed, allowlisted):
    _legacy_holding(
        seed,
        market="crypto_perp",
        symbol="BTCUSDT",
        shares=0.1,
        entry_price=100,
        stop_loss_pct=None,
    )
    _exact_settings(monkeypatch, mode="halted", market="crypto_perp", scope=seed.account)
    monkeypatch.setattr(
        settings,
        "decision_loop_legacy_protective_scopes",
        (seed.account,) if allowlisted else (),
    )
    calls = []
    adapter = type("EmptyPerpAdapter", (), {"_positions": {}})()
    monkeypatch.setattr(
        cpu_tasks,
        "_build_perp_adapter_from_db",
        lambda: calls.append("legacy") or adapter,
    )

    result = cpu_tasks.perp_liquidation_monitor.run()

    assert calls == (["legacy"] if allowlisted else [])
    if not allowlisted:
        assert result["skipped"] == "legacy_protective_scope_not_authorized"


def test_perp_allowlist_fallback_excludes_durable_lot_identity_from_legacy_close(monkeypatch, seed):
    fill_id = seed.fill(
        quantity=0.3,
        price=100,
        market="crypto_perp",
        symbol="BTCUSDT",
        instrument="BTCUSDT-PERP",
    )
    project(seed, fill_id)
    with seed.sessions() as session:
        lot = session.scalar(select(PositionLot).where(PositionLot.opening_fill_id == fill_id))
        decision_id = lot.opening_decision_id
    _set_owner_protection(seed, decision_id, leverage=3.0, margin_ratio_threshold=0.15)
    legacy_holding_id, _account_id = _legacy_holding(
        seed,
        market="crypto_perp",
        symbol="ETHUSDT",
        shares=0.2,
        entry_price=100,
        stop_loss_pct=None,
    )
    _exact_settings(monkeypatch, mode="halted", market="crypto_perp", scope=seed.account)
    monkeypatch.setattr(settings, "decision_loop_legacy_protective_scopes", (seed.account,))
    monkeypatch.setattr(cpu_tasks, "_get_perp_mark_prices", lambda _symbols: {"BTCUSDT": 10.0, "ETHUSDT": 90.0})
    durable_calls = []
    monkeypatch.setattr(
        cpu_tasks,
        "_run_durable_protective",
        lambda **kwargs: durable_calls.append(kwargs) or ["durable-order"],
    )

    class LegacyAdapter:
        def __init__(self, *_args):
            self._positions = {"BTCUSDT": object(), "ETHUSDT": object()}

        def update_mark_prices(self, _prices):
            return None

        def query_positions(self):
            return [{"symbol": "ETHUSDT", "marginRatio": 0.1}]

    captured = []

    class LegacyManager:
        def __init__(self, *_args):
            pass

        def execute_rebalance(self, orders, **_kwargs):
            captured.extend(order.symbol for order in orders)
            return [type("Result", (), {"success": False})() for _order in orders]

    monkeypatch.setattr(cpu_tasks, "_build_perp_adapter_from_db", LegacyAdapter)
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: object())
    monkeypatch.setattr("poseidon.orders.manager.OrderManager", LegacyManager)
    monkeypatch.setattr("poseidon.broker.perp_paper_adapter.PerpPaperAdapter", LegacyAdapter)

    result = cpu_tasks.perp_liquidation_monitor.run()

    assert result["durable_order_ids"] == ["durable-order"]
    assert durable_calls[0]["triggered"] == {
        ("crypto_perp", "BTCUSDT", "BTCUSDT-PERP", "long"): 10.0,
    }
    assert captured == ["ETHUSDT"]
    with seed.sessions() as session:
        assert session.get(PortfolioHoldingRecord, legacy_holding_id).closed is False


def test_exact_legacy_tracker_close_does_not_close_same_symbol_projected_row(seed):
    legacy_id, _account_id = _legacy_holding(seed, market="crypto_perp", symbol="ETHUSDT", shares=0.2)
    projected_id, _account_id = _legacy_holding(
        seed,
        market="crypto_perp",
        symbol="ETHUSDT",
        shares=0.3,
        projected=True,
    )
    tracker = PositionTracker(seed.sessions)

    tracker.apply_orders(
        [
            RebalanceOrder(
                symbol="ETHUSDT",
                action="sell",
                target_weight=0,
                current_weight=0.1,
                delta_weight=-0.1,
                side="long",
                holding_id=legacy_id,
            )
        ],
        "liquidation_protection",
        "crypto_perp",
    )

    with seed.sessions() as session:
        assert session.get(PortfolioHoldingRecord, legacy_id).closed is True
        assert session.get(PortfolioHoldingRecord, projected_id).closed is False


@pytest.mark.parametrize("source", ["missing", "projected"])
def test_exact_legacy_tracker_close_rejects_missing_or_projected_id(seed, source):
    projected_id, _account_id = _legacy_holding(seed, projected=True)
    holding_id = uuid.uuid4() if source == "missing" else projected_id
    tracker = PositionTracker(seed.sessions)

    with pytest.raises(ValueError, match="exact legacy holding"):
        tracker.apply_orders(
            [
                RebalanceOrder(
                    symbol="2330",
                    action="sell",
                    target_weight=0,
                    current_weight=0.1,
                    delta_weight=-0.1,
                    side="long",
                    holding_id=holding_id,
                )
            ],
            "liquidation_protection",
            "tw_stock",
        )

    with seed.sessions() as session:
        assert session.get(PortfolioHoldingRecord, projected_id).closed is False
