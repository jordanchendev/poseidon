"""Durable protective materialization, reservation, routing, and trace proofs."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import func, select, update

from poseidon.broker.base import BrokerAdapter, BrokerCapabilities, BrokerOrderSnapshot
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
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from poseidon.positions.lots import FillProjectionConflictError
from poseidon.workers import cpu_tasks
from poseidon.workers.cpu_tasks import _protective_execution_route
from tests.test_decision_service import worker
from tests.test_position_lot_allocation import project
from tests.test_position_lot_allocation import seed as _position_lot_seed


@pytest.fixture(name="seed")
def protective_seed():
    """Keep the shared lot fixture available regardless of collection order."""
    yield from _position_lot_seed.__wrapped__()


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


def _opening(seed, *, quantity=5, time=NOW):
    fill_id = seed.fill(quantity=quantity, time=time)
    project(seed, fill_id)
    with seed.sessions() as session:
        fill = session.get(OrderFillRecord, fill_id)
        lot = session.scalar(select(PositionLot).where(PositionLot.opening_fill_id == fill.id))
        return lot.id, lot.opening_decision_id


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
            session.scalar(select(func.count()).select_from(OrderRecord).where(OrderRecord.order_origin != "decision"))
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
            session.scalar(select(func.count()).select_from(OrderRecord).where(OrderRecord.order_origin == "stop_loss"))
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
            session.scalar(select(func.count()).select_from(OrderRecord).where(OrderRecord.order_origin == "stop_loss"))
            == 1
        )


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
            session.scalar(select(func.count()).select_from(OrderRecord).where(OrderRecord.order_origin == "stop_loss"))
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
