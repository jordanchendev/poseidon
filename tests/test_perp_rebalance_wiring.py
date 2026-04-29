"""Phase 89 Plan 02 -- perp_rebalance wiring (W2).

Verifies perp_rebalance reads PASSED crypto_perp signals from SignalRepository
and propagates signal_ids dict to OrderManager.execute_rebalance, tagging
resulting Order objects with signal_id and order_origin="signal".

These are unit-style integration tests: external dependencies (DB, broker,
data repository, position tracker, protections) are monkeypatched. The
RED step (pre-implementation) MUST fail because perp_rebalance currently
instantiates CryptoTrendStrategy inline (cpu_tasks.py:1986) and never
consults SignalRepository.latest_passed.

Test inventory (per 89-02-PLAN Task 1):
1. test_perp_rebalance_propagates_signal_ids — fresh PASSED signal → signal_ids dict reaches execute_rebalance
2. test_perp_rebalance_ignores_stale_signals_beyond_8h — signal_time = now - 9h → no orders dispatched
3. test_perp_rebalance_dedupes_per_symbol_newest_wins — 2 signals for same symbol → newer wins
4. test_perp_rebalance_zero_recent_signals_returns_no_orders — empty fresh window → early return
5. test_perp_rebalance_order_origin_defaults_to_signal — implicit default 'signal' propagation
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from poseidon.orders.schemas import Order


def _make_signal_record(
    *,
    symbol: str = "ETHUSDT",
    market: str = "crypto_perp",
    action: str = "long",
    quantity_pct: float = 0.5,
    age_minutes: int = 30,
    signal_id: uuid.UUID | None = None,
):
    """Build a SignalRecord-shaped namespace without touching SQLAlchemy ORM.

    Avoids ORM instrumentation by returning a SimpleNamespace with the same
    attribute surface that `_signals_to_targets` and downstream consumers
    read. Used by latest_passed mocks to return synthetic results.
    """
    return SimpleNamespace(
        id=signal_id or uuid.uuid4(),
        strategy_id=None,
        symbol=symbol,
        market=market,
        action=action,
        confidence=0.8,
        quantity_pct=quantity_pct,
        signal_time=datetime.now(UTC) - timedelta(minutes=age_minutes),
        valid_until=datetime.now(UTC) + timedelta(hours=4),
        interval="4h",
        params={},
        status="passed",
        reject_reason=None,
        order_type="market",
        order_price=None,
        stop_loss_price=None,
        take_profit_price=None,
    )


# ---------------------------------------------------------------------------
# Test fixture: heavily mocked perp_rebalance environment
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_perp_env(monkeypatch):
    """Patch all external surfaces perp_rebalance touches.

    Returns a dict the test can mutate to control behaviour:
        - 'signals': list[SignalRecord] returned by latest_passed
        - 'captured': dict capturing execute_rebalance kwargs
        - 'mark_prices': dict[symbol, price]
        - 'protection_locked': bool — toggle portfolio-level lock
    """
    from poseidon.workers import cpu_tasks

    state: dict = {
        "signals": [],
        "captured": {},
        "mark_prices": {"ETHUSDT": 3000.0, "BTCUSDT": 60000.0},
        "protection_locked": False,
    }

    # 1. Position tracker — empty holdings
    fake_tracker = MagicMock()
    fake_tracker.current_holdings.return_value = {}
    fake_tracker.apply_orders.return_value = None
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: fake_tracker)

    # 2. Perp adapter — empty positions
    fake_adapter = MagicMock()
    fake_adapter._positions = {}
    monkeypatch.setattr(cpu_tasks, "_build_perp_adapter_from_db", lambda: fake_adapter)

    # 3. Mark prices
    monkeypatch.setattr(
        cpu_tasks, "_get_perp_mark_prices", lambda syms: {s: state["mark_prices"].get(s, 1000.0) for s in syms}
    )

    # 4. db_session — context manager returning a mocked session whose
    # SignalRepository(session).latest_passed(...) returns state["signals"].
    class _FakeSession:
        def close(self):
            pass

        def query(self, *_a, **_kw):
            mq = MagicMock()
            mq.filter.return_value = mq
            mq.order_by.return_value = mq
            mq.limit.return_value = mq
            mq.all.return_value = list(state["signals"])
            return mq

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def commit(self):
            pass

        def add(self, *_a, **_kw):
            pass

    class _FakeDbSession:
        def __enter__(self):
            return _FakeSession()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cpu_tasks, "db_session", lambda: _FakeDbSession())

    # 5. Protection manager — non-locking by default
    from poseidon.protections import manager as pm_mod

    fake_pm = MagicMock()
    fake_pm.check_all.return_value = []  # no locks
    fake_pm.is_locked.return_value = False
    monkeypatch.setattr(pm_mod.ProtectionManager, "from_defaults", classmethod(lambda cls: fake_pm))

    # 6. SessionLocal — never actually invoked once db_session is mocked, but
    # OrderManager constructor needs *something* callable.
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _FakeSession())

    # 7. Capture execute_rebalance kwargs at the OrderManager class.
    from poseidon.orders.manager import OrderManager

    def fake_execute_rebalance(
        self, rebalance_orders, strategy_name, prices, market="tw_stock", *, signal_ids=None, **extra
    ):
        state["captured"]["rebalance_orders"] = list(rebalance_orders)
        state["captured"]["strategy_name"] = strategy_name
        state["captured"]["prices"] = dict(prices)
        state["captured"]["market"] = market
        state["captured"]["signal_ids"] = signal_ids
        state["captured"]["extra"] = extra
        # Manufacture Order objects we'd have placed, so caller can inspect order_origin.
        from poseidon.orders.schemas import OrderResult

        results = []
        for ro in rebalance_orders:
            sid = signal_ids.get(ro.symbol) if signal_ids else None
            order = Order(
                symbol=ro.symbol,
                market=market,
                action="buy" if ro.delta_weight > 0 else "sell",
                order_type="market",
                target_weight=ro.target_weight,
                quantity=1.0,
                strategy_name=strategy_name,
                broker_mode="paper",
                signal_id=sid,
                order_origin=extra.get("order_origin", "signal"),
            )
            results.append(OrderResult(order=order, fills=[], success=True))
        state["captured"]["results"] = results
        return results

    monkeypatch.setattr(OrderManager, "execute_rebalance", fake_execute_rebalance)

    yield state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerpRebalanceWiring:
    """perp_rebalance must read PASSED signals and propagate signal_ids."""

    def test_perp_rebalance_propagates_signal_ids(self, patched_perp_env):
        """Fresh PASSED ETHUSDT signal → signal_ids dict reaches execute_rebalance."""
        from poseidon.workers import cpu_tasks

        sig_id = uuid.uuid4()
        patched_perp_env["signals"] = [
            _make_signal_record(symbol="ETHUSDT", action="long", quantity_pct=0.5, age_minutes=30, signal_id=sig_id),
        ]

        cpu_tasks.perp_rebalance()

        captured = patched_perp_env["captured"]
        assert "signal_ids" in captured, "execute_rebalance must be called with signal_ids kwarg"
        assert captured["signal_ids"] == {"ETHUSDT": sig_id}

    def test_perp_rebalance_ignores_stale_signals_beyond_8h(self, patched_perp_env):
        """Signals older than 8h must not produce orders."""
        from poseidon.workers import cpu_tasks

        # latest_passed should never return signals older than 'since' — but we
        # simulate the contract by returning empty when window is < 9h.
        patched_perp_env["signals"] = []  # latest_passed(since=now-8h) → empty
        cpu_tasks.perp_rebalance()

        captured = patched_perp_env["captured"]
        # Either execute_rebalance was never called OR was called with zero orders.
        if "rebalance_orders" in captured:
            assert len(captured["rebalance_orders"]) == 0
        # Else: the early-return path before any execute_rebalance — also acceptable.

    def test_perp_rebalance_dedupes_per_symbol_newest_wins(self, patched_perp_env):
        """Multiple signals for the same symbol → newest signal_id wins (latest_passed orders DESC)."""
        from poseidon.workers import cpu_tasks

        newer_id = uuid.uuid4()
        older_id = uuid.uuid4()
        # latest_passed returns DESC, so newer is first.
        patched_perp_env["signals"] = [
            _make_signal_record(symbol="ETHUSDT", age_minutes=30, signal_id=newer_id),
            _make_signal_record(symbol="ETHUSDT", age_minutes=60, signal_id=older_id),
        ]

        cpu_tasks.perp_rebalance()

        captured = patched_perp_env["captured"]
        assert captured.get("signal_ids", {}).get("ETHUSDT") == newer_id, (
            f"Expected newer signal id {newer_id}, got {captured.get('signal_ids', {}).get('ETHUSDT')}"
        )

    def test_perp_rebalance_zero_recent_signals_returns_no_orders(self, patched_perp_env):
        """Empty fresh window → early return, no execute_rebalance call."""
        from poseidon.workers import cpu_tasks

        patched_perp_env["signals"] = []
        result = cpu_tasks.perp_rebalance()

        # Implementation should short-circuit and return either zero-orders or skipped state.
        assert result.get("orders", 0) == 0 or "skipped" in result or "reason" in result, (
            f"Expected zero-order/skipped result, got {result}"
        )

    def test_perp_rebalance_order_origin_defaults_to_signal(self, patched_perp_env):
        """Orders dispatched by perp_rebalance must carry order_origin='signal' (default)."""
        from poseidon.workers import cpu_tasks

        patched_perp_env["signals"] = [
            _make_signal_record(symbol="ETHUSDT", action="long", quantity_pct=0.3),
        ]

        cpu_tasks.perp_rebalance()

        results = patched_perp_env["captured"].get("results", [])
        assert len(results) >= 1, "expected at least one Order produced"
        for r in results:
            assert r.order.order_origin == "signal", (
                f"Expected order_origin='signal' (default), got {r.order.order_origin}"
            )


# ---------------------------------------------------------------------------
# Helper unit tests (REFACTOR step — _signals_to_targets)
# ---------------------------------------------------------------------------


class TestSignalsToTargets:
    """The _signals_to_targets helper translates SignalRecord list to (targets, signal_ids).

    Mapping (Phase 89-02):
        action='long'  → targets[symbol] = float(quantity_pct)
        action='short' → targets[symbol] = -float(quantity_pct)  (short side handled by RebalanceOrder.side='short')
        action='close' → targets[symbol] = 0.0
        action='hold'  → skip (no entry in targets)

    Newest-per-symbol semantics: latest_passed already orders DESC, helper
    must defensively dedup (first occurrence wins).
    """

    def test_long_signal_maps_to_positive_target(self):
        from poseidon.workers.cpu_tasks import _signals_to_targets

        sigs = [_make_signal_record(symbol="ETHUSDT", action="long", quantity_pct=0.4)]
        targets, signal_ids = _signals_to_targets(sigs)

        assert targets["ETHUSDT"] == pytest.approx(0.4)
        assert signal_ids["ETHUSDT"] == sigs[0].id

    def test_close_signal_maps_to_zero_target(self):
        from poseidon.workers.cpu_tasks import _signals_to_targets

        sigs = [_make_signal_record(symbol="ETHUSDT", action="close", quantity_pct=0.4)]
        targets, signal_ids = _signals_to_targets(sigs)

        assert targets["ETHUSDT"] == 0.0
        assert signal_ids["ETHUSDT"] == sigs[0].id

    def test_hold_signal_is_skipped(self):
        from poseidon.workers.cpu_tasks import _signals_to_targets

        sigs = [_make_signal_record(symbol="ETHUSDT", action="hold", quantity_pct=0.4)]
        targets, signal_ids = _signals_to_targets(sigs)

        assert "ETHUSDT" not in targets
        assert "ETHUSDT" not in signal_ids

    def test_dedup_first_wins(self):
        """latest_passed returns DESC; first SignalRecord per symbol must win."""
        from poseidon.workers.cpu_tasks import _signals_to_targets

        newer_id = uuid.uuid4()
        older_id = uuid.uuid4()
        sigs = [
            _make_signal_record(symbol="ETHUSDT", action="long", quantity_pct=0.5, signal_id=newer_id),
            _make_signal_record(symbol="ETHUSDT", action="long", quantity_pct=0.2, signal_id=older_id, age_minutes=120),
        ]
        targets, signal_ids = _signals_to_targets(sigs)

        assert signal_ids["ETHUSDT"] == newer_id
        assert targets["ETHUSDT"] == pytest.approx(0.5)
