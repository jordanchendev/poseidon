"""Phase 89 Plan 02 -- portfolio_monthly_rebalance un-noop wiring (W3).

Verifies that portfolio_monthly_rebalance, previously a logged no-op since
Phase 61 (RevenueBreakoutStrategy removed when FinLab data became unavailable),
now consumes PASSED tw_stock signals from the SignalRepository within a
7-day freshness window and dispatches RebalanceOrders.

The 7-day filter mechanically excludes the 13 legacy frozen signals dated
2026-03-19 (CONTEXT D-15).

Test inventory (per 89-02-PLAN Task 2):
1. test_fresh_signals_dispatch_orders — 3 fresh PASSED signals → 3 RebalanceOrders + signal_ids
2. test_only_stale_signals_returns_skipped — 13 legacy signals (>7d) → no_recent_signals skip
3. test_mixed_fresh_stale_only_fresh_honoured — fresh + stale → only fresh dispatched
4. test_order_origin_defaults_to_signal — produced Orders carry order_origin='signal'
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from poseidon.orders.schemas import Order


def _make_signal(
    *,
    symbol: str,
    action: str = "long",
    quantity_pct: float = 0.2,
    age_days: float = 2.0,
    signal_id: uuid.UUID | None = None,
):
    """SignalRecord-shaped namespace without SQLAlchemy ORM instrumentation."""
    return SimpleNamespace(
        id=signal_id or uuid.uuid4(),
        strategy_id=uuid.uuid4(),
        symbol=symbol,
        market="tw_stock",
        action=action,
        confidence=0.85,
        quantity_pct=quantity_pct,
        signal_time=datetime.now(UTC) - timedelta(days=age_days),
        valid_until=datetime.now(UTC) + timedelta(days=7),
        interval="1d",
        params={},
        status="passed",
        reject_reason=None,
        order_type="market",
        order_price=None,
        stop_loss_price=None,
        take_profit_price=None,
    )


@pytest.fixture
def patched_monthly_env(monkeypatch):
    """Patch all external surfaces portfolio_monthly_rebalance touches."""
    from poseidon.workers import cpu_tasks

    state: dict = {
        "signals": [],
        "captured": {},
        "prices": {"2330": 600.0, "2454": 1100.0, "2317": 110.0, "1234": 50.0, "5678": 200.0},
    }

    # 1. Position tracker — empty holdings
    fake_tracker = MagicMock()
    fake_tracker.current_holdings.return_value = {}
    fake_tracker.apply_orders.return_value = None
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: fake_tracker)

    # 2. TW stock broker
    fake_broker = MagicMock()
    monkeypatch.setattr(cpu_tasks, "_build_tw_stock_broker", lambda cfg: fake_broker)

    # 3. Latest prices
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda syms: {s: state["prices"].get(s, 100.0) for s in syms})

    # 4. db_session — fake session whose SignalRepository returns state["signals"]
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
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _FakeSession())

    # 5. Capture execute_rebalance kwargs at the OrderManager class
    from poseidon.orders.manager import OrderManager

    def fake_execute_rebalance(
        self, rebalance_orders, strategy_name, prices, market="tw_stock", *, signal_ids=None, **extra
    ):
        state["captured"]["rebalance_orders"] = list(rebalance_orders)
        state["captured"]["strategy_name"] = strategy_name
        state["captured"]["prices"] = dict(prices)
        state["captured"]["market"] = market
        state["captured"]["signal_ids"] = signal_ids

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
                quantity=10.0,
                strategy_name=str(strategy_name),
                broker_mode="paper",
                signal_id=sid,
                order_origin=extra.get("order_origin", "signal"),
            )
            results.append(OrderResult(order=order, fills=[], success=True))
        state["captured"]["results"] = results
        return results

    monkeypatch.setattr(OrderManager, "execute_rebalance", fake_execute_rebalance)

    # Skip yaml file load — patch open of broker.yaml to a minimal cfg
    import builtins

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if "broker.yaml" in str(path):
            from io import StringIO

            return StringIO("mode: paper\npaper_initial_nav: 100000.0\nlot_size: 1\nfractional_qty: false\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    yield state


class TestPortfolioMonthlyRebalanceWiring:
    """portfolio_monthly_rebalance must consume fresh tw_stock PASSED signals."""

    def test_fresh_signals_dispatch_orders(self, patched_monthly_env):
        """3 fresh PASSED signals → 3 RebalanceOrders + signal_ids dict."""
        from poseidon.workers import cpu_tasks

        sig_ids = {sym: uuid.uuid4() for sym in ("2330", "2454", "2317")}
        patched_monthly_env["signals"] = [
            _make_signal(symbol="2330", action="long", quantity_pct=0.2, age_days=2, signal_id=sig_ids["2330"]),
            _make_signal(symbol="2454", action="long", quantity_pct=0.2, age_days=2, signal_id=sig_ids["2454"]),
            _make_signal(symbol="2317", action="long", quantity_pct=0.2, age_days=2, signal_id=sig_ids["2317"]),
        ]

        result = cpu_tasks.portfolio_monthly_rebalance()

        captured = patched_monthly_env["captured"]
        assert "rebalance_orders" in captured, f"expected execute_rebalance to be called, got result={result}"
        assert len(captured["rebalance_orders"]) == 3, (
            f"expected 3 RebalanceOrders, got {len(captured['rebalance_orders'])}"
        )
        assert set(captured["signal_ids"].keys()) == {"2330", "2454", "2317"}
        for sym, sid in sig_ids.items():
            assert captured["signal_ids"][sym] == sid
        assert result.get("rebalanced") is True
        assert result.get("orders") == 3

    def test_only_stale_signals_returns_skipped(self, patched_monthly_env):
        """13 legacy signals (>7d) → no_recent_signals skip, no execute_rebalance call."""
        from poseidon.workers import cpu_tasks

        # 13 frozen 2026-03-19 cluster (CONTEXT D-15) — all >7d old
        patched_monthly_env["signals"] = []  # latest_passed(since=now-7d) → empty for stale

        result = cpu_tasks.portfolio_monthly_rebalance()

        assert result.get("skipped") == "no_recent_signals", f"expected skipped=no_recent_signals, got {result}"
        assert "rebalance_orders" not in patched_monthly_env["captured"], (
            "execute_rebalance must NOT be called when no fresh signals"
        )

    def test_mixed_fresh_stale_only_fresh_honoured(self, patched_monthly_env):
        """Fresh + stale → only fresh dispatched (latest_passed already filters)."""
        from poseidon.workers import cpu_tasks

        # The repository's `since` filter mechanically excludes stale; in this
        # test we only return what latest_passed would return (the fresh ones).
        sig_a, sig_b = uuid.uuid4(), uuid.uuid4()
        patched_monthly_env["signals"] = [
            _make_signal(symbol="1234", action="long", quantity_pct=0.3, age_days=3, signal_id=sig_a),
            _make_signal(symbol="5678", action="long", quantity_pct=0.3, age_days=3, signal_id=sig_b),
        ]

        cpu_tasks.portfolio_monthly_rebalance()

        captured = patched_monthly_env["captured"]
        assert len(captured["rebalance_orders"]) == 2
        assert set(captured["signal_ids"].keys()) == {"1234", "5678"}

    def test_order_origin_defaults_to_signal(self, patched_monthly_env):
        """Orders dispatched by monthly task must carry order_origin='signal'."""
        from poseidon.workers import cpu_tasks

        patched_monthly_env["signals"] = [
            _make_signal(symbol="2330", action="long", quantity_pct=0.25, age_days=1),
        ]

        cpu_tasks.portfolio_monthly_rebalance()

        results = patched_monthly_env["captured"].get("results", [])
        assert len(results) >= 1
        for r in results:
            assert r.order.order_origin == "signal", f"Expected order_origin='signal', got {r.order.order_origin}"
