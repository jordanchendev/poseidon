"""Tests for OrderRiskChecker -- per-order risk validation.

Tests per ORDER-02 requirements:
1. Position limit check
2. Max exposure check (buy orders only)
3. Stop-loss configured check (buy orders only)
4. Per-order isolation (one reject does not block others)
"""

import pytest

from poseidon.orders.risk_checker import OrderRiskChecker
from poseidon.orders.schemas import Order, RiskCheckResult
from poseidon.orders.state_machine import OrderStatus
from poseidon.strategies.portfolio.schemas import Holding


def _make_order(
    symbol: str = "2330",
    action: str = "buy",
    target_weight: float = 0.10,
    quantity: int = 1000,
) -> Order:
    """Helper to create a test Order."""
    return Order(
        symbol=symbol,
        market="tw_stock",
        action=action,
        order_type="market",
        target_weight=target_weight,
        quantity=quantity,
        strategy_name="test_strategy",
        broker_mode="paper",
    )


def _make_holdings(weights: dict[str, float]) -> dict[str, Holding]:
    """Helper to create holdings dict from symbol->weight mapping."""
    return {
        sym: Holding(symbol=sym, market="tw_stock", weight=w)
        for sym, w in weights.items()
    }


class TestPositionLimitCheck:
    """Test 1 & 2: Single stock position limit."""

    def test_order_within_limit_passes(self):
        """Order with weight 0.10 passes when position_limit_pct=0.15."""
        checker = OrderRiskChecker(position_limit_pct=0.15)
        order = _make_order(target_weight=0.10)
        result = checker.check(order, {}, total_nav=10_000_000.0)
        assert result.passed is True

    def test_order_exceeding_limit_rejected(self):
        """Order with weight 0.20 fails when position_limit_pct=0.15."""
        checker = OrderRiskChecker(position_limit_pct=0.15)
        order = _make_order(target_weight=0.20)
        result = checker.check(order, {}, total_nav=10_000_000.0)
        assert result.passed is False
        assert "position_limit" in result.reason


class TestMaxExposureCheck:
    """Test 3 & 4: Total portfolio exposure limit."""

    def test_buy_exceeding_max_exposure_rejected(self):
        """Buy when exposure 0.90 + order 0.10 > max 0.95 -> rejected."""
        checker = OrderRiskChecker(
            position_limit_pct=0.20,
            max_exposure=0.95,
        )
        holdings = _make_holdings({"A": 0.30, "B": 0.30, "C": 0.30})
        order = _make_order(target_weight=0.10)
        result = checker.check(order, holdings, total_nav=10_000_000.0)
        assert result.passed is False
        assert "max_exposure" in result.reason

    def test_buy_within_max_exposure_passes(self):
        """Buy when exposure 0.80 + order 0.10 <= max 0.95 -> passed."""
        checker = OrderRiskChecker(
            position_limit_pct=0.20,
            max_exposure=0.95,
        )
        holdings = _make_holdings({"A": 0.40, "B": 0.40})
        order = _make_order(target_weight=0.10)
        result = checker.check(order, holdings, total_nav=10_000_000.0)
        assert result.passed is True


class TestStopLossCheck:
    """Test 5 & 6: Stop-loss must be configured for buy orders."""

    def test_buy_without_stop_loss_rejected(self):
        """Buy order without stop_loss configured -> rejected."""
        checker = OrderRiskChecker(
            position_limit_pct=0.20,
            stop_loss_pct=None,
        )
        order = _make_order(action="buy", target_weight=0.10)
        result = checker.check(order, {}, total_nav=10_000_000.0)
        assert result.passed is False
        assert "stop_loss" in result.reason

    def test_sell_skips_stop_loss_check(self):
        """Sell order skips stop_loss check -> passed."""
        checker = OrderRiskChecker(
            position_limit_pct=0.20,
            stop_loss_pct=None,
        )
        order = _make_order(action="sell", target_weight=0.10)
        result = checker.check(order, {}, total_nav=10_000_000.0)
        assert result.passed is True


class TestOrderIsolation:
    """Test 7: Multiple orders processed independently."""

    def test_one_rejected_does_not_affect_others(self):
        """Rejected order does not block other orders."""
        checker = OrderRiskChecker(position_limit_pct=0.15)

        order_ok = _make_order(symbol="2330", target_weight=0.10)
        order_bad = _make_order(symbol="2317", target_weight=0.20)
        order_ok2 = _make_order(symbol="2454", target_weight=0.12)

        r1 = checker.check(order_ok, {}, total_nav=10_000_000.0)
        r2 = checker.check(order_bad, {}, total_nav=10_000_000.0)
        r3 = checker.check(order_ok2, {}, total_nav=10_000_000.0)

        assert r1.passed is True
        assert r2.passed is False
        assert r3.passed is True
