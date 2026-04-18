"""Unit tests for PortfolioBacktester.

Tests the monthly rebalance loop, equity curve generation, and metrics computation.
All tests use synthetic OHLCV data and mock strategies.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.portfolio_backtester import PortfolioBacktester, PortfolioBacktestResult
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.schemas import TargetPosition


# --- Fixtures ---


def _make_ohlcv(start: str = "2023-01-02", periods: int = 130, base_price: float = 100.0) -> pd.DataFrame:
    """Create synthetic OHLCV data with business day frequency."""
    dates = pd.date_range(start, periods=periods, freq="B")
    # Slight uptrend
    close = [base_price + i * 0.1 for i in range(periods)]
    return pd.DataFrame(
        {
            "open": [c - 0.5 for c in close],
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1000000] * periods,
        },
        index=dates,
    )


@pytest.fixture
def cost_model():
    return get_cost_model("tw_stock")


@pytest.fixture
def ohlcv_dict():
    return {
        "2330": _make_ohlcv(base_price=500.0),
        "2317": _make_ohlcv(base_price=100.0),
    }


class MockStrategy(PortfolioStrategy):
    """Mock strategy that returns predefined targets."""

    name = "mock_strategy"

    def __init__(self, targets: list[TargetPosition] | None = None):
        self._targets = targets or []
        self._call_count = 0

    def select_stocks(self, universe_df: pd.DataFrame, as_of: date | None = None) -> list[TargetPosition]:
        self._call_count += 1
        return self._targets

    def validate_config(self) -> bool:
        return True


class FailingStrategy(PortfolioStrategy):
    """Strategy that raises an exception on select_stocks."""

    name = "failing_strategy"

    def select_stocks(self, universe_df: pd.DataFrame, as_of: date | None = None) -> list[TargetPosition]:
        raise RuntimeError("Strategy exploded!")

    def validate_config(self) -> bool:
        return True


# --- Tests ---


class TestPortfolioBacktester:

    def test_monthly_rebalance_calls_select_stocks(self, cost_model, ohlcv_dict):
        """select_stocks should be called approximately once per month."""
        targets = [
            TargetPosition(symbol="2330", weight=0.5, reason="test"),
            TargetPosition(symbol="2317", weight=0.5, reason="test"),
        ]
        strategy = MockStrategy(targets=targets)

        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=1_000_000.0)
        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert result.status == "completed"
        # 6 months -> approximately 6 rebalance calls (Jan-Jun)
        assert strategy._call_count >= 5
        assert strategy._call_count <= 7

    def test_equity_curve_has_daily_entries(self, cost_model, ohlcv_dict):
        """Equity curve length should equal number of trading days."""
        strategy = MockStrategy(targets=[])
        backtester = PortfolioBacktester(cost_model=cost_model)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        # Count unique trading days in ohlcv_dict within range
        trading_days = set()
        for sym, df in ohlcv_dict.items():
            for idx in df.index:
                d = idx.date() if hasattr(idx, "date") else idx
                if date(2023, 1, 2) <= d <= date(2023, 6, 30):
                    trading_days.add(d)

        assert len(result.equity_curve) == len(trading_days)

    def test_metrics_sharpe_computed(self, cost_model, ohlcv_dict):
        """Result metrics should contain sharpe_ratio as a float."""
        targets = [
            TargetPosition(symbol="2330", weight=0.5, reason="test"),
        ]
        strategy = MockStrategy(targets=targets)
        backtester = PortfolioBacktester(cost_model=cost_model)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert "sharpe_ratio" in result.metrics
        assert isinstance(result.metrics["sharpe_ratio"], float)

    def test_metrics_max_drawdown_computed(self, cost_model, ohlcv_dict):
        """Result metrics should contain max_drawdown."""
        targets = [
            TargetPosition(symbol="2330", weight=0.5, reason="test"),
        ]
        strategy = MockStrategy(targets=targets)
        backtester = PortfolioBacktester(cost_model=cost_model)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert "max_drawdown" in result.metrics
        assert isinstance(result.metrics["max_drawdown"], float)

    def test_failed_strategy_returns_failed_result(self, cost_model, ohlcv_dict):
        """A strategy that raises should produce status='failed'."""
        strategy = FailingStrategy()
        backtester = PortfolioBacktester(cost_model=cost_model)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert result.status == "failed"
        assert result.error_message is not None
        assert "exploded" in result.error_message.lower()

    def test_initial_capital_preserved_no_trades(self, cost_model, ohlcv_dict):
        """With no trades (empty targets), final NAV should equal initial capital."""
        strategy = MockStrategy(targets=[])
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=1_000_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert result.status == "completed"
        # No trades -> NAV should be exactly initial capital
        assert len(result.equity_curve) > 0
        final_nav = result.equity_curve[-1][1]
        assert final_nav == pytest.approx(1_000_000.0, abs=0.01)
