"""Unit tests for PortfolioBacktester.

Tests the monthly rebalance loop, equity curve generation, and metrics computation.
All tests use synthetic OHLCV data and mock strategies.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock
from types import SimpleNamespace

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


class ConfigurableMockStrategy(MockStrategy):
    """Mock strategy with flat config fields used by portfolio strategies."""

    def __init__(
        self,
        targets: list[TargetPosition] | None = None,
        *,
        rebalance_day_of_month: int = 15,
        rebalance_frequency: str = "monthly",
        rebalance_day_of_week: int = 4,
        stop_loss_pct: float | None = None,
    ):
        super().__init__(targets=targets)
        self.config = SimpleNamespace(
            rebalance_day_of_month=rebalance_day_of_month,
            rebalance_frequency=rebalance_frequency,
            rebalance_day_of_week=rebalance_day_of_week,
            stop_loss_pct=stop_loss_pct,
        )


class HoldUntilMockStrategy(ConfigurableMockStrategy):
    """Mock strategy that supports check_hold_until with configurable per-symbol+date result."""

    def __init__(
        self,
        *args,
        hold_until_results: dict[str, bool] | None = None,
        hold_until_fail_dates: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Static per-symbol results (always True/False)
        self._hold_until_results = hold_until_results or {}
        # Date-based failure: {symbol: "YYYY-MM-DD"} -- returns False on/after this date
        self._hold_until_fail_dates = hold_until_fail_dates or {}

    def check_hold_until(self, symbol: str, as_of, holding) -> bool:
        if symbol in self._hold_until_fail_dates:
            fail_date = date.fromisoformat(self._hold_until_fail_dates[symbol])
            if as_of >= fail_date:
                return False
        return self._hold_until_results.get(symbol, True)


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

    def test_flat_rebalance_day_of_month_is_respected(self, cost_model, ohlcv_dict):
        """Portfolio strategies with flat rebalance_day_of_month should drive rebalance dates."""
        strategy = ConfigurableMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=0.5, reason="test")],
            rebalance_day_of_month=5,
        )
        backtester = PortfolioBacktester(cost_model=cost_model)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 3, 31),
        )

        assert result.status == "completed"
        rebalance_days = [entry["date"] for entry in result.rebalance_log]
        assert rebalance_days[:3] == ["2023-01-05", "2023-02-06", "2023-03-06"]

    # --- Phase 73: weekly rebalance tests ---

    def test_weekly_rebalance_dates(self, cost_model, ohlcv_dict):
        """Weekly rebalance should produce ~13 rebalances over 3 months, mostly on Fridays."""
        strategy = ConfigurableMockStrategy(
            targets=[
                TargetPosition(symbol="2330", weight=0.5, reason="test"),
                TargetPosition(symbol="2317", weight=0.5, reason="test"),
            ],
            rebalance_frequency="weekly",
            rebalance_day_of_week=4,
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=1_000_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 3, 31),
        )

        assert result.status == "completed"
        assert len(result.rebalance_log) >= 12  # ~13 weeks in 3 months
        # Most rebalance dates should be Fridays (weekday 4); some may snap backward to Thu
        friday_count = sum(
            1 for e in result.rebalance_log
            if date.fromisoformat(e["date"]).weekday() == 4
        )
        assert friday_count >= 10, f"Expected >=10 Fridays, got {friday_count}"
        # All rebalance dates must be weekdays
        assert all(
            date.fromisoformat(e["date"]).weekday() in {0, 1, 2, 3, 4}
            for e in result.rebalance_log
        )

    def test_monthly_rebalance_backward_compat(self, cost_model, ohlcv_dict):
        """Monthly rebalance with day_of_month=5 should produce identical dates as before refactor."""
        strategy = ConfigurableMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=0.5, reason="test")],
            rebalance_day_of_month=5,
            rebalance_frequency="monthly",
        )
        backtester = PortfolioBacktester(cost_model=cost_model)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict=ohlcv_dict,
            start_date=date(2023, 1, 2),
            end_date=date(2023, 3, 31),
        )

        assert result.status == "completed"
        rebalance_days = [entry["date"] for entry in result.rebalance_log]
        # Identical to test_flat_rebalance_day_of_month_is_respected
        assert rebalance_days[:3] == ["2023-01-05", "2023-02-06", "2023-03-06"]

    def test_generate_rebalance_dates_weekly_produces_fridays(self, cost_model):
        """Direct unit test: weekly frequency should produce every Friday in range."""
        backtester = PortfolioBacktester(cost_model=cost_model)
        dates = backtester._generate_rebalance_dates(
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 31),
            frequency="weekly",
            day_of_week=4,
        )
        # January 2023 has Fridays on: 6, 13, 20, 27
        assert dates == [date(2023, 1, 6), date(2023, 1, 13), date(2023, 1, 20), date(2023, 1, 27)]

    def test_generate_rebalance_dates_monthly_unchanged(self, cost_model):
        """Direct unit test: monthly frequency produces same dates as old _generate_monthly_dates."""
        backtester = PortfolioBacktester(cost_model=cost_model)
        dates = backtester._generate_rebalance_dates(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 31),
            frequency="monthly",
            day_of_month=15,
        )
        assert dates == [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]

    def test_generate_rebalance_dates_invalid_frequency(self, cost_model):
        """Invalid frequency should raise ValueError."""
        backtester = PortfolioBacktester(cost_model=cost_model)
        with pytest.raises(ValueError, match="Unsupported rebalance frequency"):
            backtester._generate_rebalance_dates(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 3, 31),
                frequency="daily",
            )

    def test_generate_rebalance_dates_weekly_wednesday(self, cost_model):
        """Weekly frequency with day_of_week=2 (Wednesday) should produce Wednesdays."""
        backtester = PortfolioBacktester(cost_model=cost_model)
        dates = backtester._generate_rebalance_dates(
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 31),
            frequency="weekly",
            day_of_week=2,
        )
        # January 2023 has Wednesdays on: 4, 11, 18, 25
        assert dates == [date(2023, 1, 4), date(2023, 1, 11), date(2023, 1, 18), date(2023, 1, 25)]
        assert all(d.weekday() == 2 for d in dates)

    def test_stop_loss_pct_triggers_exit_between_rebalances(self, cost_model):
        """Configured stop loss should liquidate holdings on daily mark-to-market breach."""
        dates = pd.to_datetime(["2023-01-02", "2023-01-05", "2023-01-06", "2023-01-09"])
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 100.0, 87.0, 86.0],
                "high": [101.0, 101.0, 88.0, 87.0],
                "low": [99.0, 99.0, 86.0, 85.0],
                "close": [100.0, 100.0, 87.0, 86.0],
                "volume": [1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        strategy = ConfigurableMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=1.0, reason="test")],
            rebalance_day_of_month=2,
            stop_loss_pct=0.10,
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=100_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 9),
        )

        assert result.status == "completed"
        sell_trades = [trade for trade in result.trades if trade["action"] == "sell"]
        assert len(sell_trades) == 1
        assert sell_trades[0]["date"] == "2023-01-06"

    # --- Phase 72: hold_until tests ---

    def test_hold_until_exit_triggers_sell(self, cost_model):
        """Strategy with check_hold_until returning False should trigger sell with reason=hold_until_exit."""
        dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        strategy = HoldUntilMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=1.0, reason="test")],
            rebalance_day_of_month=2,
            hold_until_fail_dates={"2330": "2023-01-04"},
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=100_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 5),
        )

        assert result.status == "completed"
        hold_until_sells = [t for t in result.trades if t.get("reason") == "hold_until_exit"]
        assert len(hold_until_sells) == 1
        assert hold_until_sells[0]["date"] == "2023-01-04"
        assert hold_until_sells[0]["symbol"] == "2330"

    def test_hold_until_no_exit_when_conditions_met(self, cost_model):
        """Strategy with check_hold_until always True should not trigger hold_until sells."""
        dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        strategy = HoldUntilMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=1.0, reason="test")],
            rebalance_day_of_month=2,
            hold_until_results={"2330": True},
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=100_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 5),
        )

        assert result.status == "completed"
        hold_until_sells = [t for t in result.trades if t.get("reason") == "hold_until_exit"]
        assert len(hold_until_sells) == 0

    def test_hold_until_retention_on_rebalance(self, cost_model):
        """Position still valid per hold_until should be retained even if not in new top-N."""
        # Two symbols, 130+ dates to cover at least 2 rebalance cycles
        ohlcv_2330 = _make_ohlcv(start="2023-01-02", periods=130, base_price=500.0)
        ohlcv_2317 = _make_ohlcv(start="2023-01-02", periods=130, base_price=100.0)

        # Strategy returns different targets on different calls:
        # First rebalance: buy 2330
        # Second rebalance: only 2317 (2330 dropped from top-N)
        call_count = {"n": 0}

        class SwitchingHoldUntilStrategy(HoldUntilMockStrategy):
            def select_stocks(self, universe_df, as_of=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return [TargetPosition(symbol="2330", weight=1.0, reason="test")]
                return [TargetPosition(symbol="2317", weight=1.0, reason="test")]

        strategy = SwitchingHoldUntilStrategy(
            targets=[],  # not used due to override
            rebalance_day_of_month=15,
            hold_until_results={"2330": True, "2317": True},
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=1_000_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv_2330, "2317": ohlcv_2317},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert result.status == "completed"
        # 2330 should NOT be sold at second rebalance (retained via hold_until)
        sell_2330_trades = [
            t for t in result.trades
            if t["symbol"] == "2330" and t["action"] == "sell"
        ]
        assert len(sell_2330_trades) == 0, f"2330 should be retained but was sold: {sell_2330_trades}"

    def test_hold_until_expired_position_sold_on_rebalance(self, cost_model):
        """Position with hold_until=False should be sold when not in new top-N at rebalance."""
        ohlcv_2330 = _make_ohlcv(start="2023-01-02", periods=130, base_price=500.0)
        ohlcv_2317 = _make_ohlcv(start="2023-01-02", periods=130, base_price=100.0)

        call_count = {"n": 0}

        class SwitchingExpiredStrategy(HoldUntilMockStrategy):
            def select_stocks(self, universe_df, as_of=None):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return [TargetPosition(symbol="2330", weight=1.0, reason="test")]
                return [TargetPosition(symbol="2317", weight=1.0, reason="test")]

        strategy = SwitchingExpiredStrategy(
            targets=[],
            rebalance_day_of_month=15,
            hold_until_results={"2330": False, "2317": True},
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=1_000_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv_2330, "2317": ohlcv_2317},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 6, 30),
        )

        assert result.status == "completed"
        # 2330 should be sold (either via hold_until_exit daily check or at rebalance)
        sell_2330_trades = [
            t for t in result.trades
            if t["symbol"] == "2330" and t["action"] == "sell"
        ]
        assert len(sell_2330_trades) >= 1, "2330 should have been sold when hold_until expired"

    def test_no_check_hold_until_backward_compat(self, cost_model):
        """Strategy without check_hold_until should work normally (no hold_until trades)."""
        dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        # ConfigurableMockStrategy does NOT have check_hold_until
        strategy = ConfigurableMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=1.0, reason="test")],
            rebalance_day_of_month=2,
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=100_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 5),
        )

        assert result.status == "completed"
        hold_until_sells = [t for t in result.trades if t.get("reason") == "hold_until_exit"]
        assert len(hold_until_sells) == 0

    def test_stop_loss_priority_over_hold_until(self, cost_model):
        """Stop loss should fire before hold_until; same symbol should not get double-sold."""
        # Price drops below 10% stop on 2023-01-06
        dates = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"])
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 85.0, 84.0],
                "high": [101.0, 101.0, 101.0, 101.0, 86.0, 85.0],
                "low": [99.0, 99.0, 99.0, 99.0, 84.0, 83.0],
                "close": [100.0, 100.0, 100.0, 100.0, 85.0, 84.0],
                "volume": [1000, 1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )
        # hold_until fails on same day as stop_loss would trigger
        strategy = HoldUntilMockStrategy(
            targets=[TargetPosition(symbol="2330", weight=1.0, reason="test")],
            rebalance_day_of_month=2,
            stop_loss_pct=0.10,
            hold_until_fail_dates={"2330": "2023-01-06"},
        )
        backtester = PortfolioBacktester(cost_model=cost_model, initial_capital=100_000.0)

        result = backtester.run(
            strategy=strategy,
            ohlcv_dict={"2330": ohlcv},
            start_date=date(2023, 1, 2),
            end_date=date(2023, 1, 9),
        )

        assert result.status == "completed"
        sell_trades = [t for t in result.trades if t["action"] == "sell" and t["symbol"] == "2330"]
        # Exactly 1 sell trade (not 2 -- stop_loss fires first, hold_until skips since already deleted)
        assert len(sell_trades) == 1
        assert sell_trades[0].get("reason") == "stop_loss"
