"""Tests for funding rate settlement in backtest engine.

Phase 52-02: Validates BacktestResult funding cost fields and
BacktestRunner funding settlement logic at 8h boundaries.

All tests run without DB/GPU dependencies (synthetic data only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType

# ---------------------------------------------------------------------------
# Task 1: BacktestResult schema extension tests
# ---------------------------------------------------------------------------


class TestBacktestResultFundingFields:
    """Verify BacktestResult funding cost fields are present and backward-compatible."""

    def _make_config(self) -> BacktestConfig:
        return BacktestConfig(
            strategy_type="rule",
            symbol="BTCUSDT",
            market="crypto_perp",
        )

    def test_default_funding_costs_total(self):
        """BacktestResult can be instantiated with funding_costs_total=0.0 (default)."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
        )
        assert result.funding_costs_total == 0.0

    def test_explicit_funding_costs_total(self):
        """BacktestResult can be instantiated with funding_costs_total=123.45."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
            funding_costs_total=123.45,
        )
        assert result.funding_costs_total == 123.45

    def test_funding_costs_by_trade(self):
        """BacktestResult can be instantiated with funding_costs_by_trade as list of floats."""
        costs = [10.5, -5.2, 3.1]
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
            funding_costs_by_trade=costs,
        )
        assert result.funding_costs_by_trade == costs

    def test_pnl_with_and_without_funding(self):
        """BacktestResult can be instantiated with pnl_with_funding and pnl_without_funding."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="completed",
            pnl_with_funding=950.0,
            pnl_without_funding=1000.0,
        )
        assert result.pnl_with_funding == 950.0
        assert result.pnl_without_funding == 1000.0

    def test_backward_compatibility(self):
        """Existing BacktestResult instantiation without funding fields still works."""
        result = BacktestResult(
            config=self._make_config(),
            metrics={"total_pnl": 500.0},
            trade_count=3,
            equity_curve_length=100,
            status="completed",
            trades=[{"symbol": "BTCUSDT"}],
        )
        # All funding fields have defaults
        assert result.funding_costs_total == 0.0
        assert result.funding_costs_by_trade == []
        assert result.pnl_with_funding is None
        assert result.pnl_without_funding is None
        # Existing fields still work
        assert result.trade_count == 3
        assert result.metrics["total_pnl"] == 500.0


# ---------------------------------------------------------------------------
# Task 2: BacktestRunner funding settlement integration tests
# ---------------------------------------------------------------------------


class FundingTestStrategy(BaseStrategy):
    """Strategy that opens a LONG market order at bar 5.

    Used to test funding rate settlement on open positions.
    The position stays open for the duration of the backtest.
    """

    name = "funding_test"
    strategy_type = StrategyType.RULE
    symbol = "BTCUSDT"
    market = "crypto_perp"
    interval = "1h"
    supports_backtest = True
    supports_live = False
    bias_risk = []
    stateful = False

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        bar_idx = len(features) - 1
        if bar_idx == 5:
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.LONG,
                    confidence=0.8,
                    order_type=OrderType.MARKET,
                    signal_time=features.index[-1],
                )
            ]
        return []

    def validate_config(self) -> bool:
        return True

    def get_feature_specs(self) -> list:
        return []


class NoTradeStrategy(BaseStrategy):
    """Strategy that never opens any position. Used to test no-position funding."""

    name = "no_trade_test"
    strategy_type = StrategyType.RULE
    symbol = "BTCUSDT"
    market = "crypto_perp"
    interval = "1h"
    supports_backtest = True
    supports_live = False
    bias_risk = []
    stateful = False

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        return []

    def validate_config(self) -> bool:
        return True

    def get_feature_specs(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_funding_ohlcv(n_bars: int = 72) -> pd.DataFrame:
    """Create 72 hours (3 days) of synthetic OHLCV data starting at midnight UTC.

    72 bars = 3 full days, covering 9 funding settlements (00:00, 08:00, 16:00 each day).
    Price hovers around 50000.
    """
    times = pd.date_range(
        "2025-06-01 00:00:00",
        periods=n_bars,
        freq="h",
        tz="UTC",
    )
    rng = np.random.default_rng(42)
    closes = 50000.0 + rng.normal(0, 50, n_bars).cumsum()
    opens = closes - rng.normal(0, 20, n_bars)
    highs = np.maximum(closes, opens) + rng.uniform(10, 100, n_bars)
    lows = np.minimum(closes, opens) - rng.uniform(10, 100, n_bars)
    volumes = rng.uniform(500, 1500, n_bars)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )
    df.index.name = "time"
    return df


def _make_funding_rates(start: str = "2025-06-01", days: int = 3) -> pd.DataFrame:
    """Create synthetic 8h funding rate DataFrame for testing.

    Returns DataFrame indexed by datetime at 00:00, 08:00, 16:00 UTC
    with a "funding_rate" column. Rate is fixed at 0.0001 (0.01%).
    """
    times = []
    for day_offset in range(days):
        for hour in (0, 8, 16):
            dt = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=day_offset, hours=hour)
            times.append(dt)

    df = pd.DataFrame(
        {"funding_rate": [0.0001] * len(times)},
        index=pd.DatetimeIndex(times),
    )
    df.index.name = "time"
    return df


def _make_runner_with_funding(
    strategy: BaseStrategy,
    include_funding: bool = False,
    funding_rates: pd.DataFrame | None = None,
    initial_capital: float = 100_000.0,
) -> BacktestRunner:
    """Create a BacktestRunner with optional funding settlement."""
    cost_model = get_cost_model("crypto_perp")
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine(rules=[])

    return BacktestRunner(
        strategy=strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
        include_funding=include_funding,
        funding_rates=funding_rates,
    )


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


class TestFundingSettlementRunner:
    """Test BacktestRunner funding settlement logic."""

    def test_include_funding_false_no_costs(self):
        """include_funding=False (default) produces zero funding costs."""
        strategy = FundingTestStrategy()
        runner = _make_runner_with_funding(strategy, include_funding=False)
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        assert result.status == "completed"
        assert result.funding_costs_total == 0.0
        assert result.funding_costs_by_trade == []
        assert result.pnl_with_funding is None
        assert result.pnl_without_funding is None

    def test_include_funding_true_deducts_costs(self):
        """include_funding=True with funding_rates deducts costs at 8h boundaries."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        assert result.status == "completed"
        # Position opens at bar 5 (hour 05:00), so settlements at 08:00, 16:00,
        # then day 2 (00:00, 08:00, 16:00), day 3 (00:00, 08:00, 16:00)
        # = 8 settlements with open position
        assert result.funding_costs_total > 0.0
        assert len(result.funding_costs_by_trade) > 0

    def test_funding_cost_formula(self):
        """Funding cost = qty * mark_price * rate for each settlement."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        # Each settlement cost should be: qty * close_price * 0.0001
        # Verify that each entry in funding_costs_by_trade is positive
        # (long position + positive funding rate = cost paid)
        for cost in result.funding_costs_by_trade:
            assert cost > 0.0, "Long position pays positive funding rate"

    def test_settlement_once_per_8h(self):
        """Funding deduction happens exactly once per 8h boundary, not every bar."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        # Position opens at bar 5 (05:00 UTC on day 1).
        # Settlements crossed after position open:
        # Day 1: 08:00, 16:00 (2)
        # Day 2: 00:00, 08:00, 16:00 (3)
        # Day 3: 00:00, 08:00, 16:00 (3)
        # Total = 8 settlement events
        assert len(result.funding_costs_by_trade) == 8

    def test_funding_costs_total_matches_sum(self):
        """funding_costs_total matches sum of abs(funding_costs_by_trade)."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        expected_total = sum(abs(c) for c in result.funding_costs_by_trade)
        assert abs(result.funding_costs_total - expected_total) < 1e-6

    def test_no_deduction_when_no_position(self):
        """No funding deduction when no position is open at settlement time."""
        strategy = NoTradeStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        assert result.funding_costs_total == 0.0
        assert result.funding_costs_by_trade == []

    def test_pnl_with_without_funding_populated(self):
        """pnl_with_funding and pnl_without_funding are populated when include_funding=True."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        assert result.pnl_with_funding is not None
        assert result.pnl_without_funding is not None
        # pnl_with_funding should be less than pnl_without_funding by the funding cost
        assert result.pnl_without_funding >= result.pnl_with_funding

    def test_result_reports_funding_costs_total(self):
        """BacktestResult includes correct funding_costs_total value."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        # Funding rate is 0.0001. Each settlement: qty * close * 0.0001
        # With initial_capital=100000 and notional_pct=0.1, position ~0.2 BTC
        # Each settlement cost ~ 0.2 * 50000 * 0.0001 = 1.0 USDT
        # 8 settlements ~ 8 USDT total (approximate)
        assert 0.1 < result.funding_costs_total < 100.0  # sanity range

    def test_funding_costs_by_trade_has_entries(self):
        """funding_costs_by_trade tracks per-settlement cost entries (not always empty)."""
        strategy = FundingTestStrategy()
        funding_rates = _make_funding_rates()
        runner = _make_runner_with_funding(
            strategy,
            include_funding=True,
            funding_rates=funding_rates,
        )
        ohlcv = _make_funding_ohlcv()
        result = runner.run(ohlcv, feature_specs=[])

        assert len(result.funding_costs_by_trade) > 0
        # Each entry should be a float
        for cost in result.funding_costs_by_trade:
            assert isinstance(cost, float)
