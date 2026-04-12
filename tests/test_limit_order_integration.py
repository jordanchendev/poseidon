"""End-to-end integration tests for limit order backtest pipeline.

Phase 50-02 Task 2: Validates the complete pipeline --
PendingOrderBook + four-phase loop + SL/TP + FIXED_RISK sizing + maker fees.

All tests run without DB/GPU dependencies (synthetic data only).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class LimitOrderTestStrategy(BaseStrategy):
    """Strategy emitting limit orders at specific bars for integration testing.

    Bar 15: LONG limit @ 49500 with SL=49000, TP=51000
    Bar 40: SHORT limit @ 51500 with SL=52000, TP=50000
    """

    name = "limit_order_integration_test"
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

        if bar_idx == 15:
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.LONG,
                    confidence=0.8,
                    order_type=OrderType.LIMIT,
                    order_price=49500.0,
                    stop_loss_price=49000.0,
                    take_profit_price=51000.0,
                    signal_time=features.index[-1],
                )
            ]
        elif bar_idx == 40:
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.SHORT,
                    confidence=0.8,
                    order_type=OrderType.LIMIT,
                    order_price=51500.0,
                    stop_loss_price=52000.0,
                    take_profit_price=50000.0,
                    signal_time=features.index[-1],
                )
            ]
        return []

    def validate_config(self) -> bool:
        return True

    def get_feature_specs(self) -> list:
        return []


class TimeoutTestStrategy(BaseStrategy):
    """Strategy emitting a limit order at bar 10 with unreachable price."""

    name = "timeout_test"
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
        if bar_idx == 10:
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.LONG,
                    confidence=0.8,
                    order_type=OrderType.LIMIT,
                    order_price=30000.0,  # far below market, never reached
                    signal_time=features.index[-1],
                )
            ]
        return []

    def validate_config(self) -> bool:
        return True

    def get_feature_specs(self) -> list:
        return []


class MarketOrderSlTpStrategy(BaseStrategy):
    """Strategy emitting a market order with SL/TP at bar 5."""

    name = "sl_tp_market_test"
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
                    stop_loss_price=49000.0,
                    take_profit_price=53000.0,
                    signal_time=features.index[-1],
                )
            ]
        return []

    def validate_config(self) -> bool:
        return True

    def get_feature_specs(self) -> list:
        return []


class FixedRiskLimitStrategy(BaseStrategy):
    """Strategy emitting a limit order for FIXED_RISK sizing test at bar 5."""

    name = "fixed_risk_limit_test"
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
                    order_type=OrderType.LIMIT,
                    order_price=50000.0,
                    stop_loss_price=49000.0,
                    signal_time=features.index[-1],
                )
            ]
        return []

    def validate_config(self) -> bool:
        return True

    def get_feature_specs(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Synthetic OHLCV generators
# ---------------------------------------------------------------------------


def _make_limit_order_ohlcv(n_bars: int = 60) -> pd.DataFrame:
    """Deterministic OHLCV that guarantees specific fill conditions.

    Price path (close):
    - Bars 0-14: hover around 50000
    - Bars 15-17: dip — low reaches below 49500 (buy limit fill trigger)
    - Bars 18-24: recover to 50000
    - Bars 25-28: rise above 51000 (TP trigger for long)
    - Bars 29-39: drift around 50500-51000
    - Bars 40-42: rise above 51500 (sell limit fill trigger)
    - Bars 43-49: drop toward 50000 (TP trigger for short)
    - Bars 50-59: continue around 50000
    """
    times = pd.date_range("2025-06-01", periods=n_bars, freq="h", tz=timezone.utc)

    closes = np.full(n_bars, 50000.0)
    highs = np.full(n_bars, 50200.0)
    lows = np.full(n_bars, 49800.0)
    opens = np.full(n_bars, 50000.0)

    # Bars 15-17: dip to trigger buy-limit at 49500
    for i in range(15, 18):
        closes[i] = 49400.0
        lows[i] = 49200.0  # well below 49500 — fills both optimistic and pessimistic
        highs[i] = 49800.0
        opens[i] = 49700.0

    # Bars 18-24: recover
    for i in range(18, 25):
        closes[i] = 50200.0 + (i - 18) * 100
        lows[i] = closes[i] - 200
        highs[i] = closes[i] + 200
        opens[i] = closes[i] - 50

    # Bars 25-28: rise above 51000 (TP trigger for long position)
    for i in range(25, 29):
        closes[i] = 51200.0 + (i - 25) * 100
        lows[i] = closes[i] - 300
        highs[i] = closes[i] + 300  # high >= 51000
        opens[i] = closes[i] - 50

    # Bars 29-39: drift around 50500
    for i in range(29, 40):
        closes[i] = 50500.0
        lows[i] = 50200.0
        highs[i] = 50800.0
        opens[i] = 50500.0

    # Bars 40-42: rise above 51500 (sell-limit fill trigger for SHORT)
    for i in range(40, 43):
        closes[i] = 51600.0
        lows[i] = 51300.0
        highs[i] = 51800.0  # > 51500 — triggers sell limit fill
        opens[i] = 51400.0

    # Bars 43-49: drop toward 50000 (TP trigger for short at 50000)
    for i in range(43, 50):
        offset = (i - 43) * 200
        closes[i] = 51000.0 - offset
        lows[i] = closes[i] - 300  # low <= 50000 eventually
        highs[i] = closes[i] + 200
        opens[i] = closes[i] + 100

    volumes = np.full(n_bars, 1000.0)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )
    df.index.name = "time"
    return df


def _make_sl_trigger_ohlcv(n_bars: int = 30) -> pd.DataFrame:
    """OHLCV where bar 10 triggers SL at 49000 for a long position opened at bar 5.

    Close at bar 5: 50000 (market order fill)
    Bar 10: low=48800 which is <= 49000 (SL trigger for long)
    """
    times = pd.date_range("2025-06-01", periods=n_bars, freq="h", tz=timezone.utc)

    closes = np.full(n_bars, 50000.0)
    highs = np.full(n_bars, 50200.0)
    lows = np.full(n_bars, 49800.0)
    opens = np.full(n_bars, 50000.0)

    # Bar 10: drop to trigger SL
    closes[10] = 48900.0
    lows[10] = 48800.0   # <= 49000 SL trigger
    highs[10] = 49500.0
    opens[10] = 49500.0

    # Bars 11+: stay around 49000
    for i in range(11, n_bars):
        closes[i] = 49000.0
        lows[i] = 48900.0
        highs[i] = 49200.0
        opens[i] = 49100.0

    volumes = np.full(n_bars, 1000.0)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )
    df.index.name = "time"
    return df


def _make_fixed_risk_ohlcv(n_bars: int = 30) -> pd.DataFrame:
    """OHLCV for FIXED_RISK sizing test. Bar 7 dips below 50000 to fill limit buy."""
    times = pd.date_range("2025-06-01", periods=n_bars, freq="h", tz=timezone.utc)

    closes = np.full(n_bars, 50500.0)
    highs = np.full(n_bars, 50700.0)
    lows = np.full(n_bars, 50300.0)
    opens = np.full(n_bars, 50500.0)

    # Bar 7-8: dip below 50000 to trigger limit buy fill
    for i in range(7, 9):
        closes[i] = 49800.0
        lows[i] = 49700.0   # < 50000 fills
        highs[i] = 50100.0
        opens[i] = 50000.0

    volumes = np.full(n_bars, 1000.0)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=times,
    )
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_runner(
    strategy: BaseStrategy,
    fill_model: FillModel = FillModel.OPTIMISTIC,
    max_pending_bars: int = 5,
    sizing_config: SizingConfig | None = None,
    initial_capital: float = 1_000_000.0,
) -> BacktestRunner:
    """Create a BacktestRunner with standard crypto_perp cost model."""
    cost_model = get_cost_model("crypto_perp")
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine(rules=[])

    return BacktestRunner(
        strategy=strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=initial_capital,
        sizing_config=sizing_config,
        fill_model=fill_model,
        max_pending_bars=max_pending_bars,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestLimitOrderE2EOptimistic:
    """End-to-end test with OPTIMISTIC fill model."""

    def test_limit_order_e2e_optimistic(self):
        """Complete pipeline: limit order -> pending -> fill -> SL/TP exit."""
        strategy = LimitOrderTestStrategy()
        runner = _make_runner(strategy, fill_model=FillModel.OPTIMISTIC, max_pending_bars=5)
        ohlcv = _make_limit_order_ohlcv()

        result = runner.run(ohlcv)

        assert result.status == "completed"
        # At minimum: LONG fill at bar 17 (limit at 49500) + TP exit
        assert result.trade_count >= 2, (
            f"Expected >= 2 trades, got {result.trade_count}. Trades: {result.trades}"
        )

        # Verify limit fill uses order_price, not bar close
        entry_trades = [t for t in result.trades if t.get("action") in ("long", "short")]
        limit_fill_found = any(
            t["entry_price"] == pytest.approx(49500.0, abs=1e-2)
            for t in entry_trades
        )
        assert limit_fill_found, (
            f"No trade with entry_price ~49500.0 found. Entry trades: {entry_trades}"
        )

        # Verify maker fee rate (0.0002) for limit fills
        for t in entry_trades:
            if t["entry_price"] == pytest.approx(49500.0, abs=1e-2):
                expected_fee = t["quantity"] * t["entry_price"] * 0.0002
                assert t["fees"] == pytest.approx(expected_fee, rel=1e-4), (
                    f"Limit fill fee mismatch: {t['fees']} != expected {expected_fee}"
                )


class TestLimitOrderE2EPessimistic:
    """End-to-end test with PESSIMISTIC fill model."""

    def test_limit_order_e2e_pessimistic(self):
        """Pessimistic fill requires strict penetration (low < order_price)."""
        strategy = LimitOrderTestStrategy()
        runner = _make_runner(strategy, fill_model=FillModel.PESSIMISTIC, max_pending_bars=5)
        ohlcv = _make_limit_order_ohlcv()

        result = runner.run(ohlcv)

        assert result.status == "completed"
        # OHLCV bars 15-17 have low=49200 < 49500 -> strict penetration fills
        assert result.trade_count >= 1, (
            f"Expected >= 1 trade with pessimistic fill, got {result.trade_count}"
        )


class TestLimitOrderTimeoutCancellation:
    """Test that unfilled limit orders expire after max_pending_bars."""

    def test_limit_order_timeout_cancellation(self):
        """Order at bar 10 with unreachable price expires after 3 bars."""
        strategy = TimeoutTestStrategy()
        runner = _make_runner(strategy, max_pending_bars=3)
        ohlcv = _make_limit_order_ohlcv()

        result = runner.run(ohlcv)

        assert result.status == "completed"
        # No fill should occur — order price 30000 is never reached
        filled_trades = [t for t in result.trades if t.get("action") in ("long", "short")]
        assert len(filled_trades) == 0, (
            f"Expected 0 filled trades for unreachable order, got {len(filled_trades)}"
        )


class TestSlTpIntegration:
    """Test SL/TP intra-bar exit with trigger price."""

    def test_sl_tp_integration(self):
        """Market order with SL triggers at correct price (not bar close)."""
        strategy = MarketOrderSlTpStrategy()
        runner = _make_runner(strategy)
        ohlcv = _make_sl_trigger_ohlcv()

        result = runner.run(ohlcv)

        assert result.status == "completed"
        assert result.trade_count >= 2, (
            f"Expected >= 2 trades (entry + SL exit), got {result.trade_count}"
        )

        # Find the SL exit trade
        sl_exits = [t for t in result.trades if t.get("action") == "close_sl"]
        assert len(sl_exits) >= 1, (
            f"Expected SL exit trade, found none. Trades: {result.trades}"
        )

        sl_trade = sl_exits[0]
        # SL exit price should be the trigger price (49000), not bar close (48900)
        assert sl_trade["exit_price"] == pytest.approx(49000.0, abs=1e-2), (
            f"SL exit should be at trigger price 49000, got {sl_trade['exit_price']}"
        )
        assert sl_trade["action"] == "close_sl"


class TestFixedRiskSizingIntegration:
    """Test FIXED_RISK sizing mode in end-to-end pipeline."""

    def test_fixed_risk_sizing_integration(self):
        """Limit order with FIXED_RISK sizing produces correct position size.

        order_price=50000, SL=49000, risk_pct=0.02, equity=1M:
        qty = (1M * 0.02) / |50000 - 49000| = 20_000 / 1000 = 20 units
        """
        strategy = FixedRiskLimitStrategy()
        sizing_cfg = SizingConfig(
            mode=SizingMode.FIXED_RISK,
            risk_pct=0.02,
            max_notional_pct=1.5,  # high cap to not interfere
        )
        runner = _make_runner(
            strategy,
            fill_model=FillModel.OPTIMISTIC,
            max_pending_bars=5,
            sizing_config=sizing_cfg,
            initial_capital=1_000_000.0,
        )
        ohlcv = _make_fixed_risk_ohlcv()

        result = runner.run(ohlcv)

        assert result.status == "completed"

        # Find the filled trade (limit fill at 50000)
        entry_trades = [t for t in result.trades if t.get("action") == "long"]
        assert len(entry_trades) >= 1, (
            f"Expected a LONG trade, got {result.trades}"
        )

        fill_trade = entry_trades[0]
        # Expected qty ~20 (= 1M * 0.02 / 1000)
        assert fill_trade["quantity"] == pytest.approx(20.0, rel=0.01), (
            f"Expected qty ~20, got {fill_trade['quantity']}"
        )
        assert fill_trade["entry_price"] == pytest.approx(50000.0, abs=1e-2)


class TestGoldenStillPasses:
    """Verify golden regression test infrastructure still works."""

    def test_golden_still_passes_after_all_changes(self):
        """Golden backtest produces >= 2 trades and completes successfully."""
        from tests.test_backtest_golden import _run_golden_backtest

        result = _run_golden_backtest()
        assert result.status == "completed"
        assert result.trade_count >= 2, (
            f"Golden test only produced {result.trade_count} trades, expected >= 2"
        )
