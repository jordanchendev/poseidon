"""Tests for dual-mode backtest comparison utility (LIMIT-06).

Covers:
- run_dual_mode_comparison() returns DualModeResult with both results
- delta_metrics contains correct keys and diff values
- is_viable correctness based on pessimistic Sharpe
- Both runs use identical strategy config
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.comparison import DualModeResult, run_dual_mode_comparison
from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.schemas import BacktestResult
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_default_config() -> dict:
    """Return a config dict for LiquiditySweepStrategy."""
    return {
        "name": "dual_mode_test",
        "symbol": "BTCUSDT",
        "market": "crypto_perp",
        "interval": "1h",
        "detection": {
            "lookback_bars": 24,
            "wick_ratio_min": 0.15,
            "breakout_distance_min": 0.1,
            "oi_buildup_min": 1.0,
            "confirmation_threshold": 0.5,
            "w_oi_drop": 0.4,
            "w_volume": 0.3,
            "w_funding": 0.3,
        },
        "entry": {
            "fib_level": 0.618,
            "atr_multipliers": {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0},
        },
        "exit": {
            "cooldown_bars": 4,
            "max_holding_bars": None,
        },
        "trailing": {
            "activation_r": 1.0,
            "atr_multiplier": 2.0,
        },
        "direction_mode": "bidirectional",
    }


def _make_e2e_ohlcv(n_bars: int = 200) -> pd.DataFrame:
    """Create synthetic BTC 1h OHLCV with ALL feature columns pre-populated.

    Engineered so that bar 50 triggers a downward sweep detection (long entry).
    The limit order at ~49091 gets filled when price dips at bars 52-55.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Base OHLCV
    opens = np.full(n_bars, 50000.0)
    highs = np.full(n_bars, 50200.0)
    lows = np.full(n_bars, 49800.0)
    closes = np.full(n_bars, 50100.0)
    volume = np.full(n_bars, 1000.0)

    # Feature columns -- neutral defaults
    swing_high_24 = np.full(n_bars, 50600.0)
    swing_low_24 = np.full(n_bars, 49400.0)
    oi_buildup_24 = np.full(n_bars, 0.5)
    oi_change_zscore_20 = np.full(n_bars, 0.0)
    oi_change_pct = np.full(n_bars, 0.0)
    oiwap_168 = np.full(n_bars, 50000.0)
    oiwap_distance_168 = np.full(n_bars, 0.0)
    breakout_down_24 = np.full(n_bars, 0.0)
    breakout_up_24 = np.full(n_bars, 0.0)
    fib_ext_down = np.full(n_bars, 49000.0)
    fib_ext_up = np.full(n_bars, 51000.0)
    wick_ratio_lower = np.full(n_bars, 0.05)
    wick_ratio_upper = np.full(n_bars, 0.05)
    wick_ratio_total = np.full(n_bars, 0.1)
    range_expansion_14 = np.full(n_bars, 0.8)
    vol_regime = np.full(n_bars, 1.0)
    atr_14 = np.full(n_bars, 500.0)
    volume_ratio_20 = np.full(n_bars, 1.0)
    funding_rate_daily = np.full(n_bars, 0.01)

    # --- Engineer sweep at bar 50 ---
    oi_buildup_24[50] = 5.0
    wick_ratio_lower[50] = 0.4
    breakout_down_24[50] = 0.5
    opens[50] = 49500.0
    highs[50] = 49600.0
    lows[50] = 48800.0
    closes[50] = 49500.0
    oi_change_zscore_20[50] = -2.5
    range_expansion_14[50] = 1.5
    funding_rate_daily[50] = 0.01
    volume[50] = 3000.0

    # --- Bars 51-55: dip to fill limit order at ~49091 ---
    for i in range(51, 56):
        opens[i] = 49300.0 - (i - 51) * 50
        highs[i] = 49400.0
        lows[i] = 49000.0 - (i - 51) * 30
        closes[i] = 49200.0 - (i - 51) * 40
        oi_buildup_24[i] = 0.3

    lows[53] = 49050.0
    lows[54] = 48900.0

    # --- Bars 56-70: bounce back toward TP ---
    for i in range(56, 71):
        base = 49200.0 + (i - 56) * 50
        opens[i] = base
        highs[i] = base + 200.0
        lows[i] = base - 100.0
        closes[i] = base + 100.0
        oi_buildup_24[i] = 0.3

    # --- Bars 71+: stabilize ---
    for i in range(71, n_bars):
        opens[i] = 50000.0
        highs[i] = 50200.0
        lows[i] = 49800.0
        closes[i] = 50100.0
        oi_buildup_24[i] = 0.3

    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "swing_high_24": swing_high_24,
            "swing_low_24": swing_low_24,
            "oi_buildup_24": oi_buildup_24,
            "oi_change_zscore_20": oi_change_zscore_20,
            "oi_change_pct": oi_change_pct,
            "oiwap_168": oiwap_168,
            "oiwap_distance_168": oiwap_distance_168,
            "breakout_down_24": breakout_down_24,
            "breakout_up_24": breakout_up_24,
            "fib_ext_down_0_618": fib_ext_down,
            "fib_ext_up_0_618": fib_ext_up,
            "wick_ratio_lower": wick_ratio_lower,
            "wick_ratio_upper": wick_ratio_upper,
            "wick_ratio_total": wick_ratio_total,
            "range_expansion_14": range_expansion_14,
            "vol_regime": vol_regime,
            "atr_14": atr_14,
            "volume_ratio_20": volume_ratio_20,
            "funding_rate_daily": funding_rate_daily,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDualModeComparisonResults:
    """Test 1: run_dual_mode_comparison returns DualModeResult with both results."""

    def test_returns_dual_mode_result(self):
        """Both optimistic and pessimistic results should be BacktestResult instances."""
        config = _make_default_config()

        def factory():
            return LiquiditySweepStrategy(config=config)

        ohlcv = _make_e2e_ohlcv()

        result = run_dual_mode_comparison(
            strategy_factory=factory,
            ohlcv=ohlcv,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            initial_capital=1_000_000.0,
            sizing_config=SizingConfig(
                mode=SizingMode.FIXED_RISK, risk_pct=0.01, max_notional_pct=0.3,
            ),
            max_pending_bars=10,
            feature_specs=[],
        )

        assert isinstance(result, DualModeResult)
        assert isinstance(result.optimistic_result, BacktestResult)
        assert isinstance(result.pessimistic_result, BacktestResult)
        assert result.optimistic_result.status == "completed"
        assert result.pessimistic_result.status == "completed"


class TestDualModeComparisonDelta:
    """Test 2: delta_metrics contains correct keys and diff values."""

    def test_delta_metrics_keys(self):
        """delta_metrics should have sharpe_ratio, max_drawdown, win_rate, total_pnl."""
        config = _make_default_config()

        def factory():
            return LiquiditySweepStrategy(config=config)

        ohlcv = _make_e2e_ohlcv()

        result = run_dual_mode_comparison(
            strategy_factory=factory,
            ohlcv=ohlcv,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            sizing_config=SizingConfig(
                mode=SizingMode.FIXED_RISK, risk_pct=0.01, max_notional_pct=0.3,
            ),
            max_pending_bars=10,
            feature_specs=[],
        )

        assert "sharpe_ratio" in result.delta_metrics
        assert "max_drawdown" in result.delta_metrics
        assert "win_rate" in result.delta_metrics
        assert "total_pnl" in result.delta_metrics

        # Each delta = optimistic - pessimistic
        for key in ["sharpe_ratio", "max_drawdown", "win_rate", "total_pnl"]:
            opt_val = float(result.optimistic_result.metrics.get(key, 0.0) or 0.0)
            pess_val = float(result.pessimistic_result.metrics.get(key, 0.0) or 0.0)
            expected_delta = opt_val - pess_val
            assert abs(result.delta_metrics[key] - expected_delta) < 1e-6, (
                f"Delta for {key}: expected {expected_delta}, got {result.delta_metrics[key]}"
            )


class TestDualModeViabilityTrue:
    """Test 3: is_viable is True when pessimistic Sharpe > 0."""

    def test_is_viable_when_pessimistic_sharpe_positive(self):
        """If pessimistic Sharpe > 0, is_viable should be True."""
        config = _make_default_config()

        def factory():
            return LiquiditySweepStrategy(config=config)

        ohlcv = _make_e2e_ohlcv()

        result = run_dual_mode_comparison(
            strategy_factory=factory,
            ohlcv=ohlcv,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            sizing_config=SizingConfig(
                mode=SizingMode.FIXED_RISK, risk_pct=0.01, max_notional_pct=0.3,
            ),
            max_pending_bars=10,
            feature_specs=[],
        )

        pess_sharpe = float(result.pessimistic_result.metrics.get("sharpe_ratio", 0.0) or 0.0)
        if pess_sharpe > 0:
            assert result.is_viable is True
        else:
            # If pessimistic Sharpe happens to be <= 0 with this data,
            # verify is_viable is False (correct behavior)
            assert result.is_viable is False


class TestDualModeViabilityFalse:
    """Test 4: is_viable is False when pessimistic Sharpe <= 0."""

    def test_is_viable_false_when_no_trades(self):
        """A strategy that produces no trades should have Sharpe=0 -> is_viable=False."""
        # Use a config that won't trigger any sweeps (very high thresholds)
        config = _make_default_config()
        config["detection"]["confirmation_threshold"] = 999.0  # impossible to meet

        def factory():
            return LiquiditySweepStrategy(config=config)

        ohlcv = _make_e2e_ohlcv()

        result = run_dual_mode_comparison(
            strategy_factory=factory,
            ohlcv=ohlcv,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            sizing_config=SizingConfig(
                mode=SizingMode.FIXED_RISK, risk_pct=0.01, max_notional_pct=0.3,
            ),
            max_pending_bars=10,
            feature_specs=[],
        )

        # With no trades, Sharpe should be 0 -> not viable
        assert result.is_viable is False
        assert result.optimistic_result.trade_count == 0
        assert result.pessimistic_result.trade_count == 0


class TestDualModeIdenticalConfig:
    """Test 5: Both runs use identical strategy config."""

    def test_same_strategy_config(self):
        """Both runs should use the same strategy parameters (only fill model differs)."""
        config = _make_default_config()
        created_strategies = []

        def factory():
            s = LiquiditySweepStrategy(config=config)
            created_strategies.append(s)
            return s

        ohlcv = _make_e2e_ohlcv()

        result = run_dual_mode_comparison(
            strategy_factory=factory,
            ohlcv=ohlcv,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            sizing_config=SizingConfig(
                mode=SizingMode.FIXED_RISK, risk_pct=0.01, max_notional_pct=0.3,
            ),
            max_pending_bars=10,
            feature_specs=[],
        )

        # Factory should have been called twice
        assert len(created_strategies) == 2

        s1, s2 = created_strategies
        # Both strategies should have identical config parameters
        assert s1.symbol == s2.symbol == "BTCUSDT"
        assert s1.market == s2.market == "crypto_perp"
        assert s1._lookback_bars == s2._lookback_bars
        assert s1._fib_level == s2._fib_level
        assert s1._wick_ratio_min == s2._wick_ratio_min
        assert s1._confirmation_threshold == s2._confirmation_threshold
        assert s1._trailing_activation_r == s2._trailing_activation_r
        assert s1._trail_atr_multiplier == s2._trail_atr_multiplier

        # Both backtest results should be from same config
        assert result.optimistic_result.config.symbol == result.pessimistic_result.config.symbol
        assert result.optimistic_result.config.market == result.pessimistic_result.config.market
