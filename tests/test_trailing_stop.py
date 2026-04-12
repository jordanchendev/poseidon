"""Tests for LiquiditySweepStrategy trailing stop logic (ADV-01).

Covers:
- Trailing stop activation threshold (1R unrealized profit)
- ATR-based trailing for long and short positions
- SL only tightens, never loosens
- reset() clears all trailing stop state
- Configurable trailing_activation_r and trail_atr_multiplier
- Strategy metadata includes updated_stop_loss key
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from poseidon.signals.schemas import OrderType, SignalAction
from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_trailing_config(**overrides) -> dict:
    """Return a config dict for LiquiditySweepStrategy with trailing stop."""
    config = {
        "name": "trailing_test",
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
    for key, val in overrides.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val
    return config


def make_sweep_features(n_rows: int = 200, **overrides) -> pd.DataFrame:
    """Create synthetic feature DataFrame with all liquidity-sweep columns.

    Default values represent a valid downward sweep scenario (long trigger).
    Override the LAST row only via **overrides.
    """
    idx = pd.date_range("2025-01-01", periods=n_rows, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            # OHLCV
            "open": [50000.0] * n_rows,
            "high": [50500.0] * n_rows,
            "low": [49500.0] * n_rows,
            "close": [50200.0] * n_rows,
            "volume": [1000.0] * n_rows,
            # Swing
            "swing_high_24": [50600.0] * n_rows,
            "swing_low_24": [49400.0] * n_rows,
            # OI
            "oi_buildup_24": [5.0] * n_rows,
            "oi_change_zscore_20": [-2.5] * n_rows,
            "oi_change_pct": [-0.05] * n_rows,
            "oiwap_168": [50000.0] * n_rows,
            "oiwap_distance_168": [-2.0] * n_rows,
            # Breakout
            "breakout_down_24": [0.5] * n_rows,
            "breakout_up_24": [0.0] * n_rows,
            # Fib
            "fib_ext_down_0_618": [49000.0] * n_rows,
            "fib_ext_up_0_618": [51000.0] * n_rows,
            # Wick
            "wick_ratio_lower": [0.4] * n_rows,
            "wick_ratio_upper": [0.1] * n_rows,
            "wick_ratio_total": [0.5] * n_rows,
            # Range
            "range_expansion_14": [1.5] * n_rows,
            # Vol
            "vol_regime": [1.0] * n_rows,
            "atr_14": [500.0] * n_rows,
            # Volume
            "volume_ratio_20": [2.0] * n_rows,
            # Funding
            "funding_rate_daily": [0.01] * n_rows,
        },
        index=idx,
    )
    for col, val in overrides.items():
        df.loc[df.index[-1], col] = val
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrailingStopNoActivation:
    """Test 1: trailing stop does NOT activate when unrealized profit < 1R."""

    def test_trailing_stop_inactive_below_1r(self):
        """When price has moved less than 1R, SL should remain fixed."""
        config = make_trailing_config()
        strategy = LiquiditySweepStrategy(config=config)

        # First, trigger an entry by evaluating a sweep scenario
        features = make_sweep_features(n_rows=200)
        signals = strategy.evaluate(features)

        # Should have an entry signal
        assert len(signals) == 1
        entry_signal = signals[0]
        assert entry_signal.action == SignalAction.LONG

        entry_price = entry_signal.order_price
        stop_loss = entry_signal.stop_loss_price
        one_r = abs(entry_price - stop_loss)

        # Simulate being in position -- manually set entry context
        # (evaluate() should set this internally when signal is emitted)
        assert strategy._entry_price is not None
        assert strategy._initial_stop_loss is not None
        assert strategy._trailing_active is False

        # Now simulate next bar with close < entry + 1R (not enough profit)
        close_below_1r = entry_price + one_r * 0.5  # only 0.5R profit
        features2 = make_sweep_features(n_rows=201, close=close_below_1r)
        signals2 = strategy.evaluate(features2)

        # Should get empty or HOLD, trailing should NOT be active
        assert strategy._trailing_active is False
        # SL should remain unchanged
        assert strategy._current_stop_loss == stop_loss


class TestTrailingStopLongActivation:
    """Test 2: trailing stop activates at >= 1R for long position."""

    def test_trailing_stop_activates_long(self):
        """When unrealized profit >= 1R on long, trailing SL moves up."""
        # Use a smaller trail_atr_multiplier so trailing SL can actually tighten
        # above the initial SL with reasonable price moves.
        config = make_trailing_config(trailing={
            "activation_r": 1.0,
            "atr_multiplier": 0.5,  # tight trailing
        })
        strategy = LiquiditySweepStrategy(config=config)

        # Trigger entry
        features = make_sweep_features(n_rows=200)
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        entry_signal = signals[0]
        entry_price = entry_signal.order_price
        stop_loss = entry_signal.stop_loss_price
        one_r = abs(entry_price - stop_loss)

        # Next bar: close far above entry (3R profit) so trail_sl > initial_sl
        # trail_sl = hwm - atr * 0.5 = close - 250
        # With entry ~49091, stop ~48782, close at 49091 + 309*3 = 50018
        # trail_sl = 50018 - 250 = 49768, which is > 48782
        close_high = entry_price + one_r * 3.0
        atr = 500.0
        features2 = make_sweep_features(n_rows=201, close=close_high, atr_14=atr)
        signals2 = strategy.evaluate(features2)

        # Trailing should be active
        assert strategy._trailing_active is True
        # New SL should be tighter than initial
        assert strategy._current_stop_loss > stop_loss

        # Should have a HOLD signal with updated_stop_loss
        assert len(signals2) > 0
        assert signals2[0].action == SignalAction.HOLD


class TestTrailingStopShortActivation:
    """Test 3: trailing stop activates at >= 1R for short position."""

    def test_trailing_stop_activates_short(self):
        """When unrealized profit >= 1R on short, trailing SL moves down."""
        # Use tight trailing so SL can tighten with reasonable price moves
        config = make_trailing_config(
            direction_mode="short_only",
            trailing={"activation_r": 1.0, "atr_multiplier": 0.5},
        )
        strategy = LiquiditySweepStrategy(config=config)

        # Create upward sweep scenario for short entry
        upward_overrides = {
            "wick_ratio_upper": 0.4,
            "wick_ratio_lower": 0.1,
            "breakout_up_24": 0.5,
            "breakout_down_24": 0.0,
            "close": 50200.0,  # below swing_high
            "funding_rate_daily": -0.01,  # negative = crowded short = supports upward sweep
        }
        features = make_sweep_features(n_rows=200, **upward_overrides)
        signals = strategy.evaluate(features)

        assert len(signals) == 1
        entry_signal = signals[0]
        assert entry_signal.action == SignalAction.SHORT
        entry_price = entry_signal.order_price
        stop_loss = entry_signal.stop_loss_price
        one_r = abs(entry_price - stop_loss)

        # Next bar: close drops enough for 3R profit on short
        # trail_sl = lwm + atr * 0.5 = close + 250
        # With entry ~50909, stop ~51218, one_r=309
        # close at 50909 - 309*3 = 49982; trail_sl = 49982 + 250 = 50232
        # 50232 < 51218 = stop_loss, so it tightens
        close_below_entry = entry_price - one_r * 3.0
        atr = 500.0
        features2 = make_sweep_features(
            n_rows=201, close=close_below_entry, atr_14=atr,
            wick_ratio_upper=0.4, wick_ratio_lower=0.1,
            breakout_up_24=0.5, breakout_down_24=0.0,
            funding_rate_daily=-0.01,
        )
        signals2 = strategy.evaluate(features2)

        assert strategy._trailing_active is True
        # For short: new SL should be lower (tighter) than initial
        assert strategy._current_stop_loss < stop_loss


class TestTrailingStopOnlyTightens:
    """Test 4: trailing SL only tightens -- never loosens after activation."""

    def test_trailing_sl_does_not_loosen_on_retrace(self):
        """After trailing activates and SL moves up, retrace should NOT move SL back down."""
        config = make_trailing_config(trailing={
            "activation_r": 1.0,
            "atr_multiplier": 0.5,
        })
        strategy = LiquiditySweepStrategy(config=config)

        # Trigger long entry
        features = make_sweep_features(n_rows=200)
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        entry_price = signals[0].order_price
        stop_loss = signals[0].stop_loss_price
        one_r = abs(entry_price - stop_loss)

        # Bar 2: price rises to 3R -- activates trailing with tight multiplier
        close_high = entry_price + one_r * 3.0
        features2 = make_sweep_features(n_rows=201, close=close_high, atr_14=500.0)
        strategy.evaluate(features2)
        assert strategy._trailing_active is True
        sl_after_rise = strategy._current_stop_loss
        assert sl_after_rise > stop_loss  # trailing actually tightened

        # Bar 3: price retraces (still above SL but lower than peak)
        close_retrace = entry_price + one_r * 1.5
        features3 = make_sweep_features(n_rows=202, close=close_retrace, atr_14=500.0)
        strategy.evaluate(features3)

        # SL should NOT have loosened
        assert strategy._current_stop_loss >= sl_after_rise


class TestTrailingStopReset:
    """Test 5: reset() clears all trailing stop state."""

    def test_reset_clears_trailing_state(self):
        """After reset(), all trailing stop state should be None/False."""
        config = make_trailing_config()
        strategy = LiquiditySweepStrategy(config=config)

        # Simulate some state
        strategy._trailing_active = True
        strategy._position_high_watermark = 55000.0
        strategy._position_low_watermark = 45000.0
        strategy._entry_price = 50000.0
        strategy._initial_stop_loss = 49000.0
        strategy._current_stop_loss = 49500.0

        strategy.reset()

        assert strategy._trailing_active is False
        assert strategy._position_high_watermark is None
        assert strategy._position_low_watermark is None
        assert strategy._entry_price is None
        assert strategy._initial_stop_loss is None
        assert strategy._current_stop_loss is None


class TestTrailingStopConfig:
    """Test 6: configurable trailing_activation_r and trail_atr_multiplier."""

    def test_custom_trailing_config(self):
        """Custom activation_r and atr_multiplier should be used."""
        config = make_trailing_config(trailing={
            "activation_r": 2.0,
            "atr_multiplier": 3.0,
        })
        strategy = LiquiditySweepStrategy(config=config)

        assert strategy._trailing_activation_r == 2.0
        assert strategy._trail_atr_multiplier == 3.0

    def test_default_trailing_config(self):
        """Default trailing config when not specified."""
        config = make_trailing_config()
        # Remove trailing section
        del config["trailing"]
        strategy = LiquiditySweepStrategy(config=config)

        assert strategy._trailing_activation_r == 1.0
        assert strategy._trail_atr_multiplier == 2.0


class TestTrailingStopMetadata:
    """Test 7: strategy metadata includes updated_stop_loss key when trailing SL changes."""

    def test_hold_signal_has_updated_stop_loss(self):
        """When trailing SL changes, the HOLD signal metadata should have updated_stop_loss."""
        config = make_trailing_config(trailing={
            "activation_r": 1.0,
            "atr_multiplier": 0.5,
        })
        strategy = LiquiditySweepStrategy(config=config)

        # Trigger entry
        features = make_sweep_features(n_rows=200)
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        entry_price = signals[0].order_price
        stop_loss = signals[0].stop_loss_price
        one_r = abs(entry_price - stop_loss)

        # Next bar: price at 3R -- trailing activates with tight multiplier
        close_high = entry_price + one_r * 3.0
        features2 = make_sweep_features(n_rows=201, close=close_high, atr_14=500.0)
        signals2 = strategy.evaluate(features2)

        # Should have a signal with updated_stop_loss in metadata
        assert len(signals2) > 0
        hold_signal = signals2[0]
        assert hold_signal.action == SignalAction.HOLD
        assert "updated_stop_loss" in hold_signal.metadata
        assert hold_signal.metadata["updated_stop_loss"] > stop_loss
