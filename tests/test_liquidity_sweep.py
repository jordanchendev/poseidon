"""Tests for LiquiditySweepStrategy -- three-stage maker ambush detection.

Covers SWEEP-02 (three-stage detection), SWEEP-03 (direction modes),
SWEEP-04 (volatility-adaptive distance).
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from poseidon.signals.schemas import OrderType, SignalAction
from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def make_default_config(**overrides) -> dict:
    """Return a config dict matching D-02 structure for LiquiditySweepStrategy."""
    config = {
        "name": "liquidity_sweep_strategy",
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
        "direction_mode": "bidirectional",
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val
    return config


# ---------------------------------------------------------------------------
# Upward sweep data helper
# ---------------------------------------------------------------------------


def _upward_sweep_overrides() -> dict:
    """Override values for the last row to create an upward sweep scenario."""
    return {
        "high": 51100.0,  # above swing_high_24=50600
        "close": 50400.0,  # below swing_high_24 (reversal)
        "wick_ratio_upper": 0.4,
        "wick_ratio_lower": 0.1,
        "breakout_up_24": 0.5,
        "breakout_down_24": 0.0,
        "funding_rate_daily": -0.01,  # negative = crowded short = supports upward sweep
    }


# ---------------------------------------------------------------------------
# TestThreeStageDetection
# ---------------------------------------------------------------------------


class TestThreeStageDetection:
    """SWEEP-02: Three-stage detection -- zone, sweep, ambush."""

    def test_downward_sweep_emits_long_signal(self):
        """Downward sweep with all conditions met -> 1 LONG LIMIT signal."""
        features = make_sweep_features()
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG
        assert signals[0].order_type == OrderType.LIMIT

    def test_upward_sweep_emits_short_signal(self):
        """Upward sweep with all conditions met -> 1 SHORT LIMIT signal."""
        features = make_sweep_features(**_upward_sweep_overrides())
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.SHORT
        assert signals[0].order_type == OrderType.LIMIT

    def test_mandatory_gate_wick_fails(self):
        """Wick ratio below threshold -> 0 signals."""
        features = make_sweep_features(wick_ratio_lower=0.01)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_mandatory_gate_breakout_fails(self):
        """Breakout distance below threshold -> 0 signals."""
        features = make_sweep_features(breakout_down_24=0.0)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_mandatory_gate_reversal_fails(self):
        """Close below swing_low (no reversal for downward sweep) -> 0 signals."""
        # close=49300 is below swing_low_24=49400 so no reversal
        features = make_sweep_features(close=49300.0)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_confirmation_score_below_threshold(self):
        """All gates pass but score is too low -> 0 signals."""
        features = make_sweep_features(
            oi_change_zscore_20=0.0,
            range_expansion_14=0.5,
            funding_rate_daily=0.0,
        )
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_oi_buildup_filter(self):
        """OI buildup below threshold -> 0 signals (zone not identified)."""
        features = make_sweep_features(oi_buildup_24=0.1)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_cooldown_blocks_entry(self):
        """After a signal, simulate position exit, no new signals during cooldown."""
        strategy = LiquiditySweepStrategy(config=make_default_config(
            exit={"cooldown_bars": 4, "max_holding_bars": None},
        ))

        # First call: should emit signal
        features = make_sweep_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1

        # Simulate position exit (reset position, set bars_since_exit = 0)
        strategy._position_direction = None
        strategy._bars_since_exit = 0

        # Next 4 bars should produce no signals (cooldown active)
        for _ in range(4):
            signals = strategy.evaluate(features)
            assert len(signals) == 0

        # 5th bar after exit should allow new entry
        signals = strategy.evaluate(features)
        assert len(signals) == 1

    def test_position_blocks_new_entry(self):
        """While in position, no new entry signals."""
        strategy = LiquiditySweepStrategy(config=make_default_config())
        strategy._position_direction = "long"

        features = make_sweep_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_signal_has_correct_prices(self):
        """Emitted signal has order_price, stop_loss_price, take_profit_price set."""
        features = make_sweep_features()
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.order_price is not None
        assert isinstance(sig.order_price, float)
        assert sig.stop_loss_price is not None
        assert isinstance(sig.stop_loss_price, float)
        assert sig.take_profit_price is not None
        assert isinstance(sig.take_profit_price, float)

    def test_signal_metadata_contains_sweep_info(self):
        """Signal metadata has sweep_type, zone_level, confirmation_score, vol_regime."""
        features = make_sweep_features()
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        meta = signals[0].metadata
        assert "sweep_type" in meta
        assert "zone_level" in meta
        assert "confirmation_score" in meta
        assert "vol_regime" in meta


# ---------------------------------------------------------------------------
# TestDirectionModes
# ---------------------------------------------------------------------------


class TestDirectionModes:
    """SWEEP-03: Direction modes -- long_only, short_only, bidirectional."""

    def test_long_only_skips_upward(self):
        """direction_mode='long_only' with upward sweep data -> 0 signals."""
        features = make_sweep_features(**_upward_sweep_overrides())
        strategy = LiquiditySweepStrategy(
            config=make_default_config(direction_mode="long_only"),
        )
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_long_only_detects_downward(self):
        """direction_mode='long_only' with downward sweep data -> 1 LONG signal."""
        features = make_sweep_features()
        strategy = LiquiditySweepStrategy(
            config=make_default_config(direction_mode="long_only"),
        )
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG

    def test_short_only_skips_downward(self):
        """direction_mode='short_only' with downward sweep data -> 0 signals."""
        features = make_sweep_features()
        strategy = LiquiditySweepStrategy(
            config=make_default_config(direction_mode="short_only"),
        )
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_short_only_detects_upward(self):
        """direction_mode='short_only' with upward sweep data -> 1 SHORT signal."""
        features = make_sweep_features(**_upward_sweep_overrides())
        strategy = LiquiditySweepStrategy(
            config=make_default_config(direction_mode="short_only"),
        )
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.SHORT

    def test_bidirectional_detects_both(self):
        """direction_mode='bidirectional' can detect both downward and upward sweeps."""
        # Test downward sweep
        features_down = make_sweep_features()
        strategy = LiquiditySweepStrategy(
            config=make_default_config(direction_mode="bidirectional"),
        )
        signals_down = strategy.evaluate(features_down)
        assert len(signals_down) == 1
        assert signals_down[0].action == SignalAction.LONG

        # Reset state, test upward sweep
        strategy.reset()
        features_up = make_sweep_features(**_upward_sweep_overrides())
        signals_up = strategy.evaluate(features_up)
        assert len(signals_up) == 1
        assert signals_up[0].action == SignalAction.SHORT


# ---------------------------------------------------------------------------
# TestVolatilityAdaptive
# ---------------------------------------------------------------------------


class TestVolatilityAdaptive:
    """SWEEP-04: Volatility-adaptive entry distance via VolRegime + ATR."""

    def test_low_vol_tighter_distance(self):
        """vol_regime=0, atr_14=500 -> entry distance = 500 * 0.618 * 0.5 = 154.5."""
        features = make_sweep_features(vol_regime=0.0, atr_14=500.0)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        # Long entry: sweep_level (49400) - fib_distance (154.5) = 49245.5
        expected_distance = 500.0 * 0.618 * 0.5  # 154.5
        sweep_level = 49400.0
        expected_entry = sweep_level - expected_distance
        assert abs(sig.order_price - expected_entry) < 0.01

    def test_normal_vol_standard_distance(self):
        """vol_regime=1, atr_14=500 -> entry distance = 500 * 0.618 * 1.0 = 309.0."""
        features = make_sweep_features(vol_regime=1.0, atr_14=500.0)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        expected_distance = 500.0 * 0.618 * 1.0  # 309.0
        sweep_level = 49400.0
        expected_entry = sweep_level - expected_distance
        assert abs(sig.order_price - expected_entry) < 0.01

    def test_high_vol_wider_distance(self):
        """vol_regime=2, atr_14=500 -> entry distance = 500 * 0.618 * 1.5 = 463.5."""
        features = make_sweep_features(vol_regime=2.0, atr_14=500.0)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        expected_distance = 500.0 * 0.618 * 1.5  # 463.5
        sweep_level = 49400.0
        expected_entry = sweep_level - expected_distance
        assert abs(sig.order_price - expected_entry) < 0.01

    def test_extreme_vol_widest_distance(self):
        """vol_regime=3, atr_14=500 -> entry distance = 500 * 0.618 * 2.0 = 618.0."""
        features = make_sweep_features(vol_regime=3.0, atr_14=500.0)
        strategy = LiquiditySweepStrategy(config=make_default_config())
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        expected_distance = 500.0 * 0.618 * 2.0  # 618.0
        sweep_level = 49400.0
        expected_entry = sweep_level - expected_distance
        assert abs(sig.order_price - expected_entry) < 0.01
