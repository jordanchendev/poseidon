"""Tests for VolatilityBreakoutStrategy signal logic.

Covers:
- N-bar breakout above + ATR expansion -> LONG entry
- N-bar breakout below + ATR expansion -> SHORT entry
- No signal when ATR contracting (atr < atr_sma)
- No signal when no breakout
- HOLD signal with trailing stop after entry
- Trailing stop only tightens (long: up only, short: down only)
- Direction reversal (CLOSE + new entry)
- NaN and empty guards
- Feature specs
- Validate config
"""

import pandas as pd
import pytest

from poseidon.signals.schemas import InstrumentType, SignalAction
from poseidon.strategies.tw_futures.configs import VolatilityBreakoutConfig
from poseidon.strategies.tw_futures.volatility_breakout import (
    VolatilityBreakoutStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vb_features(
    n_rows: int = 55,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    atr_vals: list[float] | None = None,
    freq: str = "30min",
) -> pd.DataFrame:
    """Build a synthetic TX features DataFrame for VolatilityBreakout testing.

    Defaults produce enough rows for breakout_period=20 + atr_sma_period=50
    with flat price/ATR (no breakout, no expansion).
    """
    idx = pd.date_range("2024-01-01", periods=n_rows, freq=freq, tz="UTC")
    if highs is None:
        highs = [22000.0] * n_rows
    if lows is None:
        lows = [21800.0] * n_rows
    if closes is None:
        closes = [21950.0] * n_rows
    if atr_vals is None:
        atr_vals = [80.0] * n_rows

    return pd.DataFrame(
        {
            "open": [21950.0] * n_rows,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [50000] * n_rows,
            "atr_14": atr_vals,
        },
        index=idx,
    )


def make_long_breakout_df(n: int = 55) -> pd.DataFrame:
    """DataFrame where last bar breaks above previous N bars' high with ATR expansion."""
    highs = [22000.0] * (n - 1) + [22300.0]
    lows = [21800.0] * (n - 1) + [21900.0]
    closes = [21950.0] * (n - 1) + [22100.0]
    atr_vals = [80.0] * (n - 1) + [120.0]
    return make_vb_features(n_rows=n, highs=highs, lows=lows, closes=closes, atr_vals=atr_vals)


def make_short_breakout_df(n: int = 55) -> pd.DataFrame:
    """DataFrame where last bar breaks below previous N bars' low with ATR expansion."""
    highs = [22000.0] * (n - 1) + [21850.0]
    lows = [21800.0] * (n - 1) + [21600.0]
    closes = [21950.0] * (n - 1) + [21700.0]
    atr_vals = [80.0] * (n - 1) + [120.0]
    return make_vb_features(n_rows=n, highs=highs, lows=lows, closes=closes, atr_vals=atr_vals)


def make_strategy(**overrides) -> VolatilityBreakoutStrategy:
    """Create a VolatilityBreakoutStrategy with default config."""
    config = VolatilityBreakoutConfig(**overrides)
    return VolatilityBreakoutStrategy(config=config)


# ---------------------------------------------------------------------------
# Entry Signal Tests
# ---------------------------------------------------------------------------


class TestEntrySignals:
    def test_long_signal_breakout_above(self):
        """close > highest_high(prev N) AND atr > atr_sma -> LONG."""
        s = make_strategy()
        df = make_long_breakout_df()
        signals = s.evaluate(df)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.LONG
        assert sig.stop_loss_price == pytest.approx(22100 - 1.5 * 120)
        assert sig.symbol == "TX"
        assert sig.market == "tw_futures"
        assert sig.instrument == InstrumentType.FUTURES

    def test_short_signal_breakout_below(self):
        """close < lowest_low(prev N) AND atr > atr_sma -> SHORT."""
        s = make_strategy()
        df = make_short_breakout_df()
        signals = s.evaluate(df)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.SHORT
        assert sig.stop_loss_price == pytest.approx(21700 + 1.5 * 120)


# ---------------------------------------------------------------------------
# No Signal Tests
# ---------------------------------------------------------------------------


class TestNoSignalConditions:
    def test_no_signal_atr_contracting(self):
        """Breakout but atr < atr_sma -> no signal."""
        s = make_strategy()
        n = 55
        # Last bar breaks above but ATR is below SMA (contracting)
        highs = [22000.0] * (n - 1) + [22300.0]
        closes = [21950.0] * (n - 1) + [22100.0]
        atr_vals = [80.0] * (n - 1) + [70.0]  # ATR drops, < SMA of 80
        df = make_vb_features(n_rows=n, highs=highs, closes=closes, atr_vals=atr_vals)
        signals = s.evaluate(df)
        assert signals == []

    def test_no_signal_no_breakout(self):
        """ATR expanding but no price breakout -> no signal."""
        s = make_strategy()
        n = 55
        # ATR jumps but close stays within range
        atr_vals = [80.0] * (n - 1) + [120.0]
        df = make_vb_features(n_rows=n, atr_vals=atr_vals)
        signals = s.evaluate(df)
        assert signals == []


# ---------------------------------------------------------------------------
# Trailing Stop Tests
# ---------------------------------------------------------------------------


class TestTrailingStop:
    def test_hold_signal_trailing_stop(self):
        """After entry, second evaluate -> HOLD with updated_stop_loss."""
        s = make_strategy()

        # First: LONG entry
        df1 = make_long_breakout_df()
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Second: still in range, no breakout opposite -> HOLD with trailing stop
        n = 55
        highs = [22000.0] * (n - 1) + [22300.0]
        closes = [21950.0] * (n - 1) + [22200.0]  # Price moved up
        atr_vals = [80.0] * (n - 1) + [100.0]
        df2 = make_vb_features(n_rows=n, highs=highs, closes=closes, atr_vals=atr_vals)
        signals2 = s.evaluate(df2)

        assert len(signals2) == 1
        assert signals2[0].action == SignalAction.HOLD
        assert "updated_stop_loss" in signals2[0].metadata

    def test_trailing_stop_tightens_long(self):
        """Long position, trailing stop only increases (tightens)."""
        s = make_strategy()

        # Enter LONG
        df1 = make_long_breakout_df()
        s.evaluate(df1)
        initial_sl = s._trailing_stop
        assert initial_sl is not None

        # Price rises -> stop should move up
        n = 55
        highs = [22000.0] * (n - 1) + [22400.0]
        closes = [21950.0] * (n - 1) + [22300.0]
        atr_vals = [80.0] * (n - 1) + [100.0]
        df2 = make_vb_features(n_rows=n, highs=highs, closes=closes, atr_vals=atr_vals)
        s.evaluate(df2)
        higher_sl = s._trailing_stop
        assert higher_sl >= initial_sl

        # Price drops -> stop should NOT move down
        closes_down = [21950.0] * (n - 1) + [22000.0]
        atr_vals_low = [80.0] * (n - 1) + [100.0]
        df3 = make_vb_features(n_rows=n, closes=closes_down, atr_vals=atr_vals_low)
        s.evaluate(df3)
        assert s._trailing_stop >= higher_sl

    def test_trailing_stop_tightens_short(self):
        """Short position, trailing stop only decreases (tightens)."""
        s = make_strategy()

        # Enter SHORT
        df1 = make_short_breakout_df()
        s.evaluate(df1)
        initial_sl = s._trailing_stop
        assert initial_sl is not None

        # Price drops -> stop should move down
        n = 55
        lows = [21800.0] * (n - 1) + [21500.0]
        closes = [21950.0] * (n - 1) + [21600.0]
        atr_vals = [80.0] * (n - 1) + [100.0]
        df2 = make_vb_features(n_rows=n, lows=lows, closes=closes, atr_vals=atr_vals)
        s.evaluate(df2)
        lower_sl = s._trailing_stop
        assert lower_sl <= initial_sl

        # Price rises -> stop should NOT move up
        closes_up = [21950.0] * (n - 1) + [21900.0]
        atr_vals_up = [80.0] * (n - 1) + [100.0]
        df3 = make_vb_features(n_rows=n, closes=closes_up, atr_vals=atr_vals_up)
        s.evaluate(df3)
        assert s._trailing_stop <= lower_sl


# ---------------------------------------------------------------------------
# Direction Reversal Tests
# ---------------------------------------------------------------------------


class TestDirectionReversal:
    def test_close_before_reverse(self):
        """In long, breakout below fires -> CLOSE + SHORT."""
        s = make_strategy()

        # Enter LONG
        df1 = make_long_breakout_df()
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Breakout below with ATR expansion -> CLOSE + SHORT
        df2 = make_short_breakout_df()
        signals2 = s.evaluate(df2)
        assert len(signals2) == 2
        assert signals2[0].action == SignalAction.CLOSE
        assert signals2[1].action == SignalAction.SHORT


# ---------------------------------------------------------------------------
# Guard Tests
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_signal_nan_guard(self):
        """NaN in ATR features -> empty list."""
        s = make_strategy()
        df = make_long_breakout_df()
        df.loc[df.index[-1], "atr_14"] = float("nan")
        signals = s.evaluate(df)
        assert signals == []

    def test_no_signal_empty_df(self):
        """Empty DataFrame -> empty signal list."""
        s = make_strategy()
        df = pd.DataFrame()
        signals = s.evaluate(df)
        assert signals == []


# ---------------------------------------------------------------------------
# Feature Specs and Config Validation
# ---------------------------------------------------------------------------


class TestFeatureSpecs:
    def test_get_feature_specs(self):
        """Returns correct feature specs for FeatureEngine."""
        s = make_strategy()
        specs = s.get_feature_specs()
        assert ("atr", {"period": 14}) in specs
        assert len(specs) == 1


class TestConfigValidation:
    def test_validate_config(self):
        """Valid config returns True."""
        s = make_strategy()
        assert s.validate_config() is True

    def test_validate_config_invalid_breakout_period(self):
        """breakout_period <= 0 raises ValueError."""
        s = make_strategy(breakout_period=0)
        with pytest.raises(ValueError, match="breakout_period must be positive"):
            s.validate_config()

    def test_validate_config_invalid_trail_r(self):
        """trail_r <= 0 raises ValueError."""
        s = make_strategy(trail_r=0.0)
        with pytest.raises(ValueError, match="trail_r must be positive"):
            s.validate_config()
