"""Tests for MeanReversionStrategy signal logic.

Covers:
- BB lower + RSI oversold -> LONG entry
- BB upper + RSI overbought -> SHORT entry
- BB middle return -> CLOSE exit
- No signal when only one condition met (BB or RSI alone)
- No double-entry when already in position
- Direction reversal (CLOSE + new entry)
- NaN and empty guards
- Feature specs
- Validate config
"""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from poseidon.signals.schemas import InstrumentType, SignalAction
from poseidon.strategies.tw_futures.configs import MeanReversionConfig
from poseidon.strategies.tw_futures.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mr_features(
    n_rows: int = 5,
    close: float = 22000.0,
    bb_upper: float = 22200.0,
    bb_middle: float = 22000.0,
    bb_lower: float = 21800.0,
    rsi: float = 50.0,
    freq: str = "1h",
) -> pd.DataFrame:
    """Build a synthetic TX features DataFrame for MeanReversion testing."""
    idx = pd.date_range("2024-01-01", periods=n_rows, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": [close] * n_rows,
            "high": [close + 50] * n_rows,
            "low": [close - 50] * n_rows,
            "close": [close] * n_rows,
            "volume": [50000] * n_rows,
            "bb_upper_20": [bb_upper] * n_rows,
            "bb_middle_20": [bb_middle] * n_rows,
            "bb_lower_20": [bb_lower] * n_rows,
            "rsi_14": [rsi] * n_rows,
        },
        index=idx,
    )


def make_strategy(**overrides) -> MeanReversionStrategy:
    """Create a MeanReversionStrategy with default config."""
    config = MeanReversionConfig(**overrides)
    return MeanReversionStrategy(config=config)


# ---------------------------------------------------------------------------
# Entry Signal Tests
# ---------------------------------------------------------------------------


class TestEntrySignals:
    def test_long_signal_bb_lower_rsi_oversold(self):
        """close < bb_lower AND rsi < 30 -> LONG signal."""
        s = make_strategy()
        df = make_mr_features(close=21750, bb_lower=21800, rsi=25)
        signals = s.evaluate(df)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.LONG
        assert sig.symbol == "TX"
        assert sig.market == "tw_futures"
        assert sig.instrument == InstrumentType.FUTURES

    def test_short_signal_bb_upper_rsi_overbought(self):
        """close > bb_upper AND rsi > 70 -> SHORT signal."""
        s = make_strategy()
        df = make_mr_features(close=22250, bb_upper=22200, rsi=75)
        signals = s.evaluate(df)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.SHORT


# ---------------------------------------------------------------------------
# No Signal Tests (single condition not enough)
# ---------------------------------------------------------------------------


class TestNoSignalConditions:
    def test_no_signal_rsi_normal(self):
        """close < bb_lower but rsi=50 (normal) -> no signal."""
        s = make_strategy()
        df = make_mr_features(close=21750, bb_lower=21800, rsi=50)
        signals = s.evaluate(df)
        assert signals == []

    def test_no_signal_bb_inside(self):
        """rsi < 30 but close within bands -> no signal."""
        s = make_strategy()
        df = make_mr_features(close=22000, bb_lower=21800, bb_upper=22200, rsi=25)
        signals = s.evaluate(df)
        assert signals == []


# ---------------------------------------------------------------------------
# Exit Tests (BB middle return)
# ---------------------------------------------------------------------------


class TestExitSignals:
    def test_exit_long_at_bb_middle(self):
        """In long position, close >= bb_middle -> CLOSE."""
        s = make_strategy()

        # First: enter long
        df1 = make_mr_features(close=21750, bb_lower=21800, rsi=25)
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Second: price returns to middle band
        df2 = make_mr_features(close=22000, bb_middle=22000, bb_lower=21800, rsi=40)
        signals2 = s.evaluate(df2)
        assert len(signals2) == 1
        assert signals2[0].action == SignalAction.CLOSE

    def test_exit_short_at_bb_middle(self):
        """In short position, close <= bb_middle -> CLOSE."""
        s = make_strategy()

        # First: enter short
        df1 = make_mr_features(close=22250, bb_upper=22200, rsi=75)
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.SHORT

        # Second: price returns to middle band
        df2 = make_mr_features(close=22000, bb_middle=22000, bb_upper=22200, rsi=60)
        signals2 = s.evaluate(df2)
        assert len(signals2) == 1
        assert signals2[0].action == SignalAction.CLOSE


# ---------------------------------------------------------------------------
# Double Entry Prevention
# ---------------------------------------------------------------------------


class TestDoubleEntry:
    def test_no_double_entry(self):
        """Already long, still oversold -> no new signal."""
        s = make_strategy()

        # Enter long
        df1 = make_mr_features(close=21750, bb_lower=21800, rsi=25)
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Still oversold but already long -> no signal
        df2 = make_mr_features(close=21700, bb_lower=21800, rsi=20)
        signals2 = s.evaluate(df2)
        assert signals2 == []


# ---------------------------------------------------------------------------
# Direction Reversal Tests
# ---------------------------------------------------------------------------


class TestDirectionReversal:
    def test_close_before_reverse(self):
        """In long, overbought signal fires -> CLOSE + SHORT."""
        s = make_strategy()

        # Enter long
        df1 = make_mr_features(close=21750, bb_lower=21800, rsi=25)
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Overbought reversal (skip exit because close > bb_upper, not >= bb_middle)
        # Set bb_middle high so exit condition doesn't trigger first
        df2 = make_mr_features(
            close=22250, bb_upper=22200, bb_middle=22300, bb_lower=21800, rsi=75
        )
        signals2 = s.evaluate(df2)
        assert len(signals2) == 2
        assert signals2[0].action == SignalAction.CLOSE
        assert signals2[1].action == SignalAction.SHORT


# ---------------------------------------------------------------------------
# Guard Tests
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_signal_nan_guard(self):
        """NaN in features -> empty list."""
        s = make_strategy()
        df = make_mr_features(close=21750, bb_lower=21800, rsi=25)
        df.loc[df.index[-1], "rsi_14"] = float("nan")
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
        assert ("bollinger", {"period": 20, "num_std": 2.0}) in specs
        assert ("rsi", {"period": 14}) in specs
        assert len(specs) == 2


class TestConfigValidation:
    def test_validate_config(self):
        """Valid config returns True."""
        s = make_strategy()
        assert s.validate_config() is True

    def test_validate_config_invalid_rsi_order(self):
        """rsi_oversold >= rsi_overbought raises ValueError."""
        s = make_strategy(rsi_oversold=70.0, rsi_overbought=30.0)
        with pytest.raises(ValueError, match="rsi_oversold"):
            s.validate_config()
