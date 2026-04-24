"""Tests for TrendFollowingStrategy signal logic.

Covers:
- EMA crossover entry (LONG/SHORT)
- ATR trailing stop via HOLD + updated_stop_loss
- Trailing stop only tightens (never loosens)
- Direction reversal (CLOSE + new entry)
- NaN and empty guards
- Feature specs
- Validate config
- InstrumentType.FUTURES
"""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from poseidon.signals.schemas import InstrumentType, SignalAction
from poseidon.strategies.tw_futures.configs import TrendFollowingConfig
from poseidon.strategies.tw_futures.trend_following import TrendFollowingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tx_features(
    n_rows: int = 5,
    close: float = 22000.0,
    ema_fast: float | list[float] = 22100.0,
    ema_slow: float | list[float] = 22000.0,
    atr: float = 100.0,
    freq: str = "1d",
) -> pd.DataFrame:
    """Build a synthetic TX features DataFrame with pre-computed indicators.

    If ema_fast/ema_slow are floats, they are broadcast to all rows.
    If lists, they must have length n_rows.
    """
    idx = pd.date_range("2024-01-01", periods=n_rows, freq=freq, tz="UTC")
    ema_fast_vals = [ema_fast] * n_rows if isinstance(ema_fast, (int, float)) else ema_fast
    ema_slow_vals = [ema_slow] * n_rows if isinstance(ema_slow, (int, float)) else ema_slow

    return pd.DataFrame(
        {
            "open": [close] * n_rows,
            "high": [close + 50] * n_rows,
            "low": [close - 50] * n_rows,
            "close": [close] * n_rows,
            "volume": [50000] * n_rows,
            "ema_20": ema_fast_vals,
            "ema_60": ema_slow_vals,
            "atr_14": [atr] * n_rows,
        },
        index=idx,
    )


def make_strategy(**overrides) -> TrendFollowingStrategy:
    """Create a TrendFollowingStrategy with default config."""
    config = TrendFollowingConfig(**overrides)
    return TrendFollowingStrategy(config=config)


# ---------------------------------------------------------------------------
# Entry Signal Tests
# ---------------------------------------------------------------------------


class TestEntrySignals:
    def test_long_signal_on_ema_crossover(self):
        """EMA(20) > EMA(60) -> LONG signal with stop_loss_price."""
        s = make_strategy()
        df = make_tx_features(ema_fast=22100, ema_slow=22000, close=22000, atr=100)
        signals = s.evaluate(df)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.LONG
        assert sig.stop_loss_price == pytest.approx(22000 - 2.0 * 100)  # close - atr*mult
        assert sig.symbol == "TX"
        assert sig.market == "tw_futures"

    def test_short_signal_on_ema_crossover(self):
        """EMA(20) < EMA(60) -> SHORT signal with stop_loss_price."""
        s = make_strategy()
        df = make_tx_features(ema_fast=21900, ema_slow=22000, close=22000, atr=100)
        signals = s.evaluate(df)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.SHORT
        assert sig.stop_loss_price == pytest.approx(22000 + 2.0 * 100)  # close + atr*mult

    def test_instrument_type_is_futures(self):
        """Signal must have instrument=InstrumentType.FUTURES."""
        s = make_strategy()
        df = make_tx_features(ema_fast=22100, ema_slow=22000)
        signals = s.evaluate(df)

        assert len(signals) == 1
        assert signals[0].instrument == InstrumentType.FUTURES


# ---------------------------------------------------------------------------
# Guard Tests
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_signal_when_empty(self):
        """Empty DataFrame -> empty signal list."""
        s = make_strategy()
        df = pd.DataFrame()
        signals = s.evaluate(df)
        assert signals == []

    def test_no_signal_when_nan(self):
        """Features with NaN EMA/ATR -> empty signal list."""
        s = make_strategy()
        df = make_tx_features()
        df.loc[df.index[-1], "ema_20"] = float("nan")
        signals = s.evaluate(df)
        assert signals == []

    def test_no_signal_all_nan(self):
        """All NaN values -> empty signal list."""
        s = make_strategy()
        df = make_tx_features()
        df["ema_20"] = float("nan")
        df["ema_60"] = float("nan")
        df["atr_14"] = float("nan")
        signals = s.evaluate(df)
        assert signals == []


# ---------------------------------------------------------------------------
# Trailing Stop Tests
# ---------------------------------------------------------------------------


class TestTrailingStop:
    def test_hold_signal_updates_trailing_stop(self):
        """After LONG entry, subsequent evaluate -> HOLD with updated_stop_loss."""
        s = make_strategy()

        # First call: LONG entry
        df1 = make_tx_features(ema_fast=22100, ema_slow=22000, close=22000, atr=100)
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Second call: same trend -> HOLD with trailing stop update
        df2 = make_tx_features(ema_fast=22200, ema_slow=22000, close=22100, atr=100)
        signals2 = s.evaluate(df2)
        assert len(signals2) == 1
        assert signals2[0].action == SignalAction.HOLD
        assert "updated_stop_loss" in signals2[0].metadata
        assert signals2[0].metadata["updated_stop_loss"] == pytest.approx(
            22100 - 2.0 * 100
        )

    def test_trailing_stop_only_tightens_long(self):
        """For long position, trailing stop only moves up, never down."""
        s = make_strategy()

        # LONG entry at close=22000, SL=21800
        df1 = make_tx_features(ema_fast=22100, ema_slow=22000, close=22000, atr=100)
        s.evaluate(df1)
        initial_sl = s._trailing_stop
        assert initial_sl == pytest.approx(21800)

        # Price rises -> SL moves up
        df2 = make_tx_features(ema_fast=22300, ema_slow=22000, close=22200, atr=100)
        signals2 = s.evaluate(df2)
        assert s._trailing_stop == pytest.approx(22000)  # 22200 - 200
        assert s._trailing_stop > initial_sl

        # Price drops -> SL stays (doesn't move down)
        df3 = make_tx_features(ema_fast=22150, ema_slow=22000, close=21900, atr=100)
        signals3 = s.evaluate(df3)
        # new_sl would be 21900-200=21700, but max(21700, 22000) = 22000
        assert s._trailing_stop == pytest.approx(22000)

    def test_trailing_stop_only_tightens_short(self):
        """For short position, trailing stop only moves down, never up."""
        s = make_strategy()

        # SHORT entry at close=22000, SL=22200
        df1 = make_tx_features(ema_fast=21900, ema_slow=22000, close=22000, atr=100)
        s.evaluate(df1)
        initial_sl = s._trailing_stop
        assert initial_sl == pytest.approx(22200)

        # Price drops -> SL moves down
        df2 = make_tx_features(ema_fast=21700, ema_slow=22000, close=21800, atr=100)
        signals2 = s.evaluate(df2)
        assert s._trailing_stop == pytest.approx(22000)  # 21800 + 200
        assert s._trailing_stop < initial_sl

        # Price rises -> SL stays (doesn't move up)
        df3 = make_tx_features(ema_fast=21850, ema_slow=22000, close=22100, atr=100)
        signals3 = s.evaluate(df3)
        # new_sl would be 22100+200=22300, but min(22300, 22000) = 22000
        assert s._trailing_stop == pytest.approx(22000)


# ---------------------------------------------------------------------------
# Direction Reversal Tests
# ---------------------------------------------------------------------------


class TestDirectionReversal:
    def test_close_before_reverse(self):
        """In long position, bearish crossover -> CLOSE + SHORT signals."""
        s = make_strategy()

        # LONG entry
        df1 = make_tx_features(ema_fast=22100, ema_slow=22000, close=22000, atr=100)
        signals1 = s.evaluate(df1)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.LONG

        # Bearish crossover -> should emit CLOSE then SHORT
        df2 = make_tx_features(ema_fast=21900, ema_slow=22000, close=22000, atr=100)
        signals2 = s.evaluate(df2)
        assert len(signals2) == 2
        assert signals2[0].action == SignalAction.CLOSE
        assert signals2[1].action == SignalAction.SHORT

    def test_close_before_reverse_short_to_long(self):
        """In short position, bullish crossover -> CLOSE + LONG signals."""
        s = make_strategy()

        # SHORT entry
        df1 = make_tx_features(ema_fast=21900, ema_slow=22000, close=22000, atr=100)
        signals1 = s.evaluate(df1)
        assert signals1[0].action == SignalAction.SHORT

        # Bullish crossover
        df2 = make_tx_features(ema_fast=22100, ema_slow=22000, close=22000, atr=100)
        signals2 = s.evaluate(df2)
        assert len(signals2) == 2
        assert signals2[0].action == SignalAction.CLOSE
        assert signals2[1].action == SignalAction.LONG


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_validate_config_valid(self):
        """Valid config returns True."""
        s = make_strategy()
        assert s.validate_config() is True

    def test_validate_config_invalid_ema_order(self):
        """ema_fast >= ema_slow raises ValueError."""
        s = make_strategy(ema_fast=60, ema_slow=20)
        with pytest.raises(ValueError, match="ema_slow must be greater"):
            s.validate_config()

    def test_validate_config_invalid_ema_fast_zero(self):
        """ema_fast <= 0 raises ValueError."""
        s = make_strategy(ema_fast=0, ema_slow=60)
        with pytest.raises(ValueError, match="ema_fast must be positive"):
            s.validate_config()

    def test_validate_config_invalid_atr_multiplier(self):
        """atr_multiplier <= 0 raises ValueError."""
        s = make_strategy(atr_multiplier=0.0)
        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            s.validate_config()


# ---------------------------------------------------------------------------
# Feature Specs Test
# ---------------------------------------------------------------------------


class TestFeatureSpecs:
    def test_get_feature_specs(self):
        """Returns correct feature specs for FeatureEngine."""
        s = make_strategy()
        specs = s.get_feature_specs()
        assert ("ema", {"period": 20}) in specs
        assert ("ema", {"period": 60}) in specs
        assert ("atr", {"period": 14}) in specs
        assert len(specs) == 3

    def test_get_feature_specs_custom(self):
        """Custom config produces custom feature specs."""
        s = make_strategy(ema_fast=10, ema_slow=30, atr_period=7)
        specs = s.get_feature_specs()
        assert ("ema", {"period": 10}) in specs
        assert ("ema", {"period": 30}) in specs
        assert ("atr", {"period": 7}) in specs
