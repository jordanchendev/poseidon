"""Tests for VotingStrategy — vote counting, ATR trailing stop, position sizing."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from poseidon.signals.schemas import SignalAction
from poseidon.strategies.voting_strategy import VotingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_features(n_rows=200, **overrides):
    """Create synthetic feature DataFrame with overridable column values at last row."""
    df = pd.DataFrame({
        "close": [100.0] * n_rows,
        "ema_7": [100.0] * n_rows,
        "ema_26": [99.0] * n_rows,
        "rsi_8": [55.0] * n_rows,
        "macd_histogram": [0.5] * n_rows,
        "bb_upper_20": [105.0] * n_rows,
        "bb_lower_20": [95.0] * n_rows,
        "atr_14": [2.0] * n_rows,
        "cum_return_6d": [0.01] * n_rows,
        "cum_return_12d": [0.02] * n_rows,
    })
    for col, val in overrides.items():
        df.loc[df.index[-1], col] = val
    return df


def _make_all_true_config(min_votes: int = 4) -> dict:
    """Config where all 6 sub-signals fire on default make_features() data."""
    return {
        "name": "test_voting",
        "symbol": "BTCUSDT",
        "market": "crypto_spot",
        "interval": "1h",
        "min_votes": min_votes,
        "position_pct": 0.08,
        "sub_signals": [
            {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 6}, "threshold": 0},
            {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 12}, "threshold": 0},
            {"type": "indicator_comparison", "indicator_a": "ema", "indicator_b": "ema",
             "params": {"period_a": 7, "period_b": 26}, "direction": "above"},
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 8}, "threshold": 50},
            {"type": "indicator_above", "indicator": "macd_histogram", "params": {}, "threshold": 0},
            {"type": "bollinger_width_percentile", "params": {"period": 20, "lookback": 168}, "threshold": 0.2},
        ],
    }


# ---------------------------------------------------------------------------
# TestVotingStrategy — core vote counting
# ---------------------------------------------------------------------------

class TestVotingStrategy:
    """Core VotingStrategy tests — vote counting, signal emission."""

    def test_4_of_6_true_emits_long(self):
        """4 of 6 conditions true, min_votes=4 -> emits 1 LONG signal."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        # Make 2 conditions false: rsi < 50, macd < 0
        features = make_features(rsi_8=45.0, macd_histogram=-0.1)
        # That leaves 4 true: cum_return_6d, cum_return_12d, ema comparison, bb squeeze
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG

    def test_3_of_6_true_emits_nothing(self):
        """3 of 6 conditions true, min_votes=4 -> emits 0 signals."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        # Make 3 conditions false
        features = make_features(rsi_8=45.0, macd_histogram=-0.1, cum_return_6d=-0.01)
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_already_in_position_no_reentry(self):
        """Already in position, 5 of 6 conditions true -> emits 0 signals."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        features = make_features(rsi_8=45.0)  # 5 of 6 true

        # First evaluate enters position
        signals1 = strategy.evaluate(features)
        assert len(signals1) == 1

        # Second evaluate — already in position, no re-entry
        signals2 = strategy.evaluate(features)
        assert len(signals2) == 0

    def test_confidence_equals_vote_ratio(self):
        """Signal confidence = vote_count / total_sub_signals."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        # 5 of 6 true (rsi false)
        features = make_features(rsi_8=45.0)
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        # 5/6 = 0.8333...
        assert abs(signals[0].confidence - 5 / 6) < 1e-6

    def test_quantity_pct_is_008(self):
        """Signal quantity_pct = 0.08 (fixed 8% position sizing)."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        features = make_features()  # all 6 true
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].quantity_pct == 0.08

    def test_signal_metadata_from_config(self):
        """Signal has correct symbol, market, interval from config."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        features = make_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].symbol == "BTCUSDT"
        assert signals[0].market == "crypto_spot"
        assert signals[0].interval == "1h"

    def test_validate_config_valid(self):
        """validate_config() returns True for valid config."""
        config = _make_all_true_config()
        strategy = VotingStrategy(config=config)
        assert strategy.validate_config() is True

    def test_validate_config_empty_sub_signals_raises(self):
        """validate_config() raises ValueError for empty sub_signals."""
        config = _make_all_true_config()
        config["sub_signals"] = []
        strategy = VotingStrategy(config=config)
        with pytest.raises(ValueError, match="at least one sub-signal"):
            strategy.validate_config()


# ---------------------------------------------------------------------------
# TestATRTrailingStop
# ---------------------------------------------------------------------------

class TestATRTrailingStop:
    """ATR trailing stop exit tests."""

    def test_price_above_stop_no_close(self):
        """Enter at 100, rises to 110, atr=2, mult=2 -> stop=106, price=107 -> no close."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter position
        features_entry = make_features()
        signals_entry = strategy.evaluate(features_entry)
        assert len(signals_entry) == 1
        assert signals_entry[0].action == SignalAction.LONG

        # Price rises to 110 — still above stop
        features_up = make_features(close=110.0, atr_14=2.0)
        signals_up = strategy.evaluate(features_up)
        # No close signal (107 not tested here, just 110 > stop)
        close_signals = [s for s in signals_up if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 0

    def test_price_drops_below_stop_emits_close(self):
        """Enter at 100, price rises to 110, atr=2, mult=2, then price drops to 105 -> CLOSE."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter position (price=100)
        features_entry = make_features()
        strategy.evaluate(features_entry)

        # Price rises to 110 (update high watermark)
        features_up = make_features(close=110.0, atr_14=2.0)
        strategy.evaluate(features_up)
        # hwm should now be 110

        # Price drops to 105, atr=2.0 -> stop = 110 - 2*2 = 106 -> 105 < 106 -> CLOSE
        features_down = make_features(close=105.0, atr_14=2.0)
        signals_down = strategy.evaluate(features_down)
        close_signals = [s for s in signals_down if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].quantity_pct is None

    def test_close_resets_state(self):
        """After CLOSE signal, in_position=False, hwm=None."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter
        strategy.evaluate(make_features())
        # Trigger stop
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        strategy.evaluate(make_features(close=105.0, atr_14=2.0))

        assert strategy._in_position is False
        assert strategy._position_high_watermark is None

    def test_reentry_after_close(self):
        """After CLOSE, new vote threshold met -> can re-enter (new LONG)."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter
        strategy.evaluate(make_features())
        # Trigger stop
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        strategy.evaluate(make_features(close=105.0, atr_14=2.0))

        # Re-enter (all conditions true again)
        signals_reentry = strategy.evaluate(make_features())
        long_signals = [s for s in signals_reentry if s.action == SignalAction.LONG]
        assert len(long_signals) == 1

    def test_trailing_stop_before_vote_counting(self):
        """Trailing stop evaluated BEFORE vote counting — close signal takes priority."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter
        strategy.evaluate(make_features())
        # Push hwm up
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))

        # Price drops below stop, but all vote conditions are true
        # Should emit CLOSE only (not LONG), proving stop checked first
        features_drop = make_features(close=105.0, atr_14=2.0)
        signals = strategy.evaluate(features_drop)
        actions = [s.action for s in signals]
        assert SignalAction.CLOSE in actions
        assert SignalAction.LONG not in actions


# ---------------------------------------------------------------------------
# TestPositionSizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    """Position sizing tests."""

    def test_entry_signals_have_008_quantity_pct(self):
        """All entry signals have quantity_pct=0.08."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config)
        features = make_features()
        signals = strategy.evaluate(features)
        for s in signals:
            if s.action == SignalAction.LONG:
                assert s.quantity_pct == 0.08

    def test_close_signals_have_none_quantity_pct(self):
        """Close signals have quantity_pct=None."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter, push hwm, trigger stop
        strategy.evaluate(make_features())
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        signals = strategy.evaluate(make_features(close=105.0, atr_14=2.0))
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].quantity_pct is None
