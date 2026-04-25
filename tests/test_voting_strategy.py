"""Tests for VotingStrategy — vote counting, ATR trailing stop, position sizing."""

import json
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
    df = pd.DataFrame(
        {
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
        }
    )
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
        "cooldown_bars": 0,
        "conviction_gap": 0,
        "sub_signals": [
            {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 6}, "threshold": 0},
            {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 12}, "threshold": 0},
            {
                "type": "indicator_comparison",
                "indicator_a": "ema",
                "indicator_b": "ema",
                "params": {"period_a": 7, "period_b": 26},
                "direction": "above",
            },
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
        """After CLOSE + global cooldown, new vote threshold met -> can re-enter (new LONG)."""
        config = _make_all_true_config(min_votes=4)
        config["cooldown_bars"] = 3  # short cooldown for test
        config["conviction_gap"] = 0  # disable gap for this test
        strategy = VotingStrategy(config=config, atr_multiplier=2.0, atr_period=14)

        # Enter
        strategy.evaluate(make_features())
        # Trigger stop
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        strategy.evaluate(make_features(close=105.0, atr_14=2.0))

        # Wait cooldown_bars for cooldown to expire (global block)
        for _ in range(3):
            strategy.evaluate(make_features(rsi_8=45.0, macd_histogram=-0.1, cum_return_6d=-0.01))

        # Re-enter (all conditions true again, cooldown expired)
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


# ---------------------------------------------------------------------------
# TestNunchiSignals — Nunchi 6-signal config integration tests
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "src" / "poseidon" / "strategies" / "configs" / "nunchi_crypto_1h.json"


def _load_nunchi_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class TestNunchiSignals:
    """Nunchi 6-signal config integration tests."""

    def test_json_loads_without_error(self):
        """The JSON file is valid JSON."""
        config = _load_nunchi_config()
        assert isinstance(config, dict)

    def test_config_creates_valid_strategy(self):
        """Load config, create VotingStrategy, validate_config() returns True."""
        config = _load_nunchi_config()
        strategy = VotingStrategy(config=config)
        assert strategy.validate_config() is True

    def test_strategy_name(self):
        """VotingStrategy.name == 'nunchi_voting_crypto_1h'."""
        config = _load_nunchi_config()
        strategy = VotingStrategy(config=config)
        assert strategy.name == "nunchi_voting_crypto_1h"

    def test_strategy_metadata(self):
        """VotingStrategy has correct symbol, market, interval."""
        config = _load_nunchi_config()
        strategy = VotingStrategy(config=config)
        assert strategy.symbol == "BTCUSDT"
        assert strategy.market == "crypto_spot"
        assert strategy.interval == "1h"

    def test_strategy_has_6_sub_signals(self):
        """len(strategy._sub_signals) == 6."""
        config = _load_nunchi_config()
        strategy = VotingStrategy(config=config)
        assert len(strategy._sub_signals) == 6

    def test_all_6_true_emits_long(self):
        """evaluate() with all 6 conditions true -> emits LONG signal."""
        config = _load_nunchi_config()
        strategy = VotingStrategy(config=config)

        # Build features where all 6 conditions are true
        n_rows = 200
        df = pd.DataFrame(
            {
                "close": [100.0] * n_rows,
                "ema_7": [105.0] * n_rows,
                "ema_26": [100.0] * n_rows,
                "rsi_8": [55.0] * n_rows,
                "macd_histogram": [0.5] * n_rows,
                "bb_upper_20": [110.0] * n_rows,
                "bb_lower_20": [90.0] * n_rows,
                "atr_14": [2.0] * n_rows,
                "cum_return_6d": [0.01] * n_rows,
                "cum_return_12d": [0.02] * n_rows,
            }
        )
        # Last row: narrow BB (squeeze) for signal 6
        df.loc[df.index[-1], "bb_upper_20"] = 101.0
        df.loc[df.index[-1], "bb_lower_20"] = 99.0

        signals = strategy.evaluate(df)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG

    def test_only_2_true_emits_nothing(self):
        """evaluate() with only 2 conditions true -> emits no signal."""
        config = _load_nunchi_config()
        strategy = VotingStrategy(config=config)

        n_rows = 200
        df = pd.DataFrame(
            {
                "close": [100.0] * n_rows,
                "ema_7": [95.0] * n_rows,  # ema_7 < ema_26 -> signal 3 false
                "ema_26": [100.0] * n_rows,
                "rsi_8": [45.0] * n_rows,  # < 50 -> signal 4 false
                "macd_histogram": [0.5] * n_rows,  # > 0 -> signal 5 true
                "bb_upper_20": [110.0] * n_rows,
                "bb_lower_20": [90.0] * n_rows,
                "atr_14": [2.0] * n_rows,
                "cum_return_6d": [-0.01] * n_rows,  # < 0 -> signal 1 false
                "cum_return_12d": [-0.02] * n_rows,  # < 0 -> signal 2 false
            }
        )
        # Last row: narrow BB (squeeze) for signal 6 -> true
        df.loc[df.index[-1], "bb_upper_20"] = 101.0
        df.loc[df.index[-1], "bb_lower_20"] = 99.0

        signals = strategy.evaluate(df)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Bear sub_signals config helper
# ---------------------------------------------------------------------------


def _make_bear_config() -> dict:
    """Config with both bull and bear sub_signals."""
    return {
        "name": "test_bidirectional",
        "symbol": "BTCUSDT",
        "market": "crypto_spot",
        "interval": "1h",
        "min_votes": 4,
        "position_pct": 0.08,
        "sub_signals": [
            {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 6}, "threshold": 0},
            {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 12}, "threshold": 0},
            {
                "type": "indicator_comparison",
                "indicator_a": "ema",
                "indicator_b": "ema",
                "params": {"period_a": 7, "period_b": 26},
                "direction": "above",
            },
            {"type": "indicator_above", "indicator": "rsi", "params": {"period": 8}, "threshold": 50},
            {"type": "indicator_above", "indicator": "macd_histogram", "params": {}, "threshold": 0},
            {"type": "bollinger_width_percentile", "params": {"period": 20, "lookback": 168}, "threshold": 0.2},
        ],
        "bear_sub_signals": [
            {"type": "indicator_below", "indicator": "cum_return", "params": {"period": 6}, "threshold": 0},
            {"type": "indicator_below", "indicator": "cum_return", "params": {"period": 12}, "threshold": 0},
            {
                "type": "indicator_comparison",
                "indicator_a": "ema",
                "indicator_b": "ema",
                "params": {"period_a": 7, "period_b": 26},
                "direction": "below",
            },
            {"type": "indicator_below", "indicator": "rsi", "params": {"period": 8}, "threshold": 50},
            {"type": "indicator_below", "indicator": "macd_histogram", "params": {}, "threshold": 0},
            {"type": "bollinger_width_percentile", "params": {"period": 20, "lookback": 168}, "threshold": 0.2},
        ],
        "bear_min_votes": 4,
        "bear_position_pct": 0.06,
        "cooldown_bars": 0,
        "conviction_gap": 0,
    }


def _make_bear_features(**overrides):
    """Features where bear conditions are true (bearish market)."""
    n_rows = 200
    df = pd.DataFrame(
        {
            "close": [100.0] * n_rows,
            "ema_7": [95.0] * n_rows,  # ema_7 < ema_26 -> bearish
            "ema_26": [100.0] * n_rows,
            "rsi_8": [35.0] * n_rows,  # < 50 -> bearish
            "macd_histogram": [-0.5] * n_rows,  # < 0 -> bearish
            "bb_upper_20": [110.0] * n_rows,
            "bb_lower_20": [90.0] * n_rows,
            "atr_14": [2.0] * n_rows,
            "cum_return_6d": [-0.02] * n_rows,  # < 0 -> bearish
            "cum_return_12d": [-0.03] * n_rows,  # < 0 -> bearish
        }
    )
    # Last row: narrow BB (squeeze) for signal 6 -> true
    df.loc[df.index[-1], "bb_upper_20"] = 101.0
    df.loc[df.index[-1], "bb_lower_20"] = 99.0
    for col, val in overrides.items():
        df.loc[df.index[-1], col] = val
    return df


# ---------------------------------------------------------------------------
# TestBearShortSignals — SHORT signal emission
# ---------------------------------------------------------------------------


class TestBearShortSignals:
    """Bear sub_signals and SHORT signal emission tests."""

    def test_bear_sub_signals_evaluated_independently(self):
        """Bear sub_signals evaluated independently from bull sub_signals."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Bearish features: bull signals should NOT fire, bear signals SHOULD fire
        features = _make_bear_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.SHORT

    def test_short_emitted_when_bear_votes_meet_threshold(self):
        """SHORT signal emitted when bear_votes >= bear_min_votes and not in position."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        features = _make_bear_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.SHORT

    def test_short_uses_bear_position_pct(self):
        """SHORT signal uses bear_position_pct for quantity_pct."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        features = _make_bear_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].quantity_pct == 0.06

    def test_no_short_when_already_in_position(self):
        """No SHORT emitted when already in short position."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        features = _make_bear_features()
        # Enter short
        signals1 = strategy.evaluate(features)
        assert len(signals1) == 1
        assert signals1[0].action == SignalAction.SHORT

        # Second eval -- already in short position
        signals2 = strategy.evaluate(features)
        assert len(signals2) == 0

    def test_no_long_when_in_short_position(self):
        """No LONG emitted when in short position."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter short first
        features_bear = _make_bear_features()
        strategy.evaluate(features_bear)

        # Now bullish features -- should not emit LONG because in position
        features_bull = make_features()
        signals = strategy.evaluate(features_bull)
        # Should not contain LONG (may contain CLOSE from exit checks)
        long_signals = [s for s in signals if s.action == SignalAction.LONG]
        assert len(long_signals) == 0

    def test_backward_compat_no_bear_sub_signals(self):
        """Config without bear_sub_signals works as before (long-only)."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        features = make_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG


# ---------------------------------------------------------------------------
# TestDefaultATRMultiplier
# ---------------------------------------------------------------------------


class TestDefaultATRMultiplier:
    """Default atr_multiplier should be 5.5 per D-05."""

    def test_default_atr_multiplier_is_5_5(self):
        """Default atr_multiplier is 5.5."""
        config = _make_all_true_config()
        strategy = VotingStrategy(config=config)
        assert strategy._atr_multiplier == 5.5


# ---------------------------------------------------------------------------
# TestRSIExit — RSI mean-reversion exits
# ---------------------------------------------------------------------------


class TestRSIExit:
    """RSI exit tests -- long exits at RSI > 69, short exits at RSI < 31."""

    def test_long_rsi_exit_above_69(self):
        """Exit longs when RSI > 69."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter long
        strategy.evaluate(make_features())
        assert strategy._position_direction == "long"

        # RSI goes above 69 -- should trigger RSI exit
        features = make_features(rsi_8=72.0, close=100.0, atr_14=2.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "rsi_exit"

    def test_short_rsi_exit_below_31(self):
        """Exit shorts when RSI < 31."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter short
        strategy.evaluate(_make_bear_features())
        assert strategy._position_direction == "short"

        # RSI drops below 31 -- should trigger RSI exit
        features = _make_bear_features(rsi_8=28.0, close=100.0, atr_14=2.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "rsi_exit"

    def test_long_rsi_at_69_no_exit(self):
        """RSI exactly at 69 does NOT trigger exit (need > 69)."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter long
        strategy.evaluate(make_features())

        features = make_features(rsi_8=69.0, close=100.0, atr_14=2.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 0


# ---------------------------------------------------------------------------
# TestSignalFlip — signal flip exits
# ---------------------------------------------------------------------------


class TestSignalFlip:
    """Signal flip exit tests -- opposing ensemble fires while in position."""

    def test_bear_fires_while_in_long_emits_close(self):
        """Bear ensemble fires while in long position -> CLOSE (signal flip)."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter long
        strategy.evaluate(make_features())
        assert strategy._position_direction == "long"

        # Now bear conditions fire -- should signal flip (CLOSE)
        features = _make_bear_features(close=100.0, atr_14=2.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "signal_flip"

    def test_bull_fires_while_in_short_emits_close(self):
        """Bull ensemble fires while in short position -> CLOSE (signal flip)."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter short
        strategy.evaluate(_make_bear_features())
        assert strategy._position_direction == "short"

        # Now bull conditions fire -- should signal flip
        features = make_features(close=100.0, atr_14=2.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "signal_flip"


# ---------------------------------------------------------------------------
# TestCooldown — global cooldown mechanism
# ---------------------------------------------------------------------------


class TestCooldown:
    """Global cooldown blocks ALL entries after ANY exit for cooldown_bars."""

    def test_no_reentry_within_cooldown_bars(self):
        """Cannot re-enter any direction within cooldown_bars of exit."""
        config = _make_all_true_config(min_votes=4)
        config["cooldown_bars"] = 3  # short cooldown for test
        config["conviction_gap"] = 0  # disable gap for this test
        strategy = VotingStrategy(config=config, atr_multiplier=2.0)
        # Enter long
        strategy.evaluate(make_features())

        # Trigger ATR trailing stop exit
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        signals_close = strategy.evaluate(make_features(close=105.0, atr_14=2.0))
        assert any(s.action == SignalAction.CLOSE for s in signals_close)

        # Bars 0-3 after exit: blocked by global cooldown
        for _ in range(3):
            signals = strategy.evaluate(make_features())
            long_signals = [s for s in signals if s.action == SignalAction.LONG]
            assert len(long_signals) == 0

        # Bar 4 after exit: cooldown expired, can re-enter
        signals_ok = strategy.evaluate(make_features())
        long_signals_ok = [s for s in signals_ok if s.action == SignalAction.LONG]
        assert len(long_signals_ok) == 1

    def test_cooldown_blocks_opposite_direction_too(self):
        """After long exit, SHORT entry is also blocked during cooldown."""
        config = _make_bear_config()
        config["cooldown_bars"] = 3
        config["conviction_gap"] = 0
        strategy = VotingStrategy(config=config, atr_multiplier=2.0)
        # Enter long
        strategy.evaluate(make_features())

        # Trigger ATR trailing stop exit from long
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        strategy.evaluate(make_features(close=105.0, atr_14=2.0))

        # Immediately try bear entry -- should BE blocked (global cooldown)
        signals = strategy.evaluate(_make_bear_features())
        short_signals = [s for s in signals if s.action == SignalAction.SHORT]
        assert len(short_signals) == 0


# ---------------------------------------------------------------------------
# TestShortTrailingStop — short-side trailing stop
# ---------------------------------------------------------------------------


class TestShortTrailingStop:
    """Short trailing stop tracks low watermark."""

    def test_short_trailing_stop_tracks_low_watermark(self):
        """Short trailing stop tracks _position_low_watermark."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=2.0)
        # Enter short at 100
        strategy.evaluate(_make_bear_features(close=100.0))
        assert strategy._position_direction == "short"
        assert strategy._position_low_watermark == 100.0

        # Price drops to 90 -- update low watermark
        strategy.evaluate(_make_bear_features(close=90.0, atr_14=2.0))
        assert strategy._position_low_watermark == 90.0

    def test_short_trailing_stop_exits_when_price_above_stop(self):
        """Short exit when close > low_watermark + atr_multiplier * atr."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=2.0)
        # Enter short at 100
        strategy.evaluate(_make_bear_features(close=100.0))
        # Price drops to 90 (lwm=90)
        strategy.evaluate(_make_bear_features(close=90.0, atr_14=2.0))
        # Price rises to 95: stop = 90 + 2*2 = 94, 95 > 94 -> CLOSE
        signals = strategy.evaluate(_make_bear_features(close=95.0, atr_14=2.0))
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "atr_trailing_stop"

    def test_long_trailing_stop_still_works(self):
        """Long trailing stop unchanged (tracks high watermark)."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0)
        # Enter long
        strategy.evaluate(make_features(close=100.0))
        # Price rises to 110
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        # Price drops to 105: stop = 110 - 2*2 = 106, 105 < 106 -> CLOSE
        signals = strategy.evaluate(make_features(close=105.0, atr_14=2.0))
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "atr_trailing_stop"


# ---------------------------------------------------------------------------
# TestExitPriority — ATR > RSI > signal flip
# ---------------------------------------------------------------------------


class TestExitPriority:
    """Exit priority: ATR trailing stop > RSI exit > signal flip."""

    def test_atr_takes_priority_over_rsi(self):
        """When both ATR stop and RSI exit trigger, ATR wins."""
        config = _make_all_true_config(min_votes=4)
        strategy = VotingStrategy(config=config, atr_multiplier=2.0)
        # Enter long
        strategy.evaluate(make_features(close=100.0))
        # Push hwm to 110
        strategy.evaluate(make_features(close=110.0, atr_14=2.0))
        # Price drops below stop AND RSI > 69
        features = make_features(close=105.0, atr_14=2.0, rsi_8=72.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "atr_trailing_stop"

    def test_rsi_takes_priority_over_signal_flip(self):
        """When both RSI exit and signal flip trigger, RSI wins."""
        config = _make_bear_config()
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Enter long
        strategy.evaluate(make_features(close=100.0))
        # RSI > 69 AND bear ensemble fires -- RSI should win
        features = _make_bear_features(close=100.0, atr_14=2.0, rsi_8=72.0)
        signals = strategy.evaluate(features)
        close_signals = [s for s in signals if s.action == SignalAction.CLOSE]
        assert len(close_signals) == 1
        assert close_signals[0].metadata["reason"] == "rsi_exit"


# ---------------------------------------------------------------------------
# TestR2FeatureSpecs — get_feature_specs() with R2 feature conditions
# ---------------------------------------------------------------------------


class TestR2FeatureSpecs:
    """Test get_feature_specs() handles feature_above/below conditions."""

    def test_r2_bull_sub_signals_in_specs(self):
        """feature_above/below columns appear in get_feature_specs() output."""
        config = _make_all_true_config(min_votes=4)
        # Add R2 sub_signals
        config["sub_signals"].extend(
            [
                {"type": "feature_above", "column": "foreign_net_buy_ratio", "threshold": 0.01},
                {"type": "feature_below", "column": "pe_ratio", "threshold": 20.0},
            ]
        )
        config["min_votes"] = 4  # keep at 4
        strategy = VotingStrategy(config=config)
        specs = strategy.get_feature_specs()
        spec_names = [name for name, _ in specs]
        assert "foreign_net_buy_ratio" in spec_names
        assert "pe_ratio" in spec_names

    def test_r2_specs_have_empty_params(self):
        """R2 feature specs have empty params dict (pre-computed columns)."""
        config = _make_all_true_config(min_votes=4)
        config["sub_signals"].append(
            {"type": "feature_above", "column": "foreign_net_buy_ratio", "threshold": 0.01},
        )
        strategy = VotingStrategy(config=config)
        specs = strategy.get_feature_specs()
        r2_specs = [(n, p) for n, p in specs if n == "foreign_net_buy_ratio"]
        assert len(r2_specs) == 1
        assert r2_specs[0] == ("foreign_net_buy_ratio", {})

    def test_ta_specs_still_present(self):
        """Standard TA specs (rsi, ema, macd, etc.) still present with R2 additions."""
        config = _make_all_true_config(min_votes=4)
        config["sub_signals"].append(
            {"type": "feature_above", "column": "foreign_net_buy_ratio", "threshold": 0.01},
        )
        strategy = VotingStrategy(config=config)
        specs = strategy.get_feature_specs()
        spec_names = [name for name, _ in specs]
        # Standard TA specs from _make_all_true_config
        assert "rsi" in spec_names
        assert "atr" in spec_names
        assert "returns" in spec_names

    def test_bear_r2_sub_signals_in_specs(self):
        """R2 conditions in bear_sub_signals also appear in get_feature_specs()."""
        config = _make_bear_config()
        config["bear_sub_signals"].append(
            {"type": "feature_below", "column": "funding_rate_daily", "threshold": -0.001},
        )
        strategy = VotingStrategy(config=config)
        specs = strategy.get_feature_specs()
        spec_names = [name for name, _ in specs]
        assert "funding_rate_daily" in spec_names
        assert ("funding_rate_daily", {}) in specs
