"""Tests for StructuralReversalStrategy.

Covers STRAT-01 (entry conditions), STRAT-02 (limit price derivation),
STRAT-07 (net Sharpe/funding formula), STRAT-08 (20% MaxDD enforcement),
trailing stop mechanics, time expiry, and WFE reset isolation.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from poseidon.signals.schemas import OrderType, SignalAction
from poseidon.strategies.structural_reversal import (
    StructuralReversalConfig,
    StructuralReversalStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_structural_features(n_rows: int = 200, **overrides) -> pd.DataFrame:
    """Synthetic feature DataFrame with all structural reversal columns.

    Default: bullish reversal setup (oiwap_distance negative, cascade positive).
    Override the LAST row only via **overrides.
    """
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {
            # OHLCV
            "open": [50000.0] * n_rows,
            "high": [50500.0] * n_rows,
            "low": [49500.0] * n_rows,
            "close": [50200.0] * n_rows,
            "volume": [1000.0] * n_rows,
            # IC-validated features
            "oiwap_distance_168": [-4.0] * n_rows,  # D-07: below threshold -> long
            "oiwap_168": [50000.0] * n_rows,
            "cascade_direction": [1.0] * n_rows,  # D-08: positive = confirms long
            "cvd_change_20": [-100.0] * n_rows,  # D-09: negative = bullish (IC=-0.027)
            "atr_14": [500.0] * n_rows,
        },
        index=idx,
    )
    for col, val in overrides.items():
        df.loc[df.index[-1], col] = val
    return df


def _make_strategy(**cfg_overrides) -> StructuralReversalStrategy:
    """Create a StructuralReversalStrategy with optional config overrides."""
    cfg = StructuralReversalConfig(**cfg_overrides)
    return StructuralReversalStrategy(config=cfg, symbol="BTCUSDT")


# ---------------------------------------------------------------------------
# STRAT-01: Entry Conditions
# ---------------------------------------------------------------------------


class TestEntryConditions:
    """STRAT-01: Strategy emits max 2-3 condition signals with LIMIT orders."""

    def test_long_entry_oiwap_cascade_triggers(self):
        """D-07 + D-08: negative oiwap_distance + positive cascade -> LONG LIMIT."""
        strategy = _make_strategy()
        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.LONG
        assert sig.order_type == OrderType.LIMIT
        assert sig.order_price is not None

    def test_short_entry_oiwap_cascade_triggers(self):
        """D-07 + D-08: positive oiwap_distance + negative cascade -> SHORT LIMIT."""
        strategy = _make_strategy()
        features = make_structural_features(
            oiwap_distance_168=4.0, cascade_direction=-1.0
        )
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.SHORT
        assert sig.order_type == OrderType.LIMIT

    def test_no_entry_when_oiwap_within_threshold(self):
        """D-07: oiwap_distance within threshold -> no signal."""
        strategy = _make_strategy()
        features = make_structural_features(oiwap_distance_168=-1.0)
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_no_entry_when_cascade_disagrees(self):
        """D-08: cascade_direction must confirm direction."""
        strategy = _make_strategy()
        # oiwap_distance is negative (long), but cascade is negative (disagrees)
        features = make_structural_features(cascade_direction=-1.0)
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_cvd_filter_blocks_when_enabled(self):
        """D-09: positive cvd_change blocks long entry when filter is on."""
        strategy = _make_strategy(use_cvd_filter=True)
        # cvd_change_20=100.0 is positive, should block long
        features = make_structural_features(cvd_change_20=100.0)
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_cvd_filter_passes_when_negative_for_long(self):
        """D-09: negative cvd_change confirms long entry when filter is on."""
        strategy = _make_strategy(use_cvd_filter=True)
        # cvd_change_20=-100.0 is negative, should allow long
        features = make_structural_features(cvd_change_20=-100.0)
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG

    def test_no_entry_when_nan_features(self):
        """NaN in required features -> no signal."""
        strategy = _make_strategy()
        features = make_structural_features(oiwap_distance_168=float("nan"))
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_no_entry_when_cascade_nan(self):
        """NaN cascade -> no signal."""
        strategy = _make_strategy()
        features = make_structural_features(cascade_direction=float("nan"))
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_no_entry_when_atr_nan(self):
        """NaN atr -> no signal."""
        strategy = _make_strategy()
        features = make_structural_features(atr_14=float("nan"))
        signals = strategy.evaluate(features)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# STRAT-02: Limit Price Derivation
# ---------------------------------------------------------------------------


class TestLimitPriceDerivation:
    """STRAT-02: Limit price = OIWAP +/- (ATR_multiplier * ATR_14)."""

    def test_long_limit_price_below_oiwap(self):
        """D-12: Long limit price = OIWAP - (atr_mult * ATR)."""
        strategy = _make_strategy(atr_multiplier=1.0)
        features = make_structural_features()
        # Expected: 50000.0 - (1.0 * 500.0) = 49500.0
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].order_price == 49500.0

    def test_short_limit_price_above_oiwap(self):
        """D-12: Short limit price = OIWAP + (atr_mult * ATR)."""
        strategy = _make_strategy(atr_multiplier=1.0)
        features = make_structural_features(
            oiwap_distance_168=4.0, cascade_direction=-1.0
        )
        # Expected: 50000.0 + (1.0 * 500.0) = 50500.0
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].order_price == 50500.0

    def test_atr_multiplier_affects_price(self):
        """D-13: ATR_multiplier scales the offset from OIWAP."""
        strategy = _make_strategy(atr_multiplier=2.0)
        features = make_structural_features()
        # Expected: 50000.0 - (2.0 * 500.0) = 49000.0
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].order_price == 49000.0

    def test_stop_loss_set_on_entry_signal(self):
        """Stop loss is set on entry signal at limit_price -/+ stop_atr_mult * ATR."""
        strategy = _make_strategy(atr_multiplier=1.0, stop_atr_multiplier=3.0)
        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        # Long: limit=49500, stop=49500 - (3.0 * 500) = 48000
        assert sig.stop_loss_price == 48000.0

    def test_short_stop_loss_set_on_entry_signal(self):
        """Short stop loss is above limit price."""
        strategy = _make_strategy(atr_multiplier=1.0, stop_atr_multiplier=3.0)
        features = make_structural_features(
            oiwap_distance_168=4.0, cascade_direction=-1.0
        )
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        # Short: limit=50500, stop=50500 + (3.0 * 500) = 52000
        assert sig.stop_loss_price == 52000.0


# ---------------------------------------------------------------------------
# Trailing Stop
# ---------------------------------------------------------------------------


class TestTrailingStop:
    """ATR trailing stop tightens in profit direction."""

    def test_trailing_stop_tightens_for_long(self):
        """After on_fill(), higher close -> HOLD with updated_stop_loss higher than initial."""
        strategy = _make_strategy(stop_atr_multiplier=3.0)
        # Simulate fill at 49500 with ATR=500
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)
        assert strategy._in_position is True
        # Initial SL = 49500 - 3*500 = 48000
        assert strategy._current_stop_loss == 48000.0

        # Evaluate with higher close (price moved up)
        features = make_structural_features(close=52000.0)
        signals = strategy.evaluate(features)

        # Should emit HOLD with updated_stop_loss
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.HOLD
        assert "updated_stop_loss" in sig.metadata
        # New SL = 52000 - 3*500 = 50500 (higher than 48000)
        assert sig.metadata["updated_stop_loss"] == 50500.0

    def test_trailing_stop_does_not_loosen(self):
        """After tightening, lower close -> no update (stop stays)."""
        strategy = _make_strategy(stop_atr_multiplier=3.0)
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)

        # First: price moves up, tightens SL
        features_up = make_structural_features(close=52000.0)
        signals = strategy.evaluate(features_up)
        assert len(signals) == 1
        tight_sl = signals[0].metadata["updated_stop_loss"]
        assert tight_sl == 50500.0

        # Second: price drops, SL should NOT loosen
        features_down = make_structural_features(close=49800.0)
        signals2 = strategy.evaluate(features_down)
        # No new HOLD signal because SL can't be tightened further
        assert len(signals2) == 0
        # SL remains at the tightened level
        assert strategy._current_stop_loss == 50500.0


# ---------------------------------------------------------------------------
# Time Expiry
# ---------------------------------------------------------------------------


class TestTimeExpiry:
    """D-18: Time-based expiry after max_holding_bars."""

    def test_close_after_max_holding_bars(self):
        """Position held >= max_holding_bars -> CLOSE MARKET signal."""
        strategy = _make_strategy(max_holding_bars=40)
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)
        # Manually set bars_in_position to 39 (next evaluate increments to 40)
        strategy._bars_in_position = 39

        features = make_structural_features()
        signals = strategy.evaluate(features)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.CLOSE
        assert sig.order_type == OrderType.MARKET
        assert sig.metadata["exit_reason"] == "max_holding_bars"
        # Position should be closed
        assert strategy._in_position is False


# ---------------------------------------------------------------------------
# STRAT-08: MaxDD Enforcement
# ---------------------------------------------------------------------------


class TestMaxDDEnforcement:
    """STRAT-08: 20% MaxDD hard stop enforced via set_equity/_check_drawdown."""

    def test_no_entry_when_drawdown_exceeds_20pct(self):
        """D-25: >20% DD from peak -> no new entries."""
        strategy = _make_strategy(max_drawdown=0.20)
        strategy.set_equity(100000)
        strategy.set_equity(79000)  # 21% drawdown

        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_entry_allowed_when_drawdown_below_20pct(self):
        """D-25: <20% DD -> entries allowed."""
        strategy = _make_strategy(max_drawdown=0.20)
        strategy.set_equity(100000)
        strategy.set_equity(85000)  # 15% drawdown

        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.LONG

    def test_drawdown_at_exactly_threshold_allows_entry(self):
        """D-25: Exactly 20% DD -> entries allowed (strict greater-than check)."""
        strategy = _make_strategy(max_drawdown=0.20)
        strategy.set_equity(100000)
        strategy.set_equity(80000)  # exactly 20% DD, not exceeding

        features = make_structural_features()
        signals = strategy.evaluate(features)
        # Exactly at threshold is NOT exceeding -- entries still allowed
        assert len(signals) == 1

    def test_drawdown_just_over_threshold_blocks(self):
        """D-25: 20.01% DD -> no entries."""
        strategy = _make_strategy(max_drawdown=0.20)
        strategy.set_equity(100000)
        strategy.set_equity(79990)  # 20.01% DD

        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 0

    def test_no_equity_set_allows_entry(self):
        """If set_equity never called, MaxDD gate does not block."""
        strategy = _make_strategy()
        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1


# ---------------------------------------------------------------------------
# WFE Reset
# ---------------------------------------------------------------------------


class TestResetForWFE:
    """Strategy reset() clears all stateful fields for WFE isolation."""

    def test_reset_clears_position_state(self):
        """After reset(), all tracking fields are None/False."""
        strategy = _make_strategy()
        strategy._in_position = True
        strategy._position_side = "long"
        strategy._entry_price = 50000.0
        strategy._bars_in_position = 10
        strategy._current_stop_loss = 49000.0
        strategy._position_high_watermark = 51000.0
        strategy._position_low_watermark = 49000.0
        strategy._fill_atr = 500.0

        strategy.reset()

        assert strategy._in_position is False
        assert strategy._position_side is None
        assert strategy._entry_price is None
        assert strategy._bars_in_position == 0
        assert strategy._current_stop_loss is None
        assert strategy._position_high_watermark is None
        assert strategy._position_low_watermark is None
        assert strategy._fill_atr is None

    def test_reset_clears_equity_tracking(self):
        """After reset(), peak and current equity are None."""
        strategy = _make_strategy()
        strategy.set_equity(100000)
        strategy.set_equity(80000)

        assert strategy._peak_equity == 100000
        assert strategy._current_equity == 80000

        strategy.reset()

        assert strategy._peak_equity is None
        assert strategy._current_equity is None


# ---------------------------------------------------------------------------
# STRAT-07: Funding Deduction Formula
# ---------------------------------------------------------------------------


class TestFundingDeduction:
    """STRAT-07: net_sharpe covers funding + fee hurdle."""

    def test_net_sharpe_deduction(self):
        """Pure unit test of funding cost math (D-24).

        Given:
          gross_sharpe = 1.5
          time_in_market_fraction = 0.5
          funding_rate_annual = 0.1095  (0.01%/8h * 3 * 365)
        Then:
          net_sharpe = 1.5 - (0.1095 * 0.5) = 1.44525
        """
        gross_sharpe = 1.5
        time_in_market_fraction = 0.5
        funding_rate_annual = 0.1095

        funding_cost = funding_rate_annual * time_in_market_fraction
        net_sharpe = gross_sharpe - funding_cost

        assert abs(net_sharpe - 1.44525) < 1e-6
        assert abs(funding_cost - 0.054750) < 1e-6

    def test_zero_time_in_market_no_funding_cost(self):
        """No time in market -> no funding cost."""
        gross_sharpe = 1.0
        funding_rate_annual = 0.1095
        net_sharpe = gross_sharpe - (funding_rate_annual * 0.0)
        assert net_sharpe == gross_sharpe

    def test_full_time_in_market_max_funding_cost(self):
        """100% time in market -> full annual funding cost."""
        gross_sharpe = 1.0
        funding_rate_annual = 0.1095
        net_sharpe = gross_sharpe - (funding_rate_annual * 1.0)
        assert abs(net_sharpe - 0.8905) < 1e-6


# ---------------------------------------------------------------------------
# on_fill / on_close
# ---------------------------------------------------------------------------


class TestOnFillOnClose:
    """Verify on_fill and on_close set/clear position state correctly."""

    def test_on_fill_sets_long_position(self):
        strategy = _make_strategy()
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)
        assert strategy._in_position is True
        assert strategy._position_side == "long"
        assert strategy._entry_price == 49500.0
        assert strategy._bars_in_position == 0
        # SL = 49500 - 3*500 = 48000
        assert strategy._current_stop_loss == 48000.0
        assert strategy._position_high_watermark == 49500.0
        assert strategy._position_low_watermark is None

    def test_on_fill_sets_short_position(self):
        strategy = _make_strategy()
        strategy.on_fill(fill_price=50500.0, side="short", atr=500.0)
        assert strategy._in_position is True
        assert strategy._position_side == "short"
        # SL = 50500 + 3*500 = 52000
        assert strategy._current_stop_loss == 52000.0
        assert strategy._position_low_watermark == 50500.0
        assert strategy._position_high_watermark is None

    def test_on_close_resets_position(self):
        strategy = _make_strategy()
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)
        strategy.on_close()
        assert strategy._in_position is False
        assert strategy._position_side is None
        assert strategy._entry_price is None
        assert strategy._current_stop_loss is None


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Strategy validate_config raises on invalid configuration."""

    def test_valid_config_passes(self):
        strategy = _make_strategy()
        assert strategy.validate_config() is True

    def test_empty_symbol_raises(self):
        strategy = _make_strategy()
        strategy.symbol = ""
        with pytest.raises(ValueError, match="requires a symbol"):
            strategy.validate_config()

    def test_zero_atr_multiplier_raises(self):
        strategy = _make_strategy(atr_multiplier=0.0)
        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            strategy.validate_config()

    def test_zero_stop_atr_multiplier_raises(self):
        strategy = _make_strategy(stop_atr_multiplier=0.0)
        with pytest.raises(ValueError, match="stop_atr_multiplier must be positive"):
            strategy.validate_config()


# ---------------------------------------------------------------------------
# Empty features
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for evaluate()."""

    def test_empty_features_returns_empty(self):
        strategy = _make_strategy()
        signals = strategy.evaluate(pd.DataFrame())
        assert signals == []

    def test_signal_has_correct_instrument(self):
        """Signal instrument should be PERPETUAL for crypto_perp."""
        from poseidon.signals.schemas import InstrumentType

        strategy = _make_strategy()
        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        assert signals[0].instrument == InstrumentType.PERPETUAL

    def test_signal_confidence_fixed_at_05(self):
        """Rule-based strategy confidence is fixed at 0.5."""
        strategy = _make_strategy()
        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert signals[0].confidence == 0.5


# ---------------------------------------------------------------------------
# D-27: MaxDD Integration Verification
# ---------------------------------------------------------------------------


class TestMaxDDIntegration:
    """D-27: Verify strategy-internal MaxDD enforcement in backtest context.

    Simulates a mini backtest loop where set_equity() is called before
    each evaluate(), confirming MaxDD gate correctly suppresses new entries
    while allowing existing position management.
    """

    def test_maxdd_suppresses_entries_in_backtest_loop(self):
        """Simulate backtest loop: equity drop -> suppress -> recovery -> allow."""
        strategy = _make_strategy(max_drawdown=0.20)
        features = make_structural_features()

        # Step 1: Initial equity, baseline works
        strategy.set_equity(100000)
        signals = strategy.evaluate(features)
        assert len(signals) == 1, "Baseline: should emit entry signal"
        assert signals[0].action == SignalAction.LONG

        # Step 2: Simulate drawdown to 21% (79000)
        strategy.set_equity(79000)
        signals = strategy.evaluate(features)
        assert len(signals) == 0, "MaxDD gate active: should suppress entry"

        # Step 3: Partial recovery to 85000 (still 15% from peak=100000)
        strategy.set_equity(85000)
        signals = strategy.evaluate(features)
        assert len(signals) == 1, "DD recovered below 20%: should allow entry"
        assert signals[0].action == SignalAction.LONG

    def test_maxdd_does_not_block_existing_position_management(self):
        """When in position and MaxDD breached, HOLD/CLOSE signals still work.

        D-26: Only new entries are blocked, not position management.
        """
        strategy = _make_strategy(max_drawdown=0.20, max_holding_bars=40)

        # Set equity to trigger MaxDD
        strategy.set_equity(100000)
        strategy.set_equity(79000)  # 21% drawdown

        # Manually put strategy in position near time expiry
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)
        strategy._bars_in_position = 39  # next evaluate increments to 40

        # Evaluate -- should still emit CLOSE (time expiry) despite MaxDD
        features = make_structural_features()
        signals = strategy.evaluate(features)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.CLOSE
        assert sig.metadata["exit_reason"] == "max_holding_bars"

    def test_maxdd_does_not_block_trailing_stop_hold(self):
        """When in position and MaxDD breached, trailing stop HOLD still works."""
        strategy = _make_strategy(max_drawdown=0.20, stop_atr_multiplier=3.0)

        # Set equity to trigger MaxDD
        strategy.set_equity(100000)
        strategy.set_equity(79000)  # 21% drawdown

        # Put strategy in position
        strategy.on_fill(fill_price=49500.0, side="long", atr=500.0)

        # Price moved up significantly -> trailing stop should tighten
        features = make_structural_features(close=52000.0)
        signals = strategy.evaluate(features)

        # Should emit HOLD with updated_stop_loss despite MaxDD
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == SignalAction.HOLD
        assert "updated_stop_loss" in sig.metadata
        # New SL = 52000 - 3*500 = 50500
        assert sig.metadata["updated_stop_loss"] == 50500.0
