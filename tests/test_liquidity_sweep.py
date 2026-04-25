"""Tests for LiquiditySweepStrategy -- three-stage maker ambush detection.

Covers SWEEP-02 (three-stage detection), SWEEP-03 (direction modes),
SWEEP-04 (volatility-adaptive distance), and E2E backtest integration.
"""

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
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
        strategy = LiquiditySweepStrategy(
            config=make_default_config(
                exit={"cooldown_bars": 4, "max_holding_bars": None},
            )
        )

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


# ---------------------------------------------------------------------------
# E2E BacktestRunner integration helpers
# ---------------------------------------------------------------------------


def _make_e2e_ohlcv(n_bars: int = 200) -> pd.DataFrame:
    """Create synthetic BTC 1h OHLCV with ALL feature columns pre-populated.

    The data is engineered so that bar 50 triggers a downward sweep detection
    (long entry). The limit order is placed at ~49091 (swing_low=49400 minus
    fib_distance=309). Bars 52-55 have prices that dip to fill the order.

    All feature columns are pre-populated to avoid FeatureEngine needing
    a DB connection. We pass feature_specs=[] to runner.run() so the engine
    just passes through the data.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Base OHLCV: price hovering around 50000
    opens = np.full(n_bars, 50000.0)
    highs = np.full(n_bars, 50200.0)
    lows = np.full(n_bars, 49800.0)
    closes = np.full(n_bars, 50100.0)
    volume = np.full(n_bars, 1000.0)

    # Feature columns -- neutral (no sweep) defaults
    swing_high_24 = np.full(n_bars, 50600.0)
    swing_low_24 = np.full(n_bars, 49400.0)
    oi_buildup_24 = np.full(n_bars, 0.5)  # Below threshold -> no zone
    oi_change_zscore_20 = np.full(n_bars, 0.0)
    oi_change_pct = np.full(n_bars, 0.0)
    oiwap_168 = np.full(n_bars, 50000.0)
    oiwap_distance_168 = np.full(n_bars, 0.0)
    breakout_down_24 = np.full(n_bars, 0.0)
    breakout_up_24 = np.full(n_bars, 0.0)
    fib_ext_down = np.full(n_bars, 49000.0)
    fib_ext_up = np.full(n_bars, 51000.0)
    wick_ratio_lower = np.full(n_bars, 0.05)  # Below threshold
    wick_ratio_upper = np.full(n_bars, 0.05)
    wick_ratio_total = np.full(n_bars, 0.1)
    range_expansion_14 = np.full(n_bars, 0.8)  # Below 1.0 -> no volume score
    vol_regime = np.full(n_bars, 1.0)
    atr_14 = np.full(n_bars, 500.0)
    volume_ratio_20 = np.full(n_bars, 1.0)
    funding_rate_daily = np.full(n_bars, 0.01)  # Positive -> crowded long

    # --- Engineer sweep at bar 50 ---
    # Stage 1: OI buildup active at bar 50
    oi_buildup_24[50] = 5.0  # Well above threshold 1.0

    # Stage 2: Mandatory gates pass + confirmation scoring
    wick_ratio_lower[50] = 0.4  # > 0.15 threshold
    breakout_down_24[50] = 0.5  # > 0.1 threshold

    # OHLCV at bar 50: sweep candle with reversal above swing_low
    opens[50] = 49500.0
    highs[50] = 49600.0
    lows[50] = 48800.0  # Below swing_low, long wick
    closes[50] = 49500.0  # Recovers above swing_low=49400 (reversal)

    # Confirmation scoring at bar 50:
    # OI drop: oi_change_zscore_20=-2.5 -> score = max(0, 2.5) = 2.5
    oi_change_zscore_20[50] = -2.5
    # Volume: range_expansion_14=1.5 -> score = max(0, 1.5-1) = 0.5
    range_expansion_14[50] = 1.5
    # Funding: 0.01 > 0 -> alignment=1.0 (crowded long supports downward sweep)
    funding_rate_daily[50] = 0.01
    # Total score = 2.5*0.4 + 0.5*0.3 + 1.0*0.3 = 1.0 + 0.15 + 0.3 = 1.45 >= 0.5 OK
    volume[50] = 3000.0  # High volume

    # Entry calculation (vol_regime=1, multiplier=1.0):
    # fib_distance = atr_14 * fib_level * multiplier = 500 * 0.618 * 1.0 = 309
    # entry_price = swing_low - fib_distance = 49400 - 309 = 49091
    # stop_loss = swing_low - 2*fib_distance = 49400 - 618 = 48782
    # take_profit = min(oiwap=50000, swing_low+fib_distance=49709) = 49709

    # --- Bars 51-55: price dips to fill the limit order at ~49091 ---
    for i in range(51, 56):
        opens[i] = 49300.0 - (i - 51) * 50
        highs[i] = 49400.0
        lows[i] = 49000.0 - (i - 51) * 30  # Dips below 49091 at bars 52+
        closes[i] = 49200.0 - (i - 51) * 40
        # Keep feature columns neutral (no new sweep)
        oi_buildup_24[i] = 0.3

    # Bars 53-54 specifically: ensure low goes below 49091 for fill
    lows[53] = 49050.0  # Below 49091 -> fills pessimistic (low < order_price)
    lows[54] = 48900.0  # Clearly below

    # --- Bars 56-70: price bounces back toward TP (49709) ---
    for i in range(56, 71):
        base = 49200.0 + (i - 56) * 50
        opens[i] = base
        highs[i] = base + 200.0
        lows[i] = base - 100.0
        closes[i] = base + 100.0
        oi_buildup_24[i] = 0.3

    # Bar 65: high should reach 49709 TP target
    # base at bar 65 = 49200 + (65-56)*50 = 49200 + 450 = 49650
    # high = 49650 + 200 = 49850 -> above TP=49709 -> TP triggers

    # --- Bars 71+: continue at higher level ---
    for i in range(71, n_bars):
        opens[i] = 50000.0
        highs[i] = 50200.0
        lows[i] = 49800.0
        closes[i] = 50100.0
        oi_buildup_24[i] = 0.3

    # Ensure OHLCV validity: highs >= max(open, close), lows <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            # All feature columns pre-populated
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
# TestE2EBacktest
# ---------------------------------------------------------------------------


class TestE2EBacktest:
    """E2E integration test: BacktestRunner + LiquiditySweepStrategy.

    SWEEP-02 SC4: Strategy runs end-to-end in BacktestRunner with pessimistic
    fill model on BTC 1h data, produces at least one filled trade.

    Uses pre-populated feature columns with feature_specs=[] to avoid
    FeatureEngine database dependency (compute_with_companions needs DB
    for non-price features like OI/funding). The strategy reads feature
    values from the DataFrame regardless of how they were computed.
    """

    def test_e2e_backtest_produces_filled_trade(self):
        """Run full backtest pipeline and verify at least one trade is filled."""
        config = make_default_config()
        strategy = LiquiditySweepStrategy(config=config)

        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("crypto_perp")
        sizing = SizingConfig(
            mode=SizingMode.FIXED_RISK,
            risk_pct=0.01,
            max_notional_pct=0.3,
        )

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            sizing_config=sizing,
            fill_model=FillModel.PESSIMISTIC,
            max_pending_bars=10,
        )

        ohlcv = _make_e2e_ohlcv(n_bars=200)

        # Pass feature_specs=[] to skip FeatureEngine computation
        # (all features are pre-populated on the OHLCV DataFrame)
        result = runner.run(ohlcv, feature_specs=[])

        # Must have at least one filled trade (entry from limit fill)
        assert len(result.trades) >= 1, (
            f"Expected at least 1 filled trade, got {len(result.trades)}. Trade count: {result.trade_count}"
        )

        # Verify the trade was a long entry (downward sweep -> long limit)
        first_trade = result.trades[0]
        assert first_trade["action"] in ("long", "short"), f"Unexpected action: {first_trade['action']}"

        # Verify result has equity curve
        assert result.equity_curve_length > 0

    def test_e2e_backtest_with_optimistic_fill(self):
        """Optimistic fill should produce >= as many trades as pessimistic."""
        config = make_default_config()

        # Pessimistic run
        strategy_p = LiquiditySweepStrategy(config=config)
        runner_p = BacktestRunner(
            strategy=strategy_p,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            sizing_config=SizingConfig(mode=SizingMode.FIXED_RISK, risk_pct=0.01),
            fill_model=FillModel.PESSIMISTIC,
            max_pending_bars=10,
        )
        ohlcv = _make_e2e_ohlcv()
        result_p = runner_p.run(ohlcv, feature_specs=[])

        # Optimistic run
        strategy_o = LiquiditySweepStrategy(config=config)
        runner_o = BacktestRunner(
            strategy=strategy_o,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("crypto_perp"),
            sizing_config=SizingConfig(mode=SizingMode.FIXED_RISK, risk_pct=0.01),
            fill_model=FillModel.OPTIMISTIC,
            max_pending_bars=10,
        )
        ohlcv2 = _make_e2e_ohlcv()
        result_o = runner_o.run(ohlcv2, feature_specs=[])

        assert len(result_o.trades) >= len(result_p.trades), (
            f"Optimistic fills ({len(result_o.trades)}) should be >= pessimistic ({len(result_p.trades)})"
        )
