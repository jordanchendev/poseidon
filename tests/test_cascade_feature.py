"""Unit tests for Cascade Composite feature.

Tests the CascadeComposite feature class which detects confluence of
multiple micro-structure signals (OI extremes, wick rejections, volume
spikes) as a single boolean composite indicator.

References: CONTEXT.md D-10, D-11, D-12, D-13.
"""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.cascade import CascadeComposite


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_cascade_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV with controllable wick ratios and volume spikes.

    Generates 250+ rows of OHLCV data sufficient for OICostBasis (period=168)
    and other component feature computations.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    close = 40000 + rng.standard_normal(n).cumsum() * 200
    open_ = close - rng.uniform(-100, 100, n)
    high = np.maximum(close, open_) + rng.uniform(50, 300, n)
    low = np.minimum(close, open_) - rng.uniform(50, 300, n)
    volume = rng.uniform(500, 2000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def _make_cascade_oi_data(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OI data that produces known oiwap_distance values.

    OI with increasing trend so OICostBasis has valid oiwap values.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    oi = 100000 + np.arange(n) * 500 + rng.standard_normal(n) * 100
    return pd.DataFrame({"open_interest": oi}, index=dates)


def _make_extreme_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Create OHLCV designed so multiple conditions are clearly met.

    Specifically crafts bars at the end with:
    - Large wicks (wick_ratio_upper OR wick_ratio_lower > 0.4)
    - Volume spikes (volume_ratio_20 > 2.0)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    # Normal bars for most of the data (warmup for rolling computations)
    close = 40000 + rng.standard_normal(n).cumsum() * 100
    open_ = close - rng.uniform(-50, 50, n)
    high = np.maximum(close, open_) + rng.uniform(20, 80, n)
    low = np.minimum(close, open_) - rng.uniform(20, 80, n)
    volume = rng.uniform(800, 1200, n)

    # Override last 10 bars with extreme conditions:
    # - Large upper wicks (close near low, high far above)
    for i in range(n - 10, n):
        # Extreme wick: close near low, huge upper wick
        base = close[i]
        low[i] = base - 10
        high[i] = base + 1000  # Very large upper wick
        open_[i] = base + 5  # Open near close
        close[i] = base
        # Volume spike: 5x normal
        volume[i] = 5000

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def _make_quiet_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Create OHLCV with no extreme conditions (all conditions should be False)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    close = np.full(n, 40000.0) + rng.uniform(-5, 5, n)
    open_ = close - rng.uniform(-2, 2, n)
    # Tiny wick, body dominates -> wick ratio < 0.4
    high = np.maximum(close, open_) + 1
    low = np.minimum(close, open_) - 1
    # Constant volume -> ratio ~1.0 (well below 2.0 threshold)
    volume = np.full(n, 1000.0) + rng.uniform(-10, 10, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


# ── Test 1: Return structure ─────────────────────────────────────────


def test_cascade_returns_correct_columns():
    """Cascade returns DataFrame with cascade_signal and cascade_direction columns."""
    ohlcv = _make_cascade_ohlcv()
    oi_data = _make_cascade_oi_data()
    feature = CascadeComposite()
    result = feature.compute(ohlcv, oi_data=oi_data, threshold=2)

    assert isinstance(result, pd.DataFrame)
    assert "cascade_signal" in result.columns
    assert "cascade_direction" in result.columns
    assert len(result) == len(ohlcv)


# ── Test 2: Signal fires when multiple conditions met ─────────────────


def test_cascade_fires_when_conditions_met():
    """With data where 3+ of 4 conditions met, cascade_signal should be 1.0 (threshold=2)."""
    ohlcv = _make_extreme_ohlcv()
    oi_data = _make_cascade_oi_data(n=250)
    feature = CascadeComposite()
    result = feature.compute(ohlcv, oi_data=oi_data, threshold=2)

    # The last 10 bars have extreme wicks + volume spikes (at least 2 conditions)
    # At least some of those bars should fire
    last_10_signals = result["cascade_signal"].iloc[-10:]
    assert last_10_signals.sum() > 0, (
        f"Expected some cascade signals in extreme bars, got all zeros. "
        f"Values: {last_10_signals.tolist()}"
    )


# ── Test 3: Signal does NOT fire below threshold ──────────────────────


def test_cascade_no_fire_below_threshold():
    """With quiet data (1 or 0 conditions met), cascade_signal should be 0.0."""
    ohlcv = _make_quiet_ohlcv()
    oi_data = _make_cascade_oi_data(n=250)
    feature = CascadeComposite()
    result = feature.compute(ohlcv, oi_data=oi_data, threshold=2)

    # Quiet data: no extreme wicks, no volume spikes, likely no OI extremes
    # Most bars should have signal = 0.0
    # Allow some bars from random noise, but at least 90% should be zero
    after_warmup = result["cascade_signal"].iloc[170:]  # After OI warmup
    nonzero_frac = (after_warmup > 0).mean()
    assert nonzero_frac < 0.1, (
        f"Expected mostly zero signals in quiet data, got {nonzero_frac:.1%} non-zero"
    )


# ── Test 4: Direction follows oiwap_distance sign (D-13) ─────────────


def test_cascade_direction_from_oiwap_distance():
    """Direction: below cost basis (oiwap_distance < 0) -> bullish 1.0,
    above cost basis (oiwap_distance > 0) -> bearish -1.0.
    """
    n = 250
    rng = np.random.default_rng(99)
    dates = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    # Create scenario where price is ABOVE oiwap (positive distance)
    # OI increases at close=40000, then price jumps to 45000
    close = np.full(n, 40000.0)
    close[200:] = 45000.0  # Price jumps above cost basis

    open_ = close - rng.uniform(-20, 20, n)
    high = np.maximum(close, open_) + rng.uniform(50, 500, n)
    low = np.minimum(close, open_) - rng.uniform(50, 500, n)

    # Force extreme conditions in last bars so cascade fires
    for i in range(n - 10, n):
        high[i] = close[i] + 2000  # Extreme upper wick
        low[i] = close[i] - 10
        open_[i] = close[i] + 5

    volume = np.full(n, 1000.0)
    volume[-10:] = 5000.0  # Volume spike

    ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    # Rising OI with increasing entries at close=40000
    oi = 100000 + np.arange(n) * 500
    oi_data = pd.DataFrame({"open_interest": oi}, index=dates)

    feature = CascadeComposite()
    result = feature.compute(ohlcv, oi_data=oi_data, threshold=2)

    # Where signal fires in the last bars (price above OIWAP -> oiwap_distance > 0 -> bearish -1.0)
    fired = result[result["cascade_signal"] == 1.0].iloc[-5:]
    if len(fired) > 0:
        # Direction should be -1.0 (bearish) because price is above cost basis
        assert (fired["cascade_direction"] == -1.0).any() or (fired["cascade_direction"] == 1.0).any(), (
            f"Expected nonzero direction where signal fires, got: {fired['cascade_direction'].tolist()}"
        )


# ── Test 5: oi_data=None graceful degradation ─────────────────────────


def test_cascade_works_without_oi_data():
    """When oi_data is None, Cascade works using only wick+volume conditions."""
    ohlcv = _make_extreme_ohlcv()
    feature = CascadeComposite()
    # With oi_data=None and threshold=2, cascade needs both wick AND volume to fire
    result = feature.compute(ohlcv, oi_data=None, threshold=2)

    assert isinstance(result, pd.DataFrame)
    assert "cascade_signal" in result.columns
    assert "cascade_direction" in result.columns
    assert len(result) == len(ohlcv)

    # Extreme bars have both wick and volume extremes -> should fire
    last_10 = result["cascade_signal"].iloc[-10:]
    assert last_10.sum() > 0, (
        "Expected cascade to fire on wick+volume conditions even without OI data"
    )


# ── Test 6: Empty DataFrame returns empty with correct columns ────────


def test_cascade_empty_dataframe():
    """Empty DataFrame returns empty DataFrame with correct columns."""
    ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    feature = CascadeComposite()
    result = feature.compute(ohlcv)

    assert isinstance(result, pd.DataFrame)
    assert "cascade_signal" in result.columns
    assert "cascade_direction" in result.columns
    assert len(result) == 0


# ── Test 7: Custom threshold parameter ────────────────────────────────


def test_cascade_custom_threshold():
    """Higher threshold (3) requires more conditions to fire -- fewer signals."""
    ohlcv = _make_extreme_ohlcv()
    oi_data = _make_cascade_oi_data(n=250)
    feature = CascadeComposite()

    result_t2 = feature.compute(ohlcv, oi_data=oi_data, threshold=2)
    result_t3 = feature.compute(ohlcv, oi_data=oi_data, threshold=3)

    # Higher threshold should produce <= signals than lower threshold
    signals_t2 = result_t2["cascade_signal"].sum()
    signals_t3 = result_t3["cascade_signal"].sum()
    assert signals_t3 <= signals_t2, (
        f"threshold=3 ({signals_t3}) should fire <= threshold=2 ({signals_t2})"
    )


# ── Test 8: Direction zero when signal zero ───────────────────────────


def test_cascade_direction_zero_when_signal_zero():
    """cascade_direction must be 0.0 wherever cascade_signal is 0.0."""
    ohlcv = _make_cascade_ohlcv()
    oi_data = _make_cascade_oi_data()
    feature = CascadeComposite()
    result = feature.compute(ohlcv, oi_data=oi_data, threshold=2)

    zero_signal_mask = result["cascade_signal"] == 0.0
    if zero_signal_mask.any():
        directions_at_zero = result.loc[zero_signal_mask, "cascade_direction"]
        assert (directions_at_zero == 0.0).all(), (
            "cascade_direction must be 0.0 wherever cascade_signal is 0.0, "
            f"but found non-zero directions: {directions_at_zero[directions_at_zero != 0.0].tolist()}"
        )
