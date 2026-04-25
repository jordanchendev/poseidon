"""Unit tests for CVD (Cumulative Volume Delta) feature.

Tests the CVD feature class which computes rolling change in cumulative
volume delta using BVC classification. Key invariant: output is a
rolling change (stationary), NOT raw cumulative sum (non-stationary).
"""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.cvd import CVD

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def bullish_ohlcv() -> pd.DataFrame:
    """30 bars where close > open consistently (buying pressure)."""
    n = 30
    opens = np.full(n, 100.0)
    closes = np.full(n, 108.0)
    highs = np.full(n, 110.0)
    lows = np.full(n, 95.0)
    volumes = np.full(n, 1000.0)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})


@pytest.fixture
def bearish_ohlcv() -> pd.DataFrame:
    """30 bars where close < open consistently (selling pressure)."""
    n = 30
    opens = np.full(n, 108.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 110.0)
    lows = np.full(n, 95.0)
    volumes = np.full(n, 1000.0)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})


@pytest.fixture
def random_ohlcv() -> pd.DataFrame:
    """100 bars with random close positions for stationarity test."""
    rng = np.random.default_rng(42)
    n = 100
    lows = rng.uniform(90, 100, n)
    highs = lows + rng.uniform(5, 20, n)
    opens = lows + rng.uniform(0, 1, n) * (highs - lows)
    closes = lows + rng.uniform(0, 1, n) * (highs - lows)
    volumes = rng.uniform(500, 3000, n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})


@pytest.fixture
def empty_ohlcv() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


# -- Test 1: Return type and naming ------------------------------------------


def test_cvd_returns_series_with_correct_name(bullish_ohlcv: pd.DataFrame):
    """CVD returns Series named cvd_change_{period}."""
    result = CVD().compute(bullish_ohlcv, period=20)
    assert isinstance(result, pd.Series)
    assert result.name == "cvd_change_20"


# -- Test 2: Bullish bars produce positive cvd_change ------------------------


def test_bullish_bars_positive_cvd_change(bullish_ohlcv: pd.DataFrame):
    """With all-bullish bars, cvd_change is positive (net buying pressure)."""
    result = CVD().compute(bullish_ohlcv, period=20)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid > 0).all(), f"Expected all positive, got: {valid.values}"


# -- Test 3: Bearish bars produce negative cvd_change -------------------------


def test_bearish_bars_negative_cvd_change(bearish_ohlcv: pd.DataFrame):
    """With all-bearish bars, cvd_change is negative (net selling pressure)."""
    result = CVD().compute(bearish_ohlcv, period=20)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid < 0).all(), f"Expected all negative, got: {valid.values}"


# -- Test 4: Rolling change is stationary (NOT raw cumulative) ----------------


def test_rolling_change_is_stationary(random_ohlcv: pd.DataFrame):
    """Rolling change should be approximately mean-zero for random data.

    Raw CVD cumsum would have a non-zero drift; the rolling change
    (cvd[t] - cvd[t-period]) should be stationary.
    """
    result = CVD().compute(random_ohlcv, period=20)
    valid = result.dropna()
    assert len(valid) > 0
    # For random data, the mean of rolling change should be near zero
    # relative to the scale of values
    mean_val = valid.mean()
    std_val = valid.std()
    # Mean should be within 2 standard deviations of zero (loose check for stationarity)
    assert abs(mean_val) < 2 * std_val, f"Rolling change appears non-stationary: mean={mean_val:.4f}, std={std_val:.4f}"


# -- Test 5: Empty DataFrame returns empty Series ----------------------------


def test_empty_dataframe_returns_empty_series(empty_ohlcv: pd.DataFrame):
    """Empty DataFrame returns empty Series with correct name."""
    result = CVD().compute(empty_ohlcv, period=20)
    assert isinstance(result, pd.Series)
    assert result.name == "cvd_change_20"
    assert len(result) == 0


# -- Test 6: First `period` values are NaN ------------------------------------


def test_warmup_period_produces_nan(bullish_ohlcv: pd.DataFrame):
    """First `period` values are NaN (insufficient warmup for rolling change)."""
    period = 20
    result = CVD().compute(bullish_ohlcv, period=period)
    # The first `period` values should be NaN because
    # cvd.shift(period) is NaN for the first period rows
    assert result.iloc[:period].isna().all()
    # After warmup, values should be valid
    assert result.iloc[period:].notna().all()
