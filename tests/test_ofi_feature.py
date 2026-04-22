"""Unit tests for OFI (Order Flow Imbalance) feature.

Tests the OFI feature class which computes rolling sum of
delta(buy_volume) - delta(sell_volume) using BVC classification.
"""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.ofi import OFI


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def constant_volume_ohlcv() -> pd.DataFrame:
    """20 bars with identical OHLCV values (no volume change)."""
    n = 20
    return pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 110.0),
            "low": np.full(n, 95.0),
            "close": np.full(n, 105.0),
            "volume": np.full(n, 1000.0),
        }
    )


@pytest.fixture
def increasing_buy_pressure_ohlcv() -> pd.DataFrame:
    """20 bars where close position within range rises progressively.

    close/open ratio increases => buy_volume fraction increases each bar,
    so delta(buy_volume) > 0 and delta(sell_volume) < 0 => positive OFI.
    """
    n = 20
    lows = np.full(n, 90.0)
    highs = np.full(n, 110.0)
    opens = np.full(n, 100.0)
    # Close rises from 95 to 109 (within the 90-110 range)
    closes = np.linspace(95, 109, n)
    volumes = np.full(n, 1000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


@pytest.fixture
def empty_ohlcv() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def simple_ohlcv() -> pd.DataFrame:
    """10 bars with varied patterns for detailed OFI testing."""
    return pd.DataFrame(
        {
            "open": [100, 102, 104, 103, 101, 105, 107, 106, 104, 108],
            "high": [110, 112, 114, 113, 111, 115, 117, 116, 114, 118],
            "low": [90, 92, 94, 93, 91, 95, 97, 96, 94, 98],
            "close": [108, 110, 112, 95, 98, 113, 115, 97, 100, 116],
            "volume": [1000, 1200, 1100, 1300, 900, 1400, 1500, 800, 1000, 1600],
        }
    )


# -- Test 1: Return type and naming ------------------------------------------


def test_ofi_returns_series_with_correct_name(constant_volume_ohlcv: pd.DataFrame):
    """OFI returns Series named ofi_{period}."""
    result = OFI().compute(constant_volume_ohlcv, period=5)
    assert isinstance(result, pd.Series)
    assert result.name == "ofi_5"


# -- Test 2: Constant volume bars produce OFI near zero ----------------------


def test_constant_volume_produces_near_zero_ofi(constant_volume_ohlcv: pd.DataFrame):
    """Constant identical bars produce OFI near zero (no delta in buy/sell)."""
    result = OFI().compute(constant_volume_ohlcv, period=5)
    valid = result.dropna()
    assert len(valid) > 0
    # With constant OHLCV, buy/sell volumes are constant, so diff() is 0
    # Rolling sum of zeros = 0
    assert (valid.abs() < 1e-6).all(), f"Expected near zero, got: {valid.values}"


# -- Test 3: Increasing buy pressure produces positive OFI -------------------


def test_increasing_buy_pressure_positive_ofi(increasing_buy_pressure_ohlcv: pd.DataFrame):
    """Increasing close position (more buying) produces positive OFI."""
    result = OFI().compute(increasing_buy_pressure_ohlcv, period=5)
    valid = result.dropna()
    assert len(valid) > 0
    # Close is rising within range => buy_pct increases each bar
    # => delta(buy_volume) > 0, delta(sell_volume) < 0
    # => OFI = delta(buy) - delta(sell) > 0
    assert (valid > 0).all(), f"Expected all positive, got: {valid.values}"


# -- Test 4: Empty DataFrame returns empty Series ----------------------------


def test_empty_dataframe_returns_empty_series(empty_ohlcv: pd.DataFrame):
    """Empty DataFrame returns empty Series with correct name."""
    result = OFI().compute(empty_ohlcv, period=5)
    assert isinstance(result, pd.Series)
    assert result.name == "ofi_5"
    assert len(result) == 0


# -- Test 5: First few values are NaN ----------------------------------------


def test_warmup_period_produces_nan(simple_ohlcv: pd.DataFrame):
    """First few values are NaN (needs diff + rolling warmup).

    diff() produces 1 NaN at the start, then rolling(window=period)
    needs (period - 1) more before producing a valid sum.
    Total NaN warmup = 1 + (period - 1) = period.
    """
    period = 5
    result = OFI().compute(simple_ohlcv, period=period)
    # First `period` values should be NaN
    assert result.iloc[:period].isna().all()


# -- Test 6: OFI with period=1 equals raw ofi_raw for valid rows -------------


def test_period_1_equals_raw_ofi(simple_ohlcv: pd.DataFrame):
    """OFI with period=1 should equal raw delta(buy) - delta(sell).

    rolling(window=1).sum() is identity, so ofi_1 == ofi_raw for valid rows.
    """
    from poseidon.data.features.bvc import classify_volume

    bvc = classify_volume(simple_ohlcv)
    ofi_raw = bvc["buy_volume"].diff() - bvc["sell_volume"].diff()

    result = OFI().compute(simple_ohlcv, period=1)

    # Compare valid (non-NaN) values
    valid_mask = result.notna() & ofi_raw.notna()
    pd.testing.assert_series_equal(
        result[valid_mask].reset_index(drop=True),
        ofi_raw[valid_mask].reset_index(drop=True),
        check_names=False,
        atol=1e-10,
    )
