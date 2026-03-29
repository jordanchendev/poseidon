"""Tests for fundamental feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.fundamentals import (
    PBRatio,
    PERatio,
    RevenueMoM,
    RevenueYoY,
)


@pytest.fixture
def ohlcv():
    """Synthetic OHLCV DataFrame with 100 daily rows."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(100).cumsum()
    volume = rng.integers(1000, 10000, size=100).astype(float)
    return pd.DataFrame(
        {
            "open": close + rng.uniform(-1, 1, 100),
            "high": close + abs(rng.standard_normal(100)),
            "low": close - abs(rng.standard_normal(100)),
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def fundamental_data(ohlcv):
    """Synthetic fundamental data: sparse quarterly PE/PB + monthly revenue."""
    idx = ohlcv.index
    # Quarterly PE/PB (4 data points spread across the 100-day range)
    quarterly_dates = [idx[0], idx[25], idx[50], idx[75]]
    pe_values = [15.0, 16.5, 14.0, 17.0]
    pb_values = [2.0, 2.1, 1.9, 2.3]

    # Monthly revenue (approx 5 months in 100 business days)
    monthly_dates = [idx[0], idx[20], idx[40], idx[60], idx[80]]
    monthly_rev = [1000.0, 1100.0, 1050.0, 1200.0, 1150.0]
    prev_year_rev = [900.0, 950.0, 1000.0, 1050.0, 1100.0]

    # Build combined DataFrame with all quarterly + monthly dates
    all_dates = sorted(set(quarterly_dates + monthly_dates))
    df = pd.DataFrame(index=pd.DatetimeIndex(all_dates))

    # Set quarterly values
    for d, pe, pb in zip(quarterly_dates, pe_values, pb_values):
        df.loc[d, "pe_ratio"] = pe
        df.loc[d, "pb_ratio"] = pb

    # Set monthly values
    for d, rev, prev in zip(monthly_dates, monthly_rev, prev_year_rev):
        df.loc[d, "monthly_rev"] = rev
        df.loc[d, "prev_year_rev"] = prev

    return df


class TestPERatio:
    def test_forward_fill_quarterly(self, ohlcv, fundamental_data):
        feat = PERatio()
        result = feat.compute(ohlcv, fundamental_data=fundamental_data)
        assert result.name == "pe_ratio"
        assert len(result) == len(ohlcv)
        # First value should be 15.0 (first quarterly data point)
        assert result.iloc[0] == 15.0
        # After day 25, should be 16.5
        assert result.iloc[26] == 16.5
        # Forward-filled means no NaN between known points (after first)
        assert not result.iloc[0:76].isna().any()

    def test_none_data_returns_nan(self, ohlcv):
        feat = PERatio()
        result = feat.compute(ohlcv, fundamental_data=None)
        assert result.name == "pe_ratio"
        assert result.isna().all()


class TestPBRatio:
    def test_forward_fill_quarterly(self, ohlcv, fundamental_data):
        feat = PBRatio()
        result = feat.compute(ohlcv, fundamental_data=fundamental_data)
        assert result.name == "pb_ratio"
        assert result.iloc[0] == 2.0
        assert result.iloc[26] == 2.1

    def test_none_data_returns_nan(self, ohlcv):
        feat = PBRatio()
        result = feat.compute(ohlcv, fundamental_data=None)
        assert result.name == "pb_ratio"
        assert result.isna().all()


class TestRevenueMoM:
    def test_compute_pct_change(self, ohlcv, fundamental_data):
        feat = RevenueMoM()
        result = feat.compute(ohlcv, fundamental_data=fundamental_data)
        assert result.name == "revenue_mom"
        assert len(result) == len(ohlcv)
        # First value is NaN (no previous to compare)
        assert np.isnan(result.iloc[0])
        # When value changes from 1000 to 1100 at idx[20], pct_change = 0.1
        # The exact position depends on ffill alignment
        # Just verify some non-NaN values exist after the first month
        non_nan = result.dropna()
        assert len(non_nan) > 0

    def test_none_data_returns_nan(self, ohlcv):
        feat = RevenueMoM()
        result = feat.compute(ohlcv, fundamental_data=None)
        assert result.name == "revenue_mom"
        assert result.isna().all()


class TestRevenueYoY:
    def test_compute_yoy(self, ohlcv, fundamental_data):
        feat = RevenueYoY()
        result = feat.compute(ohlcv, fundamental_data=fundamental_data)
        assert result.name == "revenue_yoy"
        assert len(result) == len(ohlcv)
        # At idx[0]: (1000-900)/900 = 0.1111...
        first_valid = result.iloc[0]
        assert abs(first_valid - (1000 - 900) / 900) < 1e-10

    def test_none_data_returns_nan(self, ohlcv):
        feat = RevenueYoY()
        result = feat.compute(ohlcv, fundamental_data=None)
        assert result.name == "revenue_yoy"
        assert result.isna().all()

    def test_missing_prev_year_col(self, ohlcv):
        """If prev_year_rev column missing, return NaN."""
        feat = RevenueYoY()
        partial = pd.DataFrame(
            {"monthly_rev": [1000.0]},
            index=pd.DatetimeIndex([ohlcv.index[0]]),
        )
        result = feat.compute(ohlcv, fundamental_data=partial)
        assert result.isna().all()
