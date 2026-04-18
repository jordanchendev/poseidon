"""Tests for monthly revenue feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.monthly_revenue import (
    CumulativeRevenueGrowth,
    RevenueAccelerationMonths,
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
def monthly_revenue_data(ohlcv):
    """Synthetic monthly revenue data.

    YoY values: [0.05, 0.08, 0.12, 0.10, 0.15]
    - Accelerating: 0.05 -> 0.08 -> 0.12 (diff > 0 for months 2 and 3)
    - Decelerating: 0.12 -> 0.10 (diff < 0 for month 4)
    - Accelerating again: 0.10 -> 0.15 (diff > 0 for month 5)
    """
    monthly_dates = ohlcv.index[::20]  # ~5 monthly points
    return pd.DataFrame(
        {
            "monthly_rev": [5000.0, 5400.0, 5600.0, 5500.0, 5750.0],
            "monthly_rev_yoy": [0.05, 0.08, 0.12, 0.10, 0.15],
            "cum_rev": [5000.0, 10400.0, 16000.0, 21500.0, 27250.0],
            "cum_rev_yoy": [0.05, 0.065, 0.078, 0.073, 0.082],
        },
        index=monthly_dates[:5],
    )


class TestCumulativeRevenueGrowth:
    def test_ffill_to_daily(self, ohlcv, monthly_revenue_data):
        feat = CumulativeRevenueGrowth()
        result = feat.compute(ohlcv, monthly_revenue_data=monthly_revenue_data)
        assert result.name == "cumulative_revenue_growth"
        assert len(result) == len(ohlcv)
        # First value should match first data point
        assert result.iloc[0] == 0.05
        # Should not be all NaN
        assert not result.isna().all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = CumulativeRevenueGrowth()
        result = feat.compute(ohlcv, monthly_revenue_data=None)
        assert result.name == "cumulative_revenue_growth"
        assert result.isna().all()

    def test_empty_data_returns_nan(self, ohlcv):
        feat = CumulativeRevenueGrowth()
        result = feat.compute(ohlcv, monthly_revenue_data=pd.DataFrame())
        assert result.name == "cumulative_revenue_growth"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = CumulativeRevenueGrowth()
        wrong = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, monthly_revenue_data=wrong)
        assert result.name == "cumulative_revenue_growth"
        assert result.isna().all()


class TestRevenueAccelerationMonths:
    def test_acceleration_counting(self, ohlcv, monthly_revenue_data):
        """Verify consecutive count resets on deceleration.

        YoY: [0.05, 0.08, 0.12, 0.10, 0.15]
        Diff: [NaN,  0.03, 0.04, -0.02, 0.05]
        Positive: [0, 1, 1, 0, 1]  (first is 0 because diff is NaN)
        Consecutive: [0, 1, 2, 0, 1]
        """
        feat = RevenueAccelerationMonths()
        result = feat.compute(ohlcv, monthly_revenue_data=monthly_revenue_data)
        assert result.name == "revenue_acceleration_months"
        assert len(result) == len(ohlcv)
        # After the second monthly point (accelerating 0.05->0.08), count = 1
        # After third (0.08->0.12), count = 2
        # After fourth (0.12->0.10, decelerating), count resets to 0
        # After fifth (0.10->0.15, accelerating again), count = 1
        non_nan = result.dropna()
        assert non_nan.shape[0] > 0
        # Max should be 2 (consecutive acceleration for months 2-3)
        assert non_nan.max() == 2.0
        # Last value should be 1 (only one month of acceleration after reset)
        assert result.iloc[-1] == 1.0

    def test_none_data_returns_nan(self, ohlcv):
        feat = RevenueAccelerationMonths()
        result = feat.compute(ohlcv, monthly_revenue_data=None)
        assert result.name == "revenue_acceleration_months"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = RevenueAccelerationMonths()
        wrong = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, monthly_revenue_data=wrong)
        assert result.name == "revenue_acceleration_months"
        assert result.isna().all()
