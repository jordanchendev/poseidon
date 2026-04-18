"""Tests for quality factor feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.quality_factor import (
    QualityGrowthZ,
    QualityProfitabilityZ,
    QualitySafetyZ,
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
def quality_factor_data(ohlcv):
    """Synthetic quality factor data (monthly-ish, every 20th day)."""
    monthly_dates = ohlcv.index[::20]  # ~5 monthly points
    return pd.DataFrame(
        {
            "profitability_z": [0.5, 0.7, 0.3, 0.8, 0.6],
            "growth_z": [0.2, 0.4, 0.1, 0.6, 0.3],
            "safety_z": [-0.1, 0.2, -0.3, 0.1, 0.0],
        },
        index=monthly_dates[:5],
    )


class TestQualityProfitabilityZ:
    def test_ffill_to_daily(self, ohlcv, quality_factor_data):
        feat = QualityProfitabilityZ()
        result = feat.compute(ohlcv, quality_factor_data=quality_factor_data)
        assert result.name == "quality_profitability_z"
        assert len(result) == len(ohlcv)
        # First value should match first data point
        assert result.iloc[0] == 0.5
        # Should not be all NaN
        assert not result.isna().all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = QualityProfitabilityZ()
        result = feat.compute(ohlcv, quality_factor_data=None)
        assert result.name == "quality_profitability_z"
        assert result.isna().all()

    def test_empty_data_returns_nan(self, ohlcv):
        feat = QualityProfitabilityZ()
        result = feat.compute(ohlcv, quality_factor_data=pd.DataFrame())
        assert result.name == "quality_profitability_z"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = QualityProfitabilityZ()
        wrong = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, quality_factor_data=wrong)
        assert result.name == "quality_profitability_z"
        assert result.isna().all()


class TestQualityGrowthZ:
    def test_ffill_to_daily(self, ohlcv, quality_factor_data):
        feat = QualityGrowthZ()
        result = feat.compute(ohlcv, quality_factor_data=quality_factor_data)
        assert result.name == "quality_growth_z"
        assert len(result) == len(ohlcv)
        assert result.iloc[0] == 0.2
        assert not result.isna().all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = QualityGrowthZ()
        result = feat.compute(ohlcv, quality_factor_data=None)
        assert result.name == "quality_growth_z"
        assert result.isna().all()


class TestQualitySafetyZ:
    def test_ffill_to_daily(self, ohlcv, quality_factor_data):
        feat = QualitySafetyZ()
        result = feat.compute(ohlcv, quality_factor_data=quality_factor_data)
        assert result.name == "quality_safety_z"
        assert len(result) == len(ohlcv)
        assert result.iloc[0] == -0.1
        assert not result.isna().all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = QualitySafetyZ()
        result = feat.compute(ohlcv, quality_factor_data=None)
        assert result.name == "quality_safety_z"
        assert result.isna().all()
