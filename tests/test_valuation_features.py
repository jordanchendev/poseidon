"""Tests for valuation feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.valuation import (
    DividendYield,
    DividendYieldPercentile,
    PBRPercentile,
    PEPercentile,
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
def pe_pbr_data(ohlcv):
    """Synthetic PE/PBR/dividend yield data (every 5th day, 20 rows)."""
    idx = ohlcv.index
    daily_dates = idx[::5]  # every 5th day
    rng = np.random.default_rng(77)
    n = len(daily_dates)
    return pd.DataFrame(
        {
            "pe_ratio": rng.uniform(10.0, 25.0, n),
            "pb_ratio": rng.uniform(1.0, 4.0, n),
            "dividend_yield": rng.uniform(0.01, 0.06, n),
        },
        index=daily_dates,
    )


class TestDividendYield:
    def test_ffill_to_daily(self, ohlcv, pe_pbr_data):
        feat = DividendYield()
        result = feat.compute(ohlcv, pe_pbr_data=pe_pbr_data)
        assert result.name == "dividend_yield"
        assert len(result) == len(ohlcv)
        # First value should match first data point
        assert result.iloc[0] == pe_pbr_data["dividend_yield"].iloc[0]
        assert not result.isna().all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = DividendYield()
        result = feat.compute(ohlcv, pe_pbr_data=None)
        assert result.name == "dividend_yield"
        assert result.isna().all()

    def test_empty_data_returns_nan(self, ohlcv):
        feat = DividendYield()
        result = feat.compute(ohlcv, pe_pbr_data=pd.DataFrame())
        assert result.name == "dividend_yield"
        assert result.isna().all()


class TestPEPercentile:
    def test_rolling_rank(self, ohlcv, pe_pbr_data):
        feat = PEPercentile()
        result = feat.compute(ohlcv, pe_pbr_data=pe_pbr_data)
        assert result.name == "pe_percentile"
        assert len(result) == len(ohlcv)
        # Rolling rank should produce values between 0 and 1
        non_nan = result.dropna()
        assert (non_nan >= 0.0).all()
        assert (non_nan <= 1.0).all()
        assert non_nan.shape[0] > 0

    def test_none_data_returns_nan(self, ohlcv):
        feat = PEPercentile()
        result = feat.compute(ohlcv, pe_pbr_data=None)
        assert result.name == "pe_percentile"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = PEPercentile()
        wrong = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, pe_pbr_data=wrong)
        assert result.name == "pe_percentile"
        assert result.isna().all()


class TestPBRPercentile:
    def test_rolling_rank(self, ohlcv, pe_pbr_data):
        feat = PBRPercentile()
        result = feat.compute(ohlcv, pe_pbr_data=pe_pbr_data)
        assert result.name == "pbr_percentile"
        assert len(result) == len(ohlcv)
        non_nan = result.dropna()
        assert (non_nan >= 0.0).all()
        assert (non_nan <= 1.0).all()
        assert non_nan.shape[0] > 0

    def test_none_data_returns_nan(self, ohlcv):
        feat = PBRPercentile()
        result = feat.compute(ohlcv, pe_pbr_data=None)
        assert result.name == "pbr_percentile"
        assert result.isna().all()


class TestDividendYieldPercentile:
    def test_rolling_rank(self, ohlcv, pe_pbr_data):
        feat = DividendYieldPercentile()
        result = feat.compute(ohlcv, pe_pbr_data=pe_pbr_data)
        assert result.name == "dividend_yield_percentile"
        assert len(result) == len(ohlcv)
        non_nan = result.dropna()
        assert (non_nan >= 0.0).all()
        assert (non_nan <= 1.0).all()
        assert non_nan.shape[0] > 0

    def test_none_data_returns_nan(self, ohlcv):
        feat = DividendYieldPercentile()
        result = feat.compute(ohlcv, pe_pbr_data=None)
        assert result.name == "dividend_yield_percentile"
        assert result.isna().all()
