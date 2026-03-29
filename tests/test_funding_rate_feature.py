"""Tests for FundingRateDaily feature class."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.funding_rate import FundingRateDaily


@pytest.fixture
def ohlcv_30d():
    """Synthetic 30-day daily OHLCV for a crypto spot symbol."""
    dates = pd.date_range("2024-06-01", periods=30, freq="D")
    rng = np.random.default_rng(42)
    close = 60000 + rng.normal(0, 500, 30).cumsum()
    return pd.DataFrame(
        {
            "time": dates,
            "open": close - rng.uniform(50, 200, 30),
            "high": close + rng.uniform(100, 400, 30),
            "low": close - rng.uniform(100, 400, 30),
            "close": close,
            "volume": rng.uniform(100, 1000, 30),
        },
        index=dates,
    )


@pytest.fixture
def funding_data_20d():
    """Synthetic funding data: 20 rows starting 5 days before OHLCV."""
    dates = pd.date_range("2024-05-27", periods=20, freq="D")
    rng = np.random.default_rng(99)
    rates = rng.uniform(-0.001, 0.003, 20)
    return pd.DataFrame({"funding_rate_daily": rates}, index=dates)


class TestFundingRateDaily:
    """Tests for the FundingRateDaily feature."""

    def test_compute_aligned_values(self, ohlcv_30d, funding_data_20d):
        """Aligned values match funding data where dates overlap."""
        feat = FundingRateDaily()
        result = feat.compute(ohlcv_30d, funding_data=funding_data_20d)

        assert result.name == "funding_rate_daily"
        assert len(result) == 30

        # Dates that exist in both OHLCV and funding should match exactly
        overlap = ohlcv_30d.index.intersection(funding_data_20d.index)
        for dt in overlap:
            assert result.loc[dt] == pytest.approx(
                funding_data_20d.loc[dt, "funding_rate_daily"]
            )

    def test_forward_fill_gaps(self, ohlcv_30d):
        """Gaps in funding data are forward-filled."""
        # Funding data with gaps: only every 3rd day
        dates = pd.date_range("2024-06-01", periods=10, freq="3D")
        funding = pd.DataFrame(
            {"funding_rate_daily": np.arange(10, dtype=float) * 0.001}, index=dates
        )
        feat = FundingRateDaily()
        result = feat.compute(ohlcv_30d, funding_data=funding)

        # Day after a funding date should be forward-filled (not NaN)
        assert not np.isnan(result.iloc[1])  # 2024-06-02 filled from 2024-06-01

    def test_none_funding_data_returns_nan(self, ohlcv_30d):
        """When funding_data is None, return NaN Series with correct length."""
        feat = FundingRateDaily()
        result = feat.compute(ohlcv_30d, funding_data=None)

        assert result.name == "funding_rate_daily"
        assert len(result) == 30
        assert result.isna().all()

    def test_empty_ohlcv_returns_empty(self):
        """Empty OHLCV returns empty Series."""
        feat = FundingRateDaily()
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        result = feat.compute(empty)

        assert result.name == "funding_rate_daily"
        assert len(result) == 0

    def test_missing_column_returns_nan(self, ohlcv_30d):
        """Funding data without the expected column returns NaN."""
        dates = pd.date_range("2024-06-01", periods=10, freq="D")
        bad_funding = pd.DataFrame({"wrong_col": [0.1] * 10}, index=dates)
        feat = FundingRateDaily()
        result = feat.compute(ohlcv_30d, funding_data=bad_funding)

        assert result.name == "funding_rate_daily"
        assert result.isna().all()
