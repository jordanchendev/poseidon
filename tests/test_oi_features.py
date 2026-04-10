"""Tests for OI feature computation (OIChange and OIBuildup)."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.open_interest import OIChange, OIBuildup


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 40000 + rng.standard_normal(n).cumsum() * 100
    return pd.DataFrame(
        {
            "time": dates,
            "open": close - rng.uniform(0, 50, n),
            "high": close + rng.uniform(0, 100, n),
            "low": close - rng.uniform(0, 100, n),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=dates,
    )


def _make_oi_data(n: int = 50, trend: str = "rising") -> pd.DataFrame:
    """Create synthetic OI data for testing.

    Args:
        n: Number of data points.
        trend: "rising" for monotonically increasing OI, "flat" for constant.
    """
    dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    if trend == "rising":
        oi = 100000 + np.arange(n) * 500 + np.random.default_rng(42).standard_normal(n) * 50
    elif trend == "flat":
        oi = np.full(n, 100000.0)
    else:
        oi = 100000 + np.random.default_rng(42).standard_normal(n).cumsum() * 200
    return pd.DataFrame({"open_interest": oi}, index=dates)


class TestOIChange:
    """Test OIChange feature computation."""

    def test_output_columns(self):
        """OIChange should produce oi_change_pct and oi_change_zscore_{period}."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=oi_data, period=20)
        assert isinstance(result, pd.DataFrame)
        assert "oi_change_pct" in result.columns
        assert "oi_change_zscore_20" in result.columns
        assert len(result) == len(ohlcv)

    def test_custom_period(self):
        """OIChange should respect custom period parameter."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=oi_data, period=10)
        assert "oi_change_zscore_10" in result.columns

    def test_none_oi_data_returns_nan(self):
        """OIChange with no OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=None)
        assert isinstance(result, pd.DataFrame)
        assert result["oi_change_pct"].isna().all()
        assert result["oi_change_zscore_20"].isna().all()

    def test_empty_oi_data_returns_nan(self):
        """OIChange with empty OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=pd.DataFrame())
        assert result["oi_change_pct"].isna().all()

    def test_positive_oi_change(self):
        """Rising OI should produce positive change percentages."""
        ohlcv = _make_ohlcv(n=30)
        oi_data = _make_oi_data(n=30, trend="rising")
        feature = OIChange()
        result = feature.compute(ohlcv, oi_data=oi_data, period=10)
        # Skip first row (NaN from shift) and first few from rolling
        valid = result["oi_change_pct"].dropna()
        assert (valid > 0).sum() > len(valid) * 0.8  # Mostly positive for rising OI


class TestOIBuildup:
    """Test OIBuildup feature computation."""

    def test_output_columns(self):
        """OIBuildup should produce oi_buildup_{period} and oi_price_divergence_{period}."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=24)
        assert isinstance(result, pd.DataFrame)
        assert "oi_buildup_24" in result.columns
        assert "oi_price_divergence_24" in result.columns
        assert len(result) == len(ohlcv)

    def test_custom_period(self):
        """OIBuildup should respect custom period parameter."""
        ohlcv = _make_ohlcv()
        oi_data = _make_oi_data()
        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=12)
        assert "oi_buildup_12" in result.columns
        assert "oi_price_divergence_12" in result.columns

    def test_none_oi_data_returns_nan(self):
        """OIBuildup with no OI data should return NaN columns."""
        ohlcv = _make_ohlcv()
        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=None)
        assert result["oi_buildup_24"].isna().all()
        assert result["oi_price_divergence_24"].isna().all()

    def test_rising_oi_flat_price_positive_divergence(self):
        """Rising OI + flat price should produce positive divergence."""
        n = 50
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")

        # Flat price
        ohlcv = pd.DataFrame(
            {
                "time": dates,
                "open": np.full(n, 40000.0),
                "high": np.full(n, 40050.0),
                "low": np.full(n, 39950.0),
                "close": np.full(n, 40000.0),
                "volume": np.full(n, 500.0),
            },
            index=dates,
        )

        # Rising OI
        oi = 100000 + np.arange(n) * 1000
        oi_data = pd.DataFrame({"open_interest": oi}, index=dates)

        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=10)

        # After warmup period, divergence should be positive
        # (OI rising but price flat → OI change > abs(price change))
        valid_divergence = result["oi_price_divergence_10"].dropna()
        assert len(valid_divergence) > 0
        assert (valid_divergence > 0).sum() > len(valid_divergence) * 0.9

    def test_buildup_values_are_cumulative_pct(self):
        """OI buildup values should be cumulative percentage change over period."""
        n = 30
        dates = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        ohlcv = _make_ohlcv(n)
        # Linear increasing OI: 100k, 101k, 102k, ...
        oi_values = 100000 + np.arange(n) * 1000.0
        oi_data = pd.DataFrame({"open_interest": oi_values}, index=dates)

        feature = OIBuildup()
        result = feature.compute(ohlcv, oi_data=oi_data, period=5)
        # At index 5: oi_buildup = (105000 - 100000) / 100000 * 100 = 5.0%
        buildup_at_5 = result["oi_buildup_5"].iloc[5]
        assert abs(buildup_at_5 - 5.0) < 0.01
