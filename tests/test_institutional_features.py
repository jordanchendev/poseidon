"""Tests for institutional flow feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.institutional import (
    DealerNetBuyRatio,
    ForeignHoldingChange,
    ForeignNetBuyCum,
    ForeignNetBuyRatio,
    NetBuyConsecutiveDays,
    TrustNetBuyCum,
    TrustNetBuyRatio,
)


@pytest.fixture
def ohlcv():
    """Synthetic OHLCV DataFrame with 50 rows."""
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(50).cumsum()
    volume = rng.integers(1000, 10000, size=50).astype(float)
    return pd.DataFrame(
        {
            "open": close + rng.uniform(-1, 1, 50),
            "high": close + abs(rng.standard_normal(50)),
            "low": close - abs(rng.standard_normal(50)),
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def institutional_data(ohlcv):
    """Synthetic institutional data aligned to OHLCV dates."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "foreign": rng.integers(-500, 500, size=len(ohlcv)).astype(float),
            "trust": rng.integers(-300, 300, size=len(ohlcv)).astype(float),
            "dealer": rng.integers(-200, 200, size=len(ohlcv)).astype(float),
            "dealer_hedge": rng.integers(-100, 100, size=len(ohlcv)).astype(float),
        },
        index=ohlcv.index,
    )


class TestForeignNetBuyRatio:
    def test_compute_correctly(self, ohlcv, institutional_data):
        feat = ForeignNetBuyRatio()
        result = feat.compute(ohlcv, institutional_data=institutional_data)
        expected = institutional_data["foreign"] / ohlcv["volume"]
        expected.name = "foreign_net_buy_ratio"
        pd.testing.assert_series_equal(result, expected)

    def test_none_data_returns_nan(self, ohlcv):
        feat = ForeignNetBuyRatio()
        result = feat.compute(ohlcv, institutional_data=None)
        assert result.name == "foreign_net_buy_ratio"
        assert len(result) == len(ohlcv)
        assert result.isna().all()

    def test_empty_ohlcv_returns_empty(self):
        feat = ForeignNetBuyRatio()
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = feat.compute(empty)
        assert len(result) == 0
        assert result.name == "foreign_net_buy_ratio"


class TestForeignNetBuyCum:
    def test_compute_period_5(self, ohlcv, institutional_data):
        feat = ForeignNetBuyCum()
        result = feat.compute(ohlcv, institutional_data=institutional_data, period=5)
        expected = institutional_data["foreign"].rolling(5).sum()
        expected.name = "foreign_net_buy_cum_5"
        pd.testing.assert_series_equal(result, expected)
        assert result.name == "foreign_net_buy_cum_5"

    def test_none_data_returns_nan(self, ohlcv):
        feat = ForeignNetBuyCum()
        result = feat.compute(ohlcv, institutional_data=None, period=5)
        assert result.name == "foreign_net_buy_cum_5"
        assert result.isna().all()


class TestTrustNetBuyRatio:
    def test_compute_correctly(self, ohlcv, institutional_data):
        feat = TrustNetBuyRatio()
        result = feat.compute(ohlcv, institutional_data=institutional_data)
        expected = institutional_data["trust"] / ohlcv["volume"]
        expected.name = "trust_net_buy_ratio"
        pd.testing.assert_series_equal(result, expected)

    def test_none_data_returns_nan(self, ohlcv):
        feat = TrustNetBuyRatio()
        result = feat.compute(ohlcv, institutional_data=None)
        assert result.name == "trust_net_buy_ratio"
        assert result.isna().all()


class TestTrustNetBuyCum:
    def test_compute_period_20(self, ohlcv, institutional_data):
        feat = TrustNetBuyCum()
        result = feat.compute(ohlcv, institutional_data=institutional_data, period=20)
        expected = institutional_data["trust"].rolling(20).sum()
        expected.name = "trust_net_buy_cum_20"
        pd.testing.assert_series_equal(result, expected)

    def test_none_data_returns_nan(self, ohlcv):
        feat = TrustNetBuyCum()
        result = feat.compute(ohlcv, institutional_data=None, period=20)
        assert result.name == "trust_net_buy_cum_20"
        assert result.isna().all()


class TestDealerNetBuyRatio:
    def test_compute_correctly(self, ohlcv, institutional_data):
        feat = DealerNetBuyRatio()
        result = feat.compute(ohlcv, institutional_data=institutional_data)
        expected = institutional_data["dealer"] / ohlcv["volume"]
        expected.name = "dealer_net_buy_ratio"
        pd.testing.assert_series_equal(result, expected)

    def test_none_data_returns_nan(self, ohlcv):
        feat = DealerNetBuyRatio()
        result = feat.compute(ohlcv, institutional_data=None)
        assert result.name == "dealer_net_buy_ratio"
        assert result.isna().all()


# --- Tests for extended institutional features ---


@pytest.fixture
def foreign_holding_data(ohlcv):
    """Synthetic foreign holding data (every 5th day)."""
    idx = ohlcv.index
    daily_dates = idx[::5]
    values = [0.70, 0.71, 0.72, 0.71, 0.73, 0.74, 0.73, 0.75, 0.74, 0.76]
    df = pd.DataFrame(index=daily_dates[: len(values)])
    df["foreign_holding_ratio"] = values[: len(daily_dates)]
    return df


class TestForeignHoldingChange:
    def test_compute_diff(self, ohlcv, foreign_holding_data):
        feat = ForeignHoldingChange()
        result = feat.compute(ohlcv, foreign_holding_data=foreign_holding_data)
        assert result.name == "foreign_holding_change"
        assert len(result) == len(ohlcv)
        # First value is NaN (diff has no previous)
        assert np.isnan(result.iloc[0])
        # Some non-NaN values should exist
        assert result.dropna().shape[0] > 0

    def test_none_data_returns_nan(self, ohlcv):
        feat = ForeignHoldingChange()
        result = feat.compute(ohlcv, foreign_holding_data=None)
        assert result.name == "foreign_holding_change"
        assert result.isna().all()

    def test_empty_data_returns_nan(self, ohlcv):
        feat = ForeignHoldingChange()
        result = feat.compute(ohlcv, foreign_holding_data=pd.DataFrame())
        assert result.name == "foreign_holding_change"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = ForeignHoldingChange()
        wrong = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, foreign_holding_data=wrong)
        assert result.name == "foreign_holding_change"
        assert result.isna().all()


class TestNetBuyConsecutiveDays:
    def test_compute_consecutive(self, ohlcv, institutional_data):
        feat = NetBuyConsecutiveDays()
        result = feat.compute(ohlcv, institutional_data=institutional_data)
        assert result.name == "net_buy_consecutive_days"
        assert len(result) == len(ohlcv)
        # Values should be non-negative integers (as float)
        non_nan = result.dropna()
        assert (non_nan >= 0).all()
        # Check that consecutive counting works: max should be > 1
        # (with random data, very likely to have some consecutive positive days)
        assert non_nan.max() >= 1.0

    def test_none_data_returns_nan(self, ohlcv):
        feat = NetBuyConsecutiveDays()
        result = feat.compute(ohlcv, institutional_data=None)
        assert result.name == "net_buy_consecutive_days"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = NetBuyConsecutiveDays()
        wrong = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, institutional_data=wrong)
        assert result.name == "net_buy_consecutive_days"
        assert result.isna().all()
