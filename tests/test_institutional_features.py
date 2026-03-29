"""Tests for institutional flow feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.institutional import (
    DealerNetBuyRatio,
    ForeignNetBuyCum,
    ForeignNetBuyRatio,
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
