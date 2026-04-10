"""Tests for margin feature classes."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.margin import MarginBuyRatio, MarginSellRatio


@pytest.fixture
def ohlcv():
    """Synthetic OHLCV DataFrame with 100 daily rows."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B", tz="UTC")
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
def margin_data(ohlcv):
    """Synthetic margin transaction data (daily frequency, naive datetime)."""
    # Use naive datetime to test timezone alignment (FinLab returns naive)
    naive_dates = ohlcv.index.tz_localize(None)
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "margin_buy_ratio": rng.uniform(0.3, 0.8, len(ohlcv)),
            "margin_sell_ratio": rng.uniform(0.01, 0.15, len(ohlcv)),
        },
        index=naive_dates,
    )


class TestMarginBuyRatio:
    def test_compute_with_data(self, ohlcv, margin_data):
        feat = MarginBuyRatio()
        result = feat.compute(ohlcv, margin_data=margin_data)
        assert result.name == "margin_buy_ratio"
        assert len(result) == len(ohlcv)
        # Values should be non-NaN (timezone alignment worked)
        assert not result.isna().all()
        # Values should be in the expected range
        assert result.dropna().min() >= 0.0

    def test_none_data_returns_nan(self, ohlcv):
        feat = MarginBuyRatio()
        result = feat.compute(ohlcv, margin_data=None)
        assert result.name == "margin_buy_ratio"
        assert result.isna().all()

    def test_empty_data_returns_nan(self, ohlcv):
        feat = MarginBuyRatio()
        result = feat.compute(ohlcv, margin_data=pd.DataFrame())
        assert result.name == "margin_buy_ratio"
        assert result.isna().all()

    def test_missing_column_returns_nan(self, ohlcv):
        feat = MarginBuyRatio()
        wrong_cols = pd.DataFrame({"wrong_col": [1.0]}, index=ohlcv.index[:1])
        result = feat.compute(ohlcv, margin_data=wrong_cols)
        assert result.name == "margin_buy_ratio"
        assert result.isna().all()


class TestMarginSellRatio:
    def test_compute_with_data(self, ohlcv, margin_data):
        feat = MarginSellRatio()
        result = feat.compute(ohlcv, margin_data=margin_data)
        assert result.name == "margin_sell_ratio"
        assert len(result) == len(ohlcv)
        assert not result.isna().all()

    def test_none_data_returns_nan(self, ohlcv):
        feat = MarginSellRatio()
        result = feat.compute(ohlcv, margin_data=None)
        assert result.name == "margin_sell_ratio"
        assert result.isna().all()

    def test_empty_data_returns_nan(self, ohlcv):
        feat = MarginSellRatio()
        result = feat.compute(ohlcv, margin_data=pd.DataFrame())
        assert result.name == "margin_sell_ratio"
        assert result.isna().all()
