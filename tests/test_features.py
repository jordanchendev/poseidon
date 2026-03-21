"""Tests for individual feature implementations."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features import get_feature, list_features
from poseidon.data.features.base import BaseFeature
from poseidon.data.features.returns import CumulativeReturn, Returns
from poseidon.data.features.technical import ATR, SMA, EMA, MACD, RSI, BollingerBands
from poseidon.data.features.volatility import (
    GarmanKlassVolatility,
    ParkinsonVolatility,
    StandardVolatility,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "time": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def empty_ohlcv() -> pd.DataFrame:
    """Empty OHLCV DataFrame."""
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


# --- Registry Tests ---


def test_all_features_registered():
    """All 11 features should be registered."""
    names = list_features()
    assert len(names) == 11
    expected = {
        "sma", "ema", "rsi", "macd", "bollinger", "atr",
        "returns", "cum_return", "std_vol", "parkinson_vol", "garman_klass_vol",
    }
    assert set(names) == expected


def test_get_feature_returns_class():
    cls = get_feature("sma")
    assert issubclass(cls, BaseFeature)


def test_get_unknown_feature_raises():
    with pytest.raises(KeyError, match="Unknown feature"):
        get_feature("nonexistent")


# --- SMA ---


def test_sma_basic():
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=5, freq="D"),
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1.0, 2.0, 3.0, 4.0, 5.0],
        "volume": [100] * 5,
    })
    result = SMA().compute(df, period=3)
    assert result.name == "sma_3"
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_sma_empty(empty_ohlcv):
    result = SMA().compute(empty_ohlcv, period=20)
    assert isinstance(result, pd.Series)
    assert len(result) == 0


# --- EMA ---


def test_ema_basic(sample_ohlcv):
    result = EMA().compute(sample_ohlcv, period=10)
    assert result.name == "ema_10"
    assert len(result) == len(sample_ohlcv)
    assert not np.isnan(result.iloc[-1])


# --- RSI ---


def test_rsi_range(sample_ohlcv):
    result = RSI().compute(sample_ohlcv, period=14)
    assert result.name == "rsi_14"
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_empty(empty_ohlcv):
    result = RSI().compute(empty_ohlcv, period=14)
    assert isinstance(result, pd.Series)
    assert len(result) == 0


# --- MACD ---


def test_macd_columns(sample_ohlcv):
    result = MACD().compute(sample_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["macd_line", "macd_signal", "macd_histogram"]
    assert len(result) == len(sample_ohlcv)


def test_macd_histogram_is_difference(sample_ohlcv):
    result = MACD().compute(sample_ohlcv)
    diff = result["macd_line"] - result["macd_signal"]
    pd.testing.assert_series_equal(diff, result["macd_histogram"], check_names=False)


# --- Bollinger Bands ---


def test_bollinger_columns(sample_ohlcv):
    result = BollingerBands().compute(sample_ohlcv, period=20)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["bb_upper_20", "bb_middle_20", "bb_lower_20"]


def test_bollinger_ordering(sample_ohlcv):
    result = BollingerBands().compute(sample_ohlcv, period=20)
    valid = result.dropna()
    assert (valid["bb_upper_20"] >= valid["bb_middle_20"]).all()
    assert (valid["bb_middle_20"] >= valid["bb_lower_20"]).all()


# --- ATR ---


def test_atr_positive(sample_ohlcv):
    result = ATR().compute(sample_ohlcv, period=14)
    assert result.name == "atr_14"
    valid = result.dropna()
    assert (valid > 0).all()


# --- Returns ---


def test_returns_columns(sample_ohlcv):
    result = Returns().compute(sample_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["return_1d", "log_return_1d"]


def test_returns_first_is_nan(sample_ohlcv):
    result = Returns().compute(sample_ohlcv)
    assert np.isnan(result["return_1d"].iloc[0])
    assert np.isnan(result["log_return_1d"].iloc[0])


# --- Cumulative Return ---


def test_cum_return_naming(sample_ohlcv):
    result = CumulativeReturn().compute(sample_ohlcv, period=5)
    assert result.name == "cum_return_5d"


# --- Volatility ---


def test_std_vol_positive(sample_ohlcv):
    result = StandardVolatility().compute(sample_ohlcv, period=20)
    assert result.name == "std_vol_20"
    valid = result.dropna()
    assert (valid >= 0).all()


def test_parkinson_vol_positive(sample_ohlcv):
    result = ParkinsonVolatility().compute(sample_ohlcv, period=20)
    assert result.name == "parkinson_vol_20"
    valid = result.dropna()
    assert (valid >= 0).all()


def test_garman_klass_vol_positive(sample_ohlcv):
    result = GarmanKlassVolatility().compute(sample_ohlcv, period=20)
    assert result.name == "garman_klass_vol_20"
    valid = result.dropna()
    assert (valid >= 0).all()
