"""Unit tests for BVC (Bulk Volume Classification) helper module.

Tests the classify_volume() function which classifies each OHLCV bar's
volume into buyer- and seller-initiated volume using the normal CDF
approximation from Easley/LdP/O'Hara (2016).
"""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.bvc import classify_volume


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Mixed bullish/bearish bars with varied close positions."""
    return pd.DataFrame(
        {
            "open": [100, 105, 98, 110, 95, 102, 108, 99, 103, 107],
            "high": [110, 112, 105, 115, 105, 110, 115, 106, 110, 115],
            "low": [95, 100, 92, 105, 90, 98, 102, 94, 97, 100],
            "close": [108, 101, 103, 112, 98, 109, 103, 104, 108, 102],
            "volume": [1000, 2000, 1500, 3000, 2500, 1800, 2200, 1600, 1900, 2100],
        }
    )


@pytest.fixture
def doji_ohlcv() -> pd.DataFrame:
    """Bars where high == low (zero range doji)."""
    return pd.DataFrame(
        {
            "open": [100, 100, 100],
            "high": [100, 100, 100],
            "low": [100, 100, 100],
            "close": [100, 100, 100],
            "volume": [500, 1000, 750],
        }
    )


@pytest.fixture
def empty_ohlcv() -> pd.DataFrame:
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def nan_volume_ohlcv() -> pd.DataFrame:
    """Bars with NaN volume values."""
    return pd.DataFrame(
        {
            "open": [100, 105],
            "high": [110, 112],
            "low": [95, 100],
            "close": [108, 101],
            "volume": [np.nan, 2000],
        }
    )


# ── Test: return structure ────────────────────────────────────────────


def test_returns_dataframe_with_buy_sell_columns(sample_ohlcv: pd.DataFrame):
    """classify_volume returns DataFrame with buy_volume and sell_volume."""
    result = classify_volume(sample_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert "buy_volume" in result.columns
    assert "sell_volume" in result.columns
    assert len(result) == len(sample_ohlcv)


# ── Test: volume conservation ─────────────────────────────────────────


def test_volume_conservation(sample_ohlcv: pd.DataFrame):
    """buy_volume + sell_volume == total volume for every row."""
    result = classify_volume(sample_ohlcv)
    total = result["buy_volume"] + result["sell_volume"]
    pd.testing.assert_series_equal(
        total, sample_ohlcv["volume"], check_names=False, atol=1e-10
    )


# ── Test: bullish bars ────────────────────────────────────────────────


def test_bullish_bar_has_more_buy_volume(sample_ohlcv: pd.DataFrame):
    """Bullish bar (close > open) should have buy_volume > sell_volume."""
    result = classify_volume(sample_ohlcv)
    # Row 0: open=100, close=108 -> bullish
    assert result.loc[0, "buy_volume"] > result.loc[0, "sell_volume"]


# ── Test: bearish bars ────────────────────────────────────────────────


def test_bearish_bar_has_more_sell_volume(sample_ohlcv: pd.DataFrame):
    """Bearish bar (close < open) should have sell_volume > buy_volume."""
    result = classify_volume(sample_ohlcv)
    # Row 1: open=105, close=101 -> bearish
    assert result.loc[1, "sell_volume"] > result.loc[1, "buy_volume"]


# ── Test: doji bars ───────────────────────────────────────────────────


def test_doji_bars_produce_50_50_split(doji_ohlcv: pd.DataFrame):
    """Doji bar (high == low) produces 50/50 volume split."""
    result = classify_volume(doji_ohlcv)
    for i in range(len(doji_ohlcv)):
        expected_half = doji_ohlcv.loc[i, "volume"] * 0.5
        assert abs(result.loc[i, "buy_volume"] - expected_half) < 1e-10
        assert abs(result.loc[i, "sell_volume"] - expected_half) < 1e-10


# ── Test: empty DataFrame ────────────────────────────────────────────


def test_empty_dataframe_returns_empty_with_columns(empty_ohlcv: pd.DataFrame):
    """Empty input returns empty DataFrame with correct columns."""
    result = classify_volume(empty_ohlcv)
    assert isinstance(result, pd.DataFrame)
    assert "buy_volume" in result.columns
    assert "sell_volume" in result.columns
    assert len(result) == 0


# ── Test: non-negative volumes ────────────────────────────────────────


def test_all_volumes_non_negative(sample_ohlcv: pd.DataFrame):
    """All buy_volume and sell_volume values are non-negative."""
    result = classify_volume(sample_ohlcv)
    assert (result["buy_volume"] >= 0).all()
    assert (result["sell_volume"] >= 0).all()


# ── Test: NaN volume bars ─────────────────────────────────────────────


def test_nan_volume_produces_nan_buy_sell(nan_volume_ohlcv: pd.DataFrame):
    """NaN volume bars produce NaN buy/sell volume (not errors)."""
    result = classify_volume(nan_volume_ohlcv)
    # Row 0 has NaN volume -> buy/sell should be NaN
    assert pd.isna(result.loc[0, "buy_volume"])
    assert pd.isna(result.loc[0, "sell_volume"])
    # Row 1 has valid volume -> should be valid
    assert pd.notna(result.loc[1, "buy_volume"])
    assert pd.notna(result.loc[1, "sell_volume"])
