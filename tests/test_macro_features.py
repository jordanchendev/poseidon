"""Tests for macro index feature classes (VIX, DXY, TNX, TWDUSD)."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.macro import MacroDXY, MacroTNX, MacroTWDUSD, MacroVIX


@pytest.fixture
def ohlcv_50d():
    """Synthetic 50-day daily OHLCV."""
    dates = pd.date_range("2024-06-01", periods=50, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 2, 50).cumsum()
    return pd.DataFrame(
        {
            "time": dates,
            "open": close - rng.uniform(0.5, 1.5, 50),
            "high": close + rng.uniform(1, 3, 50),
            "low": close - rng.uniform(1, 3, 50),
            "close": close,
            "volume": rng.uniform(1e6, 5e6, 50),
        },
        index=dates,
    )


@pytest.fixture
def macro_data_40d():
    """Synthetic macro data with 4 columns and some NaN gaps."""
    dates = pd.date_range("2024-05-25", periods=40, freq="D")
    rng = np.random.default_rng(123)
    data = pd.DataFrame(
        {
            "macro_vix": rng.uniform(12, 30, 40),
            "macro_dxy": rng.uniform(100, 110, 40),
            "macro_tnx": rng.uniform(3.5, 5.0, 40),
            "macro_twdusd": rng.uniform(30, 33, 40),
        },
        index=dates,
    )
    # Introduce NaN gaps for forward-fill testing
    data.iloc[10:13, 0] = np.nan  # VIX gap
    data.iloc[20:22, 1] = np.nan  # DXY gap
    return data


class TestMacroVIX:
    def test_compute_aligned(self, ohlcv_50d, macro_data_40d):
        feat = MacroVIX()
        result = feat.compute(ohlcv_50d, macro_data=macro_data_40d)
        assert result.name == "macro_vix"
        assert len(result) == 50

    def test_none_macro_data(self, ohlcv_50d):
        feat = MacroVIX()
        result = feat.compute(ohlcv_50d, macro_data=None)
        assert result.name == "macro_vix"
        assert result.isna().all()


class TestMacroDXY:
    def test_compute_aligned(self, ohlcv_50d, macro_data_40d):
        feat = MacroDXY()
        result = feat.compute(ohlcv_50d, macro_data=macro_data_40d)
        assert result.name == "macro_dxy"
        assert len(result) == 50

    def test_none_macro_data(self, ohlcv_50d):
        feat = MacroDXY()
        result = feat.compute(ohlcv_50d, macro_data=None)
        assert result.isna().all()


class TestMacroTNX:
    def test_compute_aligned(self, ohlcv_50d, macro_data_40d):
        feat = MacroTNX()
        result = feat.compute(ohlcv_50d, macro_data=macro_data_40d)
        assert result.name == "macro_tnx"
        assert len(result) == 50

    def test_none_macro_data(self, ohlcv_50d):
        feat = MacroTNX()
        result = feat.compute(ohlcv_50d, macro_data=None)
        assert result.isna().all()


class TestMacroTWDUSD:
    def test_compute_aligned(self, ohlcv_50d, macro_data_40d):
        feat = MacroTWDUSD()
        result = feat.compute(ohlcv_50d, macro_data=macro_data_40d)
        assert result.name == "macro_twdusd"
        assert len(result) == 50

    def test_none_macro_data(self, ohlcv_50d):
        feat = MacroTWDUSD()
        result = feat.compute(ohlcv_50d, macro_data=None)
        assert result.isna().all()


class TestForwardFill:
    def test_vix_forward_fill_across_gaps(self, ohlcv_50d, macro_data_40d):
        """NaN gaps in macro source should be forward-filled."""
        feat = MacroVIX()
        result = feat.compute(ohlcv_50d, macro_data=macro_data_40d)
        # After forward-fill, the NaN gap in source (rows 10-12) should be
        # filled by the last valid value *before* the gap.  Dates beyond the
        # gap should not all be NaN.
        non_nan_count = result.notna().sum()
        assert non_nan_count > 30  # most values should be filled

    def test_empty_ohlcv(self):
        """Empty OHLCV returns empty Series."""
        feat = MacroVIX()
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        result = feat.compute(empty)
        assert len(result) == 0
