"""Tests for MacroIndexLoader."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from poseidon.data.loaders.macro_loader import MacroIndexLoader


def _make_mock_yf_data() -> pd.DataFrame:
    """Create synthetic yfinance MultiIndex DataFrame for 4 tickers."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    tickers = ["^VIX", "DX-Y.NYB", "^TNX", "TWDUSD=X"]

    # Build MultiIndex columns: (Price, Ticker)
    arrays = []
    for ticker in tickers:
        arrays.append(("Close", ticker))
        arrays.append(("Volume", ticker))

    mi = pd.MultiIndex.from_tuples(arrays, names=["Price", "Ticker"])

    data = {}
    for ticker in tickers:
        data[("Close", ticker)] = [20.0 + i for i in range(5)]
        data[("Volume", ticker)] = [1000 + i for i in range(5)]

    df = pd.DataFrame(data, index=dates, columns=mi)
    return df


def _make_mock_yf_data_with_nan() -> pd.DataFrame:
    """Create synthetic data with NaN gaps (different market calendars)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    tickers = ["^VIX", "DX-Y.NYB", "^TNX", "TWDUSD=X"]

    data = {}
    for ticker in tickers:
        vals = [20.0 + i for i in range(5)]
        data[("Close", ticker)] = vals
        data[("Volume", ticker)] = [1000 + i for i in range(5)]

    mi = pd.MultiIndex.from_tuples(
        [(p, t) for t in tickers for p in ["Close", "Volume"]],
        names=["Price", "Ticker"],
    )

    # Rebuild with proper structure
    df = pd.DataFrame(index=dates)
    for ticker in tickers:
        df[("Close", ticker)] = [20.0 + i for i in range(5)]
        df[("Volume", ticker)] = [1000 + i for i in range(5)]

    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Price", "Ticker"])

    # Introduce NaN on different days for different tickers
    df.iloc[1, df.columns.get_loc(("Close", "^VIX"))] = np.nan
    df.iloc[3, df.columns.get_loc(("Close", "DX-Y.NYB"))] = np.nan

    return df


class TestGetMacroData:
    @patch("poseidon.data.loaders.macro_loader.yf", create=True)
    def test_returns_four_columns(self, mock_yf_module):
        """get_macro_data returns 4 macro columns."""
        mock_data = _make_mock_yf_data()

        with patch("poseidon.data.loaders.macro_loader.MacroIndexLoader.get_macro_data") as mock_method:
            # Actually test the real method by calling it directly
            pass

        # Test with real method by mocking yfinance.download
        loader = MacroIndexLoader()

        import poseidon.data.loaders.macro_loader as mod

        with patch.object(mod, "yf", create=True) as mock_yf:
            # We need to mock the import inside the method
            pass

        # Simpler approach: mock at import level
        loader = MacroIndexLoader()
        loader._cache = None

        # Directly test with pre-made data by simulating what get_macro_data does
        mock_data = _make_mock_yf_data()

        with patch("yfinance.download", return_value=mock_data):
            result = loader.get_macro_data(start="2024-01-01")

        assert not result.empty
        expected_cols = {"macro_vix", "macro_dxy", "macro_tnx", "macro_twdusd"}
        assert set(result.columns) == expected_cols
        assert len(result) == 5

    def test_caching_prevents_redundant_downloads(self):
        """Second call uses cache and does not invoke yf.download."""
        loader = MacroIndexLoader()

        # Pre-populate cache
        cache_df = pd.DataFrame(
            {
                "macro_vix": [20.0, 21.0],
                "macro_dxy": [100.0, 101.0],
                "macro_tnx": [4.0, 4.1],
                "macro_twdusd": [31.0, 31.1],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        cache_df.index.name = "date"
        loader._cache = cache_df

        with patch("yfinance.download") as mock_dl:
            result = loader.get_macro_data()
            mock_dl.assert_not_called()

        pd.testing.assert_frame_equal(result, cache_df)

    def test_forward_fill_nan_gaps(self):
        """NaN gaps from different market calendars are forward-filled."""
        mock_data = _make_mock_yf_data_with_nan()
        loader = MacroIndexLoader()

        with patch("yfinance.download", return_value=mock_data):
            result = loader.get_macro_data(start="2024-01-01")

        assert not result.empty
        # After ffill, there should be no NaN (except possibly the first row
        # if the first value was NaN, which it's not in our test data)
        assert not result.iloc[1:].isna().any().any()

    def test_download_error_returns_empty(self):
        """yfinance error returns empty DataFrame."""
        loader = MacroIndexLoader()

        with patch("yfinance.download", side_effect=Exception("Network error")):
            result = loader.get_macro_data()

        assert result.empty
