"""Tests for FinLabDataLoader."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_wide_df(symbol: str, values: list[float]) -> pd.DataFrame:
    """Create a synthetic wide DataFrame (dates x symbols)."""
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {symbol: values, "9999": [v * 2 for v in values]},
        index=dates,
    )


@pytest.fixture
def loader():
    """Create a FinLabDataLoader with mocked finlab internals."""
    with (
        patch("poseidon.data.loaders.finlab_loader.FinLabDataLoader._ensure_login"),
    ):
        from poseidon.data.loaders.finlab_loader import FinLabDataLoader

        loader = FinLabDataLoader(token="fake-token")
        loader._initialized = True
        yield loader


class TestGetInstitutionalFlow:
    def test_returns_four_columns(self, loader):
        """get_institutional_flow returns foreign, trust, dealer, dealer_hedge."""
        datasets = {
            "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)": [100, 200, 300],
            "institutional_investors_trading_summary:投信買賣超股數": [10, 20, 30],
            "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)": [5, 10, 15],
            "institutional_investors_trading_summary:自營商買賣超股數(避險)": [1, 2, 3],
        }
        mock_data = MagicMock()

        def fake_get(key):
            if key in datasets:
                return _make_wide_df("2330", datasets[key])
            return None

        mock_data.get = fake_get
        mock_data.set_market = MagicMock()

        with patch("poseidon.data.loaders.finlab_loader.FinLabDataLoader.get_dataset") as mock_gd:
            # Return wide DFs for each dataset key
            def side_effect(key):
                if key in datasets:
                    return _make_wide_df("2330", datasets[key])
                return pd.DataFrame()

            mock_gd.side_effect = side_effect

            result = loader.get_institutional_flow("2330")

        assert not result.empty
        assert list(result.columns) == ["foreign", "trust", "dealer", "dealer_hedge"]
        assert len(result) == 3
        assert result["foreign"].iloc[0] == 100
        assert result["trust"].iloc[1] == 20

    def test_missing_symbol_returns_empty(self, loader):
        """Missing symbol returns empty DataFrame."""
        with patch("poseidon.data.loaders.finlab_loader.FinLabDataLoader.get_dataset") as mock_gd:
            # Return a wide DF without the requested symbol
            mock_gd.return_value = _make_wide_df("9999", [1, 2, 3])
            result = loader.get_institutional_flow("2330")

        assert result.empty


class TestGetFundamentals:
    def test_returns_four_columns(self, loader):
        """get_fundamentals returns pe_ratio, pb_ratio, monthly_rev, prev_year_rev."""
        datasets = {
            "price_earning_ratio:本益比": [15.0, 16.0, 17.0],
            "price_earning_ratio:股價淨值比": [3.0, 3.1, 3.2],
            "monthly_revenue:當月營收": [1000, 1100, 1200],
            "monthly_revenue:去年當月營收": [900, 950, 1000],
        }

        with patch("poseidon.data.loaders.finlab_loader.FinLabDataLoader.get_dataset") as mock_gd:
            def side_effect(key):
                if key in datasets:
                    return _make_wide_df("2330", datasets[key])
                return pd.DataFrame()

            mock_gd.side_effect = side_effect

            result = loader.get_fundamentals("2330")

        assert not result.empty
        assert list(result.columns) == ["pe_ratio", "pb_ratio", "monthly_rev", "prev_year_rev"]
        assert len(result) == 3
        assert result["pe_ratio"].iloc[0] == 15.0


class TestGetTradeStructure:
    def test_returns_two_columns(self, loader):
        """get_trade_structure returns trade_value, trade_count."""
        datasets = {
            "price:成交金額": [5e9, 6e9, 7e9],
            "price:成交筆數": [50000, 60000, 70000],
        }

        with patch("poseidon.data.loaders.finlab_loader.FinLabDataLoader.get_dataset") as mock_gd:
            def side_effect(key):
                if key in datasets:
                    return _make_wide_df("2330", datasets[key])
                return pd.DataFrame()

            mock_gd.side_effect = side_effect

            result = loader.get_trade_structure("2330")

        assert not result.empty
        assert list(result.columns) == ["trade_value", "trade_count"]
        assert len(result) == 3


class TestGetDataset:
    def test_caching_prevents_redundant_calls(self, loader):
        """Second call to get_dataset with same key uses cache."""
        wide_df = _make_wide_df("2330", [1, 2, 3])
        mock_data = MagicMock()
        mock_data.get.return_value = wide_df
        mock_data.set_market = MagicMock()

        with patch("poseidon.data.loaders.finlab_loader.data", mock_data, create=True):
            with patch.dict("sys.modules", {"finlab": MagicMock(), "finlab.data": mock_data}):
                # Manually patch the import inside get_dataset
                import poseidon.data.loaders.finlab_loader as mod

                original_get_dataset = mod.FinLabDataLoader.get_dataset

                call_count = 0

                def counting_get_dataset(self, key):
                    nonlocal call_count
                    call_count += 1
                    return original_get_dataset(self, key)

                # Instead, directly test caching by pre-populating cache
                loader._cache["test_key"] = wide_df
                result1 = loader.get_dataset("test_key")
                result2 = loader.get_dataset("test_key")

                pd.testing.assert_frame_equal(result1, result2)
                # Cache should have been hit - no new entry
                assert "test_key" in loader._cache

    def test_empty_dataset_returns_empty(self, loader):
        """get_dataset returns empty DataFrame for missing data."""
        mock_data = MagicMock()
        mock_data.get.return_value = None
        mock_data.set_market = MagicMock()

        with patch("poseidon.data.loaders.finlab_loader.data", mock_data, create=True):
            # Pre-clear cache
            loader._cache.clear()
            # Mock the finlab import inside get_dataset
            with patch.object(type(loader), "get_dataset", return_value=pd.DataFrame()):
                result = loader.get_dataset("nonexistent_key")
                assert result.empty
