"""Tests for FeatureEngine."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.feature_engine import DEFAULT_FEATURES, FeatureEngine


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
    return pd.DataFrame(
        {
            "time": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def engine() -> FeatureEngine:
    return FeatureEngine()


class TestComputeFromDf:
    """Tests for compute_from_df (in-memory computation)."""

    def test_default_features(self, engine, sample_ohlcv):
        result = engine.compute_from_df(sample_ohlcv)
        # Original 6 columns + feature columns
        assert len(result.columns) > 6
        # Check some expected column names from default features
        assert "sma_5" in result.columns
        assert "sma_20" in result.columns
        assert "sma_60" in result.columns
        assert "ema_12" in result.columns
        assert "rsi_14" in result.columns
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "bb_upper_20" in result.columns
        assert "atr_14" in result.columns
        assert "return_1d" in result.columns
        assert "std_vol_20" in result.columns

    def test_custom_specs(self, engine, sample_ohlcv):
        specs = [("sma", {"period": 10}), ("rsi", {"period": 7})]
        result = engine.compute_from_df(sample_ohlcv, feature_specs=specs)
        assert "sma_10" in result.columns
        assert "rsi_7" in result.columns
        # Should NOT have default features
        assert "sma_60" not in result.columns

    def test_empty_df(self, engine):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        result = engine.compute_from_df(empty)
        assert result.empty

    def test_preserves_original_columns(self, engine, sample_ohlcv):
        result = engine.compute_from_df(sample_ohlcv, feature_specs=[("sma", {"period": 5})])
        for col in ["time", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_row_count_unchanged(self, engine, sample_ohlcv):
        result = engine.compute_from_df(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_multiple_same_indicator_different_params(self, engine, sample_ohlcv):
        specs = [("sma", {"period": 5}), ("sma", {"period": 20}), ("sma", {"period": 60})]
        result = engine.compute_from_df(sample_ohlcv, feature_specs=specs)
        assert "sma_5" in result.columns
        assert "sma_20" in result.columns
        assert "sma_60" in result.columns
        # Values should differ
        assert not result["sma_5"].equals(result["sma_20"])

    def test_unknown_feature_raises(self, engine, sample_ohlcv):
        with pytest.raises(KeyError, match="Unknown feature"):
            engine.compute_from_df(sample_ohlcv, feature_specs=[("nonexistent", {})])


class TestDefaultFeatures:
    """Tests for DEFAULT_FEATURES constant."""

    def test_default_features_is_list(self):
        assert isinstance(DEFAULT_FEATURES, list)
        assert len(DEFAULT_FEATURES) > 0

    def test_default_features_are_tuples(self):
        for item in DEFAULT_FEATURES:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], dict)

    def test_default_features_all_valid(self, engine, sample_ohlcv):
        """All default features should compute without error."""
        result = engine.compute_from_df(sample_ohlcv)
        assert not result.empty


class TestGetR2Specs:
    """Tests for get_r2_specs() market-conditional feature generation."""

    def test_tw_stock_includes_fundamental_expansion(self):
        from poseidon.data.feature_engine import get_r2_specs

        specs = get_r2_specs("2330", "tw_stock")
        spec_names = [name for name, _ in specs]
        assert "roe" in spec_names, "ROE missing from tw_stock R2 specs"
        assert "roa" in spec_names, "ROA missing from tw_stock R2 specs"
        assert "margin_buy_ratio" in spec_names, "margin_buy_ratio missing from tw_stock R2 specs"
        assert "margin_sell_ratio" in spec_names, "margin_sell_ratio missing from tw_stock R2 specs"

    def test_crypto_spot_excludes_margin(self):
        from poseidon.data.feature_engine import get_r2_specs

        specs = get_r2_specs("BTCUSDT", "crypto_spot")
        spec_names = [name for name, _ in specs]
        assert "margin_buy_ratio" not in spec_names
        assert "margin_sell_ratio" not in spec_names
        assert "roe" not in spec_names
        assert "roa" not in spec_names

    def test_nonprice_spec_detection(self):
        from poseidon.data.feature_engine import is_nonprice_spec, nonprice_data_key

        assert is_nonprice_spec("roe") is True
        assert is_nonprice_spec("roa") is True
        assert is_nonprice_spec("margin_buy_ratio") is True
        assert is_nonprice_spec("margin_sell_ratio") is True
        assert nonprice_data_key("roe") == "fundamental_data"
        assert nonprice_data_key("roa") == "fundamental_data"
        assert nonprice_data_key("margin_buy_ratio") == "margin_data"
        assert nonprice_data_key("margin_sell_ratio") == "margin_data"
