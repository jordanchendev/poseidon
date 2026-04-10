"""Tests for DatasetBuilder feature_specs integration."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from poseidon.qlib.dataset_builder import DatasetBuilder


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session."""
    return MagicMock()


@pytest.fixture
def sample_ohlcv():
    """Create synthetic OHLCV data for a single symbol."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(50).cumsum()
    return pd.DataFrame(
        {
            "open": close + rng.uniform(-1, 1, 50),
            "high": close + abs(rng.standard_normal(50)),
            "low": close - abs(rng.standard_normal(50)),
            "close": close,
            "volume": rng.integers(1000, 10000, size=50).astype(float),
        },
        index=dates,
    )


class TestBuildWithoutFeatureSpecs:
    """DatasetBuilder.build() backward compatibility (feature_specs=None)."""

    def test_returns_standard_qlib_columns(self, mock_session, sample_ohlcv):
        builder = DatasetBuilder(session=mock_session, market="tw_stock", interval="1d")
        with patch("poseidon.qlib.dataset_builder.read_ohlcv", return_value=sample_ohlcv):
            result = builder.build(symbols=["2330"])
        expected_cols = {"$open", "$high", "$low", "$close", "$volume", "$vwap"}
        assert set(result.columns) == expected_cols

    def test_empty_feature_specs_same_as_none(self, mock_session, sample_ohlcv):
        builder = DatasetBuilder(session=mock_session, market="tw_stock", interval="1d")
        with patch("poseidon.qlib.dataset_builder.read_ohlcv", return_value=sample_ohlcv):
            result_none = builder.build(symbols=["2330"], feature_specs=None)
            result_empty = builder.build(symbols=["2330"], feature_specs=[])
        assert set(result_none.columns) == set(result_empty.columns)


class TestBuildWithFeatureSpecs:
    """DatasetBuilder.build() with feature_specs computes and merges features."""

    def test_adds_feature_columns_with_dollar_prefix(self, mock_session, sample_ohlcv):
        builder = DatasetBuilder(session=mock_session, market="tw_stock", interval="1d")

        # Mock FeatureEngine to return OHLCV + feature columns
        mock_feature_df = sample_ohlcv.copy()
        mock_feature_df["roe"] = 0.15
        mock_feature_df["roa"] = 0.08

        with patch("poseidon.qlib.dataset_builder.read_ohlcv", return_value=sample_ohlcv), \
             patch("poseidon.qlib.dataset_builder.FeatureEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.compute_with_companions.return_value = mock_feature_df
            result = builder.build(
                symbols=["2330"],
                feature_specs=[("roe", {}), ("roa", {})],
            )

        assert "$roe" in result.columns, "Expected $roe column in output"
        assert "$roa" in result.columns, "Expected $roa column in output"
        assert "$open" in result.columns, "Standard OHLCV columns must remain"
        assert "$vwap" in result.columns, "$vwap must remain"

    def test_feature_engine_called_per_symbol(self, mock_session, sample_ohlcv):
        builder = DatasetBuilder(session=mock_session, market="tw_stock", interval="1d")

        mock_feature_df = sample_ohlcv.copy()
        mock_feature_df["roe"] = 0.15

        with patch("poseidon.qlib.dataset_builder.read_ohlcv", return_value=sample_ohlcv), \
             patch("poseidon.qlib.dataset_builder.FeatureEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.compute_with_companions.return_value = mock_feature_df
            builder.build(
                symbols=["2330", "2317"],
                feature_specs=[("roe", {})],
            )
            # FeatureEngine.compute_with_companions should be called once per symbol
            assert mock_engine_instance.compute_with_companions.call_count == 2

    def test_multi_symbol_independent_features(self, mock_session):
        """Feature values are computed per-symbol, not mixed across symbols."""
        builder = DatasetBuilder(session=mock_session, market="tw_stock", interval="1d")

        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        ohlcv_a = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0,
        }, index=dates)
        ohlcv_b = pd.DataFrame({
            "open": 200.0, "high": 201.0, "low": 199.0,
            "close": 200.0, "volume": 2000.0,
        }, index=dates)

        feat_a = ohlcv_a.copy()
        feat_a["roe"] = 0.10
        feat_b = ohlcv_b.copy()
        feat_b["roe"] = 0.25

        def mock_read_ohlcv(session, symbol, *args, **kwargs):
            return ohlcv_a if symbol == "A" else ohlcv_b

        def mock_compute(ohlcv, symbol, *args, **kwargs):
            return feat_a if symbol == "A" else feat_b

        with patch("poseidon.qlib.dataset_builder.read_ohlcv", side_effect=mock_read_ohlcv), \
             patch("poseidon.qlib.dataset_builder.FeatureEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.compute_with_companions.side_effect = mock_compute
            result = builder.build(
                symbols=["A", "B"],
                feature_specs=[("roe", {})],
            )

        # Check symbol A has roe=0.10, symbol B has roe=0.25
        a_rows = result.xs("A", level="instrument")
        b_rows = result.xs("B", level="instrument")
        assert (a_rows["$roe"] == 0.10).all(), "Symbol A should have $roe=0.10"
        assert (b_rows["$roe"] == 0.25).all(), "Symbol B should have $roe=0.25"


class TestBuildBackwardCompatible:
    """Verify existing callers (without feature_specs) still work."""

    def test_no_feature_specs_no_feature_engine(self, mock_session, sample_ohlcv):
        builder = DatasetBuilder(session=mock_session, market="tw_stock", interval="1d")
        with patch("poseidon.qlib.dataset_builder.read_ohlcv", return_value=sample_ohlcv), \
             patch("poseidon.qlib.dataset_builder.FeatureEngine") as MockEngine:
            builder.build(symbols=["2330"])
            # FeatureEngine should NOT be instantiated when feature_specs is None
            MockEngine.assert_not_called()
