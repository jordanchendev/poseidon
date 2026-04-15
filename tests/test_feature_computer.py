"""Tests for FeatureComputer purity, computation correctness, and public API promotion.

Covers:
- FEAT-01: FeatureComputer has zero I/O imports (pure computation)
- FEAT-03: No autoresearch import in feature_engine/ directory
- FEAT-04: Private APIs promoted to public (no _is_nonprice_spec etc.)
"""

import ast
import inspect
import os

import numpy as np
import pandas as pd
import pytest


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


class TestFeatureComputerPurity:
    """Verify FeatureComputer has zero I/O imports (FEAT-01)."""

    def test_no_io_imports_in_source(self):
        """Parse computer.py AST and check no forbidden imports."""
        import poseidon.data.feature_engine.computer as mod

        source_file = inspect.getfile(mod)
        with open(source_file) as f:
            tree = ast.parse(f.read())
        forbidden = {
            "storage", "requests", "ccxt", "yfinance", "finlab",
            "sqlalchemy", "Session", "SessionLocal", "db_session",
            "repository", "autoresearch",
        }
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = node.module.split(".")
                    imports.update(parts)
                    for alias in node.names:
                        imports.add(alias.name)
        violations = imports & forbidden
        assert not violations, f"FeatureComputer has forbidden imports: {violations}"

    def test_no_autoresearch_import(self):
        """No autoresearch reference anywhere in feature_engine/ directory."""
        import poseidon.data.feature_engine as pkg

        pkg_dir = os.path.dirname(inspect.getfile(pkg))
        for fname in os.listdir(pkg_dir):
            if fname.endswith(".py"):
                with open(os.path.join(pkg_dir, fname)) as f:
                    content = f.read()
                assert "autoresearch" not in content, (
                    f"autoresearch found in feature_engine/{fname}"
                )

    def test_computer_module_has_no_db_or_network_classes(self):
        """FeatureComputer module dir() should not contain DB/network objects."""
        import poseidon.data.feature_engine.computer as mod

        members = dir(mod)
        forbidden_members = {"Session", "SessionLocal", "db_session", "DataRepository"}
        found = forbidden_members & set(members)
        assert not found, f"FeatureComputer module contains forbidden members: {found}"


class TestFeatureComputerComputation:
    """Verify FeatureComputer produces correct output (FEAT-01)."""

    def test_compute_from_df_default(self, sample_ohlcv):
        """Default features produce expected columns."""
        from poseidon.data.feature_engine.computer import FeatureComputer

        computer = FeatureComputer()
        result = computer.compute_from_df(sample_ohlcv)
        assert len(result.columns) > 6
        assert "sma_5" in result.columns
        assert "rsi_14" in result.columns

    def test_compute_from_df_custom_specs(self, sample_ohlcv):
        """Custom specs produce only specified features."""
        from poseidon.data.feature_engine.computer import FeatureComputer

        computer = FeatureComputer()
        specs = [("sma", {"period": 10}), ("rsi", {"period": 7})]
        result = computer.compute_from_df(sample_ohlcv, feature_specs=specs)
        assert "sma_10" in result.columns
        assert "rsi_7" in result.columns
        assert "sma_60" not in result.columns

    def test_compute_from_df_empty(self):
        """Empty DataFrame returns empty."""
        from poseidon.data.feature_engine.computer import FeatureComputer

        computer = FeatureComputer()
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        result = computer.compute_from_df(empty)
        assert result.empty

    def test_compute_from_df_preserves_row_count(self, sample_ohlcv):
        """Feature computation does not change row count."""
        from poseidon.data.feature_engine.computer import FeatureComputer

        computer = FeatureComputer()
        result = computer.compute_from_df(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_backward_compat_via_alias(self, sample_ohlcv):
        """FeatureEngine alias produces same results as FeatureComputer."""
        from poseidon.data.feature_engine import FeatureEngine
        from poseidon.data.feature_engine.computer import FeatureComputer

        engine = FeatureEngine()  # This is FeatureOrchestrator
        computer = FeatureComputer()
        specs = [("sma", {"period": 5}), ("rsi", {"period": 14})]
        result_engine = engine.compute_from_df(sample_ohlcv, feature_specs=specs)
        result_computer = computer.compute_from_df(sample_ohlcv, feature_specs=specs)
        pd.testing.assert_frame_equal(result_engine, result_computer)

    def test_compute_with_data_separates_specs(self, sample_ohlcv):
        """compute_with_data() handles single-symbol specs correctly."""
        from poseidon.data.feature_engine.computer import FeatureComputer

        computer = FeatureComputer()
        specs = [("sma", {"period": 5}), ("rsi", {"period": 14})]
        result = computer.compute_with_data(ohlcv=sample_ohlcv, feature_specs=specs)
        assert "sma_5" in result.columns
        assert "rsi_14" in result.columns


class TestPublicAPIPromotion:
    """Verify private APIs have been promoted to public (FEAT-04)."""

    def test_is_nonprice_spec_public(self):
        """is_nonprice_spec is importable from the package."""
        from poseidon.data.feature_engine import is_nonprice_spec

        assert is_nonprice_spec("roe") is True
        assert is_nonprice_spec("sma") is False

    def test_nonprice_data_key_public(self):
        """nonprice_data_key is importable from the package."""
        from poseidon.data.feature_engine import nonprice_data_key

        assert nonprice_data_key("roe") == "fundamental_data"
        assert nonprice_data_key("macro_vix") == "macro_data"

    def test_no_private_names_exported(self):
        """No underscore-prefixed names in __all__."""
        from poseidon.data.feature_engine import __all__

        private = [n for n in __all__ if n.startswith("_")]
        assert not private, f"Private names in __all__: {private}"

    def test_constants_publicly_accessible(self):
        """All spec constants are importable from the package."""
        from poseidon.data.feature_engine import (
            CROSS_ASSET_PAIRS,
            DEFAULT_FEATURES,
            EXPANDED_FEATURES,
            FUNDAMENTAL_NAMES,
            FUNDING_NAMES,
            INSTITUTIONAL_PREFIXES,
            MACRO_PREFIX,
            MARGIN_NAMES,
            OI_NAMES,
            PREDICTION_NAMES,
            TRADE_STRUCTURE_NAMES,
        )

        assert isinstance(DEFAULT_FEATURES, list)
        assert isinstance(FUNDAMENTAL_NAMES, frozenset)
        assert isinstance(FUNDING_NAMES, frozenset)
        assert isinstance(INSTITUTIONAL_PREFIXES, tuple)
        assert isinstance(MACRO_PREFIX, str)
        assert isinstance(MARGIN_NAMES, frozenset)
        assert isinstance(OI_NAMES, frozenset)
        assert isinstance(PREDICTION_NAMES, frozenset)
        assert isinstance(TRADE_STRUCTURE_NAMES, frozenset)
        assert isinstance(CROSS_ASSET_PAIRS, dict)
        assert isinstance(EXPANDED_FEATURES, list)
