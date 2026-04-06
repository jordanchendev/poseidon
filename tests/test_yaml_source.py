"""Tests for YamlSource wrapping existing load_symbols()."""

import pytest

from poseidon.data.symbols import SymbolInfo, load_symbols
from poseidon.universe.registry import _source_registry, get_source


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure YamlSource is registered fresh for each test."""
    _source_registry.clear()
    # Re-import to trigger @register_source decorator
    import importlib

    import poseidon.universe.yaml_source as mod

    importlib.reload(mod)
    yield
    _source_registry.clear()


class TestYamlSource:
    """Tests for YamlSource backward-compatible symbol source."""

    def test_resolve_tw_stock_matches_load_symbols(self):
        """Test 1: YamlSource().resolve("tw_stock") matches load_symbols() exactly."""
        from poseidon.universe.yaml_source import YamlSource

        source = YamlSource()
        result = source.resolve("tw_stock")
        expected = load_symbols().markets["tw_stock"].symbols

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.name == e.name

    def test_resolve_crypto_spot_preserves_ccxt_symbol(self):
        """Test 2: YamlSource().resolve("crypto_spot") preserves ccxt_symbol."""
        from poseidon.universe.yaml_source import YamlSource

        source = YamlSource()
        result = source.resolve("crypto_spot")
        expected = load_symbols().markets["crypto_spot"].symbols

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.id == e.id
            assert r.name == e.name
            assert r.ccxt_symbol == e.ccxt_symbol

    def test_resolve_nonexistent_market_returns_empty(self):
        """Test 3: YamlSource().resolve("nonexistent") returns empty list."""
        from poseidon.universe.yaml_source import YamlSource

        source = YamlSource()
        result = source.resolve("nonexistent_market")
        assert result == []

    def test_registered_as_yaml(self):
        """Test 4: YamlSource is registered in source registry as 'yaml'."""
        from poseidon.universe.yaml_source import YamlSource

        cls = get_source("yaml")
        assert cls is YamlSource

    def test_capability_metadata(self):
        """Test 5: YamlSource has correct capability metadata."""
        from poseidon.universe.yaml_source import YamlSource

        src = YamlSource()
        assert src.supports_backtest is True
        assert src.supports_live is True
        assert src.bias_risk == []
        assert src.stateful is False
