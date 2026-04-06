"""Tests for universe base classes and registry."""

import pytest

from poseidon.data.symbols import SymbolInfo
from poseidon.universe.base import UniverseFilter, UniverseSource
from poseidon.universe.registry import (
    _filter_registry,
    _source_registry,
    get_filter,
    get_source,
    list_filters,
    list_sources,
    register_filter,
    register_source,
)


class TestUniverseSourceABC:
    """Test 1: UniverseSource is abstract, cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            UniverseSource()

    def test_concrete_subclass_works(self):
        class DummySource(UniverseSource):
            name = "dummy"

            def resolve(self, market: str) -> list[SymbolInfo]:
                return []

        src = DummySource()
        assert src.resolve("any") == []

    def test_default_capability_metadata(self):
        class DummySource(UniverseSource):
            name = "dummy"

            def resolve(self, market: str) -> list[SymbolInfo]:
                return []

        src = DummySource()
        assert src.supports_backtest is True
        assert src.supports_live is False
        assert src.bias_risk == []
        assert src.stateful is False


class TestUniverseFilterABC:
    """Test 2: UniverseFilter is abstract, cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            UniverseFilter()

    def test_concrete_subclass_works(self):
        class DummyFilter(UniverseFilter):
            name = "dummy"

            def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
                return symbols

        f = DummyFilter()
        assert f.filter([], "any") == []

    def test_default_capability_metadata(self):
        class DummyFilter(UniverseFilter):
            name = "dummy"

            def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
                return symbols

        f = DummyFilter()
        assert f.supports_backtest is True
        assert f.supports_live is True
        assert f.bias_risk == []
        assert f.stateful is False


class TestSourceRegistry:
    """Tests 3, 5, 6: Source registry functionality."""

    def setup_method(self):
        """Clear registries before each test."""
        _source_registry.clear()

    def test_register_source_decorator(self):
        """Test 3: register_source registers a concrete source by name."""

        @register_source
        class MySource(UniverseSource):
            name = "my_source"

            def resolve(self, market: str) -> list[SymbolInfo]:
                return []

        assert get_source("my_source") is MySource

    def test_register_source_missing_name_raises(self):
        with pytest.raises(ValueError, match="must define a 'name' attribute"):

            @register_source
            class BadSource(UniverseSource):
                def resolve(self, market: str) -> list[SymbolInfo]:
                    return []

    def test_get_source_nonexistent_raises(self):
        """Test 5: get_source("nonexistent") raises KeyError with available names."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_source("nonexistent")

    def test_list_sources_sorted(self):
        """Test 6: list_sources() returns sorted list."""

        @register_source
        class BSource(UniverseSource):
            name = "beta"

            def resolve(self, market: str) -> list[SymbolInfo]:
                return []

        @register_source
        class ASource(UniverseSource):
            name = "alpha"

            def resolve(self, market: str) -> list[SymbolInfo]:
                return []

        assert list_sources() == ["alpha", "beta"]


class TestFilterRegistry:
    """Tests 4, 7: Filter registry functionality."""

    def setup_method(self):
        """Clear registries before each test."""
        _filter_registry.clear()

    def test_register_filter_decorator(self):
        """Test 4: register_filter registers a concrete filter by name."""

        @register_filter
        class MyFilter(UniverseFilter):
            name = "my_filter"

            def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
                return symbols

        assert get_filter("my_filter") is MyFilter

    def test_register_filter_missing_name_raises(self):
        with pytest.raises(ValueError, match="must define a 'name' attribute"):

            @register_filter
            class BadFilter(UniverseFilter):
                def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
                    return symbols

    def test_list_filters_sorted(self):
        """Test 7: list_filters() returns sorted list."""

        @register_filter
        class ZFilter(UniverseFilter):
            name = "zeta"

            def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
                return symbols

        @register_filter
        class AFilter(UniverseFilter):
            name = "alpha"

            def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
                return symbols

        assert list_filters() == ["alpha", "zeta"]
