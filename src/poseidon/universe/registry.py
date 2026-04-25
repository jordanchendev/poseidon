"""Source and filter registries for the universe pipeline.

Follows the same decorator-based pattern as ``data/features/base.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poseidon.universe.base import UniverseFilter, UniverseSource

# Module-level registries
_source_registry: dict[str, type[UniverseSource]] = {}
_filter_registry: dict[str, type[UniverseFilter]] = {}


def register_source(cls: type[UniverseSource]) -> type[UniverseSource]:
    """Decorator to register a universe source class by its ``name`` attribute."""
    if not getattr(cls, "name", None):
        raise ValueError(f"Source class {cls.__name__} must define a 'name' attribute")
    _source_registry[cls.name] = cls
    return cls


def register_filter(cls: type[UniverseFilter]) -> type[UniverseFilter]:
    """Decorator to register a universe filter class by its ``name`` attribute."""
    if not getattr(cls, "name", None):
        raise ValueError(f"Filter class {cls.__name__} must define a 'name' attribute")
    _filter_registry[cls.name] = cls
    return cls


def get_source(name: str) -> type[UniverseSource]:
    """Look up a registered source by name."""
    if name not in _source_registry:
        raise KeyError(f"Unknown universe source: '{name}'. Available: {sorted(_source_registry.keys())}")
    return _source_registry[name]


def get_filter(name: str) -> type[UniverseFilter]:
    """Look up a registered filter by name."""
    if name not in _filter_registry:
        raise KeyError(f"Unknown universe filter: '{name}'. Available: {sorted(_filter_registry.keys())}")
    return _filter_registry[name]


def list_sources() -> list[str]:
    """List all registered source names, sorted."""
    return sorted(_source_registry.keys())


def list_filters() -> list[str]:
    """List all registered filter names, sorted."""
    return sorted(_filter_registry.keys())
