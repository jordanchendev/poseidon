"""Protection registry for the stateful protection layer.

Follows the same decorator-based pattern as ``universe/registry.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poseidon.protections.base import BaseProtection

# Module-level registry
_protection_registry: dict[str, type[BaseProtection]] = {}


def register_protection(cls: type[BaseProtection]) -> type[BaseProtection]:
    """Decorator to register a protection class by its ``name`` attribute."""
    if not getattr(cls, "name", None):
        raise ValueError(
            f"Protection class {cls.__name__} must define a 'name' attribute"
        )
    _protection_registry[cls.name] = cls
    return cls


def get_protection(name: str) -> type[BaseProtection]:
    """Look up a registered protection by name."""
    if name not in _protection_registry:
        raise KeyError(
            f"Unknown protection: '{name}'. "
            f"Available: {sorted(_protection_registry.keys())}"
        )
    return _protection_registry[name]


def list_protections() -> list[str]:
    """List all registered protection names, sorted."""
    return sorted(_protection_registry.keys())
