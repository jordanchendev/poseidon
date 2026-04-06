"""Strategy registry -- decorator-based registration for strategy types."""

from poseidon.strategies.base import BaseStrategy

_registry: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls):
    """Decorator to register a strategy class in the global registry."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Strategy class {cls.__name__} must define a 'name' attribute")
    _registry[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[BaseStrategy]:
    """Look up a registered strategy class by name."""
    if name not in _registry:
        raise KeyError(f"Unknown strategy: '{name}'. Available: {sorted(_registry.keys())}")
    return _registry[name]


def list_strategies() -> list[str]:
    """List all registered strategy type names."""
    return sorted(_registry.keys())
