"""Model registry — decorator-based registration for ML model types."""

from poseidon.ml.base import BaseModel

_registry: dict[str, type[BaseModel]] = {}


def register_model(cls):
    """Decorator to register a model class in the global registry."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Model class {cls.__name__} must define a 'name' attribute")
    _registry[cls.name] = cls
    return cls


def get_model(name: str) -> type[BaseModel]:
    """Look up a registered model class by name."""
    if name not in _registry:
        raise KeyError(f"Unknown model: '{name}'. Available: {sorted(_registry.keys())}")
    return _registry[name]


def list_models() -> list[str]:
    """List all registered model type names."""
    return sorted(_registry.keys())
