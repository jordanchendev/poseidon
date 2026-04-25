"""Portfolio strategy registry -- decorator-based registration for portfolio strategies."""

from poseidon.strategies.portfolio.base import PortfolioStrategy

_registry: dict[str, type[PortfolioStrategy]] = {}


def register_portfolio_strategy(cls):
    """Decorator to register a portfolio strategy class in the global registry."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Portfolio strategy class {cls.__name__} must define a 'name' attribute")
    _registry[cls.name] = cls
    return cls


def get_portfolio_strategy(name: str) -> type[PortfolioStrategy]:
    """Look up a registered portfolio strategy class by name."""
    if name not in _registry:
        raise KeyError(f"Unknown portfolio strategy: '{name}'. Available: {sorted(_registry.keys())}")
    return _registry[name]


def list_portfolio_strategies() -> list[str]:
    """List all registered portfolio strategy type names."""
    return sorted(_registry.keys())
