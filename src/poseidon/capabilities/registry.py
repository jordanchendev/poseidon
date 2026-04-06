"""Unified capability registry -- collects from all component registries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComponentCapability:
    """Capability metadata for a single registered component."""

    name: str
    component_type: str  # "feature" | "strategy" | "model" | "rule" | "portfolio_strategy"
    supports_backtest: bool
    supports_live: bool
    bias_risk: list[str]
    stateful: bool


def _extract_capability(cls: type, component_type: str) -> ComponentCapability:
    """Extract capability metadata from a component class."""
    return ComponentCapability(
        name=getattr(cls, "name", cls.__name__),
        component_type=component_type,
        supports_backtest=getattr(cls, "supports_backtest", True),
        supports_live=getattr(cls, "supports_live", False),
        bias_risk=list(getattr(cls, "bias_risk", [])),
        stateful=getattr(cls, "stateful", False),
    )


# --- Registry accessor functions (thin wrappers for testability) ---


def _get_feature_registry() -> dict:
    from poseidon.data.features.base import _registry
    return _registry


def _get_strategy_registry() -> dict:
    from poseidon.strategies.registry import _registry
    return _registry


def _get_model_registry() -> dict:
    from poseidon.ml.registry import _registry
    return _registry


def _get_rule_registry() -> dict:
    from poseidon.risk.rules import RULE_REGISTRY
    return RULE_REGISTRY


def _get_portfolio_registry() -> dict:
    from poseidon.strategies.portfolio.registry import _registry
    return _registry


def get_all_capabilities() -> list[ComponentCapability]:
    """Collect capability metadata from all 5 component registries.

    Returns a flat list sorted by (component_type, name).
    """
    results: list[ComponentCapability] = []

    for cls in _get_feature_registry().values():
        results.append(_extract_capability(cls, "feature"))
    for cls in _get_strategy_registry().values():
        results.append(_extract_capability(cls, "strategy"))
    for cls in _get_model_registry().values():
        results.append(_extract_capability(cls, "model"))
    for cls in _get_rule_registry().values():
        results.append(_extract_capability(cls, "rule"))
    for cls in _get_portfolio_registry().values():
        results.append(_extract_capability(cls, "portfolio_strategy"))

    return sorted(results, key=lambda c: (c.component_type, c.name))
