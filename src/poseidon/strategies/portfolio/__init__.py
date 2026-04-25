"""Lazy exports for portfolio strategy package.

Avoid eager imports here. Runtime paths often import submodules such as
``poseidon.strategies.portfolio.schemas``; package-level side effects must not
pull in optional strategies or research helpers.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AllocationConfig",
    "CryptoTrendStrategy",
    "FundamentalSelectionStrategy",
    "Holding",
    "PortfolioRebalancer",
    "PortfolioStrategy",
    "PositionTracker",
    "PredictionRankingStrategy",
    "RebalanceConfig",
    "RebalanceOrder",
    "RevenueBreakoutStrategy",
    "SelectionConfig",
    "TargetPosition",
]

_EXPORTS = {
    "PortfolioStrategy": ("poseidon.strategies.portfolio.base", "PortfolioStrategy"),
    "CryptoTrendStrategy": (
        "poseidon.strategies.portfolio.crypto_trend",
        "CryptoTrendStrategy",
    ),
    "FundamentalSelectionStrategy": (
        "poseidon.strategies.portfolio.fundamental_selection",
        "FundamentalSelectionStrategy",
    ),
    "PredictionRankingStrategy": (
        "poseidon.strategies.portfolio.prediction_ranking",
        "PredictionRankingStrategy",
    ),
    "RevenueBreakoutStrategy": (
        "poseidon.strategies.portfolio.revenue_breakout",
        "RevenueBreakoutStrategy",
    ),
    "PortfolioRebalancer": (
        "poseidon.strategies.portfolio.rebalancer",
        "PortfolioRebalancer",
    ),
    "PositionTracker": (
        "poseidon.strategies.portfolio.position_tracker",
        "PositionTracker",
    ),
    "TargetPosition": ("poseidon.strategies.portfolio.schemas", "TargetPosition"),
    "Holding": ("poseidon.strategies.portfolio.schemas", "Holding"),
    "RebalanceOrder": ("poseidon.strategies.portfolio.schemas", "RebalanceOrder"),
    "SelectionConfig": ("poseidon.strategies.portfolio.schemas", "SelectionConfig"),
    "AllocationConfig": ("poseidon.strategies.portfolio.schemas", "AllocationConfig"),
    "RebalanceConfig": ("poseidon.strategies.portfolio.schemas", "RebalanceConfig"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover
        raise AttributeError(name) from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
