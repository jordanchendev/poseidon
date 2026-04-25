"""StructuralStrategyFactory -- create StructuralReversalStrategy from config dict.

Bridges WalkForwardAnalyzer and GridSearchOptimizer with StructuralReversalStrategy.
"""

from __future__ import annotations

from collections.abc import Callable

from poseidon.strategies.structural_reversal import (
    StructuralReversalConfig,
    StructuralReversalStrategy,
)


class StructuralStrategyFactory:
    """Factory for WalkForwardAnalyzer + GridSearchOptimizer integration.

    WalkForwardAnalyzer calls:
      - to_config_dict(strategy) -> dict  (before window loop)
      - from_config(config_dict) -> StructuralReversalStrategy  (per window)
    """

    @staticmethod
    def from_config(config: dict) -> StructuralReversalStrategy:
        """Reconstruct strategy from config dict (per-window in WFE)."""
        cfg_data = config.get("config", config)  # handle both flat and nested
        symbol = config.get("symbol", "BTCUSDT")
        cfg = StructuralReversalConfig(**cfg_data)
        strategy = StructuralReversalStrategy(config=cfg, symbol=symbol)
        strategy.validate_config()
        return strategy

    @staticmethod
    def to_config_dict(strategy: StructuralReversalStrategy) -> dict:
        """Extract config dict from strategy for WFE window reconstruction."""
        return {
            "config": strategy.config.model_dump(),
            "symbol": strategy.symbol,
        }

    @staticmethod
    def make_factory(base_params: dict, symbol: str) -> Callable[[dict], StructuralReversalStrategy]:
        """Create a strategy factory callable for GridSearchOptimizer.

        Args:
            base_params: Base parameter dict (e.g. {"use_cvd_filter": False}).
            symbol: Symbol for this run ("BTCUSDT" or "ETHUSDT").

        Returns:
            Callable that takes a trial_params dict and returns a configured strategy.

        Usage:
            factory = StructuralStrategyFactory.make_factory(
                {"use_cvd_filter": False}, "BTCUSDT"
            )
            strategy = factory({"atr_multiplier": 1.5, "stop_atr_multiplier": 3.0})
        """

        def _factory(trial_params: dict) -> StructuralReversalStrategy:
            merged = {**base_params, **trial_params}
            cfg = StructuralReversalConfig(**merged)
            return StructuralReversalStrategy(config=cfg, symbol=symbol)

        return _factory
