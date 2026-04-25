"""ModelStrategyFactory for Optuna parameter search (FACT-02).

Mirrors LiquiditySweepStrategyFactory duck-type pattern with PARAM_BOUNDS,
build_from_trial(), from_config(), to_config_dict(), and build_trial_factory()
for integration with ParameterSearchPipeline.

Key design decisions (D-06 ~ D-09):
- Dynamic model_version_idx injection when multiple models available
- PARAM_BOUNDS: prediction_threshold, position_pct, stop_loss_pct, take_profit_pct
- build_trial_factory() returns config dicts (not strategy instances) because
  ModelStrategy requires a loaded model; BacktestRunner handles model loading
  via prediction cache mechanism (Phase 43/44/45).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


PARAM_BOUNDS: dict[str, tuple[int | float, int | float, str]] = {
    "prediction_threshold": (0.3, 0.8, "float"),
    "position_pct": (0.03, 0.15, "float"),
    "stop_loss_pct": (0.02, 0.10, "float"),
    "take_profit_pct": (0.05, 0.25, "float"),
}


class ModelStrategyFactory:
    """Factory for building ModelStrategy configs from Optuna trials.

    Mirrors LiquiditySweepStrategyFactory interface (D-08):
    - from_config(config) -> config dict (pass-through)
    - build_from_trial(trial, ...) -> config dict with model_version_id
    - to_config_dict(config_or_strategy) -> dict representation
    - build_trial_factory(symbol, market, interval) -> (callable, param_bounds)

    NOTE: build_trial_factory() is an INSTANCE method because it reads
    self._available_models for dynamic model_version_idx bound injection (D-06).

    Unlike LiquiditySweepStrategyFactory which returns strategy instances,
    ModelStrategyFactory returns config dicts. ModelStrategy requires a loaded
    MLBaseModel instance, and model loading is handled by BacktestRunner's
    prediction cache mechanism. The config dict contains model_version_id +
    search params for the pipeline to use.
    """

    def __init__(
        self,
        *,
        available_models: list | None = None,
    ):
        """Initialize factory with available models for dynamic bound injection.

        Args:
            available_models: List of ModelVersion objects from
                ModelManager.list_ready_models(). Each must have .id attribute.
                If None or empty, model_version_idx is not added to bounds.
        """
        self._available_models = list(available_models) if available_models else []

    @staticmethod
    def from_config(config: dict) -> dict:
        """Return config dict as-is.

        ModelStrategy needs a loaded model, but for autoresearch we pass
        config dicts through the pipeline. BacktestRunner handles model
        loading via prediction cache.
        """
        return config

    @staticmethod
    def build_from_trial(
        trial: Any,
        *,
        symbol: str,
        market: str,
        interval: str,
        available_models: list | None = None,
    ) -> dict:
        """Build config dict by suggesting parameters from Optuna trial.

        Args:
            trial: Optuna trial object.
            symbol: Trading symbol.
            market: Market identifier.
            interval: Time interval.
            available_models: Optional list of ModelVersion objects for
                model selection.

        Returns:
            Config dict with model_version_id and search parameters.
        """
        params: dict[str, Any] = {}
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))

        # Dynamic model selection
        model_version_id = None
        if available_models and len(available_models) > 1:
            idx = trial.suggest_int("model_version_idx", 0, len(available_models) - 1)
            model_version_id = str(available_models[idx].id)
        elif available_models and len(available_models) == 1:
            model_version_id = str(available_models[0].id)

        return {
            "name": "model_optuna_search",
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "model_version_id": model_version_id,
            "prediction_threshold": float(params["prediction_threshold"]),
            "position_pct": float(params["position_pct"]),
            "stop_loss_pct": float(params["stop_loss_pct"]),
            "take_profit_pct": float(params["take_profit_pct"]),
        }

    @staticmethod
    def to_config_dict(config_or_strategy: Any) -> dict:
        """Extract config dict from strategy instance or config dict.

        Handles both raw config dicts and strategy instances with
        attribute-based access.
        """
        if isinstance(config_or_strategy, dict):
            return config_or_strategy

        # Strategy instance -- extract relevant attributes
        result = {
            "name": getattr(config_or_strategy, "name", "model_strategy"),
            "symbol": getattr(config_or_strategy, "symbol", ""),
            "market": getattr(config_or_strategy, "market", ""),
            "interval": getattr(config_or_strategy, "interval", "1d"),
        }
        if hasattr(config_or_strategy, "model_id"):
            result["model_version_id"] = str(config_or_strategy.model_id)
        return result

    def build_trial_factory(
        self,
        *,
        symbol: str,
        market: str,
        interval: str,
    ) -> tuple[Callable[[dict], dict], dict]:
        """Return (trial_strategy_factory, param_bounds) for ParameterSearchPipeline.

        Instance method -- reads self._available_models for dynamic
        model_version_idx bound injection (D-06).

        If len(available_models) > 1, adds model_version_idx to bounds.
        If len(available_models) <= 1, no model selection needed.
        """
        available_models = self._available_models

        # Build effective bounds with dynamic model_version_idx
        effective_bounds = dict(PARAM_BOUNDS)
        if len(available_models) > 1:
            effective_bounds["model_version_idx"] = (0, len(available_models) - 1, "int")

        def trial_strategy_factory(params: dict) -> dict:
            # Resolve model_version_id from idx
            model_version_id = None
            if available_models:
                if "model_version_idx" in params and len(available_models) > 1:
                    idx = int(params["model_version_idx"])
                    idx = min(idx, len(available_models) - 1)
                    model_version_id = str(available_models[idx].id)
                else:
                    model_version_id = str(available_models[0].id)

            return {
                "name": "model_optuna_search",
                "symbol": symbol,
                "market": market,
                "interval": interval,
                "model_version_id": model_version_id,
                "prediction_threshold": float(params.get("prediction_threshold", 0.5)),
                "position_pct": float(params.get("position_pct", 0.10)),
                "stop_loss_pct": float(params.get("stop_loss_pct", 0.05)),
                "take_profit_pct": float(params.get("take_profit_pct", 0.10)),
            }

        return trial_strategy_factory, effective_bounds
