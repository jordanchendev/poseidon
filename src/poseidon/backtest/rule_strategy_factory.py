"""RuleStrategyFactory for Optuna parameter search (FACT-01).

Mirrors LiquiditySweepStrategyFactory duck-type pattern with PARAM_BOUNDS,
build_from_trial(), from_config(), to_config_dict(), and build_trial_factory()
for integration with ParameterSearchPipeline.

Key design decisions (D-01 ~ D-05):
- Fixed N feature_above conditions with Optuna-searched thresholds
- feature_above condition type uses direct column name (no period param)
- Composite voting via min_votes threshold
- feature_names stored on factory instance (param_search.py only passes symbol/market/interval)
- RuleConfig Pydantic validation at config build time
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from poseidon.strategies.dsl.schema import RuleConfig
from poseidon.strategies.rule_strategy import RuleStrategy

PARAM_BOUNDS: dict[str, tuple[int | float, int | float, str]] = {
    "threshold_0": (0.0, 100.0, "float"),
    "threshold_1": (0.0, 100.0, "float"),
    "threshold_2": (-1.0, 1.0, "float"),
    "composite_threshold": (1, 3, "int"),
    "position_pct": (0.03, 0.15, "float"),
}


def _build_config_from_params(
    params: dict[str, Any],
    *,
    symbol: str,
    market: str,
    interval: str,
    feature_names: list[str],
    direction_mode: str = "long_only",
) -> dict:
    """Map flat Optuna parameter names to RuleConfig-compatible dict.

    Uses feature_above condition type which operates on pre-computed feature
    columns (e.g., revenue_yoy_growth) directly via condition["column"].
    No period parameter needed -- feature_above doesn't use one.

    Args:
        params: Flat dict from Optuna trial (threshold_0, threshold_1, ...).
        symbol: Trading symbol.
        market: Market identifier.
        interval: Time interval.
        feature_names: List of feature column names for each condition.
        direction_mode: "long_only" or "bidirectional".

    Returns:
        Dict that passes RuleConfig(**config) validation.
    """
    # Build conditions from available features
    conditions = []
    for i, fname in enumerate(feature_names):
        threshold_key = f"threshold_{i}"
        if threshold_key not in params:
            break
        conditions.append(
            {
                "type": "feature_above",
                "column": fname,
                "threshold": float(params[threshold_key]),
            }
        )

    if not conditions:
        raise ValueError("No conditions could be built: feature_names or params empty")

    # Composite voting: min_votes capped to number of actual conditions
    min_votes = min(
        int(params.get("composite_threshold", 1)),
        len(conditions),
    )

    # Long rule entry with composite voting
    long_rule = {
        "condition": {
            "vote": {
                "conditions": conditions,
                "min_votes": min_votes,
            }
        },
        "action": "long",
        "quantity_pct": float(params.get("position_pct", 0.10)),
        "confidence": 1.0,
    }

    rules = [long_rule]

    # Bidirectional: add short rule with inverted conditions (feature_below)
    if direction_mode == "bidirectional":
        short_conditions = []
        for i, fname in enumerate(feature_names):
            threshold_key = f"threshold_{i}"
            if threshold_key not in params:
                break
            short_conditions.append(
                {
                    "type": "feature_below",
                    "column": fname,
                    "threshold": float(params[threshold_key]),
                }
            )

        if short_conditions:
            short_rule = {
                "condition": {
                    "vote": {
                        "conditions": short_conditions,
                        "min_votes": min_votes,
                    }
                },
                "action": "short",
                "quantity_pct": float(params.get("position_pct", 0.10)),
                "confidence": 1.0,
            }
            rules.append(short_rule)

    config = {
        "name": "rule_optuna_search",
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "rules": rules,
    }

    # Validate against RuleConfig Pydantic model (D-05)
    RuleConfig(**config)

    return config


class RuleStrategyFactory:
    """Factory for building RuleStrategy from Optuna trials or config dicts.

    Mirrors LiquiditySweepStrategyFactory interface (D-03):
    - from_config(config_dict) -> strategy instance
    - build_from_trial(trial, ...) -> strategy instance (with Optuna suggest)
    - to_config_dict(strategy) -> round-trippable config dict
    - build_trial_factory(symbol, market, interval) -> (callable, param_bounds)

    NOTE: build_trial_factory() is an INSTANCE method (not classmethod) because
    it reads self._feature_names. ParameterSearchPipeline calls it with only
    symbol/market/interval kwargs (line 174 of param_search.py).
    """

    def __init__(
        self,
        *,
        feature_names: list[str],
        direction_mode: str = "long_only",
    ):
        """Initialize factory with feature names for condition building.

        Args:
            feature_names: Column names for feature_above conditions.
            direction_mode: "long_only" or "bidirectional".
        """
        self._feature_names = list(feature_names)
        self._direction_mode = direction_mode

    @staticmethod
    def from_config(config: dict) -> RuleStrategy:
        """Build strategy from a config dict."""
        return RuleStrategy(config=config)

    @staticmethod
    def build_from_trial(
        trial: Any,
        *,
        symbol: str,
        market: str,
        interval: str,
        feature_names: list[str] | None = None,
        direction_mode: str | None = None,
    ) -> RuleStrategy:
        """Build strategy by suggesting parameters from Optuna trial.

        Args:
            trial: Optuna trial object.
            symbol: Trading symbol.
            market: Market identifier.
            interval: Time interval.
            feature_names: Feature column names (required).
            direction_mode: "long_only" or "bidirectional".
        """
        if feature_names is None:
            raise ValueError("feature_names is required for RuleStrategyFactory.build_from_trial()")

        # Suggest params from PARAM_BOUNDS, trimming unused threshold_N
        params = {}
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            # Skip threshold_N keys beyond available features
            if name.startswith("threshold_"):
                idx = int(name.split("_")[1])
                if idx >= len(feature_names):
                    continue
            if ptype == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))

        config = _build_config_from_params(
            params,
            symbol=symbol,
            market=market,
            interval=interval,
            feature_names=feature_names,
            direction_mode=direction_mode or "long_only",
        )
        return RuleStrategy(config=config)

    @staticmethod
    def to_config_dict(strategy: RuleStrategy) -> dict:
        """Extract config dict from strategy instance for serialization."""
        return strategy.config.model_dump()

    def build_trial_factory(
        self,
        *,
        symbol: str,
        market: str,
        interval: str,
    ) -> tuple[Callable[[dict], RuleStrategy], dict]:
        """Return (trial_strategy_factory, param_bounds) for ParameterSearchPipeline.

        Instance method -- reads self._feature_names and self._direction_mode.
        Dynamically trims PARAM_BOUNDS if fewer features than default N=3.
        """
        feature_names = self._feature_names
        direction_mode = self._direction_mode

        # Build effective bounds: remove unused threshold_N keys
        effective_bounds = {}
        for name, bounds in PARAM_BOUNDS.items():
            if name.startswith("threshold_"):
                idx = int(name.split("_")[1])
                if idx >= len(feature_names):
                    continue
            effective_bounds[name] = bounds

        def trial_strategy_factory(params: dict) -> RuleStrategy:
            config_dict = _build_config_from_params(
                params,
                symbol=symbol,
                market=market,
                interval=interval,
                feature_names=feature_names,
                direction_mode=direction_mode,
            )
            return RuleStrategyFactory.from_config(config_dict)

        return trial_strategy_factory, effective_bounds
