"""LiquiditySweepStrategy factory for Optuna parameter search (SWEEP-05, D-01).

Mirrors VotingStrategyFactory pattern with PARAM_BOUNDS, build_from_trial(), from_config(),
to_config_dict(), and build_trial_factory() for integration with ParameterSearchPipeline.

IMPORTANT: Method is named build_from_trial() per D-01 (not from_trial()).
"""

from __future__ import annotations

from typing import Any, Callable

from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy


PARAM_BOUNDS: dict[str, tuple[int | float, int | float, str]] = {
    # Detection parameters
    "lookback_bars": (12, 48, "int"),
    "wick_ratio_min": (0.10, 0.30, "float"),
    "breakout_distance_min": (0.05, 0.30, "float"),
    "oi_buildup_min": (0.5, 3.0, "float"),
    "confirmation_threshold": (0.3, 0.8, "float"),
    "w_oi_drop": (0.2, 0.6, "float"),
    "w_volume": (0.1, 0.5, "float"),
    "w_funding": (0.1, 0.5, "float"),
    # Entry parameters
    "fib_level": (0.382, 0.786, "float"),
    "atr_mult_regime_0": (0.3, 1.0, "float"),   # Low vol
    "atr_mult_regime_1": (0.5, 1.5, "float"),   # Normal vol
    "atr_mult_regime_2": (1.0, 2.5, "float"),   # High vol
    "atr_mult_regime_3": (1.5, 3.0, "float"),   # Extreme vol
    # Exit parameters
    "cooldown_bars": (2, 12, "int"),
    # Trailing stop parameters (D-09)
    "trailing_activation_r": (0.5, 2.0, "float"),
    "trail_atr_multiplier": (1.0, 3.0, "float"),
}


def _build_config_from_params(
    params: dict[str, Any],
    *,
    symbol: str,
    market: str,
    interval: str,
    direction_mode: str = "bidirectional",
) -> dict:
    """Map flat Optuna parameter names to nested LiquiditySweepStrategy config.

    Flat PARAM_BOUNDS names -> nested config structure:
    - lookback_bars, wick_ratio_min, etc. -> detection.*
    - fib_level, atr_mult_regime_* -> entry.*
    - cooldown_bars -> exit.*
    - trailing_activation_r, trail_atr_multiplier -> trailing.*
    """
    return {
        "name": "liquidity_sweep_strategy",
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "direction_mode": direction_mode,
        "detection": {
            "lookback_bars": int(params.get("lookback_bars", 24)),
            "wick_ratio_min": float(params.get("wick_ratio_min", 0.15)),
            "breakout_distance_min": float(params.get("breakout_distance_min", 0.1)),
            "oi_buildup_min": float(params.get("oi_buildup_min", 1.0)),
            "confirmation_threshold": float(params.get("confirmation_threshold", 0.5)),
            "w_oi_drop": float(params.get("w_oi_drop", 0.4)),
            "w_volume": float(params.get("w_volume", 0.3)),
            "w_funding": float(params.get("w_funding", 0.3)),
        },
        "entry": {
            "fib_level": float(params.get("fib_level", 0.618)),
            "atr_multipliers": {
                0: float(params.get("atr_mult_regime_0", 0.5)),
                1: float(params.get("atr_mult_regime_1", 1.0)),
                2: float(params.get("atr_mult_regime_2", 1.5)),
                3: float(params.get("atr_mult_regime_3", 2.0)),
            },
        },
        "exit": {
            "cooldown_bars": int(params.get("cooldown_bars", 4)),
        },
        "trailing": {
            "activation_r": float(params.get("trailing_activation_r", 1.0)),
            "atr_multiplier": float(params.get("trail_atr_multiplier", 2.0)),
        },
    }


class LiquiditySweepStrategyFactory:
    """Factory for building LiquiditySweepStrategy from Optuna trials or config dicts.

    Mirrors VotingStrategyFactory interface (D-01):
    - from_config(config_dict) -> strategy instance
    - build_from_trial(trial, ...) -> strategy instance (with Optuna suggest) -- named per D-01
    - to_config_dict(strategy) -> round-trippable config dict
    - build_trial_factory(symbol, market, interval) -> (callable, param_bounds) for pipeline injection (D-02)
    """

    @staticmethod
    def from_config(config: dict) -> LiquiditySweepStrategy:
        """Build strategy from a config dict."""
        return LiquiditySweepStrategy(config=config)

    @staticmethod
    def build_from_trial(
        trial: Any,
        *,
        symbol: str,
        market: str,
        interval: str,
        direction_mode: str = "bidirectional",
    ) -> LiquiditySweepStrategy:
        """Build strategy by suggesting parameters from Optuna trial (D-01).

        Uses PARAM_BOUNDS for suggest_int/suggest_float ranges.
        Named build_from_trial() per D-01 locked decision.
        """
        params = {}
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))

        config = _build_config_from_params(
            params,
            symbol=symbol,
            market=market,
            interval=interval,
            direction_mode=direction_mode,
        )
        return LiquiditySweepStrategy(config=config)

    @staticmethod
    def to_config_dict(strategy: LiquiditySweepStrategy) -> dict:
        """Extract config dict from strategy instance for serialization."""
        return {
            "name": strategy.name,
            "symbol": strategy.symbol,
            "market": strategy.market,
            "interval": strategy.interval,
            "direction_mode": strategy._direction_mode,
            "detection": {
                "lookback_bars": strategy._lookback_bars,
                "wick_ratio_min": strategy._wick_ratio_min,
                "breakout_distance_min": strategy._breakout_distance_min,
                "oi_buildup_min": strategy._oi_buildup_min,
                "confirmation_threshold": strategy._confirmation_threshold,
                "w_oi_drop": strategy._w_oi_drop,
                "w_volume": strategy._w_volume,
                "w_funding": strategy._w_funding,
            },
            "entry": {
                "fib_level": strategy._fib_level,
                "atr_multipliers": dict(strategy._atr_multipliers),
            },
            "exit": {
                "cooldown_bars": strategy._cooldown_bars,
            },
            "trailing": {
                "activation_r": strategy._trailing_activation_r,
                "atr_multiplier": strategy._trail_atr_multiplier,
            },
        }

    @classmethod
    def build_trial_factory(
        cls,
        *,
        symbol: str,
        market: str,
        interval: str,
        direction_mode: str = "bidirectional",
    ) -> tuple[Callable[[dict], LiquiditySweepStrategy], dict]:
        """Return (trial_strategy_factory, param_bounds) for ParameterSearchPipeline (D-02).

        The returned callable accepts a flat params dict and returns a strategy instance.
        The returned param_bounds dict is PARAM_BOUNDS for Optuna suggest calls.

        This method enables truly polymorphic pipeline injection -- the pipeline
        calls self.strategy_factory.build_trial_factory() instead of importing
        LiquiditySweep internals. No strategy-specific code leaks into param_search.py.
        """
        def trial_strategy_factory(params: dict) -> LiquiditySweepStrategy:
            config_dict = _build_config_from_params(
                params,
                symbol=symbol,
                market=market,
                interval=interval,
                direction_mode=direction_mode,
            )
            return cls.from_config(config_dict)

        return trial_strategy_factory, dict(PARAM_BOUNDS)
