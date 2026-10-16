"""RangeReversionStrategy factory for config dispatch and Optuna parameter search.

Mirrors LiquiditySweepStrategyFactory: PARAM_BOUNDS, from_config(),
build_from_trial(), to_config_dict(), build_trial_factory().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from poseidon.strategies.range_reversion import RangeReversionStrategy

PARAM_BOUNDS: dict[str, tuple[int | float, int | float, str]] = {
    "lookback_days": (5, 40, "int"),
    "min_sweeps": (1, 3, "int"),
    "sweep_valid_bars": (3, 24, "int"),
    "fvg_max_age_bars": (6, 48, "int"),
    "stop_buffer_points": (0.0, 20.0, "float"),
    "leg_lookback_bars": (12, 96, "int"),
    "min_rr": (1.0, 3.0, "float"),
    "breakeven_offset_points": (0.0, 10.0, "float"),
}


def _build_config_from_params(
    params: dict[str, Any],
    *,
    symbol: str,
    market: str,
    interval: str,
    direction_mode: str = "bidirectional",
    base_config: dict | None = None,
) -> dict:
    """Overlay flat Optuna params onto a nested RangeReversionStrategy config."""
    cfg: dict = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (base_config or {}).items()}
    cfg.update({"name": "range_50_reversion", "symbol": symbol, "market": market, "interval": interval})
    cfg["direction_mode"] = direction_mode
    cfg.setdefault("range", {})["lookback_days"] = int(params.get("lookback_days", 20))
    sweep = cfg.setdefault("sweep", {})
    sweep["min_sweeps"] = int(params.get("min_sweeps", 2))
    sweep["sweep_valid_bars"] = int(params.get("sweep_valid_bars", 12))
    cfg.setdefault("confirm", {})["fvg_max_age_bars"] = int(params.get("fvg_max_age_bars", 24))
    risk = cfg.setdefault("risk", {})
    risk["stop_buffer_points"] = float(params.get("stop_buffer_points", 5.0))
    risk["leg_lookback_bars"] = int(params.get("leg_lookback_bars", 48))
    risk["min_rr"] = float(params.get("min_rr", 1.5))
    cfg.setdefault("manage", {})["breakeven_offset_points"] = float(params.get("breakeven_offset_points", 2.0))
    return cfg


class RangeReversionStrategyFactory:
    """Duck-typed factory consumed by cpu_tasks dispatch and ParameterSearchPipeline."""

    @staticmethod
    def from_config(config: dict, *, strategy_id: Any = None) -> RangeReversionStrategy:
        if strategy_id is None:
            return RangeReversionStrategy(config=config)
        return RangeReversionStrategy(config=config, strategy_id=strategy_id)

    @staticmethod
    def build_from_trial(
        trial: Any,
        *,
        symbol: str,
        market: str,
        interval: str,
        direction_mode: str = "bidirectional",
        base_config: dict | None = None,
    ) -> RangeReversionStrategy:
        params = {}
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))
        return RangeReversionStrategy(
            config=_build_config_from_params(
                params,
                symbol=symbol,
                market=market,
                interval=interval,
                direction_mode=direction_mode,
                base_config=base_config,
            )
        )

    @staticmethod
    def to_config_dict(strategy: RangeReversionStrategy) -> dict:
        return {
            "name": strategy.name,
            "symbol": strategy.symbol,
            "market": strategy.market,
            "interval": strategy.interval,
            "direction_mode": strategy._direction_mode,
            "session": {
                "tz": strategy._tz,
                "day_rollover": strategy._day_rollover_minute,
                "entry_start": strategy._entry_start,
                "entry_end": strategy._entry_end,
                "flat_at": strategy._flat_at,
            },
            "range": {"lookback_days": strategy._lookback_days, "bias_mode": strategy._bias_mode},
            "sweep": {
                "require_h1": strategy._require_h1,
                "require_h4": strategy._require_h4,
                "require_d1": strategy._require_d1,
                "min_sweeps": strategy._min_sweeps,
                "sweep_valid_bars": strategy._sweep_valid_bars,
            },
            "confirm": {
                "smt_mode": strategy._smt_mode,
                "companion_symbol": strategy._companion_symbol,
                "companion_market": strategy._companion_market,
                "entry_trigger": strategy._entry_trigger,
                "fvg_max_age_bars": strategy._fvg_max_age_bars,
            },
            "risk": {
                "stop_buffer_points": strategy._stop_buffer_points,
                "leg_lookback_bars": strategy._leg_lookback_bars,
                "min_rr": strategy._min_rr,
                "max_stop_points": strategy._max_stop_points,
            },
            "manage": {
                "breakeven_on_h1_flip": strategy._breakeven_on_h1_flip,
                "breakeven_offset_points": strategy._breakeven_offset_points,
                "max_trades_per_day": strategy._max_trades_per_day,
                "stop_after_win": strategy._stop_after_win,
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
        base_config: dict | None = None,
    ) -> tuple[Callable[[dict], RangeReversionStrategy], dict]:
        def trial_strategy_factory(params: dict) -> RangeReversionStrategy:
            return cls.from_config(
                _build_config_from_params(
                    params,
                    symbol=symbol,
                    market=market,
                    interval=interval,
                    direction_mode=direction_mode,
                    base_config=base_config,
                )
            )

        return trial_strategy_factory, dict(PARAM_BOUNDS)
