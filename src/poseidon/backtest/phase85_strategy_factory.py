"""Phase 85 — 9-dim D-01 PARAM_SPACE → 17-key liquidity_sweep_factory remapper.

Why this exists: D-01 freezes a 9-dim Optuna search space; the underlying
``liquidity_sweep_factory`` exposes 17 ``PARAM_BOUNDS`` keys with regime-indexed
ATR names. This wrapper bridges the two without mutating the factory.

Per D-01 locked decision (CONTEXT.md), 8 ``PARAM_BOUNDS`` keys (e.g.
``confirmation_threshold``, ``w_oi_drop``, ``atr_mult_regime_3`` ...)
intentionally take factory default values. The driver writes both
``best_params`` (9-dim) AND ``factory_params_resolved`` (17-key with
defaults) to the artifact for Phase 86 audit reproducibility
(RESEARCH.md Pitfall 5).

A1 note (BayesianOptimizer ``bars_per_year`` support): NOT resolved here —
plan 85-03 will decide whether to extend ``BayesianOptimizer.optimize`` or
post-recompute via ``compute_metrics(returns, bars_per_year=525_600)`` on
trial user_attrs (RESEARCH.md Pitfall 6).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from poseidon.backtest.liquidity_sweep_factory import (
    PARAM_BOUNDS,  # 17 entries — referenced for default lookup only
    LiquiditySweepStrategyFactory,
    _build_config_from_params,
)

# D-01 locked search space (CONTEXT.md). Phase 85 ONLY samples these 9 dims.
PARAM_SPACE: dict[str, tuple[float, float, str]] = {
    "lookback_bars":         (720, 10080, "int"),
    "cooldown_bars":         (60, 1440, "int"),
    "wick_ratio_min":        (0.05, 0.30, "float"),
    "breakout_distance_min": (0.05, 0.30, "float"),
    "oi_buildup_min":        (0.5, 2.0, "float"),
    "fib_level":             (0.382, 0.786, "float"),
    "atr_multiplier_low":    (1.0, 3.0, "float"),
    "atr_multiplier_mid":    (1.5, 4.0, "float"),
    "atr_multiplier_high":   (2.0, 6.0, "float"),
}

# D-01 names → liquidity_sweep_factory PARAM_BOUNDS names (RESEARCH.md Pitfall 5).
# Note: ``atr_mult_regime_3`` is intentionally absent — falls back to factory
# default 2.0 (liquidity_sweep_factory.py line 78).
D01_TO_FACTORY_NAME: dict[str, str] = {
    "atr_multiplier_low":  "atr_mult_regime_0",
    "atr_multiplier_mid":  "atr_mult_regime_1",
    "atr_multiplier_high": "atr_mult_regime_2",
    # 6 others are pass-through:
    #   lookback_bars, cooldown_bars, wick_ratio_min,
    #   breakout_distance_min, oi_buildup_min, fib_level
}


# Snapshot of factory defaults extracted from
# ``liquidity_sweep_factory._build_config_from_params`` ``params.get(name, default)``
# fallbacks. These are the values the factory uses when a key is absent from
# the inbound params dict; mirroring them here lets the driver write a
# faithful ``factory_params_resolved`` artefact for Phase 86 audit.
#
# IMPORTANT: keep in sync with liquidity_sweep_factory.py:60-87. If the
# factory defaults move, an audit replay would diverge silently.
_FACTORY_DEFAULTS: dict[str, float | int] = {
    "lookback_bars": 24,
    "wick_ratio_min": 0.15,
    "breakout_distance_min": 0.10,
    "oi_buildup_min": 1.0,
    "confirmation_threshold": 0.5,
    "w_oi_drop": 0.4,
    "w_volume": 0.3,
    "w_funding": 0.3,
    "fib_level": 0.618,
    "atr_mult_regime_0": 0.5,
    "atr_mult_regime_1": 1.0,
    "atr_mult_regime_2": 1.5,
    "atr_mult_regime_3": 2.0,
    "cooldown_bars": 6,  # not present in factory default fallback; use mid-PARAM_BOUNDS
    "trailing_activation_r": 1.0,
    "trail_atr_multiplier": 2.0,
}


def remap_d01_to_factory(d01_params: dict[str, Any]) -> dict[str, Any]:
    """Remap 9-dim D-01 keys to factory PARAM_BOUNDS namespace.

    Pure rename; values pass through unchanged. Keys absent from
    ``D01_TO_FACTORY_NAME`` (the 6 pass-through dims) keep their D-01 name
    which already matches the factory PARAM_BOUNDS key.
    """
    return {D01_TO_FACTORY_NAME.get(k, k): v for k, v in d01_params.items()}


def resolve_factory_params(d01_params: dict[str, Any]) -> dict[str, Any]:
    """Return the full 17-key dict with factory defaults applied.

    Used by Phase 85 driver to write ``factory_params_resolved`` to the
    artifact. Phase 86 replays it to reconstruct the exact strategy config
    that produced the OOS metrics (RESEARCH.md Pitfall 5).
    """
    remapped = remap_d01_to_factory(d01_params)
    resolved: dict[str, Any] = {}
    for name in PARAM_BOUNDS:
        if name in remapped:
            resolved[name] = remapped[name]
        else:
            # Mirror liquidity_sweep_factory._build_config_from_params default
            # policy. ``_FACTORY_DEFAULTS`` snapshots those defaults; if the
            # name is somehow missing from the snapshot fall back to mid-range
            # so callers never see a KeyError.
            if name in _FACTORY_DEFAULTS:
                resolved[name] = _FACTORY_DEFAULTS[name]
            else:
                low, high, kind = PARAM_BOUNDS[name]
                resolved[name] = (
                    int((low + high) // 2) if kind == "int" else (low + high) / 2.0
                )
    return resolved


def make_phase85_strategy_factory(
    symbol: str,
    market: str = "crypto_perp",
    interval: str = "1m",
    *,
    direction_mode: str = "bidirectional",
) -> Callable[[dict[str, Any]], Any]:
    """Return a callable: D-01 9-dim params dict → ``LiquiditySweepStrategy``.

    Use as
    ``BayesianOptimizer(strategy_factory=make_phase85_strategy_factory("BTCUSDT"))``.
    The returned callable accepts a flat 9-dim dict (D-01 names); it remaps
    to factory namespace and delegates to ``LiquiditySweepStrategyFactory.from_config``.
    """

    def _factory(d01_params: dict[str, Any]) -> Any:
        factory_params = remap_d01_to_factory(d01_params)
        cfg = _build_config_from_params(
            factory_params,
            symbol=symbol,
            market=market,
            interval=interval,
            direction_mode=direction_mode,
        )
        return LiquiditySweepStrategyFactory.from_config(cfg)

    return _factory


def build_phase85_factory(
    symbol: str,
    market: str = "crypto_perp",
    interval: str = "1m",
    *,
    direction_mode: str = "bidirectional",
) -> tuple[Callable[[dict[str, Any]], Any], dict[str, tuple[float, float, str]]]:
    """Mirror ``liquidity_sweep_factory.build_trial_factory`` signature.

    Returns ``(factory_callable, PARAM_SPACE)``. The driver hands both to
    ``BayesianOptimizer.optimize(strategy_factory=..., param_space=...)``.
    """
    factory = make_phase85_strategy_factory(
        symbol, market, interval, direction_mode=direction_mode
    )
    return factory, dict(PARAM_SPACE)


__all__ = [
    "PARAM_SPACE",
    "D01_TO_FACTORY_NAME",
    "remap_d01_to_factory",
    "resolve_factory_params",
    "make_phase85_strategy_factory",
    "build_phase85_factory",
]
