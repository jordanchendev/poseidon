"""StrategyMutator -- thin wrapper over VotingStrategyFactory for autoresearch.

Per D-01: NOT a new search engine. Delegates to VotingStrategyFactory + Optuna.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from poseidon.backtest.voting_strategy_factory import (
    PARAM_BOUNDS,
    VotingStrategyFactory,
    _build_config_from_params,
)

if TYPE_CHECKING:
    import optuna
    from poseidon.strategies.voting_strategy import VotingStrategy


class StrategyMutator:
    """Thin wrapper for strategy parameter mutation during autoresearch."""

    @staticmethod
    def mutate_via_optuna(
        trial: optuna.Trial,
        *,
        symbol: str,
        market: str,
        interval: str,
        model_version_id: int | None = None,
        available_models: list | None = None,
    ) -> VotingStrategy:
        """Bayesian-guided mutation via Optuna trial suggest API (D-02)."""
        return VotingStrategyFactory.from_trial(
            trial, symbol=symbol, market=market, interval=interval,
            model_version_id=model_version_id,
        )

    @staticmethod
    def mutate_random(
        seed: int,
        *,
        symbol: str,
        market: str,
        interval: str,
        model_version_id: int | None = None,
        available_models: list | None = None,
    ) -> dict:
        """Generate random config within PARAM_BOUNDS (D-03).

        Returns validated config dict. Deterministic for a given seed.
        """
        rng = random.Random(seed)
        params: dict = {}
        models = available_models or []
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            if name == "qlib_model_enabled" and not models and model_version_id is None:
                params[name] = 0
                continue
            if ptype == "int":
                params[name] = rng.randint(int(low), int(high))
            else:
                params[name] = rng.uniform(float(low), float(high))

        mv_id = model_version_id
        if mv_id is None and models and params.get("qlib_model_enabled") == 1:
            version_idx = rng.randint(0, len(models) - 1) if len(models) > 1 else 0
            params["qlib_model_version"] = version_idx
            mv_id = models[version_idx].id

        config = _build_config_from_params(
            params, symbol=symbol, market=market, interval=interval,
            model_version_id=mv_id,
        )
        # D-04: validate via existing Pydantic validation
        strategy = VotingStrategyFactory.from_config(config)
        strategy.validate_config()
        return config
