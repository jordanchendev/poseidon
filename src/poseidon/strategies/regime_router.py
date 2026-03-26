"""RegimeRouter -- wraps VotingStrategy with per-regime parameter overrides.

Dynamically adjusts min_votes and position_pct based on regime model
predictions. Preserves trailing stop state across regime changes by
mutating existing strategy attributes rather than re-instantiating.

When disabled, passes through to static base config values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from poseidon.backtest.voting_strategy_factory import VotingStrategyFactory
from poseidon.signals.schemas import Signal
from poseidon.strategies.base import BaseStrategy, StrategyType

if TYPE_CHECKING:
    from poseidon.ml.implementations.xgboost_regime import XGBoostRegimeModel

logger = logging.getLogger(__name__)


DEFAULT_REGIME_CONFIGS: dict[str, dict] = {
    "high_vol": {"min_votes": 5, "position_pct": 0.05},
    "medium_vol": {"min_votes": 4, "position_pct": 0.08},
    "low_vol": {"min_votes": 3, "position_pct": 0.10},
}


class RegimeRouter(BaseStrategy):
    """Strategy wrapper that applies per-regime min_votes/position_pct overrides.

    Wraps a single VotingStrategy instance created from base_config.
    On each evaluate() call, queries the regime model for the current regime
    and mutates the underlying strategy's _min_votes and _position_pct
    attributes accordingly. This preserves trailing stop state (D-05).

    When enabled=False, resets to base config values on every call.
    """

    strategy_type = StrategyType.VOTING

    def __init__(
        self,
        base_config: dict,
        regime_model: XGBoostRegimeModel,
        regime_configs: dict[str, dict] | None = None,
        enabled: bool = True,
    ) -> None:
        self._base_config = base_config
        self._regime_model = regime_model
        self._regime_configs = regime_configs or dict(DEFAULT_REGIME_CONFIGS)
        self.enabled = enabled

        # Single instance -- state preserved across regime changes (D-05)
        self._strategy = VotingStrategyFactory.from_config(base_config)

        # Copy identity from inner strategy
        self.name = self._strategy.name
        self.symbol = self._strategy.symbol
        self.market = self._strategy.market
        self.interval = self._strategy.interval

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate with per-regime parameter overrides."""
        if self.enabled and not features.empty:
            regime_pred = self._regime_model.predict(features)
            current_regime = regime_pred.iloc[-1]["prediction"]
            overrides = self._regime_configs.get(current_regime, {})
            self._strategy._min_votes = overrides.get(
                "min_votes", self._base_config.get("min_votes", 4)
            )
            self._strategy._position_pct = overrides.get(
                "position_pct", self._base_config.get("position_pct", 0.08)
            )
        else:
            # Disabled: reset to base config values
            self._strategy._min_votes = self._base_config.get("min_votes", 4)
            self._strategy._position_pct = self._base_config.get("position_pct", 0.08)

        return self._strategy.evaluate(features)

    def reset(self) -> None:
        """Reset trailing stop state -- delegates to underlying strategy."""
        self._strategy.reset()

    def validate_config(self) -> bool:
        """Validate config -- delegates to underlying strategy."""
        return self._strategy.validate_config()
