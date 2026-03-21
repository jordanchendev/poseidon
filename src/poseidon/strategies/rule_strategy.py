"""RuleStrategy — evaluates JSON DSL conditions against feature data.

Data flow:
    DSL JSON -> RuleConfig (validated)
        -> For each rule: evaluate_condition(rule.condition, features, last_row)
        -> If True: emit Signal with rule's action and quantity_pct
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.dsl.executor import evaluate_condition
from poseidon.strategies.dsl.schema import RuleConfig

logger = logging.getLogger(__name__)


class RuleStrategy(BaseStrategy):
    """Strategy that evaluates JSON DSL conditions against feature data."""

    strategy_type = StrategyType.RULE

    def __init__(
        self,
        *,
        config: RuleConfig | dict,
        instrument: InstrumentType = InstrumentType.SPOT,
        strategy_id: UUID | None = None,
    ):
        if isinstance(config, dict):
            config = RuleConfig(**config)
        self.config = config
        self.name = config.name
        self.symbol = config.symbol
        self.market = config.market
        self.interval = config.interval
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate all rules against the last row of the feature DataFrame."""
        if features.empty:
            return []

        row_idx = len(features) - 1
        signals: list[Signal] = []

        for rule in self.config.rules:
            try:
                triggered = evaluate_condition(rule.condition, features, row_idx)
            except (ValueError, KeyError) as e:
                logger.warning("RuleStrategy '%s' rule evaluation error: %s", self.name, e)
                continue

            if triggered:
                signal_time = (
                    features.index[row_idx]
                    if isinstance(features.index[row_idx], datetime)
                    else datetime.now(timezone.utc)
                )
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        market=self.market,
                        instrument=self.instrument,
                        action=SignalAction(rule.action),
                        confidence=rule.confidence,
                        quantity_pct=rule.quantity_pct,
                        interval=self.interval,
                        signal_time=signal_time,
                    )
                )

        logger.info(
            "RuleStrategy '%s' evaluated %d rules -> %d signals",
            self.name, len(self.config.rules), len(signals),
        )
        return signals

    def validate_config(self) -> bool:
        if not self.config.rules:
            raise ValueError("RuleStrategy requires at least one rule")
        if not self.symbol:
            raise ValueError("RuleStrategy requires a symbol")
        if not self.market:
            raise ValueError("RuleStrategy requires a market")
        return True
