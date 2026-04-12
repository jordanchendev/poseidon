"""Strategy framework for Poseidon.

Three strategy types share a common interface:
- ModelStrategy: wraps a BaseModel, converts predictions to Signals
- RuleStrategy: parses DSL JSON, evaluates conditions against features
- VotingStrategy: multi-signal voting with ATR trailing stop
- LiquiditySweepStrategy: three-stage maker ambush with limit orders
"""

from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy
from poseidon.strategies.model_strategy import ModelStrategy
from poseidon.strategies.regime_router import RegimeRouter
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.voting_strategy import VotingStrategy

__all__ = [
    "BaseStrategy",
    "LiquiditySweepStrategy",
    "ModelStrategy",
    "RegimeRouter",
    "RuleStrategy",
    "StrategyType",
    "VotingStrategy",
]
