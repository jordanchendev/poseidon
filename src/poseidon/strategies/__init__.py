"""Strategy framework for Poseidon.

Strategy types sharing a common interface:
- ModelStrategy: wraps a BaseModel, converts predictions to Signals
- RuleStrategy: parses DSL JSON, evaluates conditions against features
- VotingStrategy: multi-signal voting with ATR trailing stop
- StructuralReversalStrategy: 2-3 condition limit order reversal (Phase 77)
"""

from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.model_strategy import ModelStrategy
from poseidon.strategies.regime_router import RegimeRouter
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.structural_reversal import StructuralReversalStrategy
from poseidon.strategies.voting_strategy import VotingStrategy

__all__ = [
    "BaseStrategy",
    "ModelStrategy",
    "RegimeRouter",
    "RuleStrategy",
    "StrategyType",
    "StructuralReversalStrategy",
    "VotingStrategy",
]
