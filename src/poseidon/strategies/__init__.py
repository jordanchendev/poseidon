"""Strategy framework for Poseidon.

Two strategy types share a common interface:
- ModelStrategy: wraps a BaseModel, converts predictions to Signals
- RuleStrategy: parses DSL JSON, evaluates conditions against features
"""

from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.model_strategy import ModelStrategy
from poseidon.strategies.rule_strategy import RuleStrategy

__all__ = ["BaseStrategy", "StrategyType", "ModelStrategy", "RuleStrategy"]
