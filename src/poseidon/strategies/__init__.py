"""Strategy framework for Poseidon.

Two strategy types share a common interface:
- ModelStrategy: wraps a BaseModel, converts predictions to Signals
- RuleStrategy: parses DSL JSON, evaluates conditions against features
"""

from poseidon.strategies.base import BaseStrategy, StrategyType

__all__ = ["BaseStrategy", "StrategyType"]
