"""Risk engine package.

Public API for risk management: BaseRule, RuleResult, RiskEngine,
VirtualPortfolio, SignalPipeline, and RULE_REGISTRY.
"""

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.risk.engine import RiskEngine
from poseidon.risk.pipeline import SignalPipeline
from poseidon.risk.portfolio import VirtualPortfolio
from poseidon.risk.rules import RULE_REGISTRY

__all__ = [
    "BaseRule",
    "RULE_REGISTRY",
    "RiskEngine",
    "RuleResult",
    "SignalPipeline",
    "VirtualPortfolio",
]
