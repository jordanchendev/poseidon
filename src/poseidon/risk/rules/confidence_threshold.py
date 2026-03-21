"""Confidence threshold risk rule.

Rejects signals with confidence below a configurable minimum.
Pure signal check -- no portfolio state needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio


class ConfidenceThresholdRule(BaseRule):
    """Reject signals with confidence below min_confidence."""

    name = "confidence_threshold"

    def __init__(self) -> None:
        self.min_confidence: float = 0.5

    def load_params(self, params: dict) -> None:
        self.min_confidence = params.get("min_confidence", 0.5)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        if signal.confidence < self.min_confidence:
            return RuleResult(
                passed=False,
                rule_name=self.name,
                reason=(
                    f"Confidence {signal.confidence:.2f} below "
                    f"threshold {self.min_confidence:.2f}"
                ),
            )
        return RuleResult(passed=True, rule_name=self.name)
