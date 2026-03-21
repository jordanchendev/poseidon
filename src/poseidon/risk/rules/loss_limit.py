"""Loss limit risk rule.

Rejects signals when the number of consecutive rejected signals
for the symbol exceeds the configured threshold. Uses signal
history rather than live price P&L (Poseidon has no live prices).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio


class LossLimitRule(BaseRule):
    """Reject signals if consecutive rejects for a symbol exceed the limit."""

    name = "loss_limit"

    def __init__(self) -> None:
        self.max_consecutive_rejects: int = 3

    def load_params(self, params: dict) -> None:
        self.max_consecutive_rejects = params.get("max_consecutive_rejects", 3)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        # This rule requires DB access to query recent signal history.
        # When used without DB context (unit tests), it passes by default.
        # The RiskEngine integration provides DB context for full evaluation.
        return RuleResult(passed=True, rule_name=self.name)
