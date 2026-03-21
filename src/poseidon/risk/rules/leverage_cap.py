"""Leverage cap risk rule.

Rejects signals when total portfolio exposure (sum of quantity_pct
across open positions) plus the new signal's quantity_pct exceeds
the configured maximum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio


class LeverageCapRule(BaseRule):
    """Reject signals if total exposure would exceed max_total_exposure."""

    name = "leverage_cap"

    def __init__(self) -> None:
        self.max_total_exposure: float = 1.0

    def load_params(self, params: dict) -> None:
        self.max_total_exposure = params.get("max_total_exposure", 1.0)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        current_exposure = portfolio.total_exposure()
        new_qty = signal.quantity_pct or 0.0
        projected = current_exposure + new_qty
        if projected > self.max_total_exposure:
            return RuleResult(
                passed=False,
                rule_name=self.name,
                reason=(
                    f"Projected exposure {projected:.2f} exceeds "
                    f"cap {self.max_total_exposure:.2f}"
                ),
            )
        return RuleResult(passed=True, rule_name=self.name)
