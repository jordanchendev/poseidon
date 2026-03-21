"""Position limit risk rule.

Rejects signals when open position count reaches the configured maximum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio


class PositionLimitRule(BaseRule):
    """Reject signals when open positions count >= max_positions."""

    name = "position_limit"

    def __init__(self) -> None:
        self.max_positions: int = 10

    def load_params(self, params: dict) -> None:
        self.max_positions = params.get("max_positions", 10)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        count = portfolio.open_position_count
        if count >= self.max_positions:
            return RuleResult(
                passed=False,
                rule_name=self.name,
                reason=(
                    f"Open positions ({count}) >= limit ({self.max_positions})"
                ),
            )
        return RuleResult(passed=True, rule_name=self.name)
