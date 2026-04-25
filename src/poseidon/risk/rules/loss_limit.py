"""Loss limit risk rule.

Rejects signals when the number of consecutive rejected signals
for the symbol exceeds the configured threshold. Uses signal
history rather than live price P&L (Poseidon has no live prices).

**Status: Placeholder** — This rule always passes because it requires
DB access to query recent signal history for a symbol. The ``check()``
method receives only the current signal and the in-memory portfolio.
To fully implement, inject a ``SignalRepository`` or DB session so
the rule can query the last N signals for the symbol and count
consecutive rejections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)


class LossLimitRule(BaseRule):
    """Reject signals if consecutive rejects for a symbol exceed the limit.

    .. warning::
        Placeholder implementation — always passes.  Requires DB session
        injection to query recent signal history.  See module docstring.
    """

    name = "loss_limit"
    supports_live = True

    def __init__(self) -> None:
        self.max_consecutive_rejects: int = 3

    def load_params(self, params: dict) -> None:
        self.max_consecutive_rejects = params.get("max_consecutive_rejects", 3)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        # Placeholder: DB access required to query recent signal history.
        logger.warning(
            "LossLimitRule is a placeholder — always passes. Inject DB session to enable loss limiting for %s.",
            signal.symbol,
        )
        return RuleResult(passed=True, rule_name=self.name)
