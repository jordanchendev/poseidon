"""Frequency control risk rule.

Rejects signals when a symbol has received too many signals
within a configured time window.

**Status: Placeholder** — This rule always passes because it requires
DB access to count recent passed signals for a symbol. The ``check()``
method receives only the current signal and the in-memory portfolio,
which has no signal history.  To fully implement, inject a
``SignalRepository`` or DB session so the rule can query
``SELECT COUNT(*) FROM signals WHERE symbol = ? AND status = 'passed'
AND signal_time > now() - interval ...``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from poseidon.risk.base import BaseRule, RuleResult
from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio

logger = logging.getLogger(__name__)


class FrequencyRule(BaseRule):
    """Reject signals if symbol has >= max_signals within time_window_hours.

    .. warning::
        Placeholder implementation — always passes.  Requires DB session
        injection to query recent signal history.  See module docstring.
    """

    name = "frequency"
    supports_live = True

    def __init__(self) -> None:
        self.max_signals: int = 5
        self.time_window_hours: int = 24

    def load_params(self, params: dict) -> None:
        self.max_signals = params.get("max_signals", 5)
        self.time_window_hours = params.get("time_window_hours", 24)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        # Placeholder: DB access required to count recent passed signals.
        logger.warning(
            "FrequencyRule is a placeholder — always passes. "
            "Inject DB session to enable frequency limiting for %s.",
            signal.symbol,
        )
        return RuleResult(passed=True, rule_name=self.name)
