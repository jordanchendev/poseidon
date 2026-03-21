"""Base classes for risk rules.

BaseRule ABC defines the interface for all risk rules.
RuleResult is the standard return type from rule checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from poseidon.signals.schemas import Signal

if TYPE_CHECKING:
    from poseidon.risk.portfolio import VirtualPortfolio


@dataclass
class RuleResult:
    """Result of a risk rule check."""

    passed: bool
    rule_name: str
    reason: str | None = None


class BaseRule(ABC):
    """Abstract base class for risk rules.

    Each concrete rule implements check() to evaluate a signal
    against the current portfolio state, and load_params() to
    configure itself from DB-stored JSONB parameters.
    """

    name: str = ""
    enabled: bool = True

    @abstractmethod
    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        """Evaluate signal against this rule.

        Returns RuleResult indicating pass/reject with reason.
        """
        ...

    @abstractmethod
    def load_params(self, params: dict) -> None:
        """Load rule parameters from DB-stored config."""
        ...
