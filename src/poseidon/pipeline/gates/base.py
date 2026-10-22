"""Shared input/output types for the six validation gates.

Every gate is a pure function ``evaluate(inputs, thresholds) -> GateResult``:
no database, no Celery, no backtest execution. ``thresholds`` is the
``config_json["thresholds"]`` mapping frozen into the run; gates read nothing
else. Reports are finite JSON (``canonical_json`` rejects NaN), so non-finite
numbers are written as ``None`` and every required non-finite input fails the
gate with ``invalid_input`` instead of degrading silently.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from poseidon.pipeline.state import Verdict

PASS_CODE = "pass"
INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class GateResult:
    """Verdict of one gate. ``headline`` is the short Telegram phrase."""

    gate: str
    verdict: Verdict
    reason_code: str
    reason: str
    headline: str
    report: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


def passed(gate: str, headline: str, report: dict, reason: str = "all checks passed") -> GateResult:
    return GateResult(gate, Verdict.PASS, PASS_CODE, reason, headline, report)


def failed(gate: str, code: str, reason: str, headline: str, report: dict) -> GateResult:
    return GateResult(gate, Verdict.FAIL, code, reason, headline, report)


def invalid(gate: str, reason: str, report: dict | None = None) -> GateResult:
    return failed(gate, INVALID_INPUT, reason, "invalid input", report or {})


@dataclass(frozen=True)
class Trade:
    """One closed trade. ``pnl_twd`` is net of fees; ``fees_twd`` is informational."""

    entry_time: datetime
    exit_time: datetime
    side: Literal["long", "short"]
    pnl_twd: float
    fees_twd: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    commission_twd: float = 0.0
    tax_twd: float = 0.0
    slippage_twd: float = 0.0

    @property
    def total_twd(self) -> float:
        return self.commission_twd + self.tax_twd + self.slippage_twd

    def as_report(self) -> dict:
        return {
            "commission_twd": finite_or_none(self.commission_twd),
            "tax_twd": finite_or_none(self.tax_twd),
            "slippage_twd": finite_or_none(self.slippage_twd),
            "total_twd": finite_or_none(self.total_twd),
        }


@dataclass(frozen=True)
class ResultSummary:
    """Headline numbers of one backtest result (a train, holdout or stressed run)."""

    total_pnl_twd: float
    sharpe: float
    trade_count: int
    costs: CostBreakdown = field(default_factory=CostBreakdown)


def is_finite(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def finite_or_none(value):
    return float(value) if is_finite(value) else None


def non_finite(named: Iterable[tuple[str, object]]) -> list[str]:
    """Names whose value is missing, NaN, infinite or not a number."""

    return [name for name, value in named if not is_finite(value)]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"
