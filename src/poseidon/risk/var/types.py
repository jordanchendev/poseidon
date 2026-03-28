"""Shared VaR types used across all VaR computation modules.

Convention: VaR/CVaR stored as positive numbers representing loss fraction.
Example: 0.05 = 5% loss at the given confidence level.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class VaRMethod(str, Enum):
    """Supported VaR computation methods."""

    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    CORNISH_FISHER = "cornish_fisher"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """Result of a VaR computation.

    All VaR/CVaR values are positive loss fractions (e.g. 0.05 = 5% loss).
    """

    method: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    as_of: datetime
    computed_at: datetime
    portfolio_value: float
    holding_period: int = 1
    details: dict | None = None
