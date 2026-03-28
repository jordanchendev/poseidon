"""Stress test type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from poseidon.risk.var.types import VaRResult


@dataclass
class ScenarioConfig:
    """Loaded from JSON config files (per D-01)."""

    name: str
    type: str  # "historical" | "hypothetical" | "correlation_stress"
    description: str
    # Historical scenario fields (D-02)
    date_range: dict | None = None  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    # Hypothetical scenario fields (D-03)
    shocks: dict[str, float] | None = None  # market -> shock factor
    # Correlation stress fields (D-04)
    target_correlation: float | None = None


@dataclass
class StressTestResult:
    """Result of running a stress test scenario."""

    scenario_name: str
    scenario_type: str
    var_result: VaRResult | None = None
    portfolio_pnl: float = 0.0
    worst_case_loss: float = 0.0
    details: dict = field(default_factory=dict)
    computed_at: datetime | None = None
