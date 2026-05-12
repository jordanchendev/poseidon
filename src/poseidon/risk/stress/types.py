"""Stress test type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from poseidon.risk.var.types import VaRResult


@dataclass
class ScenarioConfig:
    """Loaded from JSON config files."""

    name: str
    type: str  # "historical" | "hypothetical" | "correlation_stress"
    description: str
    # Historical scenario fields
    date_range: dict | None = None  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    # Hypothetical scenario fields
    shocks: dict[str, float] | None = None  # market -> shock factor
    # Correlation stress fields
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
