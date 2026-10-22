"""Gate 3 -- cost and slippage stress.

The caller re-runs the selected strategy with commission, tax and slippage all
multiplied by ``cost_stress_multiplier`` and passes both results. The gate
refuses a stressed run built with a different multiplier, then passes iff the
stressed ``total_pnl_twd`` is > 0 (when ``cost_stress_pnl_positive``).
Erosion = (base pnl - stressed pnl) / |base pnl|.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from poseidon.pipeline.gates.base import GateResult, ResultSummary, failed, invalid, non_finite, passed, pct

GATE = "cost_stress"


@dataclass(frozen=True)
class CostStressInputs:
    base: ResultSummary
    stressed: ResultSummary
    cost_multiplier: float  # the multiplier the stressed run was actually built with


def evaluate(inputs: CostStressInputs, thresholds: Mapping) -> GateResult:
    multiplier = float(thresholds["cost_stress_multiplier"])
    require_pnl = bool(thresholds["cost_stress_pnl_positive"])
    base, stressed = inputs.base, inputs.stressed
    bad = non_finite(
        [
            ("base.total_pnl_twd", base.total_pnl_twd),
            ("stressed.total_pnl_twd", stressed.total_pnl_twd),
            ("cost_multiplier", inputs.cost_multiplier),
            *[(f"base.costs.{k}", v) for k, v in base.costs.as_report().items()],
            *[(f"stressed.costs.{k}", v) for k, v in stressed.costs.as_report().items()],
        ]
    )
    if bad:
        return invalid(GATE, "non-finite inputs: " + ", ".join(bad))

    erosion = (base.total_pnl_twd - stressed.total_pnl_twd) / abs(base.total_pnl_twd) if base.total_pnl_twd else None
    report = {
        "cost_multiplier": multiplier,
        "stressed_run_multiplier": float(inputs.cost_multiplier),
        "base_total_pnl_twd": float(base.total_pnl_twd),
        "stressed_total_pnl_twd": float(stressed.total_pnl_twd),
        "pnl_erosion_pct": None if erosion is None else erosion * 100.0,
        "base_costs": base.costs.as_report(),
        "stressed_costs": stressed.costs.as_report(),
        "base_trade_count": base.trade_count,
        "stressed_trade_count": stressed.trade_count,
        "pnl_positive_required": require_pnl,
    }
    if not math.isclose(inputs.cost_multiplier, multiplier, rel_tol=1e-9):
        return failed(
            GATE,
            "multiplier_mismatch",
            f"stressed run used costs x{inputs.cost_multiplier:g}, config requires x{multiplier:g}",
            "cost multiplier mismatch",
            report,
        )
    headline = f"costs x{multiplier:g} PnL {stressed.total_pnl_twd:,.0f} (erosion {pct(erosion)})"
    if require_pnl and not stressed.total_pnl_twd > 0:
        return failed(
            GATE,
            "stressed_not_profitable",
            f"pnl with costs x{multiplier:g} is {stressed.total_pnl_twd:,.0f} TWD <= 0",
            headline,
            report,
        )
    return passed(GATE, headline, report)
