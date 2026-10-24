"""Gate 3 -- cost and slippage stress.

The caller re-runs the selected strategy with commission, tax and slippage all
multiplied by ``cost_stress_multiplier`` and passes both results. The gate
refuses a stressed run built with a different multiplier, then passes iff the
stressed ``total_pnl_twd`` is > 0 (when ``cost_stress_pnl_positive``).
Erosion = (base pnl - stressed pnl) / |base pnl|.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass

from poseidon.pipeline.gates.base import (
    INVALID_INPUT,
    GateResult,
    ResultSummary,
    failed,
    finite_or_none,
    invalid,
    non_finite,
    passed,
    pct,
)
from poseidon.pipeline.gates.daily import Basis, DailyResult, Rerun, RerunRequest, sharpe

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


# --- daily classes (design 2026-10-24 §7) -----------------------------------------------------
#
# Same criterion (stressed total PnL > 0, i.e. final equity > initial with every cost x the
# multiplier). The stressed run comes from the stage's re-run hook: the selected params over the
# same train window with ``RerunRequest(cost_multiplier=...)``, which must scale commission, tax
# and slippage of every fill (rebalance, roll, margin call, forced liquidation). Trade counts in
# the report are days with at least one fill.


@dataclass(frozen=True)
class CostStressDailyInputs:
    base: DailyResult
    stressed: DailyResult
    cost_multiplier: float  # the multiplier the stressed run was actually built with
    basis: Basis


def summary_daily(result: DailyResult, basis: Basis) -> ResultSummary:
    value = sharpe(result, basis)
    return ResultSummary(
        total_pnl_twd=result.total_pnl_twd,
        sharpe=float("nan") if value is None else value,
        trade_count=result.exposure_changes,
        costs=result.costs,
    )


def evaluate_daily(inputs: CostStressDailyInputs, thresholds: Mapping) -> GateResult:
    result = evaluate(
        CostStressInputs(
            summary_daily(inputs.base, inputs.basis),
            summary_daily(inputs.stressed, inputs.basis),
            inputs.cost_multiplier,
        ),
        thresholds,
    )
    if result.reason_code == INVALID_INPUT:
        return result
    report = {
        **result.report,
        "basis": inputs.basis,
        "trade_count_unit": "days with a fill",
        "base_total_return": finite_or_none(inputs.base.total_return),
        "stressed_total_return": finite_or_none(inputs.stressed.total_return),
    }
    return dataclasses.replace(result, report=report)


def run_daily(rerun: Rerun, params: Mapping, base: DailyResult, basis: Basis, thresholds: Mapping) -> GateResult:
    """Re-run ``params`` with costs x ``cost_stress_multiplier`` and judge it against ``base``."""

    multiplier = float(thresholds["cost_stress_multiplier"])
    stressed = rerun(RerunRequest(params=dict(params), cost_multiplier=multiplier))
    return evaluate_daily(CostStressDailyInputs(base, stressed, multiplier, basis), thresholds)
