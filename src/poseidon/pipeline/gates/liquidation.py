"""Daily classes -- forced liquidation / ruin in the train backtest (design 2026-10-24 §7, §12 #5).

Judged on the train backtest (``backtest`` stage) and on the optimizer's selected point. Fails
when the run has a ``forced_liquidation`` event (intraday risk ratio below the engine's
``liquidation_risk_ratio``) or was ruined (equity <= 0), with ``liquidation_fails`` frozen
true. ``margin_call`` events (close equity below maintenance margin, next fill capped at the
initial-margin level) are only reported. A run frozen without ``liquidation_fails`` reports
and passes.
"""

from __future__ import annotations

from collections.abc import Mapping

from poseidon.pipeline.gates.base import GateResult, failed, passed
from poseidon.pipeline.gates.daily import DailyResult

GATE = "liquidation"


def evaluate(result: DailyResult, thresholds: Mapping) -> GateResult:
    enabled = bool(thresholds.get("liquidation_fails", False))
    liquidations = result.event_days(["forced_liquidation"])
    margin_calls = result.event_days(["margin_call"])
    report = {
        "criterion_enabled": enabled,
        "days": result.n_days,
        "first_day": result.dates[0].isoformat() if result.dates else None,
        "last_day": result.dates[-1].isoformat() if result.dates else None,
        "forced_liquidations": liquidations,
        "ruined": result.ruined.isoformat() if result.ruined is not None else None,
        "margin_calls": margin_calls,
        "margin_call_count": len(margin_calls),
    }
    note = f", {len(margin_calls)} margin calls (reported)" if margin_calls else ""
    if result.ruined is not None:
        headline = f"ruined on {result.ruined.isoformat()}"
        if enabled:
            return failed(GATE, "ruin", f"equity reached 0 on {result.ruined.isoformat()} in train", headline, report)
        return passed(GATE, f"{headline} (report only)", report, reason="liquidation_fails not frozen in this run")
    if liquidations:
        headline = f"{len(liquidations)} forced liquidations (first {liquidations[0]['date']})"
        if enabled:
            return failed(GATE, "forced_liquidation", f"{headline} in train", headline, report)
        return passed(GATE, f"{headline} (report only)", report, reason="liquidation_fails not frozen in this run")
    return passed(GATE, f"no forced liquidation{note}", report)
