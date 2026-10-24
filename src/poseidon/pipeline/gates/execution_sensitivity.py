"""Daily classes -- execution-timing sensitivity, report only (design 2026-10-24 §12 #1).

The selected params re-run with the other execution component (``next_open`` <-> ``signal_close``)
over the same window. The report records both runs' PnL, total return and daily Sharpe and the
deltas (alternative - base). No threshold is registered and nothing passes or fails here; the
owner may later make it a gate (that would be a new pre-registered key).
"""

from __future__ import annotations

from collections.abc import Mapping

from poseidon.pipeline.gates.base import finite_or_none
from poseidon.pipeline.gates.daily import Basis, DailyResult, Execution, Rerun, RerunRequest, sharpe

NAME = "execution_sensitivity"
ALTERNATIVE: dict[str, Execution] = {"next_open": "signal_close", "signal_close": "next_open"}


def _row(result: DailyResult, basis: Basis) -> dict:
    return {
        "total_pnl_twd": finite_or_none(result.total_pnl_twd),
        "total_return": finite_or_none(result.total_return),
        "sharpe_daily": finite_or_none(sharpe(result, basis)),
        "days": result.n_days,
        "exposure_changes": result.exposure_changes,
    }


def _delta(a, b):
    return b - a if a is not None and b is not None else None


def compare(base: DailyResult, alternative: DailyResult, *, base_execution: Execution, basis: Basis) -> dict:
    """The report block: both runs and alternative - base deltas."""

    first, second = _row(base, basis), _row(alternative, basis)
    return {
        "report_only": True,
        "basis": basis,
        "base_execution": base_execution,
        "alternative_execution": ALTERNATIVE[base_execution],
        "base": first,
        "alternative": second,
        "delta_total_pnl_twd": _delta(first["total_pnl_twd"], second["total_pnl_twd"]),
        "delta_total_return": _delta(first["total_return"], second["total_return"]),
        "delta_sharpe_daily": _delta(first["sharpe_daily"], second["sharpe_daily"]),
    }


def measure(rerun: Rerun, params: Mapping, base: DailyResult, *, base_execution: Execution, basis: Basis) -> dict:
    """Re-run ``params`` with the alternative execution (costs x1) and :func:`compare`."""

    alternative = rerun(RerunRequest(params=dict(params), execution=ALTERNATIVE[base_execution]))
    return compare(base, alternative, base_execution=base_execution, basis=basis)
