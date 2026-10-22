"""Gate 5 -- parameter stability around the optimizer's best point.

Neighbours are one-at-a-time +/-1 grid steps of each searchable parameter
(``one_step_neighbours`` builds them from an explicit ordered grid). A
neighbour outside the grid is excluded (``total_pnl_twd is None``); an
evaluated neighbour with a non-finite pnl is an input error, never silently
dropped. The share is taken over the evaluated neighbours only -- the best
point is reported but not counted, so it cannot pad its own neighbourhood.
Pass iff evaluated >= ``param_min_neighbours`` and the share with
``total_pnl_twd`` > 0 is >= ``param_neighbourhood_profitable_min``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from poseidon.pipeline.gates.base import GateResult, failed, finite_or_none, invalid, is_finite, passed, pct

GATE = "param_stability"


@dataclass(frozen=True)
class NeighbourResult:
    param: str
    step: int  # -1 or +1 grid step from the best value
    params: Mapping[str, Any]
    total_pnl_twd: float | None  # None: out of bounds / not evaluated


@dataclass(frozen=True)
class ParamStabilityInputs:
    best_params: Mapping[str, Any]
    best_total_pnl_twd: float
    neighbours: Sequence[NeighbourResult]


def one_step_neighbours(
    best_params: Mapping[str, Any], grid: Mapping[str, Sequence[Any]]
) -> list[tuple[str, int, dict | None]]:
    """``(param, step, params)`` for each searchable param and step in (-1, +1).

    ``params`` is ``None`` when the step leaves the grid (out of bounds).
    ``grid`` maps each searchable parameter to its ordered candidate values.
    """

    out: list[tuple[str, int, dict | None]] = []
    for name in sorted(grid):
        values = list(grid[name])
        if best_params[name] not in values:
            raise ValueError(f"best value {best_params[name]!r} of {name} is not on its grid")
        position = values.index(best_params[name])
        for step in (-1, 1):
            target = position + step
            if 0 <= target < len(values):
                out.append((name, step, {**best_params, name: values[target]}))
            else:
                out.append((name, step, None))
    return out


def evaluate(inputs: ParamStabilityInputs, thresholds: Mapping) -> GateResult:
    share_min = float(thresholds["param_neighbourhood_profitable_min"])
    min_neighbours = int(thresholds["param_min_neighbours"])
    bad = [
        f"{n.param}{n.step:+d}"
        for n in inputs.neighbours
        if n.total_pnl_twd is not None and not is_finite(n.total_pnl_twd)
    ]
    if not is_finite(inputs.best_total_pnl_twd):
        bad.insert(0, "best")
    if bad:
        return invalid(GATE, "non-finite neighbour pnl: " + ", ".join(bad))

    rows = [
        {
            "param": n.param,
            "step": n.step,
            "params": dict(n.params),
            "total_pnl_twd": finite_or_none(n.total_pnl_twd),
            "evaluated": n.total_pnl_twd is not None,
            "profitable": n.total_pnl_twd is not None and n.total_pnl_twd > 0,
        }
        for n in inputs.neighbours
    ]
    evaluated = [row for row in rows if row["evaluated"]]
    profitable = [row for row in evaluated if row["profitable"]]
    share = len(profitable) / len(evaluated) if evaluated else None
    report = {
        "best_params": dict(inputs.best_params),
        "best_total_pnl_twd": float(inputs.best_total_pnl_twd),
        "neighbours": rows,
        "evaluated": len(evaluated),
        "excluded": len(rows) - len(evaluated),
        "profitable": len(profitable),
        "profitable_share": share,
        "profitable_share_min": share_min,
        "min_neighbours": min_neighbours,
    }
    if len(evaluated) < min_neighbours:
        return failed(
            GATE,
            "insufficient_neighbours",
            f"{len(evaluated)} evaluated neighbours < {min_neighbours}",
            f"{len(evaluated)} neighbours",
            report,
        )
    headline = f"{len(profitable)}/{len(evaluated)} neighbours profitable ({pct(share)})"
    if share < share_min:
        return failed(
            GATE, "unstable_neighbourhood", f"profitable share {share:.0%} < {share_min:.0%}", headline, report
        )
    return passed(GATE, headline, report)
