"""Gate 5 -- parameter stability around the optimizer's best point.

Neighbours are one-at-a-time +/-1 grid steps of each searchable parameter
(``one_step_neighbours`` builds them from an explicit ordered grid). A
neighbour outside the grid is excluded (``total_pnl_twd is None``); an
evaluated neighbour with a non-finite pnl is an input error, never silently
dropped. The share is taken over the evaluated neighbours only -- the best
point is reported but not counted, so it cannot pad its own neighbourhood.
Pass iff evaluated >= ``param_min_neighbours`` and the share with
``total_pnl_twd`` > 0 is >= ``param_neighbourhood_profitable_min``.

Edge at a component bound (``param_edge_at_bound_pass``, runs frozen with it only): a neighbour is
excluded as ``component_bound`` when the component's own hard range, not the spec's declared one, cuts
it off. When such exclusions are the only reason fewer than ``param_min_neighbours`` were evaluated
(evaluated + excluded-by-bound >= min, at least one evaluated), the gate judges the in-range neighbours
alone and passes only if all of them are profitable. ``edge_at_bound`` lists those params in the report.
Any other shortfall stays ``insufficient_neighbours``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from poseidon.pipeline.gates.base import GateResult, failed, finite_or_none, invalid, is_finite, passed, pct
from poseidon.pipeline.gates.daily import DailyResult

GATE = "param_stability"
EDGE_AT_BOUND_KEY = "param_edge_at_bound_pass"
COMPONENT_BOUND = "component_bound"


@dataclass(frozen=True)
class NeighbourResult:
    param: str
    step: int  # -1 or +1 grid step from the best value
    params: Mapping[str, Any]
    total_pnl_twd: float | None  # None: out of bounds / not evaluated
    excluded_kind: str | None = None  # COMPONENT_BOUND when a component's hard bound cut it off


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
    at_bound = (
        sorted({n.param for n in inputs.neighbours if n.excluded_kind == COMPONENT_BOUND and n.total_pnl_twd is None})
        if thresholds.get(EDGE_AT_BOUND_KEY)
        else []
    )
    if at_bound:
        report["edge_at_bound"] = at_bound
        bound_excluded = sum(1 for n in inputs.neighbours if n.excluded_kind == COMPONENT_BOUND)
        if evaluated and len(evaluated) < min_neighbours <= len(evaluated) + bound_excluded:
            edge = f"{len(profitable)}/{len(evaluated)} in-range neighbours profitable, edge_at_bound: {', '.join(at_bound)}"
            if len(profitable) < len(evaluated):
                return failed(
                    GATE,
                    "unstable_neighbourhood",
                    f"{len(evaluated) - len(profitable)} of {len(evaluated)} in-range neighbours lose money "
                    f"(edge_at_bound: {', '.join(at_bound)})",
                    edge,
                    report,
                )
            return passed(
                GATE,
                edge,
                report,
                reason=f"all in-range neighbours profitable; {', '.join(at_bound)} at a component bound",
            )
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


def neighbours_daily(
    rows: Sequence[Mapping[str, Any]], run: Callable[[Mapping[str, Any]], DailyResult | None]
) -> tuple[list[NeighbourResult], list[dict]]:
    """Daily classes (§7): neighbours judged on final equity > initial (= total PnL > 0).

    ``rows`` are ``optimization.ParamGrid.neighbours(best, steps)`` over the spec v3 ``$param``
    grid (``param``, ``direction``, ``in_grid``, ``params``); ``run(params)`` returns the train
    backtest of a neighbour, or None when it failed -- counted as not profitable (pnl 0), never
    excluded, as in the intraday stage. Returns the neighbours and the failed ones.
    """

    neighbours, failed_counted = [], []
    for row in rows:
        if not row.get("in_grid"):
            neighbours.append(NeighbourResult(row["param"], int(row["direction"]), {}, None))
            continue
        result = run(row["params"])
        if result is None:
            failed_counted.append({"param": row["param"], "direction": int(row["direction"])})
            pnl = 0.0
        else:
            pnl = result.total_pnl_twd
        neighbours.append(NeighbourResult(row["param"], int(row["direction"]), dict(row["params"]), pnl))
    return neighbours, failed_counted
