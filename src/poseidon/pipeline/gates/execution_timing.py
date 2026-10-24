"""Daily classes -- execution-timing gate, blocking (owner decision 2026-10-24 afternoon #2).

The selected params re-run over the same train window under each timing scenario:

* ``signal_close`` -- fill at the decision close (left out when it is the spec's own execution);
* ``next_open``    -- fill at the next session's open (left out when it is the spec's own execution);
* ``signal_lag_1d`` -- the spec's own execution, every signal series one row late (the decision at
  close t reads signals up to t-1).

Pass iff every scenario has total PnL > 0 (``execution_timing_pnl_positive``) and daily Sharpe >=
``execution_timing_sharpe_ratio_min`` x the base run's daily Sharpe. A run frozen without a key has
that criterion off; without both, the numbers are reported and the gate passes ("report only").
Pre-registered before any real strategy result. Why: before the 2017-05 after-hours session,
``next_open`` meant the next 08:45 open and the acceptance strategy's switch trades (~2.5x equity
each) carried whole overnight gaps, moving final equity by 15-35 %; an edge that lives only in one
fill convention or one day of signal freshness is a timing artefact, not a strategy.

API (the stage wiring calls these; nothing here runs an engine):

``scenarios_for(params, *, base_execution) -> dict[str, RerunRequest]``
    The re-runs to make, keyed by scenario name. ``base_execution`` is the spec's own fill
    (``compiler.Plan.execution``: ``"next_open"`` for every daily spec today).
``evaluate_daily(inputs: ExecutionTimingInputs, thresholds) -> GateResult``
    The verdict from the base run and one ``DailyResult`` per scenario name. Refuses
    (``invalid_input``) a scenario set that is not exactly ``scenarios_for``'s.
``run_daily(rerun, params, base, *, base_execution, basis, thresholds) -> GateResult``
    Both steps through the stage's :data:`~poseidon.pipeline.gates.daily.Rerun` hook.

A spec v3 hook maps a request onto ``v3.runtime.Conventions(execution=request.execution,
signal_lag=request.signal_lag_extra)`` (the compiler adds the lag to the warm-up).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from poseidon.pipeline.gates.base import GateResult, failed, finite_or_none, invalid, is_finite, passed
from poseidon.pipeline.gates.daily import Basis, DailyResult, Execution, Rerun, RerunRequest, sharpe

GATE = "execution_timing"
EXECUTIONS: tuple[Execution, ...] = ("signal_close", "next_open")
LAG_SCENARIO = "signal_lag_1d"
SHARPE_RATIO_KEY = "execution_timing_sharpe_ratio_min"
PNL_POSITIVE_KEY = "execution_timing_pnl_positive"


def scenarios_for(params: Mapping, *, base_execution: Execution) -> dict[str, RerunRequest]:
    if base_execution not in EXECUTIONS:
        raise ValueError(f"unknown execution {base_execution!r}")
    out = {
        execution: RerunRequest(params=dict(params), execution=execution)
        for execution in EXECUTIONS
        if execution != base_execution
    }
    out[LAG_SCENARIO] = RerunRequest(params=dict(params), signal_lag_extra=1)
    return out


@dataclass(frozen=True)
class ExecutionTimingInputs:
    base: DailyResult
    scenarios: Mapping[str, DailyResult]  # scenario name (scenarios_for) -> re-run
    basis: Basis
    base_execution: Execution


def _row(result: DailyResult, basis: Basis) -> dict:
    return {
        "total_pnl_twd": finite_or_none(result.total_pnl_twd),
        "total_return": finite_or_none(result.total_return),
        "sharpe_daily": finite_or_none(sharpe(result, basis)),
        "days": result.n_days,
        "exposure_changes": result.exposure_changes,
    }


def evaluate_daily(inputs: ExecutionTimingInputs, thresholds: Mapping) -> GateResult:
    ratio_min = thresholds.get(SHARPE_RATIO_KEY)
    pnl_positive = bool(thresholds.get(PNL_POSITIVE_KEY, False))
    if ratio_min is not None and not is_finite(ratio_min):
        return invalid(GATE, f"{SHARPE_RATIO_KEY} is not a finite number ({ratio_min!r})")
    expected = set(scenarios_for({}, base_execution=inputs.base_execution))
    if set(inputs.scenarios) != expected:
        return invalid(GATE, f"scenarios {sorted(inputs.scenarios)} are not the registered set {sorted(expected)}")
    base = _row(inputs.base, inputs.basis)
    base_sharpe = base["sharpe_daily"]
    report: dict = {
        "basis": inputs.basis,
        "base_execution": inputs.base_execution,
        "sharpe_ratio_min": float(ratio_min) if ratio_min is not None else None,
        "pnl_positive_required": pnl_positive,
        "criterion_enabled": ratio_min is not None or pnl_positive,
        "base": base,
        "scenarios": {},
    }
    if ratio_min is not None and base_sharpe is None:
        return invalid(GATE, "the base run's daily Sharpe is undefined (no variance or non-finite PnL)", report)
    sharpe_floor = float(ratio_min) * base_sharpe if ratio_min is not None else None
    report["sharpe_floor"] = sharpe_floor
    failures: list[tuple[str, str]] = []  # (code, text)
    for name in sorted(inputs.scenarios):
        row = _row(inputs.scenarios[name], inputs.basis)
        row["delta_total_pnl_twd"] = _delta(base["total_pnl_twd"], row["total_pnl_twd"])
        row["delta_sharpe_daily"] = _delta(base_sharpe, row["sharpe_daily"])
        row["sharpe_vs_base"] = (
            row["sharpe_daily"] / base_sharpe if row["sharpe_daily"] is not None and base_sharpe else None
        )
        problems = []
        if pnl_positive and not (row["total_pnl_twd"] is not None and row["total_pnl_twd"] > 0):
            problems.append(("scenario_not_profitable", f"{name} PnL {_money(row['total_pnl_twd'])} <= 0"))
        if sharpe_floor is not None and not (row["sharpe_daily"] is not None and row["sharpe_daily"] >= sharpe_floor):
            problems.append(
                (
                    "scenario_sharpe_below_ratio",
                    f"{name} Sharpe {_num(row['sharpe_daily'])} < {float(ratio_min):g} x base {base_sharpe:.4f}",
                )
            )
        row["passed"] = not problems
        report["scenarios"][name] = row
        failures += problems
    headline = ", ".join(_phrase(name, row) for name, row in report["scenarios"].items())
    if not report["criterion_enabled"]:
        return passed(GATE, f"{headline} (report only)", report, reason="execution_timing thresholds not frozen")
    if failures:
        codes = [code for code, _text in failures]
        code = "scenario_not_profitable" if "scenario_not_profitable" in codes else codes[0]
        return failed(GATE, code, "; ".join(text for _code, text in failures), headline, report)
    return passed(GATE, headline, report)


def run_daily(
    rerun: Rerun, params: Mapping, base: DailyResult, *, base_execution: Execution, basis: Basis, thresholds: Mapping
) -> GateResult:
    """Re-run ``params`` under every :func:`scenarios_for` scenario and judge them against ``base``."""

    requests = scenarios_for(params, base_execution=base_execution)
    scenarios = {name: rerun(request) for name, request in requests.items()}
    return evaluate_daily(ExecutionTimingInputs(base, scenarios, basis, base_execution), thresholds)


def _delta(a, b):
    return b - a if a is not None and b is not None else None


def _num(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _money(value) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _phrase(name: str, row: dict) -> str:
    ratio = row["sharpe_vs_base"]
    share = "n/a" if ratio is None else f"{ratio * 100:.0f}%"
    return f"{name} Sharpe {share} of base, PnL {_money(row['total_pnl_twd'])}"
