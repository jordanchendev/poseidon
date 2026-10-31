"""The validation gates (Step 3; ``benchmark`` from CONFIG_VERSION 5) as pure verdict functions.

Each module exposes ``evaluate(inputs, thresholds) -> GateResult``; the stage
wiring that runs the backtests and builds the inputs lives elsewhere.

Strategy Pipeline v2 (design 2026-10-24 §7): ``GATES_BY_CLASS`` picks the verdict
functions by the run's frozen ``strategy_class``. ``intraday_fixed`` (v1 / DSL runs and
every run frozen at CONFIG_VERSION <= 5) keeps ``GATES``; the daily classes judge
``gates.daily.DailyResult`` inputs and add the ``liquidation`` and ``execution_timing`` gates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from poseidon.pipeline.gates import (
    benchmark,
    cost_stress,
    dsr,
    execution_timing,
    holdout,
    liquidation,
    param_stability,
    regime_split,
    wfe,
)
from poseidon.pipeline.gates.base import GateResult

GATES: dict[str, Callable[..., GateResult]] = {
    wfe.GATE: wfe.evaluate,
    holdout.GATE: holdout.evaluate,
    cost_stress.GATE: cost_stress.evaluate,
    dsr.GATE: dsr.evaluate,
    param_stability.GATE: param_stability.evaluate,
    regime_split.GATE: regime_split.evaluate,
    benchmark.GATE: benchmark.evaluate,
}

# Inputs: wfe -- WfeInputs of wfe.window_metrics_daily; holdout -- HoldoutDailyInputs;
# cost_stress -- CostStressDailyInputs (or cost_stress.run_daily with the re-run hook);
# dsr -- dsr.inputs_daily; param_stability -- ParamStabilityInputs of
# param_stability.neighbours_daily; regime_split -- RegimeDailyInputs; benchmark --
# BenchmarkDailyInputs; liquidation -- the train DailyResult; execution_timing --
# ExecutionTimingInputs (or execution_timing.run_daily with the re-run hook).
DAILY_GATES: dict[str, Callable[..., GateResult]] = {
    wfe.GATE: wfe.evaluate,
    holdout.GATE: holdout.evaluate_daily,
    cost_stress.GATE: cost_stress.evaluate_daily,
    dsr.GATE: dsr.evaluate,
    param_stability.GATE: param_stability.evaluate,
    regime_split.GATE: regime_split.evaluate_daily,
    benchmark.GATE: benchmark.evaluate_daily,
    liquidation.GATE: liquidation.evaluate,
    execution_timing.GATE: execution_timing.evaluate_daily,
}

GATES_BY_CLASS: dict[str, dict[str, Callable[..., GateResult]]] = {
    "intraday_fixed": GATES,
    "daily_equity": DAILY_GATES,
    "daily_fixed": DAILY_GATES,
}


def summary_line(result: GateResult) -> str:
    """``"holdout ❌ Sharpe decay 62%"`` -- the per-gate Telegram one-liner.

    A holdout passed on its standard-error criterion reads e.g.
    ``"holdout ✅ decay 76% but within 1.96·SE (Δ 0.102 ≤ 0.105)"``.
    """

    mark = "✅" if result.passed else "❌"
    return f"{result.gate} {mark} {result.headline}".rstrip()


def summarize(results: Iterable[GateResult]) -> list[str]:
    return [summary_line(result) for result in results]


__all__ = ["DAILY_GATES", "GATES", "GATES_BY_CLASS", "GateResult", "summarize", "summary_line"]
