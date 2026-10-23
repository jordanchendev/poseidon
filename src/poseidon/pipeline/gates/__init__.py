"""The validation gates (Step 3; ``benchmark`` from CONFIG_VERSION 5) as pure verdict functions.

Each module exposes ``evaluate(inputs, thresholds) -> GateResult``; the stage
wiring that runs the backtests and builds the inputs lives elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from poseidon.pipeline.gates import benchmark, cost_stress, dsr, holdout, param_stability, regime_split, wfe
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


def summary_line(result: GateResult) -> str:
    """``"holdout ❌ Sharpe decay 62%"`` -- the per-gate Telegram one-liner.

    A holdout passed on its standard-error criterion reads e.g.
    ``"holdout ✅ decay 76% but within 1.96·SE (Δ 0.102 ≤ 0.105)"``.
    """

    mark = "✅" if result.passed else "❌"
    return f"{result.gate} {mark} {result.headline}".rstrip()


def summarize(results: Iterable[GateResult]) -> list[str]:
    return [summary_line(result) for result in results]


__all__ = ["GATES", "GateResult", "summarize", "summary_line"]
