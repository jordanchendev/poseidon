"""The execution-timing gate on the M1 acceptance spec over real refdata (owner decisions 2026-10-24 afternoon).

The spec as written (TMF, both signals on ``index:taiex``, ``execution.next_open``, default ``$param``s)
is the base run; ``gates.execution_timing`` re-runs it per scenario through the compiler
(``Conventions(execution=..., signal_lag=...)``) and judges it with the registered ``daily_equity``
thresholds over each whole window. The same spec with both signals on ``core:adjusted`` is reported
alongside for comparison.

Usage (stormtrooper container, refdata mounted):
  POSEIDON_PIPELINE_REFDATA_ROOT=/refdata python scripts/pipeline_v3_execution_timing_report.py /out/timing.json
"""

from __future__ import annotations

import json
import sys

from poseidon.pipeline.gates import execution_timing
from poseidon.pipeline.gates.daily import Rerun, RerunRequest, from_engine
from poseidon.pipeline.plan import class_settings
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec, resolve
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.runtime import Conventions

WINDOWS = {"2014-11..2026-10": ("2014-11-03", None), "2000..2014": ("2000-01-04", "2014-10-31")}
SIGNALS = ("index:taiex", "core:adjusted")


def spec_rerun(template: dict, *, start: str, end: str | None, store: RefdataStore) -> Rerun:
    """A ``gates.daily.Rerun`` for a spec v3 template over ``[start, end]`` (costs x1 only)."""

    def rerun(request: RerunRequest):
        if request.cost_multiplier != 1.0:
            raise NotImplementedError("cost scaling is not needed for the timing report")
        conventions = Conventions(execution=request.execution, signal_lag=request.signal_lag_extra)
        compiled = compile_spec(
            resolve(template, request.params), start=start, end=end, store=store, conventions=conventions
        )
        return from_engine(compiled.run())

    return rerun


def main(out_path: str) -> None:
    store = RefdataStore()
    thresholds = class_settings("daily_equity")["thresholds"]
    report: dict = {}
    for signal in SIGNALS:
        template = tmf_dynamic_leverage_3x()
        for link in template["sleeves"][0]["leverage"][1:]:
            link["params"]["signal"] = signal
        for name, (start, end) in WINDOWS.items():
            rerun = spec_rerun(template, start=start, end=end, store=store)
            base = rerun(RerunRequest(params={}))
            result = execution_timing.run_daily(
                rerun, {}, base, base_execution="next_open", basis="return", thresholds=thresholds
            )
            key = f"{signal} {name}"
            report[key] = {
                "verdict": result.verdict.value,
                "code": result.reason_code,
                "reason": result.reason,
                "headline": result.headline,
                "first_day": base.dates[0].isoformat(),
                "last_day": base.dates[-1].isoformat(),
                **result.report,
            }
            print(key, result.verdict.value, result.reason_code, "|", result.headline, flush=True)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "execution_timing_report.json")
