"""Seed sweep of the daily-class controls (owner decisions 2026-10-24 afternoon): every seed, no picking.

    python -m tests.pipeline.daily_gate_sweep --seeds 1-20 --out /out/daily_gate_sweep.json [--jobs 4]

Runs each control of ``test_pipeline_daily_gate_controls`` for every seed and records, per run, the
failed gates, the benchmark alpha HAC t and the execution-timing scenarios; prints one line per
control with the count of runs that failed the gate the control is about.
"""

from __future__ import annotations

import argparse
import functools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tests.pipeline import test_pipeline_daily_gate_controls as controls

CONTROLS = {
    # name -> (runner, what the control expects)
    "planted_edge": (controls.planted_edge, "pass every gate"),
    "planted_edge_010": (functools.partial(controls.planted_edge, edge=0.001), "pass every gate"),
    "one_day_edge": (controls.one_day_edge, "fail execution_timing"),
    "overnight_gap_edge": (controls.overnight_gap_edge, "fail execution_timing"),
    "static_leverage": (controls.static_leverage, "fail benchmark"),
    "random_exposure": (controls.random_exposure, "fail a gate"),
}


def _seeds(text: str) -> list[int]:
    low, _, high = text.partition("-")
    return list(range(int(low), int(high or low) + 1))


def _one(job: tuple[str, int]) -> dict:
    name, seed = job
    judged = CONTROLS[name][0](seed)
    timing = judged["results"]["execution_timing"].report
    return {
        "control": name,
        "seed": seed,
        "failed": controls.failed_gates(judged),
        "best_params": judged["best_params"],
        "benchmark_alpha_t": judged["results"]["benchmark"].report.get("alpha_t_hac"),
        "base_sharpe_daily": timing["base"]["sharpe_daily"],
        "timing": {
            scenario: {key: row[key] for key in ("total_pnl_twd", "sharpe_daily", "sharpe_vs_base", "passed")}
            for scenario, row in timing["scenarios"].items()
        },
    }


def _hit(name: str, run: dict) -> bool:
    failed = run["failed"]
    return {
        "planted_edge": not failed,
        "planted_edge_010": not failed,
        "one_day_edge": "execution_timing" in failed,
        "overnight_gap_edge": "execution_timing" in failed,
        "static_leverage": "benchmark" in failed,
        "random_exposure": bool(failed),
    }[name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-20")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--controls", default=",".join(CONTROLS))
    args = parser.parse_args(argv)
    jobs = [(name, seed) for name in args.controls.split(",") for seed in _seeds(args.seeds)]
    with ProcessPoolExecutor(args.jobs) as pool:
        runs = list(pool.map(_one, jobs))
    summary = {}
    for name in args.controls.split(","):
        mine = [run for run in runs if run["control"] == name]
        hits = [run["seed"] for run in mine if _hit(name, run)]
        summary[name] = {"expects": CONTROLS[name][1], "as_expected": len(hits), "runs": len(mine)}
        misses = [(run["seed"], run["failed"]) for run in mine if not _hit(name, run)]
        print(f"{name}: {len(hits)}/{len(mine)} {CONTROLS[name][1]}; otherwise {misses}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
