"""Null calibration of the beta-matched risk route over random windows (holdout and train lengths), and its replay.

    python -m tests.pipeline.holdout_risk_calibration nulls --lengths 252,330,520,1260 --sims 200 --jobs 6 \\
        --out /out/holdout_nulls.json
    python -m tests.pipeline.holdout_risk_calibration replay /out/runs/run17.json /out/runs/run19.json \\
        --out /out/holdout_replay.json

``nulls`` uses the nulls of ``benchmark_risk_calibration`` (TMF on TX prices, 1M TWD, next open): constant leverage,
two-state random switching with the spell lengths of the 200-day MA band filter, and that filter's own on/off path
circularly shifted. Each one runs with fresh capital over a random window of ``length`` trading days inside
[2001-01-02, 2026-10-27] (refdata at ``POSEIDON_PIPELINE_REFDATA_ROOT``), as a holdout does, and is judged by the
holdout's benchmark condition: holdout alpha > 0 (``benchmark.regress_daily`` on the window) and the risk route
(``benchmark.calmar_draws`` against ``daily_stages.beta_matched_run`` on the window) for every block length in
``--blocks`` and confidence in ``--confidences`` (``pass``: lower bound > 0 and tracking >= the frozen
``benchmark_risk_min_tracking``; ``lower`` and ``tracking`` are kept raw). ``frozen`` is ``benchmark.risk_route``
under the holdout keys of ``plan.DAILY_THRESHOLDS`` once they are registered there; ``frozen_train`` is the train
route as frozen (``benchmark.risk_route`` under the train keys, judged with alpha > 0 and HAC t >=
``benchmark_alpha_t_min``), for train lengths:

    python -m tests.pipeline.holdout_risk_calibration nulls --lengths 1260,2137,3840,5000 --sims 200 \\
        --constant-sims 100 --jobs 6 --blocks 252 --confidences 0.95 --out /out/train_nulls.json

``replay`` judges the dumped runs (``{"config", "stages": {name: {"artifact": {"report"}}}}``, one file per run) through
``DailyHoldoutStage``: with the frozen config, then with ``holdout_benchmark_rule`` (plus the benchmark rule keys of
``plan.DAILY_THRESHOLDS``) at ``--block`` and each of ``--confidences``. Nothing is written anywhere but ``--out``.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from poseidon.pipeline import optimization as opt
from poseidon.pipeline.daily_stages import (
    DailyHoldoutStage,
    SpecRunner,
    WindowStore,
    beta_matched_run,
    beta_matched_spec,
    underlying_daily,
)
from poseidon.pipeline.family import MemoryFamily
from poseidon.pipeline.gates import benchmark, holdout
from poseidon.pipeline.gates.daily import from_engine, underlying_returns
from poseidon.pipeline.plan import DAILY_THRESHOLDS, effective_config
from poseidon.pipeline.v3.compiler import compile_spec
from poseidon.pipeline.v3.config import session_warmup, trading_calendar_record, warmup_record
from poseidon.pipeline.v3.validation import FeasibilityContext
from tests.pipeline.benchmark_risk_calibration import (
    LEVELS,
    PathLeverage,
    filter_path,
    filter_spec,
    markov_path,
    spells,
)

START, LAST = date(2001, 1, 2), date(2026, 10, 27)  # the refdata of 2026-10-28; runs #17 / #19 hold out to LAST
HAC_LAGS = 5

_store: WindowStore | None = None
_underlying: dict | None = None
_warmup = None


def window_runner(first: date, last: date) -> SpecRunner:
    """The filter spec over ``[first, last]`` on one shared store: an engine run there starts with fresh capital."""

    global _store, _underlying, _warmup
    if _store is None:
        from poseidon.pipeline.refdata.store import RefdataStore

        _store = WindowStore(RefdataStore(), LAST)
        _underlying = underlying_returns(underlying_daily(_store, "TX", through=LAST)["close"])
        _warmup = session_warmup(warmup_record(), _store)
    return SpecRunner(
        template=filter_spec(),
        grid=opt.ParamGrid({}),
        store=_store,
        start=first,
        end=last,
        basis="return",
        feasibility=FeasibilityContext(warmup=_warmup),
        calendar_rule=trading_calendar_record(),
    )


def judge(job: dict) -> dict:
    """One null over its window: alpha > 0, the risk route at every block length and confidence, and ``frozen``."""

    r = window_runner(job["first"], job["last"])
    one = beta_matched_spec(r.template, "TX", 1.0)
    compiled = compile_spec(
        one, start=r.start, end=r.end, store=r.store, sessions=r.feasibility.warmup, calendar_rule=r.calendar_rule
    )
    path = dict(zip(job["days"], job["path"], strict=True))
    result = from_engine(compiled.run(PathLeverage(path)))
    stats = benchmark.regress_daily(result, _underlying, basis="return", lags=HAC_LAGS)
    beta, alpha = stats["beta"], stats["alpha_per_day"]
    row = {
        "null": job["name"],
        "seed": job["seed"],
        "length": job["length"],
        "first": job["first"].isoformat(),
        "n_days": result.n_days,
        "mean_leverage": float(np.mean(job["path"])),
        "beta": beta,
        "alpha": alpha,
        "alpha_t": stats["alpha_t_hac"],
        "alpha_pass": bool(alpha is not None and alpha > 0),
        "risk": {},
        "frozen": None,
        "frozen_train": None,
    }
    runs: dict[float, object] = {}

    def matched(leverage: float):
        if leverage not in runs:
            runs[leverage] = beta_matched_run(r, "TX", leverage)
        return runs[leverage]

    if holdout.RULE_KEY in DAILY_THRESHOLDS:
        risk = benchmark.risk_route(result, "return", beta, matched, DAILY_THRESHOLDS, holdout.RISK_KEYS)
        row["frozen"] = {
            "status": risk["status"],
            "reason": risk["reason"],
            "calmar_diff_lower": risk.get("calmar_diff_lower"),
        }
    train = benchmark.risk_route(result, "return", beta, matched, DAILY_THRESHOLDS)
    row["frozen_train"] = {k: train.get(k) for k in ("status", "reason", "calmar_diff_lower", "tracking")}
    if not beta >= DAILY_THRESHOLDS["benchmark_risk_min_beta"]:
        row["not_applicable"] = f"beta {beta:.2f}"
        return row
    run = matched(round(beta, 2))
    if isinstance(run, benchmark.NotApplicable):
        row["not_applicable"] = run.reason
        return row
    s, m = result.returns, run.returns
    tracking = float(np.std(s - m) / np.std(m))
    applies = tracking >= DAILY_THRESHOLDS["benchmark_risk_min_tracking"]
    row.update(
        calmar=float(benchmark.calmar_rows(s)[2]), matched_calmar=float(benchmark.calmar_rows(m)[2]), tracking=tracking
    )
    for block in job["blocks"]:
        if len(s) < 2 * block:
            continue
        draws = benchmark.calmar_draws(s, m, block_days=block, resamples=job["resamples"])
        lower = {str(c): benchmark.lower_bound(draws, c) for c in job["confidences"]}
        row["risk"][str(block)] = {"lower": lower, "pass": {c: applies and bound > 0 for c, bound in lower.items()}}
    return row


def window_job(rng: np.random.Generator, days: list[date], length: int, path_of, **fields) -> dict:
    """A null over a random window of ``length`` days; ``path_of(first, length)``: its leverage per window day."""

    first = int(rng.integers(0, len(days) - length + 1))
    span = days[first : first + length]
    path = [float(v) for v in path_of(first, length)]
    return {**fields, "length": length, "first": span[0], "last": span[-1], "days": span, "path": path}


def null_jobs(args) -> tuple[list[dict], dict]:
    days, on = filter_path(window_runner(START, LAST))
    stats = spells(on)
    common = {"blocks": args.blocks, "confidences": args.confidences, "resamples": args.resamples}
    jobs = []
    for length in args.lengths:
        rng = np.random.default_rng(length)
        for leverage in (0.5, 1.0, 2.0, 3.0):
            for seed in range(1, args.constant_sims + 1):
                path_of = lambda _first, n, lev=leverage: [lev] * n  # noqa: E731
                jobs.append(window_job(rng, days, length, path_of, name=f"constant_{leverage:g}x", seed=seed, **common))
        for name, (high, low) in LEVELS.items():
            for seed in range(1, args.sims + 1):
                path_of = lambda _first, n, h=high, lo=low, s=seed: lo + (h - lo) * markov_path(stats, n, s)  # noqa: E731
                jobs.append(window_job(rng, days, length, path_of, name=name, seed=seed, **common))
        for seed in range(1, args.sims + 1):
            shifted = 0.5 + 1.5 * np.roll(on, int(rng.integers(252, len(days) - 252)))
            path_of = lambda first, n, p=shifted: p[first : first + n]  # noqa: E731
            jobs.append(window_job(rng, days, length, path_of, name="shift_2x_0.5x", seed=seed, **common))
    return jobs, stats


def rates(rows: list[dict], block: int, c: float) -> dict:
    risk = [bool(row["risk"].get(str(block), {}).get("pass", {}).get(str(c), False)) for row in rows]
    alpha = [row["alpha_pass"] for row in rows]
    either = [a or b for a, b in zip(alpha, risk, strict=True)]
    added = [b and not a for a, b in zip(alpha, risk, strict=True)]
    applicable = [str(block) in row["risk"] for row in rows]
    n = len(rows)
    return {
        "n": n,
        "applicable": sum(applicable),
        "risk": sum(risk) / n,
        "alpha": sum(alpha) / n,
        "either": sum(either) / n,
        "risk_not_alpha": sum(added) / n,
    }


def table(rows: list[dict], args) -> list[str]:
    lines = []
    names = sorted({row["null"] for row in rows})
    switching = [name for name in names if not name.startswith("constant")]
    for length in args.lengths:
        at = [row for row in rows if row["length"] == length]
        for block in args.blocks:
            for c in args.confidences:
                lines.append(f"length {length}, block {block}, confidence {c}:")
                groups = [(name, [row for row in at if row["null"] == name]) for name in names]
                groups.append(("switching (pooled)", [row for row in at if row["null"] in switching]))
                for name, group in filter(lambda g: g[1], groups):  # --sims 0: constant nulls only
                    q = rates(group, block, c)
                    lines.append(
                        f"  {name:<20} n={q['n']:>4} applicable {q['applicable']:>4}  risk {q['risk']:6.1%}  "
                        f"alpha>0 {q['alpha']:6.1%}  either {q['either']:6.1%}  risk&!alpha {q['risk_not_alpha']:6.1%}"
                    )
    return lines


def train_table(rows: list[dict], lengths: list[int]) -> list[str]:
    """``frozen_train`` per length and null: the train route as frozen, next to the train alpha route."""

    t_min = DAILY_THRESHOLDS["benchmark_alpha_t_min"]
    lines = [f"train route as frozen (alpha: alpha > 0 and HAC t >= {t_min:g}):"]
    names = sorted({row["null"] for row in rows})
    switching = [name for name in names if not name.startswith("constant")]
    for length in lengths:
        at = [row for row in rows if row["length"] == length]
        groups = [(name, [row for row in at if row["null"] == name]) for name in names]
        groups.append(("switching (pooled)", [row for row in at if row["null"] in switching]))
        for name, group in filter(lambda g: g[1], groups):
            alpha = [bool(row["alpha_pass"] and (row["alpha_t"] or 0.0) >= t_min) for row in group]
            status = [row["frozen_train"]["status"] for row in group]
            risk = [s == "pass" for s in status]
            n = len(group)
            lines.append(
                f"  {length:>5} {name:<20} n={n:>4} n/a {status.count('not_applicable') / n:6.1%}  "
                f"risk {sum(risk) / n:6.1%}  alpha {sum(alpha) / n:6.1%}  "
                f"either {sum(a or b for a, b in zip(alpha, risk, strict=True)) / n:6.1%}  "
                f"risk&!alpha {sum(b and not a for a, b in zip(alpha, risk, strict=True)) / n:6.1%}"
            )
    return lines


def nulls(args) -> int:
    jobs, stats = null_jobs(args)
    print("filter path:", json.dumps(stats), "jobs:", len(jobs), flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(judge, jobs, chunksize=4))
    lines = table(rows, args) + train_table(rows, args.lengths)
    print("\n".join(lines))
    frozen = [row for row in rows if row["frozen"] is not None]
    if frozen:
        for length in args.lengths:
            at = [row for row in frozen if row["length"] == length]
            statuses = {
                s: sum(1 for row in at if row["frozen"]["status"] == s) for s in ("pass", "fail", "not_applicable")
            }
            either = sum(1 for row in at if row["alpha_pass"] or row["frozen"]["status"] == "pass")
            print(f"frozen length {length}: n={len(at)} {statuses} either {either / len(at):.1%}")
    out = {"filter": stats, "args": _args(args), "rows": rows, "table": lines}
    args.out.write_text(json.dumps(out, indent=1, default=str))
    return 0


def replay(args) -> int:
    out = []
    for path in args.runs:
        dump = json.loads(Path(path).read_text())
        stages = {name: SimpleNamespace(report=s["artifact"]["report"]) for name, s in dump["stages"].items()}
        route = stages["spec_generate"].report.get("route") if "spec_generate" in stages else None
        frozen = copy.deepcopy(effective_config(dump["config"], route))
        variants = [("frozen", frozen)]
        for confidence in args.confidences:
            added = copy.deepcopy(frozen)
            added["thresholds"].update({k: DAILY_THRESHOLDS[k] for k in (benchmark.RULE_KEY, *benchmark.RISK_KEYS)})
            added["thresholds"].update(
                {
                    holdout.RULE_KEY: benchmark.RULE_ALPHA_OR_RISK,
                    holdout.BLOCK_KEY: args.block,
                    holdout.CONFIDENCE_KEY: confidence,
                }
            )
            variants.append((f"holdout_rule block {args.block} confidence {confidence}", added))
        for label, config in variants:
            ctx = SimpleNamespace(
                config=config,
                upstream=stages,
                family=MemoryFamily(run_id=dump["run_no"]),
                trials=None,
                run_id=dump["run_no"],
                attempt=1,
                stage_run_id=None,
            )
            result = DailyHoldoutStage().evaluate(ctx)
            report = result.report
            gate = report["gate_report"]
            out.append(
                {
                    "run_no": dump["run_no"],
                    "config": label,
                    "verdict": result.verdict.value,
                    "reason": result.reason,
                    "summary": report["summary"],
                    "gate_report": gate,
                }
            )
            print(dump["run_no"], label, result.verdict.value, report["summary"])
            print("   ", result.reason)
            print("   ", json.dumps(gate.get("risk_route"), default=str)[:1500])
    args.out.write_text(json.dumps(out, indent=1, default=str))
    return 0


def _args(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True)
    n = sub.add_parser("nulls")
    n.add_argument("--lengths", type=_ints, default=[252, 330, 520, 1260])
    n.add_argument("--sims", type=int, default=200)
    n.add_argument("--constant-sims", type=int, default=25)
    n.add_argument("--jobs", type=int, default=4)
    n.add_argument("--blocks", type=_ints, default=[21, 42, 63, 125])
    n.add_argument("--confidences", type=lambda t: [float(x) for x in t.split(",")], default=[0.95, 0.975])
    n.add_argument("--resamples", type=int, default=2000)
    n.add_argument("--out", type=Path, required=True)
    n.set_defaults(func=nulls)
    r = sub.add_parser("replay")
    r.add_argument("runs", nargs="+")
    r.add_argument("--block", type=int, required=True)
    r.add_argument("--confidences", type=lambda t: [float(x) for x in t.split(",")], required=True)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=replay)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
