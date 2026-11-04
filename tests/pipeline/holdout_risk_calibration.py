"""Null calibration of the beta-matched risk route over random windows (holdout and train lengths), and its replay.

    python -m tests.pipeline.holdout_risk_calibration nulls --lengths 252,330,520,1260 --sims 200 --jobs 6 \\
        --out /out/holdout_nulls.json
    python -m tests.pipeline.holdout_risk_calibration replay /out/runs/run17.json /out/runs/run19.json \\
        --out /out/holdout_replay.json

``nulls`` uses the nulls of ``benchmark_risk_calibration`` (``--instrument`` on TX prices, default TMF, with
``--capital`` TWD, default 1M, next open): constant leverage, two-state random switching with the spell lengths of the
200-day MA band filter, and that filter's own on/off path circularly shifted; ``--filter-sims`` adds the filter's
own path unshifted (``filter_2x_0.5x``, the route's power on the timing it was built for). Each one runs with fresh capital over a random window of ``length`` trading days inside
[2001-01-02, 2026-10-27] (refdata at ``POSEIDON_PIPELINE_REFDATA_ROOT``), as a holdout does, and is judged by the
holdout's benchmark condition: holdout alpha > 0 (``benchmark.regress_daily`` on the window) and the risk route
(``benchmark.calmar_draws`` against ``daily_gates.beta_matched_run`` on the window) for every block length in
``--blocks`` and confidence in ``--confidences`` (``pass``: lower bound > 0 and tracking >= the frozen
``benchmark_risk_min_tracking``; ``lower`` and ``tracking`` are kept raw). ``frozen`` is ``benchmark.risk_route``
under the holdout keys (``HOLDOUT_THRESHOLDS``); ``frozen_train`` is the train route as frozen
(``benchmark.risk_route`` under the train keys, judged with alpha > 0 and HAC t >= ``benchmark_alpha_t_min``), for
train lengths. ``--set key=value`` overrides a threshold for both. ``feasible`` is ``spec_validate``'s sizing and
margin verdict on the null's largest leverage over the window (a null it rejects never reaches the gate):

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
from poseidon.pipeline.daily_gates import DailyHoldoutStage, beta_matched_run, beta_matched_spec, underlying_daily
from poseidon.pipeline.daily_stages import SpecRunner, WindowStore
from poseidon.pipeline.daily_validate import price_points
from poseidon.pipeline.family import MemoryFamily
from poseidon.pipeline.gates import benchmark, holdout
from poseidon.pipeline.gates.daily import from_engine, underlying_returns
from poseidon.pipeline.plan import DAILY_THRESHOLDS, effective_config
from poseidon.pipeline.v3.compiler import compile_spec
from poseidon.pipeline.v3.components_sizing import round_units
from poseidon.pipeline.v3.config import session_warmup, trading_calendar_record, warmup_record
from poseidon.pipeline.v3.instruments import INSTRUMENTS
from poseidon.pipeline.v3.validation import FeasibilityContext, validate_spec_v3
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
# The holdout route as the owner set it on 2026-10-29 (21-day blocks, 99 %) until plan.DAILY_THRESHOLDS carries it.
HOLDOUT_THRESHOLDS = {holdout.BLOCK_KEY: 21, holdout.CONFIDENCE_KEY: 0.99, **DAILY_THRESHOLDS}

_store: WindowStore | None = None
_underlying: dict | None = None
_warmup = None


def window_runner(first: date, last: date, symbol: str = "TMF", capital: int = 1_000_000) -> SpecRunner:
    """The filter spec on ``symbol`` and ``capital`` over ``[first, last]`` on one shared store: an engine run there
    starts with fresh capital."""

    global _store, _underlying, _warmup
    if _store is None:
        from poseidon.pipeline.refdata.store import RefdataStore

        _store = WindowStore(RefdataStore(), LAST)
        _underlying = underlying_returns(underlying_daily(_store, "TX", through=LAST)["close"])
        _warmup = session_warmup(warmup_record(), _store)
    return SpecRunner(
        template=filter_spec(symbol, capital),
        grid=opt.ParamGrid({}),
        store=_store,
        start=first,
        end=last,
        basis="return",
        feasibility=FeasibilityContext(warmup=_warmup),
        calendar_rule=trading_calendar_record(),
    )


def feasible(r: SpecRunner, points: tuple[float, float], high: float, low: float) -> list[str]:
    """``spec_validate``'s errors on ``r``'s filter spec at ``high`` x, ``low`` x below the band (``ma_days`` the one
    ``$param`` a spec needs), sized at the window's ``daily_validate.price_points``: a null it rejects never reaches
    the gate."""

    spec = copy.deepcopy(r.template)
    base, band = spec["sleeves"][0]["leverage"]
    base["params"]["value"] = high
    band["params"].update(ma_days={"$param": "ma_days"}, off_leverage=low)
    spec["params"] = {"ma_days": {"type": "int", "low": 150, "high": 250, "step": 25, "default": 200}}
    context = FeasibilityContext(start=r.start, end=r.end, prices={"TX": points}, warmup=_warmup)
    checked = validate_spec_v3(spec, feasibility=context)
    return [issue.text() for issue in (*checked.errors, *checked.unsupported)]


def judge(job: dict) -> dict:
    """One null over its window: alpha > 0, the risk route at every block length and confidence, and ``frozen``."""

    symbol, capital, thresholds = job["symbol"], job["capital"], job["thresholds"]
    r = window_runner(job["first"], job["last"], symbol, capital)
    one = beta_matched_spec(r.template, "TX", 1.0)
    compiled = compile_spec(
        one, start=r.start, end=r.end, store=r.store, sessions=r.feasibility.warmup, calendar_rule=r.calendar_rule
    )
    path = dict(zip(job["days"], job["path"], strict=True))
    result = from_engine(compiled.run(PathLeverage(path, symbol=symbol)))
    stats = benchmark.regress_daily(result, _underlying, basis="return", lags=HAC_LAGS)
    beta, alpha = stats["beta"], stats["alpha_per_day"]
    instrument = INSTRUMENTS[symbol]
    points = price_points(r.store, instrument, r.start, r.end)
    rejected = feasible(r, points, max(job["path"]), min(job["path"]))
    row = {
        "null": job["name"],
        "seed": job["seed"],
        "length": job["length"],
        "symbol": symbol,
        "capital": capital,
        "first": job["first"].isoformat(),
        "n_days": result.n_days,
        "mean_leverage": float(np.mean(job["path"])),
        "prices": points,  # first close, max close
        "feasible": not rejected,
        "rejected": rejected[:1],
        "flat_share": float(np.mean(result.returns == 0.0)),
        "beta": beta,
        "alpha": alpha,
        "alpha_t": stats["alpha_t_hac"],
        "alpha_pass": bool(alpha is not None and alpha > 0),
        "risk": {},
        "frozen": None,
        "frozen_train": None,
    }
    runs: dict[float, object] = {}
    min_contracts = {**DAILY_THRESHOLDS, **thresholds}.get(benchmark.MIN_CONTRACTS_KEY)

    def matched(leverage: float):
        if leverage not in runs:
            runs[leverage] = beta_matched_run(r, "TX", leverage, min_contracts)
        return runs[leverage]

    fields = ("status", "reason", "calmar_diff_lower", "tracking", "matched_leverage")
    risk = benchmark.risk_route(
        result, "return", beta, matched, {**HOLDOUT_THRESHOLDS, **thresholds}, holdout.RISK_KEYS
    )
    row["frozen"] = {k: risk.get(k) for k in fields}
    train = benchmark.risk_route(result, "return", beta, matched, {**DAILY_THRESHOLDS, **thresholds})
    row["frozen_train"] = {k: train.get(k) for k in fields}
    if not beta >= DAILY_THRESHOLDS["benchmark_risk_min_beta"]:
        row["not_applicable"] = f"beta {beta:.2f}"
        return row
    run = matched(round(beta, 2))
    if isinstance(run, benchmark.NotApplicable):
        row["not_applicable"] = run.reason
        return row
    s, m = result.returns, run.returns
    with np.errstate(divide="ignore", invalid="ignore"):
        tracking = float(np.std(s - m) / np.std(m))
    applies = tracking >= ({**DAILY_THRESHOLDS, **thresholds}.get("benchmark_risk_min_tracking") or 0.0)
    row.update(
        calmar=float(benchmark.calmar_rows(s)[2]),
        matched_calmar=float(benchmark.calmar_rows(m)[2]),
        tracking=tracking,
        matched_flat_share=float(np.mean(m == 0.0)),
        matched_units=[round_units(capital * round(beta, 2) / instrument.notional_twd(p), "nearest") for p in points],
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
    common = {
        "blocks": args.blocks,
        "confidences": args.confidences,
        "resamples": args.resamples,
        "symbol": args.instrument,
        "capital": args.capital,
        "thresholds": dict(args.set),
    }
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
        for seed in range(1, args.filter_sims + 1):
            path_of = lambda first, n, p=0.5 + 1.5 * on: p[first : first + n]  # noqa: E731
            jobs.append(window_job(rng, days, length, path_of, name="filter_2x_0.5x", seed=seed, **common))
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
    switching = [name for name in names if not name.startswith(("constant", "filter"))]
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


def route_table(rows: list[dict], lengths: list[int], route: str, alpha_of, title: str) -> list[str]:
    """The route ``row[route]`` as frozen per length and null, next to the alpha condition ``alpha_of(row)``;
    ``feasible``: the share of nulls ``spec_validate`` accepts, ``risk|feasible`` the route's pass rate among them."""

    lines = [title]
    names = sorted({row["null"] for row in rows})
    switching = [name for name in names if not name.startswith(("constant", "filter"))]
    for length in lengths:
        at = [row for row in rows if row["length"] == length]
        groups = [(name, [row for row in at if row["null"] == name]) for name in names]
        groups.append(("switching (pooled)", [row for row in at if row["null"] in switching]))
        for name, group in filter(lambda g: g[1], groups):
            alpha = [alpha_of(row) for row in group]
            status = [row[route]["status"] for row in group]
            risk = [s == "pass" for s in status]
            held = [r for r, row in zip(risk, group, strict=True) if row["feasible"]]
            n = len(group)
            lines.append(
                f"  {length:>5} {name:<20} n={n:>4} n/a {status.count('not_applicable') / n:6.1%}  "
                f"risk {sum(risk) / n:6.1%}  alpha {sum(alpha) / n:6.1%}  "
                f"either {sum(a or b for a, b in zip(alpha, risk, strict=True)) / n:6.1%}  "
                f"risk&!alpha {sum(b and not a for a, b in zip(alpha, risk, strict=True)) / n:6.1%}  "
                f"feasible {len(held) / n:6.1%}  risk|feasible {sum(held) / max(len(held), 1):6.1%}"
            )
    return lines


def nulls(args) -> int:
    jobs, stats = null_jobs(args)
    print("filter path:", json.dumps(stats), "jobs:", len(jobs), flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(judge, jobs, chunksize=4))
    t_min = DAILY_THRESHOLDS["benchmark_alpha_t_min"]
    lines = table(rows, args)
    lines += route_table(
        rows,
        args.lengths,
        "frozen_train",
        lambda row: bool(row["alpha_pass"] and (row["alpha_t"] or 0.0) >= t_min),
        f"train route as frozen (alpha: alpha > 0 and HAC t >= {t_min:g}):",
    )
    lines += route_table(
        rows,
        args.lengths,
        "frozen",
        lambda row: row["alpha_pass"],
        f"holdout route (block {HOLDOUT_THRESHOLDS[holdout.BLOCK_KEY]}, confidence "
        f"{HOLDOUT_THRESHOLDS[holdout.CONFIDENCE_KEY]}; alpha: alpha > 0):",
    )
    print("\n".join(lines))
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
            added["thresholds"].update(
                {
                    k: DAILY_THRESHOLDS[k]
                    for k in (benchmark.RULE_KEY, *benchmark.RISK_KEYS, benchmark.MIN_CONTRACTS_KEY)
                }
            )
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


def _setting(text: str) -> tuple[str, object]:
    key, value = text.split("=", 1)
    return key, json.loads(value)


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
    n.add_argument("--instrument", choices=("TMF", "MTX", "TX"), default="TMF")
    n.add_argument("--capital", type=int, default=1_000_000)
    n.add_argument("--filter-sims", type=int, default=0)
    n.add_argument("--set", type=_setting, action="append", default=[], help="threshold override key=json")
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
