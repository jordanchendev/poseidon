"""Null calibration of the benchmark gate's beta-matched risk route, and its replay on frozen runs.

    python -m tests.pipeline.benchmark_risk_calibration nulls --sims 200 --jobs 6 --out /out/nulls.json
    python -m tests.pipeline.benchmark_risk_calibration replay /out/runs/run19.json ... --out /out/replay.json

``nulls`` runs strategies without timing skill on real TX (refdata at ``POSEIDON_PIPELINE_REFDATA_ROOT``, TMF on TX
prices, 1M TWD, next open, run #19's train window) through the daily engine, then through the gate's own pieces:
the HAC regression (alpha route), the beta-matched run (``daily_stages.beta_matched_run``) and the bootstrap
(``benchmark.calmar_difference``) for every block length in ``--blocks``, every confidence in ``--confidences``:

* ``constant``: constant leverage 0.5 / 1 / 2 / 3, rebalanced daily or on the first trading day of a month;
* ``markov_*``: two-state random switching whose on-share and mean spell lengths are those of the 200-day MA
  band filter (``regime.ma_band`` 200 days, band 2 %, TAIEX) over the window; ``--sims`` seeds per level pair;
* ``shift_2x_0.5x``: that filter's own on/off path, circularly shifted by a random 1-19 years.

``replay`` judges the dumped runs (``{"config", "stages": {name: {"artifact": {"report"}}}}``, one file per run)
with their frozen config plus the risk-route keys of ``plan.DAILY_THRESHOLDS``, through ``DailyBenchmarkStage``.
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
import pandas as pd

from poseidon.pipeline import optimization as opt
from poseidon.pipeline.daily_stages import (
    DailyBenchmarkStage,
    SpecRunner,
    WindowStore,
    beta_matched_run,
    beta_matched_spec,
)
from poseidon.pipeline.engine import Decision
from poseidon.pipeline.gates import benchmark
from poseidon.pipeline.gates.daily import from_engine
from poseidon.pipeline.plan import DAILY_THRESHOLDS, effective_config
from poseidon.pipeline.v3.compiler import compile_spec
from poseidon.pipeline.v3.components import round_units
from poseidon.pipeline.v3.config import session_warmup, trading_calendar_record, warmup_record
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.instruments import INSTRUMENTS
from poseidon.pipeline.v3.validation import FeasibilityContext

START, TRAIN_LAST = date(2001, 1, 2), date(2021, 8, 27)  # run #19's train window
ALPHA_T_MIN = DAILY_THRESHOLDS["benchmark_alpha_t_min"]
HAC_LAGS = 5
LEVELS = {"markov_2x_0.5x": (2.0, 0.5), "markov_2x_0x": (2.0, 0.0), "markov_3x_0.5x": (3.0, 0.5)}

_runner: SpecRunner | None = None
_underlying: dict | None = None


def filter_spec(symbol: str = "TMF", capital: int = 1_000_000) -> dict:
    """2x ``symbol``, 0.5x below the 200-day MA band of the TAIEX, on ``capital`` TWD: run #19's spec at its
    defaults (TMF, 1M TWD)."""

    spec = tmf_dynamic_leverage_3x()
    spec["capital"] = {"initial_twd": capital}
    spec["instruments"]["core"]["symbol"] = symbol
    spec["sleeves"][0]["leverage"] = [
        {"component": "leverage.base", "params": {"value": 2.0}},
        {
            "component": "regime.ma_band",
            "params": {"signal": "index:taiex", "ma_days": 200, "band_pct": 0.02, "off_leverage": 0.5},
        },
    ]
    spec["params"] = {}
    return spec


def runner() -> SpecRunner:
    global _runner, _underlying
    if _runner is None:
        from poseidon.pipeline.daily_stages import underlying_daily
        from poseidon.pipeline.gates.daily import underlying_returns
        from poseidon.pipeline.refdata.store import RefdataStore

        store = WindowStore(RefdataStore(), TRAIN_LAST)
        _runner = SpecRunner(
            template=filter_spec(),
            grid=opt.ParamGrid({}),
            store=store,
            start=START,
            end=TRAIN_LAST,
            basis="return",
            feasibility=FeasibilityContext(warmup=session_warmup(warmup_record(), store)),
            calendar_rule=trading_calendar_record(),
        )
        _underlying = underlying_returns(underlying_daily(store, "TX", through=TRAIN_LAST)["close"])
    return _runner


class PathLeverage:
    """Long ``symbol`` at ``path[day]`` x equity (floor), every day or only on a month's first trading day."""

    def __init__(self, path: dict[date, float], *, monthly: bool = False, symbol: str = "TMF"):
        self.path, self.monthly, self.month = path, monthly, None
        self.instrument = INSTRUMENTS[symbol]

    def on_bar(self, view, book) -> Decision:
        day = view.date.date()
        leverage = self.path[day]
        if self.monthly:
            if self.month == (day.year, day.month):
                return Decision({}, {"core": leverage})
            self.month = (day.year, day.month)
        price = float(view["core"].close[-1])
        raw = book.equity * leverage / self.instrument.notional_twd(price) if book.equity > 0 else 0.0
        return Decision({"core": round_units(raw, "floor")}, {"core": leverage})


def compiled():
    spec = beta_matched_spec(runner().template, "TX", 1.0)
    r = runner()
    return compile_spec(
        spec, start=r.start, end=r.end, store=r.store, sessions=r.feasibility.warmup, calendar_rule=r.calendar_rule
    )


def filter_path(over: SpecRunner | None = None) -> tuple[list[date], np.ndarray]:
    """The 200-day MA band filter's state (1 on, 0 off) per day of ``over`` (default: the train window)."""

    spec = (over or runner()).compile({})
    strategy = spec.strategy()
    spec.run(strategy)
    trace = pd.DataFrame(strategy.trace)
    on = np.array([float(links[1] == links[0]) for links in trace["links"]])
    return [pd.Timestamp(t).date() for t in trace["date"]], on


def spells(on: np.ndarray) -> dict:
    changes = int(np.sum(on[1:] != on[:-1]))
    runs = np.split(on, np.flatnonzero(on[1:] != on[:-1]) + 1)
    on_runs = [len(r) for r in runs if r[0] == 1]
    off_runs = [len(r) for r in runs if r[0] == 0]
    return {
        "days": len(on),
        "on_share": float(on.mean()),
        "switches": changes,
        "mean_on_spell": float(np.mean(on_runs)),
        "mean_off_spell": float(np.mean(off_runs)),
    }


def markov_path(stats: dict, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    leave_on, leave_off = 1.0 / stats["mean_on_spell"], 1.0 / stats["mean_off_spell"]
    state = float(rng.random() < stats["on_share"])
    out = np.empty(n)
    for i, u in enumerate(rng.random(n)):
        out[i] = state
        if u < (leave_on if state else leave_off):
            state = 1.0 - state
    return out


def judge(job: dict) -> dict:
    """One null through the engine and the gate pieces; ``job``: name, path (per-day leverage), monthly.

    ``frozen``: the verdict of ``benchmark.evaluate_daily`` under ``plan.DAILY_THRESHOLDS`` (None before the
    risk-route keys are registered there); ``risk``: the risk route at every block length and confidence.
    """

    r = runner()
    path = dict(zip(job["days"], job["path"], strict=True))
    result = from_engine(compiled().run(PathLeverage(path, monthly=job.get("monthly", False))))
    alpha = benchmark.evaluate_daily(
        benchmark.BenchmarkDailyInputs(result, _underlying, "return", HAC_LAGS),
        {"benchmark_alpha_t_min": ALPHA_T_MIN},
    )
    beta = alpha.report["beta"]
    row = {
        "null": job["name"],
        "seed": job.get("seed"),
        "mean_leverage": float(np.mean(job["path"])),
        "beta": beta,
        "alpha_t": alpha.report["alpha_t_hac"],
        "alpha_pass": alpha.passed,
        "risk": {},
        "frozen": None,
    }
    runs: dict[float, object] = {}

    def matched(leverage: float):
        if leverage not in runs:
            runs[leverage] = beta_matched_run(r, "TX", leverage)
        return runs[leverage]

    if benchmark.RULE_KEY in DAILY_THRESHOLDS:
        gate = benchmark.evaluate_daily(
            benchmark.BenchmarkDailyInputs(result, _underlying, "return", HAC_LAGS, matched=matched), DAILY_THRESHOLDS
        )
        risk = gate.report["risk_route"]
        row["frozen"] = {
            "verdict": gate.verdict.value,
            "routes": gate.report["passed_routes"],
            "risk_status": risk["status"],
            "calmar_diff_lower": risk.get("calmar_diff_lower"),
        }
    run = matched(round(beta, 2))
    if isinstance(run, benchmark.NotApplicable):
        row["not_applicable"] = run.reason
        return row
    s, m = result.returns, run.returns
    (s_cagr, s_dd, s_calmar), (m_cagr, m_dd, m_calmar) = benchmark.calmar_rows(s), benchmark.calmar_rows(m)
    row.update(
        cagr=float(s_cagr),
        max_drawdown=float(s_dd),
        calmar=float(s_calmar),
        matched_cagr=float(m_cagr),
        matched_max_drawdown=float(m_dd),
        matched_calmar=float(m_calmar),
    )
    for block in job["blocks"]:
        draws = benchmark.calmar_draws(s, m, block_days=block, resamples=job["resamples"])
        lower = {str(c): benchmark.lower_bound(draws, c) for c in job["confidences"]}
        row["risk"][str(block)] = {"lower": lower, "pass": {c: bound > 0 for c, bound in lower.items()}}
    return row


def null_jobs(args) -> tuple[list[dict], dict]:
    days, on = filter_path()
    stats = spells(on)
    n = len(days)
    common = {"days": days, "blocks": args.blocks, "confidences": args.confidences, "resamples": args.resamples}
    jobs = []
    for leverage in (0.5, 1.0, 2.0, 3.0):
        for monthly in (False, True):
            name = f"constant_{leverage:g}x_{'monthly' if monthly else 'daily'}"
            jobs.append({**common, "name": name, "path": [leverage] * n, "monthly": monthly})
    for name, (high, low) in LEVELS.items():
        for seed in range(1, args.sims + 1):
            path = low + (high - low) * markov_path(stats, n, seed)
            jobs.append({**common, "name": name, "seed": seed, "path": path.tolist()})
    rng = np.random.default_rng(0)
    for seed in range(1, args.sims + 1):
        shift = int(rng.integers(252, n - 252))
        path = 0.5 + 1.5 * np.roll(on, shift)
        jobs.append({**common, "name": "shift_2x_0.5x", "seed": seed, "shift": shift, "path": path.tolist()})
    return jobs, stats


def _strip(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("days", "path")}


def table(rows: list[dict], blocks: list[int], confidences: list[float]) -> list[str]:
    lines = []
    names = sorted({row["null"] for row in rows})
    for block in blocks:
        for c in confidences:
            lines.append(f"block {block} days, confidence {c}:")
            for name in names:
                group = [row for row in rows if row["null"] == name]
                risk = [row["risk"].get(str(block), {}).get("pass", {}).get(str(c), False) for row in group]
                alpha = [row["alpha_pass"] for row in group]
                either = [a or b for a, b in zip(alpha, risk, strict=True)]
                lines.append(
                    f"  {name:<24} n={len(group):>3}  risk {sum(risk):>3} ({np.mean(risk):6.1%})  "
                    f"alpha {sum(alpha):>3} ({np.mean(alpha):6.1%})  either {sum(either):>3} ({np.mean(either):6.1%})"
                )
    return lines


def nulls(args) -> int:
    jobs, stats = null_jobs(args)
    print("filter path:", json.dumps(stats))
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(judge, jobs, chunksize=4))
    for row, job in zip(rows, jobs, strict=True):
        row.update({k: v for k, v in _strip(job).items() if k in ("shift", "monthly")})
    lines = table(rows, args.blocks, args.confidences)
    print("\n".join(lines))
    args.out.write_text(json.dumps({"filter": stats, "args": _args(args), "rows": rows, "table": lines}, indent=1))
    return 0


def replay(args) -> int:
    out = []
    for path in args.runs:
        dump = json.loads(Path(path).read_text())
        stages = {name: SimpleNamespace(report=s["artifact"]["report"]) for name, s in dump["stages"].items()}
        route = stages["spec_generate"].report.get("route") if "spec_generate" in stages else None
        config = copy.deepcopy(effective_config(dump["config"], route))
        config["thresholds"].update(
            {k: DAILY_THRESHOLDS[k] for k in (benchmark.RULE_KEY, *benchmark.RISK_KEYS, benchmark.MIN_CONTRACTS_KEY)}
        )
        ctx = SimpleNamespace(config=config, upstream=stages)
        result = DailyBenchmarkStage().evaluate(ctx)
        report = result.report
        gate = report["gate_report"]
        out.append({"run_no": dump["run_no"], "verdict": result.verdict.value, "summary": report["summary"], **gate})
        print(dump["run_no"], result.verdict.value, report["summary"])
        print("   ", json.dumps(gate.get("risk_route"), default=str)[:1500])
    args.out.write_text(json.dumps(out, indent=1, default=str))
    return 0


def _args(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(required=True)
    n = sub.add_parser("nulls")
    n.add_argument("--sims", type=int, default=200)
    n.add_argument("--jobs", type=int, default=4)
    n.add_argument("--blocks", type=lambda t: [int(x) for x in t.split(",")], default=[63, 126, 252])
    n.add_argument("--confidences", type=lambda t: [float(x) for x in t.split(",")], default=[0.95, 0.975, 0.99])
    n.add_argument("--resamples", type=int, default=2000)
    n.add_argument("--out", type=Path, required=True)
    n.set_defaults(func=nulls)
    r = sub.add_parser("replay")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=replay)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
