"""DSL v2 parity of pipeline runs: v1 runner vs target-position engine, trade for trade (design §10 step 7).

For each run in the runs file (default ``tests/pipeline/fixtures/dsl_parity_runs.json``: runs #3 and #6
as stored in the live ``poseidon_pipeline`` DB, frozen config plus optimize's best spec), backtest the
best spec on the run's frozen train window with both engines and compare every trade, once with the
frozen costs and once as the cost_stress gate runs it (costs x the gate multiplier, stops slipped by the
gate's ticks). The window is
read from an existing OhlcvCache (never fetched) and its sha256 must equal the one the run recorded;
the settlement calendar comes from the refdata store (``POSEIDON_PIPELINE_REFDATA_ROOT``). With
``--reference-dir`` the v1 rerun's ``trades.csv`` bytes are also checked against the run's stored
``best_trades.csv`` (``<dir>/<run_id>/best_trades.csv``).

Exit status 0 only when every run matches; the JSON report goes to stdout and ``--out``.

    python scripts/pv2_dsl_parity.py --cache-root /pdata [--reference-dir /pdata/ref] [--out /out/parity.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
from poseidon.pipeline.backtesting import settlement_expiries, stressed_cost_model, trades_csv
from poseidon.pipeline.data import OhlcvCache, load_bars
from poseidon.pipeline.dsl_parity import run_parity

DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "tests" / "pipeline" / "fixtures" / "dsl_parity_runs.json"


class CacheOnly:
    def fetch(self, *args, **kwargs):
        raise SystemExit("window not in the cache; this script never fetches from Thalassa")


def check_run(run: dict, cache: OhlcvCache, reference_dir: Path | None) -> dict:
    config = run["config_json"]
    bars = load_bars("backtest", config, "train", cache)
    if bars.data_sha256 != run["data"]["data_sha256"] or len(bars.frame) != run["data"]["bars"]:
        raise SystemExit(f"run {run['run_no']}: cached window differs from the one the run recorded")
    spec = run["best_spec"]
    factory = DslV2StrategyFactory(spec=spec, bar_label=config["backtest"]["bar_label"])
    expiries = settlement_expiries(spec["symbol"])
    started = time.monotonic()
    report = run_parity(factory.build, bars.frame, config["backtest"], expiries=expiries)
    stress = config["gates"]["cost_stress"]
    stressed = run_parity(
        factory.build,
        bars.frame,
        config["backtest"],
        expiries=expiries,
        cost_model=stressed_cost_model(
            config["backtest"]["cost_model"], config["thresholds"]["cost_stress_multiplier"]
        ),
        stop_slippage_ticks=float(stress["stop_slippage_ticks"]),
    )
    out = {
        "run_no": run["run_no"],
        "run_id": run["run_id"],
        "spec": spec["name"],
        "params": run["best_params"],
        "bars": len(bars.frame),
        "data_sha256": bars.data_sha256,
        "seconds": round(time.monotonic() - started, 1),
        **report.summary(),
        "cost_stress": stressed.summary(),
    }
    out["ok"] = report.ok and stressed.ok
    stored = run["best_metrics"]
    out["v1_matches_stored_metrics"] = (
        report.v1.metrics["n_trades"] == stored["n_trades"]
        and abs(report.v1.metrics["total_pnl_twd"] - stored["total_pnl_twd"]) < 1e-6
    )
    if reference_dir is not None:
        path = reference_dir / run["run_id"] / "best_trades.csv"
        stored_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        out["v1_trades_csv_equals_stored"] = hashlib.sha256(trades_csv(report.v1.trades)).hexdigest() == stored_sha
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--cache-root", type=Path, required=True, help="OhlcvCache root (holds cache/ohlcv/)")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--run", type=int, action="append", help="only these run_no (repeatable)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    cache = OhlcvCache(args.cache_root, CacheOnly())
    runs = [run for run in json.loads(args.runs.read_text()) if not args.run or run["run_no"] in args.run]
    if not runs:
        raise SystemExit("no runs selected")
    results = [check_run(run, cache, args.reference_dir) for run in runs]
    text = json.dumps(results, indent=1, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    passed = all(
        r["ok"] and r["v1_matches_stored_metrics"] and r.get("v1_trades_csv_equals_stored", True) for r in results
    )
    print("PARITY", "PASS" if passed else "FAIL", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
