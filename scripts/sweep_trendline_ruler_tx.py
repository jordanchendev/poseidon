#!/usr/bin/env python3
"""Parameter sweep for the Trendline Ruler strategy on TX 5m.

Fetches the OHLCV frame from Thalassa ONCE and runs every variant in-process
through ``backtest_trendline_ruler_tx.run_variant``. Variants are a JSON list:

  [{"label": "p5_line", "set": {"pattern.pivot_left": 5, "risk.stop_mode": "line"}}, ...]

``set`` keys are dotted config paths applied on top of --config. Writes
``<out-dir>/sweep_<name>.json`` (all summaries) and ``sweep_<name>.md`` (table
sorted by total_pnl_twd).

Run on stormtrooper in a throwaway container (see the driver's docstring):
  python scripts/sweep_trendline_ruler_tx.py --variants scripts/sweeps/trendline_ruler_stage1.json \\
      --name stage1 --start 2020-03-02 --jobs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_trendline_ruler_tx import DEFAULT_CONFIG, fetch_ohlcv, load_config, run_variant

from poseidon.strategies.trendline_ruler import TrendlineRulerStrategy

logger = logging.getLogger("sweep_trendline_ruler_tx")

_OHLCV: pd.DataFrame | None = None  # shared with forked workers (copy-on-write)


def _run_one(args: tuple[Path, dict, int]) -> dict:
    config_path, variant, contracts = args
    assert _OHLCV is not None
    overrides = variant.get("set", {})
    cfg = load_config(config_path, overrides)
    t0 = datetime.now(UTC)
    try:
        summary, _ = run_variant(cfg, _OHLCV, label=variant["label"], overrides=overrides, contracts=contracts)
    except Exception as exc:  # one broken variant must not kill the sweep
        logger.exception("variant %s failed", variant["label"])
        return {"variant": variant["label"], "overrides": overrides, "error": repr(exc)}
    summary["variant"] = variant["label"]
    logger.info(
        "variant %s done in %.0fs: trades=%d pnl=%.0f",
        variant["label"],
        (datetime.now(UTC) - t0).total_seconds(),
        summary["closed_trades"],
        summary["total_pnl_twd"],
    )
    return summary


def _fmt(v, nd: int = 1) -> str:
    return "-" if v is None else f"{v:,.{nd}f}"


def markdown_table(summaries: list[dict]) -> str:
    ok = sorted((s for s in summaries if "error" not in s), key=lambda s: s["total_pnl_twd"], reverse=True)
    head = (
        "| label | trades | entered | win% | total_pnl_twd | avg_pts | avg_win/avg_loss | trade_dd_twd "
        "| tp/sl/close | sl med bars | avg stop/target pts | pnl_by_year (pts) | +years |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for s in ok:
        ex = s["by_exit"]
        n = {k: ex.get(k, {}).get("n", 0) for k in ("close_tp", "close_sl", "close")}
        years = s["pnl_by_year"]
        compact = " ".join(f"{y[2:]}:{v:+.0f}" for y, v in years.items())
        rows.append(
            f"| {s['variant']} | {s['closed_trades']} | {s['reject_counts'].get('entered', 0)} "
            f"| {_fmt((s['win_rate'] or 0) * 100)} | {_fmt(s['total_pnl_twd'], 0)} | {_fmt(s['avg_pnl_points'], 2)} "
            f"| {_fmt(s['avg_win_points'])}/{_fmt(s['avg_loss_points'])} | {_fmt(s['trade_dd_twd'], 0)} "
            f"| {n['close_tp']}/{n['close_sl']}/{n['close']} "
            f"| {_fmt(s['median_holding_bars_by_exit'].get('close_sl'), 0)} "
            f"| {_fmt(s['avg_stop_points'])}/{_fmt(s['avg_target_points'])} "
            f"| {compact} | {sum(v > 0 for v in years.values())}/{len(years)} |"
        )
    errors = [f"- {s['variant']}: {s['error']}" for s in summaries if "error" in s]
    return head + "\n".join(rows) + ("\n\nFailed variants:\n" + "\n".join(errors) if errors else "") + "\n"


def main() -> int:
    global _OHLCV
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=Path, required=True, help="JSON list of {label, set}")
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--start", default="2020-03-02")
    ap.add_argument("--end", default=None)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (fork)")
    ap.add_argument("--out-dir", type=Path, default=Path("scripts/output"))
    args = ap.parse_args()

    variants = json.loads(args.variants.read_text())
    labels = [v["label"] for v in variants]
    if len(set(labels)) != len(labels):
        raise SystemExit("variant labels must be unique")
    for v in variants:  # fail fast on bad overrides before fetching data
        TrendlineRulerStrategy(config=load_config(args.config, v.get("set", {}))).validate_config()

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else datetime.now(UTC)
    _OHLCV = fetch_ohlcv(json.loads(args.config.read_text()), start, end)
    if _OHLCV.empty:
        logger.error("no OHLCV returned")
        return 1

    tasks = [(args.config, v, args.contracts) for v in variants]
    t0 = datetime.now(UTC)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("fork")) as pool:
            summaries = list(pool.map(_run_one, tasks))
    else:
        summaries = [_run_one(t) for t in tasks]
    logger.info("sweep %s: %d variants in %.0fs", args.name, len(variants), (datetime.now(UTC) - t0).total_seconds())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"sweep_{args.name}.json").write_text(json.dumps(summaries, indent=2, default=str))
    table = markdown_table(summaries)
    (args.out_dir / f"sweep_{args.name}.md").write_text(f"# sweep {args.name}\n\n{table}")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
