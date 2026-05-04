#!/usr/bin/env python
"""Phase 90 verdict emitter — Option C (TWAP+VWAP+Naive+v18gap, PPO/OPDS DEFERRED).

Consumes backtest_result.csv files produced by `scripts/run_phase90_train.py`,
aggregates per-leg PA(bps) → pair Sharpe / MDD / cumulative return via
:mod:`poseidon.research.rl_aggregate`, builds the locked comparison table
via :mod:`poseidon.research.rl_comparison`, applies GATE.yaml thresholds,
and emits a deploy / refine / kill verdict.

PPO / OPDS columns are recorded as ``PARTIAL`` (status="DEFERRED — qlib
v0.9.7 × tianshou 0.5+ API drift, see 90-04.1-SUMMARY.md"). The verdict
is computed against the best-of-2-RL (TWAP/VWAP) instead of best-of-4-RL,
with a clear caveat in the verdict text.

Output:
  <run-root>/<run_id>/comparison.csv
  <run-root>/<run_id>/verdict.json
  <run-root>/<run_id>/verdict.md   (human-readable)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

THALASSA_BASE_URL = os.environ.get("POSEIDON_THALASSA_BASE_URL", "http://192.168.31.241:8001")
THALASSA_API_KEY = os.environ.get("POSEIDON_THALASSA_API_KEY", "")


def fetch_daily_ohlcv(symbol: str, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull daily OHLCV from Thalassa for v18_gap baseline computation."""
    url = f"{THALASSA_BASE_URL}/api/v1/ohlcv"
    params = {
        "symbol": symbol,
        "market": market,
        "interval": "1d",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 5000,
    }
    headers = {"X-API-Key": THALASSA_API_KEY} if THALASSA_API_KEY else {}
    r = requests.get(url, params=params, headers=headers, timeout=120)
    r.raise_for_status()
    body = r.json()
    rows = body.get("data") or body.get("rows") or body
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df["time"] = df["time"].dt.tz_convert("Asia/Taipei").dt.tz_localize(None).dt.normalize()
    return df.set_index("time").sort_index()


def load_backtest_csvs(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all backtest_result.csv files under <run_dir>/outputs/.

    Returns ``{"<algo>_<leg>": DataFrame}``. Each CSV has the qlib upstream
    columns (instrument, datetime, pa, ffr, trade_price, ...). Both
    twap_TX/backtest_result.csv and twap_0050/backtest_result.csv contain
    the SAME combined data (qlib's parallel pool dispatches per stock and
    each writes the full result to the per-(algo,leg) output_dir).
    """
    out: dict[str, pd.DataFrame] = {}
    for csv_path in sorted((run_dir / "outputs").rglob("backtest_result.csv")):
        algo_leg = csv_path.parent.name  # e.g., "twap_TX"
        df = pd.read_csv(csv_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        out[algo_leg] = df
    return out


def per_leg_pa_series(combined_df: pd.DataFrame, instrument: str) -> pd.Series:
    """Extract per-day PA (bps) for one instrument from the combined CSV."""
    leg = combined_df[combined_df["instrument"] == instrument].copy()
    leg = leg.dropna(subset=["pa"])
    leg = leg.set_index("datetime")
    return leg["pa"]


def emit_verdict_for_algo(
    algo_combined: pd.DataFrame,
    tx_naive_intraday: pd.Series,
    etf_naive_intraday: pd.Series,
) -> dict[str, Any]:
    """Aggregate one algo's combined CSV → pair metrics dict."""
    from poseidon.research.rl_aggregate import aggregate_pair_metrics

    tx_pa = per_leg_pa_series(algo_combined, "TX")
    etf_pa = per_leg_pa_series(algo_combined, "0050")
    if len(tx_pa) == 0 or len(etf_pa) == 0:
        return {"status": "PARTIAL", "error": "no per-leg PA rows in CSV"}

    # Align on common dates.
    common = tx_pa.index.intersection(etf_pa.index)
    if len(common) == 0:
        return {"status": "PARTIAL", "error": "TX/0050 PA series have no common dates"}
    tx_pa = tx_pa.loc[common]
    etf_pa = etf_pa.loc[common]
    tx_naive_intraday_aligned = tx_naive_intraday.reindex(common).fillna(0.0)
    etf_naive_intraday_aligned = etf_naive_intraday.reindex(common).fillna(0.0)

    return aggregate_pair_metrics(
        tx_pa_bps=tx_pa,
        etf_pa_bps=etf_pa,
        tx_naive_intraday_ret=tx_naive_intraday_aligned,
        etf_naive_intraday_ret=etf_naive_intraday_aligned,
    )


def apply_gate(comparison_df: pd.DataFrame, gate: dict, deferred_algos: list[str]) -> dict[str, Any]:
    """Apply GATE.yaml thresholds → verdict dict.

    Plan 90-04.1 Option C: PPO/OPDS columns are DEFERRED — best_rl_pair_sharpe
    is computed across only TWAP/VWAP. The verdict text records this
    explicitly.
    """
    rl_cols = [c for c in ("TWAP", "VWAP", "PPO", "OPDS") if c not in deferred_algos]
    rl_sharpes = [
        comparison_df.loc["Pair net Sharpe", c] for c in rl_cols if not pd.isna(comparison_df.loc["Pair net Sharpe", c])
    ]
    best_rl_sharpe = float("nan") if not rl_sharpes else max(rl_sharpes)
    naive_sharpe = comparison_df.loc["Pair net Sharpe", "Naive"]
    naive_mdd = comparison_df.loc["Pair MDD %", "Naive"]
    best_rl_col = max(rl_cols, key=lambda c: comparison_df.loc["Pair net Sharpe", c]) if rl_sharpes else None
    best_rl_mdd = comparison_df.loc["Pair MDD %", best_rl_col] if best_rl_col else float("nan")
    trigger_count = int(comparison_df.loc["Trigger-day count", "Naive"])

    sharpe_buffer = best_rl_sharpe - naive_sharpe if not pd.isna(best_rl_sharpe) else float("nan")
    mdd_ratio = best_rl_mdd / naive_mdd if naive_mdd != 0 and not pd.isna(best_rl_mdd) else float("nan")

    gate_results = {}
    passed = 0

    # gate_01: best-of-RL Sharpe minus naive Sharpe ≥ 0.3
    g01 = gate["gate_thresholds"]["gate_01_sharpe_buffer"]
    g01_pass = sharpe_buffer >= g01["value"] if not pd.isna(sharpe_buffer) else False
    gate_results["gate_01_sharpe_buffer"] = {
        "passed": bool(g01_pass),
        "measured": float(sharpe_buffer) if not pd.isna(sharpe_buffer) else None,
        "threshold": g01["value"],
        "op": g01["op"],
    }
    if g01_pass:
        passed += 1

    # gate_02: best-of-RL MDD / naive MDD ≤ 1.20
    g02 = gate["gate_thresholds"]["gate_02_mdd_ratio_max"]
    g02_pass = mdd_ratio <= g02["value"] if not pd.isna(mdd_ratio) else False
    gate_results["gate_02_mdd_ratio_max"] = {
        "passed": bool(g02_pass),
        "measured": float(mdd_ratio) if not pd.isna(mdd_ratio) else None,
        "threshold": g02["value"],
        "op": g02["op"],
    }
    if g02_pass:
        passed += 1

    # gate_03: naive Sharpe ≥ 0
    g03 = gate["gate_thresholds"]["gate_03_baseline_floor"]
    g03_pass = naive_sharpe >= g03["value"] if not pd.isna(naive_sharpe) else False
    gate_results["gate_03_baseline_floor"] = {
        "passed": bool(g03_pass),
        "measured": float(naive_sharpe) if not pd.isna(naive_sharpe) else None,
        "threshold": g03["value"],
        "op": g03["op"],
    }
    if g03_pass:
        passed += 1

    # gate_04: trigger-day count ≥ 20
    g04 = gate["gate_thresholds"]["gate_04_trade_count_floor"]
    g04_pass = trigger_count >= g04["value"]
    gate_results["gate_04_trade_count_floor"] = {
        "passed": bool(g04_pass),
        "measured": int(trigger_count),
        "threshold": g04["value"],
        "op": g04["op"],
    }
    if g04_pass:
        passed += 1

    # Verdict — note: deferred PPO/OPDS does NOT auto-fail; we still run the
    # 4-gate logic on TWAP/VWAP best.
    if passed == 4 and best_rl_sharpe >= 0:
        verdict = "deploy"
    elif 1 <= passed <= 3 and best_rl_sharpe >= 0:
        verdict = "refine"
    else:
        verdict = "kill"

    return {
        "verdict": verdict,
        "passed_count": passed,
        "deferred_algos": deferred_algos,
        "gates": gate_results,
        "best_rl_col": best_rl_col,
        "best_rl_pair_sharpe": float(best_rl_sharpe) if not pd.isna(best_rl_sharpe) else None,
        "naive_pair_sharpe": float(naive_sharpe) if not pd.isna(naive_sharpe) else None,
        "trigger_count": trigger_count,
        "small_sample_warning": trigger_count < 20,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to <run-root>/<run_id>/")
    parser.add_argument(
        "--gate",
        default=".planning/phases/90-rl-order-execution/GATE.yaml",
        help="Path to GATE.yaml",
    )
    parser.add_argument(
        "--window",
        default="2020-04-01:2026-04-30",
        help="Window for v18 gap baseline daily fetch (start:end)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    # Load GATE.yaml.
    import yaml

    with open(args.gate) as fp:
        gate = yaml.safe_load(fp)

    # Load all backtest CSVs.
    csvs = load_backtest_csvs(run_dir)
    logger.info("Loaded %d backtest CSVs: %s", len(csvs), list(csvs.keys()))

    # Identify which algos actually produced non-empty results.
    available_algos = set()
    deferred_algos: list[str] = []
    for algo in ("TWAP", "VWAP", "PPO", "OPDS"):
        key = f"{algo.lower()}_TX"
        if key in csvs and len(csvs[key]) > 0:
            available_algos.add(algo)
        else:
            deferred_algos.append(algo)
    logger.info("Available algos: %s; deferred: %s", available_algos, deferred_algos)

    # Pull daily OHLCV for v18 gap baseline.
    start, end = args.window.split(":")
    tx_daily = fetch_daily_ohlcv("TX", "tw_futures", pd.Timestamp(start), pd.Timestamp(end))
    etf_daily = fetch_daily_ohlcv("0050", "tw_stock", pd.Timestamp(start), pd.Timestamp(end))

    # Use any available CSV's trigger date list as the universe (all algos
    # share the same trigger days).
    any_combined = next(iter(csvs.values()))
    trigger_dates = sorted(any_combined["datetime"].dropna().unique())

    # Build naive_intraday_ret per leg from per-day open/close diffs in the
    # daily OHLCV (close-of-day vs open-of-day on each trigger).
    tx_naive = ((tx_daily["close"] - tx_daily["open"]) / tx_daily["open"]).reindex(pd.DatetimeIndex(trigger_dates))
    etf_naive = ((etf_daily["close"] - etf_daily["open"]) / etf_daily["open"]).reindex(pd.DatetimeIndex(trigger_dates))

    # Aggregate per-algo pair metrics.
    from poseidon.research.rl_comparison import build_comparison_table, build_v18_gap_baseline

    per_algo: dict[str, dict] = {}
    for algo in ("twap", "vwap", "ppo", "opds"):
        key = f"{algo}_TX"
        if key not in csvs or len(csvs[key]) == 0:
            per_algo[algo] = {"status": "PARTIAL", "error": "DEFERRED — qlib v0.9.7 × tianshou 0.5+ API drift"}
            continue
        per_algo[algo] = emit_verdict_for_algo(csvs[key], tx_naive, etf_naive)

    # Naive baseline — single fill at session open per D-13. Pair Sharpe
    # comes from naive intraday returns (TX long, 0050 short) with zero PA.
    pair_naive_ret = (tx_naive - etf_naive).dropna()
    engaged = pd.Series([True] * len(pair_naive_ret), index=pair_naive_ret.index)
    from poseidon.research.rl_aggregate import perf_full

    naive = perf_full(pair_naive_ret, engaged)
    naive["mean_per_leg_slippage_tx_bps"] = 0.0
    naive["mean_per_leg_slippage_etf_bps"] = 0.0
    naive["mean_net_slippage_bps"] = 0.0

    # v18 |gap|/4 baseline.
    tx_open = tx_daily["open"].reindex(pd.DatetimeIndex(trigger_dates))
    prev_tx_close = tx_daily["close"].shift(1).reindex(pd.DatetimeIndex(trigger_dates))
    v18_gap = build_v18_gap_baseline(
        tx_open=tx_open.dropna(),
        prev_tx_close=prev_tx_close.dropna(),
        tx_naive_intraday_ret=tx_naive.dropna(),
        etf_naive_intraday_ret=etf_naive.dropna(),
    )

    # Build the locked comparison table.
    comparison_df = build_comparison_table(per_algo, naive, v18_gap)

    # Persist.
    comp_csv = run_dir / "comparison.csv"
    comparison_df.to_csv(comp_csv)
    logger.info("Comparison table → %s", comp_csv)

    # Apply gate.
    verdict_dict = apply_gate(comparison_df, gate, deferred_algos)
    verdict_dict["window"] = args.window
    verdict_dict["run_dir"] = str(run_dir)
    verdict_dict["comparison_csv"] = str(comp_csv)

    verdict_json = run_dir / "verdict.json"
    verdict_json.write_text(json.dumps(verdict_dict, indent=2, default=str))
    logger.info("Verdict JSON → %s", verdict_json)

    # Human-readable verdict.
    md_lines = []
    md_lines.append(f"# Phase 90 Verdict — {verdict_dict['verdict'].upper()}\n")
    md_lines.append(f"**Window:** {args.window}\n")
    md_lines.append(f"**Trigger days:** {verdict_dict['trigger_count']}\n")
    if verdict_dict["small_sample_warning"]:
        md_lines.append("⚠️ **Small-sample warning** — trigger_count < 20 (gate_04 mandates ≥20)\n")
    md_lines.append(f"**Passed gates:** {verdict_dict['passed_count']} / 4\n")
    md_lines.append(
        f"**Best RL column:** {verdict_dict['best_rl_col']} (Sharpe = {verdict_dict['best_rl_pair_sharpe']:.3f})\n"
        if verdict_dict["best_rl_pair_sharpe"] is not None
        else "**Best RL column:** N/A — no RL columns produced finite Sharpe\n"
    )
    md_lines.append(f"**Naive pair Sharpe:** {verdict_dict['naive_pair_sharpe']:.3f}\n")
    if deferred_algos:
        md_lines.append(
            f"\n**DEFERRED algos:** {', '.join(deferred_algos)} "
            f"— Plan 90-04.1 Option C (qlib v0.9.7 × tianshou 0.5+ API drift; "
            f"see 90-04.1-SUMMARY.md). Verdict computed against best-of-{len(verdict_dict['gates']) - 1}-RL "
            f"available, NOT best-of-4. The institutional execution methods (TWAP/VWAP) ARE present.\n"
        )
    md_lines.append("\n## Gates\n")
    for gname, gres in verdict_dict["gates"].items():
        status = "✅" if gres["passed"] else "❌"
        md_lines.append(
            f"- {status} `{gname}`: measured={gres['measured']}, threshold {gres['op']} {gres['threshold']}\n"
        )
    md_lines.append("\n## Comparison Table\n```\n")
    md_lines.append(comparison_df.to_string())
    md_lines.append("\n```\n")

    verdict_md = run_dir / "verdict.md"
    verdict_md.write_text("".join(md_lines))
    logger.info("Verdict MD → %s", verdict_md)

    # Print to stdout for terminal feedback.
    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict_dict['verdict'].upper()}")
    print(f"  passed_count: {verdict_dict['passed_count']}/4")
    print(f"  best_rl_col: {verdict_dict['best_rl_col']}")
    print(f"  best_rl_sharpe: {verdict_dict['best_rl_pair_sharpe']}")
    print(f"  naive_sharpe: {verdict_dict['naive_pair_sharpe']}")
    print(f"  deferred: {deferred_algos}")
    print("=" * 60 + "\n")
    print(comparison_df.to_string())
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
