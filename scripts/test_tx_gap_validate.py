#!/usr/bin/env python3
"""TX gap-fade R2 — cross-asset + execution validation.

After v2 PASS verdict (6/6 thresholds, 4/4 std windows, 2.5/3 sub-periods),
two final checks before promoting to strategy class:

  CROSS-ASSET: Same R2 rule on TX (futures), MTX (mini-futures, same
  underlying), and 0050 (TW stock ETF tracking TAIEX). If pattern shows
  on all three → TAIEX market structural effect, not a TX contract artifact.

  EXECUTION COST STRESS: Re-run R2 at 3.2/5/10/20 bps round-trip + a
  gap-conditional model (slippage scales with |gap|, mimicking thinner
  opening auction on panic days). Does alpha survive realistic assumptions?

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_gap_validate.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.data.remote_repository import RemoteDataRepository

START = datetime(2021, 3, 22)
END = datetime.now()
BARS_PER_YEAR = 240
WARMUP_BARS = 252
GAP_STD_WIN = 60
THRESHOLD = -1.0


def annualised(mean: float, std: float) -> float:
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(BARS_PER_YEAR))


def perf(net: pd.Series, engaged: pd.Series) -> dict:
    n_total = len(net)
    if n_total == 0:
        return {}
    eq = (1.0 + net).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())
    cum = float(eq.iloc[-1] - 1.0)
    sh = annualised(net.mean(), net.std())
    downside = net[net < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = annualised(net.mean(), dstd) if dstd > 0 else 0.0
    n_eng = int(engaged.sum())
    freq = n_eng / n_total
    calmar = cum / abs(mdd) if mdd < 0 else (float("inf") if cum > 0 else 0.0)
    return {
        "n_total": n_total,
        "n_engaged": n_eng,
        "freq": freq,
        "sh_full": sh,
        "sortino": sortino,
        "cum": cum,
        "mdd": mdd,
        "calmar": calmar,
    }


def build_df(repo: RemoteDataRepository, symbol: str, market: str) -> pd.DataFrame:
    raw = repo.read_ohlcv(symbol=symbol, market=market, interval="1d", start=START, end=END)
    if raw.empty:
        raise RuntimeError(f"No data for {symbol}/{market}")
    idx = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    raw = raw.copy()
    raw.index = pd.to_datetime(idx).normalize()
    raw = raw[~raw.index.duplicated(keep="last")]
    df = raw[["open", "close"]].rename(columns={"open": "px_open", "close": "px_close"}).dropna()
    df["prev_close"] = df["px_close"].shift(1)
    df["gap"] = (df["px_open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["px_close"] - df["px_open"]) / df["px_open"]
    df["gap_std"] = df["gap"].rolling(GAP_STD_WIN, min_periods=GAP_STD_WIN).std()
    df["gap_z"] = df["gap"] / df["gap_std"]
    df = df.dropna(subset=["gap_z", "intraday_ret"]).iloc[WARMUP_BARS:].copy()
    return df


def run_r2(df: pd.DataFrame, threshold: float, cost: pd.Series | float) -> tuple[dict, dict]:
    """Run R2 (long when gap_z < threshold) with given cost.

    cost can be a scalar (uniform per-trade round-trip) or a Series indexed
    like df (per-day round-trip cost — for gap-conditional model).
    Returns (R2_perf, baseline_perf).
    """
    intraday = df["intraday_ret"]
    gz = df["gap_z"]

    # Baseline: long every day
    base_eng = pd.Series(True, index=df.index)
    base_cost = cost if isinstance(cost, pd.Series) else pd.Series(cost, index=df.index)
    base_net = intraday - base_eng.astype(float) * base_cost
    base_p = perf(base_net, base_eng)

    # R2
    pos = pd.Series(0.0, index=df.index)
    pos[gz < threshold] = 1.0
    eng = pos != 0
    r2_net = pos * intraday - eng.astype(float) * base_cost
    r2_p = perf(r2_net, eng)
    return r2_p, base_p


def main():
    repo = RemoteDataRepository.from_settings()

    # ===================================================================
    # SECTION 1: Cross-asset (TX, MTX, 0050) — same R2 rule
    # ===================================================================
    print("=" * 80)
    print("SECTION 1: Cross-asset validation (R2: long when gap_z<-1, std_win=60d)")
    print("=" * 80)
    print("If R2 fires on TX/MTX/0050 with similar Sh → TAIEX structural effect")
    print("If only TX shows it → TX-contract-specific artifact (probably overfit)\n")

    cross_results = {}
    for symbol, market, label in [
        ("TX", "tw_futures", "TX (large futures)"),
        ("MTX", "tw_futures", "MTX (mini futures)"),
        ("0050", "tw_stock", "0050 (TW ETF tracking TAIEX)"),
    ]:
        try:
            df = build_df(repo, symbol, market)
        except Exception as e:
            print(f"[skip] {label}: {e}")
            continue
        r2_p, base_p = run_r2(df, THRESHOLD, cost=0.00032)
        cross_results[label] = {"r2": r2_p, "baseline": base_p, "bars": len(df)}
        d_sh = r2_p["sh_full"] - base_p["sh_full"]
        d_sort = r2_p["sortino"] - base_p["sortino"]
        d_mdd = (r2_p["mdd"] - base_p["mdd"]) * 100
        print(f"\n  {label}  ({len(df)} OOS bars, {df.index.min().date()} → {df.index.max().date()})")
        print(
            f"    baseline  freq=100.0%  sh={base_p['sh_full']:+.2f}  sort={base_p['sortino']:+.2f}  "
            f"cum={base_p['cum'] * 100:+6.2f}%  mdd={base_p['mdd'] * 100:+6.2f}%"
        )
        print(
            f"    R2        freq={r2_p['freq'] * 100:5.1f}%  sh={r2_p['sh_full']:+.2f}  sort={r2_p['sortino']:+.2f}  "
            f"cum={r2_p['cum'] * 100:+6.2f}%  mdd={r2_p['mdd'] * 100:+6.2f}%  "
            f"Δsh={d_sh:+.2f}  Δsort={d_sort:+.2f}  Δmdd={d_mdd:+.2f}pp"
        )

    # ===================================================================
    # SECTION 2: Cost stress (TX R2 at 3.2 / 5 / 10 / 20 bps + gap-conditional)
    # ===================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Cost stress (TX R2 alpha vs realistic execution costs)")
    print("=" * 80)
    print("3.2bps: optimistic estimate from v1")
    print("5.0bps:  +tax (0.002%/side TX futures = 4bps RT) + ~1bp commission")
    print("10bps:   conservative — assumes some slippage on opening auction")
    print("20bps:   pessimistic stress — should still beat baseline if alpha is real")
    print("gap-cond: cost = max(3.2bps, |gap| / 4)  — slippage scales with gap magnitude\n")

    df_tx = build_df(repo, "TX", "tw_futures")
    cost_results = {}

    for cost_label, cost in [
        ("3.2 bps RT", 0.00032),
        ("5.0 bps RT", 0.00050),
        ("10  bps RT", 0.00100),
        ("20  bps RT", 0.00200),
    ]:
        r2_p, base_p = run_r2(df_tx, THRESHOLD, cost=cost)
        cost_results[cost_label] = {"r2": r2_p, "baseline": base_p}
        print(
            f"  {cost_label}  baseline_sh={base_p['sh_full']:+.2f}  "
            f"R2_sh={r2_p['sh_full']:+.2f}  R2_sort={r2_p['sortino']:+.2f}  "
            f"R2_cum={r2_p['cum'] * 100:+6.2f}%  R2_mdd={r2_p['mdd'] * 100:+6.2f}%  "
            f"Δsh={r2_p['sh_full'] - base_p['sh_full']:+.2f}"
        )

    # Gap-conditional cost: slippage scales with |gap|. Floor at 3.2bps.
    # For R2 (long when gap_z<-1), this hurts most because R2 specifically
    # trades on big-gap days where slippage assumption matters.
    cost_series = np.maximum(0.00032, df_tx["gap"].abs() / 4)
    cost_series = pd.Series(cost_series, index=df_tx.index)
    r2_p, base_p = run_r2(df_tx, THRESHOLD, cost=cost_series)
    cost_results["gap-conditional"] = {"r2": r2_p, "baseline": base_p}
    avg_cost_engaged = float(cost_series[df_tx["gap_z"] < THRESHOLD].mean()) * 1e4
    avg_cost_all = float(cost_series.mean()) * 1e4
    print(
        f"  gap-cond  baseline_sh={base_p['sh_full']:+.2f}  "
        f"R2_sh={r2_p['sh_full']:+.2f}  R2_sort={r2_p['sortino']:+.2f}  "
        f"R2_cum={r2_p['cum'] * 100:+6.2f}%  R2_mdd={r2_p['mdd'] * 100:+6.2f}%  "
        f"Δsh={r2_p['sh_full'] - base_p['sh_full']:+.2f}"
    )
    print(f"            (avg cost: all_days={avg_cost_all:.2f}bps  R2_engaged_days={avg_cost_engaged:.2f}bps)")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Cross-asset: count how many show R2 Δsh > +0.3
    n_cross = sum(1 for r in cross_results.values() if r["r2"]["sh_full"] > r["baseline"]["sh_full"] + 0.3)
    print(f"  Cross-asset:    {n_cross}/{len(cross_results)} assets show R2 Δsh > +0.3 vs baseline")

    # Cost stress: highest cost level where R2 still beats baseline by Δsh > +0.3
    surviving_cost = None
    for k, r in cost_results.items():
        if r["r2"]["sh_full"] > r["baseline"]["sh_full"] + 0.3:
            surviving_cost = k
    print(f"  Cost stress:    R2 alpha survives up to: {surviving_cost or 'NONE'}")

    overall_pass = (n_cross >= 2) and (
        cost_results.get("10  bps RT", {}).get("r2", {}).get("sh_full", -99)
        > cost_results.get("10  bps RT", {}).get("baseline", {}).get("sh_full", 99) + 0.3
    )
    print(
        f"\n  OVERALL: {'PASS — R2 robust across assets and execution costs, ready for strategy class' if overall_pass else 'PARTIAL/FAIL — investigate further before promoting'}"
    )

    out_path = Path("scripts/output/tx_gap_validate_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "config": {
                    "warmup_bars": WARMUP_BARS,
                    "gap_std_win": GAP_STD_WIN,
                    "threshold": THRESHOLD,
                },
                "cross_asset": cross_results,
                "cost_stress": cost_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
