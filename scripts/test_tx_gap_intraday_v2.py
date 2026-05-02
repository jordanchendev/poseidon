#!/usr/bin/env python3
"""TX gap-fade v2 — robustness checks before claiming R2 alpha.

v1 (test_tx_gap_intraday.py) found:
  R2 buy gap_z<-1 → next intraday: Sh +1.24 vs baseline +0.56,
  driven by significant rebound after big gap-down (z<-1, t=+2.02).

But threshold -1 was eyeballed and z<-2 cell only had n=30.
This v2 checks robustness:

  A. Threshold sensitivity   — does Sh stay > 1 for thresholds [-2, -0.5]?
  B. Std window sensitivity  — does Sh hold for 60d / 120d / 252d gap-std?
  C. Tail-exclusion          — is the alpha just z<-2 outliers, or broader?
  D. Sub-period stability    — does R2 win baseline in each OOS third?

If R2 holds across A/B/C/D → real alpha candidate, promote to next step
(strategy class or pair with another signal).
If R2 only works at threshold = -1, std = 60d, and only with z<-2 included
→ overfit to one cell, kill it.

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_gap_intraday_v2.py
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
ROUND_TRIP_COST = 0.00032
WARMUP_BARS = 252


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


def run_long_below(gz: pd.Series, intraday: pd.Series, threshold: float) -> dict:
    """Strategy: long intraday when gap_z < threshold, else flat."""
    pos = pd.Series(0.0, index=gz.index)
    pos[gz < threshold] = 1.0
    eng = pos != 0
    net = pos * intraday - eng.astype(float) * ROUND_TRIP_COST
    return perf(net, eng)


def run_long_band(gz: pd.Series, intraday: pd.Series, lo: float, hi: float) -> dict:
    """Strategy: long intraday when lo <= gap_z < hi (band), else flat."""
    pos = pd.Series(0.0, index=gz.index)
    pos[(gz >= lo) & (gz < hi)] = 1.0
    eng = pos != 0
    net = pos * intraday - eng.astype(float) * ROUND_TRIP_COST
    return perf(net, eng)


def run_sym_fade(gz: pd.Series, intraday: pd.Series, threshold: float) -> dict:
    """Strategy: long intraday when gap_z < -threshold, short when > +threshold."""
    pos = pd.Series(0.0, index=gz.index)
    pos[gz > threshold] = -1.0
    pos[gz < -threshold] = 1.0
    eng = pos != 0
    net = pos * intraday - eng.astype(float) * ROUND_TRIP_COST
    return perf(net, eng)


def fmt_perf_row(label: str, p: dict, base: dict | None = None) -> str:
    base_part = ""
    if base is not None and base:
        d_sh = p["sh_full"] - base["sh_full"]
        d_sort = p["sortino"] - base["sortino"]
        base_part = f"  Δsh={d_sh:+5.2f}  Δsort={d_sort:+5.2f}"
    return (
        f"  {label:42}  freq={p['freq'] * 100:5.1f}%  n={p['n_engaged']:4}  "
        f"sh={p['sh_full']:+5.2f}  sort={p['sortino']:+5.2f}  "
        f"cum={p['cum'] * 100:+7.2f}%  mdd={p['mdd'] * 100:+6.2f}%  cal={p['calmar']:+5.2f}{base_part}"
    )


def make_window_df(tx: pd.DataFrame, std_win: int) -> pd.DataFrame:
    df = tx[["open", "close"]].rename(columns={"open": "tx_open", "close": "tx_close"}).dropna()
    df["prev_close"] = df["tx_close"].shift(1)
    df["gap"] = (df["tx_open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["tx_close"] - df["tx_open"]) / df["tx_open"]
    df["gap_std"] = df["gap"].rolling(std_win, min_periods=std_win).std()
    df["gap_z"] = df["gap"] / df["gap_std"]
    df = df.dropna(subset=["gap_z", "intraday_ret"]).iloc[WARMUP_BARS:].copy()
    return df


def baseline_perf(df: pd.DataFrame) -> dict:
    intraday = df["intraday_ret"]
    eng = pd.Series(True, index=df.index)
    net = intraday - ROUND_TRIP_COST
    return perf(net, eng)


def main():
    repo = RemoteDataRepository.from_settings()
    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    if tx.empty:
        raise RuntimeError("No TX data")
    tx_idx = tx.index.tz_localize(None) if tx.index.tz is not None else tx.index
    tx = tx.copy()
    tx.index = pd.to_datetime(tx_idx).normalize()
    tx = tx[~tx.index.duplicated(keep="last")]

    # === A. Threshold sensitivity (std_win = 60) ===
    print("=== A. Threshold sensitivity (std_win=60d, R2-style long-only) ===")
    df60 = make_window_df(tx, 60)
    base60 = baseline_perf(df60)
    print(f"  OOS window: {df60.index.min().date()} → {df60.index.max().date()}  ({len(df60)} bars)")
    print(fmt_perf_row("baseline long-every-day", base60))
    a_results = {}
    for thr in [-2.0, -1.5, -1.25, -1.0, -0.75, -0.5]:
        p = run_long_below(df60["gap_z"], df60["intraday_ret"], thr)
        a_results[thr] = p
        print(fmt_perf_row(f"long when gap_z<{thr:+.2f}", p, base=base60))

    # Symmetric fade for completeness
    print("  --- symmetric fade (long if gz<-T, short if gz>+T) ---")
    sym_results = {}
    for thr in [0.5, 0.75, 1.0, 1.5, 2.0]:
        p = run_sym_fade(df60["gap_z"], df60["intraday_ret"], thr)
        sym_results[thr] = p
        print(fmt_perf_row(f"sym_fade |gz|>{thr:+.2f}", p, base=base60))

    # === B. Std window sensitivity (long when gap_z < -1) ===
    print("\n=== B. Std-window sensitivity (long when gap_z < -1) ===")
    b_results = {}
    for win in [30, 60, 120, 252]:
        df_w = make_window_df(tx, win)
        base_w = baseline_perf(df_w)
        p = run_long_below(df_w["gap_z"], df_w["intraday_ret"], -1.0)
        b_results[win] = {"base": base_w, "rule": p, "bars": len(df_w)}
        print(
            f"  std_win={win:3}d  bars={len(df_w):3}  "
            f"baseline_sh={base_w['sh_full']:+.2f}  "
            f"R2_sh={p['sh_full']:+.2f}  R2_sort={p['sortino']:+.2f}  "
            f"R2_cum={p['cum'] * 100:+6.2f}%  R2_mdd={p['mdd'] * 100:+6.2f}%  "
            f"Δsh={p['sh_full'] - base_w['sh_full']:+.2f}"
        )

    # === C. Tail-exclusion (is alpha just z<-2 outliers?) ===
    print("\n=== C. Tail-exclusion (std_win=60d) ===")
    print("  Is R2 alpha driven by z<-2 cell (n=30) or by full z<-1 distribution?")
    c_results = {}
    for label, lo, hi in [
        ("z<-1 ALL (incl. z<-2)", -np.inf, -1.0),
        ("z<-2 only (extreme tail)", -np.inf, -2.0),
        ("-2 ≤ z < -1 only (mid-tail)", -2.0, -1.0),
        ("-1.5 ≤ z < -1 only (near-tail)", -1.5, -1.0),
        ("z<-1.5 only (deeper tail)", -np.inf, -1.5),
    ]:
        p = run_long_band(df60["gap_z"], df60["intraday_ret"], lo, hi)
        c_results[label] = p
        print(fmt_perf_row(label, p, base=base60))

    # === D. Sub-period stability (R2 across thirds) ===
    print("\n=== D. Sub-period stability — R2 (long when gz<-1, std_win=60d) ===")
    n = len(df60)
    cuts = [0, n // 3, 2 * n // 3, n]
    d_results = {}
    intraday60 = df60["intraday_ret"]
    gz60 = df60["gap_z"]
    for i in range(3):
        lo, hi = cuts[i], cuts[i + 1]
        idx = df60.index[lo:hi]
        period_label = f"P{i + 1} {idx.min().date()} → {idx.max().date()}"
        sub_int = intraday60.loc[idx]
        sub_gz = gz60.loc[idx]
        sub_base = baseline_perf(df60.loc[idx])
        sub_r2 = run_long_below(sub_gz, sub_int, -1.0)
        sub_r3 = run_sym_fade(sub_gz, sub_int, 1.0)
        d_results[period_label] = {"baseline": sub_base, "R2": sub_r2, "R3": sub_r3}
        print(f"  {period_label}  ({hi - lo} bars)")
        print(fmt_perf_row("baseline", sub_base))
        print(fmt_perf_row("R2 long gz<-1", sub_r2, base=sub_base))
        print(fmt_perf_row("R3 sym fade |gz|>1", sub_r3, base=sub_base))

    # === Verdict ===
    print("\n=== Verdict ===")
    # A: threshold robustness — count thresholds where R2-style sh > base + 0.3
    robust_a = sum(1 for thr, p in a_results.items() if p["sh_full"] > base60["sh_full"] + 0.3)
    print(f"  A. Threshold robustness: {robust_a}/{len(a_results)} thresholds beat baseline by Δsh>+0.3")

    # B: window robustness
    robust_b = sum(1 for w, r in b_results.items() if r["rule"]["sh_full"] > r["base"]["sh_full"] + 0.3)
    print(f"  B. Std-window robustness: {robust_b}/{len(b_results)} windows beat baseline by Δsh>+0.3")

    # C: is the mid-tail alone profitable?
    mid_tail_p = c_results.get("-2 ≤ z < -1 only (mid-tail)", {})
    extreme_p = c_results.get("z<-2 only (extreme tail)", {})
    print(
        f"  C. Tail decomposition: mid-tail (-2≤z<-1) sh={mid_tail_p.get('sh_full', 0):+.2f}, "
        f"extreme (z<-2) sh={extreme_p.get('sh_full', 0):+.2f}"
    )

    # D: sub-period — count thirds where R2 beats baseline
    robust_d = sum(1 for p, r in d_results.items() if r["R2"]["sh_full"] > r["baseline"]["sh_full"] + 0.3)
    print(f"  D. Sub-period robustness: R2 beats baseline by Δsh>+0.3 in {robust_d}/3 thirds")

    overall = (robust_a >= 4) and (robust_b >= 3) and (robust_d >= 2) and (mid_tail_p.get("sh_full", -99) > 0.5)
    print(
        f"\n  OVERALL: {'PASS — promote R2 to next-stage validation' if overall else 'FAIL or PARTIAL — see breakdown above'}"
    )

    # === Persist ===
    out_path = Path("scripts/output/tx_gap_intraday_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "warmup_bars": WARMUP_BARS,
            "round_trip_cost_bps": ROUND_TRIP_COST * 1e4,
        },
        "A_threshold_sensitivity": {str(k): v for k, v in a_results.items()},
        "A_sym_fade": {str(k): v for k, v in sym_results.items()},
        "B_std_window_sensitivity": {str(k): v for k, v in b_results.items()},
        "C_tail_exclusion": c_results,
        "D_sub_period": d_results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
