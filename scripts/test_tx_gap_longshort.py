#!/usr/bin/env python3
"""TX gap-fade R2 — long-short basis arb version.

Mystery script (test_tx_gap_mystery.py) showed R2 alpha decomposes as:
  TX intraday on R2 days  = +48 bps
  0050 intraday on R2 days = -87 bps
  basis change             = +135 bps
  → +135bps is the pure basis convergence; -87bps cash drag dilutes R2

This script tests the proper market-neutral version: long TX 1 unit notional
+ short 0050 1 unit notional (intraday only, exit at close), to capture pure
basis convergence.

Cost models (round-trip per pair):
  0  bps  — pure alpha, no cost
  5  bps  — TX-only cost (4bps tax + ~1bp commission, no 0050 leg)
  15 bps  — both legs: TX 5bps + 0050 short ~10bps (commission + tax + 융券 fee/day)
  25 bps  — pessimistic: opening auction slippage on both legs

Cost on 0050 short side:
  - TWSE commission ~1.425bps + 8bps trans tax + 8bps 융권 fee per side
  - Round-trip ≈ 25-30bps for retail
  - We model 10bps as best-case institutional access

Decision tree (auto-applied):
  long-short Sh @ 15bps cost ≥ 2.0  → real basis arb → BUILD strategy class
  long-short Sh @ 15bps cost 1.5-2.0 → marginal → BUILD long-only R2 with caveat
  long-short Sh @ 15bps cost < 1.5  → mostly directional → KILL R2 thread

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_gap_longshort.py
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
    return {
        "n_total": n_total,
        "n_engaged": int(engaged.sum()),
        "freq": int(engaged.sum()) / n_total,
        "sh_full": sh,
        "sortino": sortino,
        "cum": cum,
        "mdd": mdd,
        "calmar": cum / abs(mdd) if mdd < 0 else 0.0,
    }


def main():
    repo = RemoteDataRepository.from_settings()

    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    etf = repo.read_ohlcv(symbol="0050", market="tw_stock", interval="1d", start=START, end=END)
    if tx.empty or etf.empty:
        raise RuntimeError("Missing TX or 0050 data")

    for d in (tx, etf):
        idx = d.index.tz_localize(None) if d.index.tz is not None else d.index
        d.index = pd.to_datetime(idx).normalize()

    tx = tx[~tx.index.duplicated(keep="last")][["open", "close"]].rename(
        columns={"open": "tx_open", "close": "tx_close"}
    )
    etf = etf[~etf.index.duplicated(keep="last")][["open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    df = pd.concat([tx, etf], axis=1, join="inner").dropna()

    # Per-leg intraday returns
    df["tx_intraday"] = (df["tx_close"] - df["tx_open"]) / df["tx_open"]
    df["etf_intraday"] = (df["etf_close"] - df["etf_open"]) / df["etf_open"]
    # Pair return (long TX, short 0050, equal notional): tx_ret - etf_ret
    df["pair_intraday"] = df["tx_intraday"] - df["etf_intraday"]

    # TX gap_z signal (R2)
    df["prev_tx_close"] = df["tx_close"].shift(1)
    df["tx_gap"] = (df["tx_open"] - df["prev_tx_close"]) / df["prev_tx_close"]
    df["tx_gap_std"] = df["tx_gap"].rolling(GAP_STD_WIN, min_periods=GAP_STD_WIN).std()
    df["tx_gap_z"] = df["tx_gap"] / df["tx_gap_std"]

    df = df.dropna(subset=["tx_gap_z", "pair_intraday"]).iloc[WARMUP_BARS:].copy()
    print(f"OOS bars: {len(df)}  {df.index.min().date()} → {df.index.max().date()}\n")

    sig = df["tx_gap_z"] < THRESHOLD
    eng = sig.astype(bool)
    n_eng = int(eng.sum())
    print(f"R2 engaged: {n_eng} bars ({n_eng / len(df) * 100:.1f}%)\n")

    # ===================================================================
    # Cost stress on long-only TX vs long-short pair
    # ===================================================================
    print("=" * 80)
    print("R2 long-short basis arb (long TX, short 0050, equal notional, intraday)")
    print("=" * 80)
    print(f"{'cost (RT bps)':<15}  {'strategy':<22}  {'sh':>6}  {'sort':>6}  {'cum%':>7}  {'mdd%':>7}  {'calmar':>6}")

    cost_rows = []
    for cost_bps in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0]:
        cost = cost_bps / 1e4
        # Long-only TX (R2 original): 1 leg cost
        lo_net = df["tx_intraday"].where(eng, 0.0) - eng.astype(float) * cost
        lo_p = perf(lo_net, eng)
        # Long-short pair: 2 legs, cost is double per pair
        ls_net = df["pair_intraday"].where(eng, 0.0) - eng.astype(float) * cost * 2
        ls_p = perf(ls_net, eng)

        cost_rows.append({"cost_bps": cost_bps, "long_only": lo_p, "long_short": ls_p})

        for label, p in [("long TX only (1 leg)", lo_p), ("long TX + short 0050 (2 leg)", ls_p)]:
            print(
                f"{cost_bps:>13.1f}  {label:<22}  "
                f"{p['sh_full']:>+6.2f}  {p['sortino']:>+6.2f}  "
                f"{p['cum'] * 100:>+7.2f}  {p['mdd'] * 100:>+7.2f}  {p['calmar']:>+6.2f}"
            )
        print()

    # ===================================================================
    # Sub-period stability of long-short pair (15bps cost, realistic)
    # ===================================================================
    print("=" * 80)
    print("Long-short stability (15bps RT cost per leg = 30bps round-trip pair)")
    print("=" * 80)
    realistic_cost = 15.0 / 1e4
    pair_net = df["pair_intraday"].where(eng, 0.0) - eng.astype(float) * realistic_cost * 2
    n = len(df)
    cuts = [0, n // 3, 2 * n // 3, n]
    sub_results = {}
    for i in range(3):
        lo, hi = cuts[i], cuts[i + 1]
        idx = df.index[lo:hi]
        sub_net = pair_net.loc[idx]
        sub_eng = eng.loc[idx]
        p = perf(sub_net, sub_eng)
        period_label = f"P{i + 1} {idx.min().date()} → {idx.max().date()}"
        sub_results[period_label] = p
        print(
            f"  {period_label}  ({hi - lo} bars)  "
            f"freq={p['freq'] * 100:5.1f}%  sh={p['sh_full']:+.2f}  "
            f"sort={p['sortino']:+.2f}  cum={p['cum'] * 100:+6.2f}%  mdd={p['mdd'] * 100:+6.2f}%"
        )

    # ===================================================================
    # Auto-verdict (decision tree)
    # ===================================================================
    print("\n" + "=" * 80)
    print("VERDICT (auto-applied decision tree)")
    print("=" * 80)
    realistic_row = next(r for r in cost_rows if r["cost_bps"] == 15.0)
    ls_sh_15 = realistic_row["long_short"]["sh_full"]
    lo_sh_15 = realistic_row["long_only"]["sh_full"]

    print(
        f"  Long-only @ 5bps:  Sh = {next(r for r in cost_rows if r['cost_bps'] == 5.0)['long_only']['sh_full']:+.2f}"
    )
    print(f"  Long-short @ 15bps (per leg, realistic): Sh = {ls_sh_15:+.2f}")
    print(f"  Long-short - long-only (Sh delta):       {ls_sh_15 - lo_sh_15:+.2f}\n")

    if ls_sh_15 >= 2.0:
        verdict = "PROMOTE BOTH — real basis arb"
        action = "Build strategy class for both: (a) long-only R2 retail-friendly, (b) long-short institutional/margin"
    elif ls_sh_15 >= 1.5:
        verdict = "PROMOTE LONG-ONLY — basis arb works but doesn't justify short-leg cost"
        action = (
            "Build long-only R2 strategy class with documented basis-convergence mechanism. Skip long-short for now."
        )
    elif ls_sh_15 >= 0.7:
        verdict = "MARGINAL — basis convergence is real but weak after costs"
        action = "Build long-only R2 with caveat that mechanism is half-hedged basis arb. Monitor live."
    else:
        verdict = "KILL — basis convergence does not survive realistic costs"
        action = "R2's apparent alpha is mostly directional dressing. Mark thread dead and pivot."

    print(f"  Verdict: {verdict}")
    print(f"  Action:  {action}")

    # Persist
    out_path = Path("scripts/output/tx_gap_longshort_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "config": {
                    "warmup_bars": WARMUP_BARS,
                    "gap_std_win": GAP_STD_WIN,
                    "threshold": THRESHOLD,
                    "n_oos_bars": len(df),
                    "n_engaged": n_eng,
                },
                "cost_stress": cost_rows,
                "sub_period_long_short_15bps": sub_results,
                "verdict": verdict,
                "action": action,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
