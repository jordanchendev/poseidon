#!/usr/bin/env python3
"""TX overnight gap → intraday reaction.

Pure microstructure test on TX 1d bars (no extra data needed).

Hypothesis space:
  Gap_t = (open_t - prev_close_t) / prev_close_t
  Intraday_t = (close_t - open_t) / open_t

  Q1. Does big gap UP fade intraday? (overshoot → mean-revert)
  Q2. Does big gap DOWN rebound intraday? (oversold bounce)
  Q3. Or do gaps continue (momentum)?
  Q4. Is the alpha in long-side, short-side, or both?

Method:
  - Standardise gap by rolling 60d std (point-in-time, no peek)
  - Bin gap_z, report next-intraday return + t-stat per bin
  - Build directional rules (long/short/conditional) and compare vs baseline
  - 252-bar warmup for OOS-style results, 3.2bps round-trip cost

Crucial difference from v1/v2: this produces **conditional long/short/flat**
positions, not just an entry filter on long-only — so if it works it's real
alpha, not smart-beta repackaging.

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_gap_intraday.py
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
GAP_STD_WIN = 60
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


def fmt_perf(name: str, p: dict, base: dict | None = None) -> str:
    base_part = ""
    if base is not None:
        d_sh = p["sh_full"] - base["sh_full"]
        d_sort = p["sortino"] - base["sortino"]
        d_cum = (p["cum"] - base["cum"]) * 100
        d_mdd = (p["mdd"] - base["mdd"]) * 100
        base_part = f"  Δsh={d_sh:+5.2f}  Δsort={d_sort:+5.2f}  Δcum={d_cum:+6.2f}pp  Δmdd={d_mdd:+6.2f}pp"
    return (
        f"  {name:38}  freq={p['freq'] * 100:5.1f}%  n_eng={p['n_engaged']:4}  "
        f"sh={p['sh_full']:+5.2f}  sort={p['sortino']:+5.2f}  "
        f"cum={p['cum'] * 100:+7.2f}%  mdd={p['mdd'] * 100:+6.2f}%  calmar={p['calmar']:+5.2f}{base_part}"
    )


def main():
    repo = RemoteDataRepository.from_settings()
    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    if tx.empty:
        raise RuntimeError("No TX data")

    tx_idx = tx.index.tz_localize(None) if tx.index.tz is not None else tx.index
    tx = tx.copy()
    tx.index = pd.to_datetime(tx_idx).normalize()
    tx = tx[~tx.index.duplicated(keep="last")]

    df = tx[["open", "close"]].rename(columns={"open": "tx_open", "close": "tx_close"}).dropna()
    df["prev_close"] = df["tx_close"].shift(1)
    df["gap"] = (df["tx_open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["tx_close"] - df["tx_open"]) / df["tx_open"]

    # PIT gap standardisation: rolling 60d std (no peek)
    df["gap_std60"] = df["gap"].rolling(GAP_STD_WIN, min_periods=GAP_STD_WIN).std()
    df["gap_z"] = df["gap"] / df["gap_std60"]

    # Drop warmup
    df = df.dropna(subset=["gap_z", "intraday_ret"])
    df = df.iloc[WARMUP_BARS:].copy()

    print(f"Bars: {len(df)}  {df.index.min().date()} → {df.index.max().date()}")
    print(
        f"Gap stats: mean={df['gap'].mean() * 1e4:+.2f}bps  "
        f"std={df['gap'].std() * 1e4:.2f}bps  "
        f"min={df['gap'].min() * 1e4:+.0f}bps  "
        f"max={df['gap'].max() * 1e4:+.0f}bps\n"
    )

    # === Q1-Q3: bin by gap_z, look at intraday return per bin ===
    print("=== Intraday return by gap_z bin (TODAY's gap → TODAY's intraday) ===")
    bins = [-np.inf, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, np.inf]
    labels = ["z<-2", "-2..-1", "-1..-.3", "-.3..+.3", "+.3..+1", "+1..+2", "z>+2"]
    df["gz_bin"] = pd.cut(df["gap_z"], bins=bins, labels=labels)
    grouped = df.groupby("gz_bin", observed=True).agg(
        n=("intraday_ret", "count"),
        intra_mean=("intraday_ret", "mean"),
        intra_t=(
            "intraday_ret",
            lambda s: s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else 0,
        ),
        hit_rate=("intraday_ret", lambda s: float((s > 0).mean())),
    )
    print(
        grouped.to_string(
            formatters={
                "intra_mean": lambda x: f"{x * 1e4:+8.2f}bps",
                "intra_t": lambda x: f"t={x:+5.2f}",
                "hit_rate": lambda x: f"{x * 100:5.1f}%",
            }
        )
    )

    # === Build rule-based strategies ===
    print("\n=== Strategies (3.2bps round-trip per engaged day) ===")
    intraday = df["intraday_ret"]
    gz = df["gap_z"]

    # Baseline: long every day intraday (the OOS Sh 1.05 reference)
    base_pos = pd.Series(1.0, index=df.index)
    base_eng = pd.Series(True, index=df.index)
    base_net = base_pos * intraday - base_eng.astype(float) * ROUND_TRIP_COST
    base_perf = perf(base_net, base_eng)

    rules = {}

    # R1. Fade big gap up: gap_z > +1 → SHORT intraday
    pos = pd.Series(0.0, index=df.index)
    pos[gz > 1.0] = -1.0
    eng = pos != 0
    rules["R1 fade gap_z>+1 (short)"] = (pos, eng)

    # R2. Buy big gap down: gap_z < -1 → LONG intraday
    pos = pd.Series(0.0, index=df.index)
    pos[gz < -1.0] = 1.0
    eng = pos != 0
    rules["R2 buy gap_z<-1 (long)"] = (pos, eng)

    # R3. Symmetric fade: gap_z > +1 short, gap_z < -1 long
    pos = pd.Series(0.0, index=df.index)
    pos[gz > 1.0] = -1.0
    pos[gz < -1.0] = 1.0
    eng = pos != 0
    rules["R3 sym fade |gz|>1"] = (pos, eng)

    # R4. Symmetric momentum (opposite of R3): continuation
    pos = pd.Series(0.0, index=df.index)
    pos[gz > 1.0] = 1.0
    pos[gz < -1.0] = -1.0
    eng = pos != 0
    rules["R4 sym momentum |gz|>1"] = (pos, eng)

    # R5. Quiet-day long only: |gap_z| < 0.3 → LONG (avoid noisy gap days)
    pos = pd.Series(0.0, index=df.index)
    pos[gz.abs() < 0.3] = 1.0
    eng = pos != 0
    rules["R5 long when |gz|<0.3 (quiet)"] = (pos, eng)

    # R6. Stronger thresholds
    pos = pd.Series(0.0, index=df.index)
    pos[gz > 2.0] = -1.0
    pos[gz < -2.0] = 1.0
    eng = pos != 0
    rules["R6 sym fade |gz|>2 (extreme)"] = (pos, eng)

    # R7. One-sided fade up only (per Q1)
    pos = pd.Series(0.0, index=df.index)
    pos[gz > 0.5] = -1.0
    eng = pos != 0
    rules["R7 fade gap_z>+0.5 (short)"] = (pos, eng)

    # R8. Conditional baseline boost: long when gap negative, short when gap positive (mild)
    # tests the broad sign relationship
    pos = pd.Series(0.0, index=df.index)
    pos[gz > 0] = -1.0
    pos[gz < 0] = 1.0
    eng = pos != 0
    rules["R8 fade by gap sign"] = (pos, eng)

    print(fmt_perf("baseline long-every-day intraday", base_perf))
    perf_results = {}
    for name, (pos, eng) in rules.items():
        net = pos * intraday - eng.astype(float) * ROUND_TRIP_COST
        p = perf(net, eng)
        perf_results[name] = p
        print(fmt_perf(name, p, base=base_perf))

    # === Verdict ===
    print("\n=== Verdict ===")
    print(f"  Baseline (long every day):    sh={base_perf['sh_full']:+.2f}, sort={base_perf['sortino']:+.2f}")
    winners = [
        (n, p)
        for n, p in perf_results.items()
        if p["sh_full"] > base_perf["sh_full"] + 0.1 and p["sortino"] > base_perf["sortino"] + 0.1
    ]
    if winners:
        print("  Strategies beating baseline (Δsh > +0.1 AND Δsort > +0.1):")
        for n, p in winners:
            print(f"    {n}  sh={p['sh_full']:+.2f}  sort={p['sortino']:+.2f}  cum={p['cum'] * 100:+.2f}%")
    else:
        print("  NO strategy beats baseline on both Sh AND Sortino by >0.1 → gap-based rules dead at daily resolution")

    # === Persist ===
    out_path = Path("scripts/output/tx_gap_intraday_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "warmup_bars": WARMUP_BARS,
            "gap_std_win": GAP_STD_WIN,
            "round_trip_cost_bps": ROUND_TRIP_COST * 1e4,
        },
        "window": {
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "bars": len(df),
        },
        "bin_summary": {
            label: {
                "n": int(g.loc["n"]) if "n" in g.index else 0,
                "intra_mean_bps": float(g.loc["intra_mean"]) * 1e4 if "intra_mean" in g.index else 0,
                "intra_t": float(g.loc["intra_t"]) if "intra_t" in g.index else 0,
                "hit_rate": float(g.loc["hit_rate"]) if "hit_rate" in g.index else 0,
            }
            for label, g in grouped.iterrows()
        },
        "baseline": base_perf,
        "rules": perf_results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
