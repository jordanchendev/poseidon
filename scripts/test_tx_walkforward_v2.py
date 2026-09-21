#!/usr/bin/env python3
"""TX walk-forward v2 — answer two open questions from v1:

(a) Was v1's "filter loses to baseline" finding driven by a too-short
    OOS window (197 bars)? Re-run with WARMUP={252, 504} to compare.

(b) Are the filters actually drawdown-reducers rather than return-enhancers?
    Add Sortino, downside vol, %time-in-DD, max DD duration so we can see
    whether filtering gives you less DD per unit return given up.

Signals (same as v1, point-in-time):
  A. weak-bounce × rolling-252-bar vol p67
  B. basis_z<-1 long, basis built from 0050 RAW close
  C. A | B

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_walkforward_v2.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.research.tx_basis_rule import (
    basis_signal_from_z,
    normalize_taipei_daily_index,
    perf_full,
)

START = datetime(2021, 3, 22)
END = datetime.now()
BARS_PER_YEAR = 240
ROUND_TRIP_COST = 0.00032

VOL_QUANTILE_WIN = 252
VOL_QUANTILE = 0.67
BASIS_WIN = 60
BASIS_THRESHOLD = -1.0
SMA_WIN = 20
VOL_WIN = 20

WARMUP_OPTIONS = [252, 504]  # (a) — compare 1yr vs 2yr warmup → OOS lengths


def fmt_row(name: str, p: dict, base: dict | None = None) -> str:
    base_part = ""
    if base is not None:
        # Δ vs baseline: cum delta, MDD delta (positive = better, less negative)
        d_cum = (p["cum"] - base["cum"]) * 100
        d_mdd = (p["mdd"] - base["mdd"]) * 100  # less negative = improvement (positive Δ)
        d_sh = p["sh_full"] - base["sh_full"]
        d_sortino = p["sortino"] - base["sortino"]
        base_part = f"  Δsh={d_sh:+5.2f}  Δsort={d_sortino:+5.2f}  Δcum={d_cum:+6.2f}pp  Δmdd={d_mdd:+6.2f}pp"
    return (
        f"  {name:30}  freq={p['freq'] * 100:5.1f}%  n_eng={p['n_engaged']:4}  "
        f"sh={p['sh_full']:+5.2f}  sort={p['sortino']:+5.2f}  "
        f"cum={p['cum'] * 100:+7.2f}%  mdd={p['mdd'] * 100:+6.2f}%  "
        f"calmar={p['calmar']:+5.2f}  %dd≥1%={p['pct_time_in_dd_ge_1pct'] * 100:5.1f}%  "
        f"ddDur={p['max_dd_duration_bars']:3}{base_part}"
    )


def build_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df["sma20"] = df["tx_close"].rolling(SMA_WIN).mean()
    df["vol20"] = df["fullday_ret"].rolling(VOL_WIN).std() * np.sqrt(BARS_PER_YEAR)
    df["vol_p67_rolling"] = df["vol20"].rolling(VOL_QUANTILE_WIN, min_periods=VOL_QUANTILE_WIN).quantile(VOL_QUANTILE)
    df["basis_raw"] = np.log(df["tx_close"]) - np.log(df["etf_close_raw"])
    df["basis_ma60"] = df["basis_raw"].rolling(BASIS_WIN).mean()
    df["basis_std60"] = df["basis_raw"].rolling(BASIS_WIN).std()
    df["basis_z"] = (df["basis_raw"] - df["basis_ma60"]) / df["basis_std60"]
    yest_close = df["tx_close"].shift(1)
    yest_sma20 = df["sma20"].shift(1)
    yest_vol20 = df["vol20"].shift(1)
    yest_vol_p67 = df["vol_p67_rolling"].shift(1)
    sig_A = (yest_close < yest_sma20) & (yest_vol20 > yest_vol_p67)
    sig_B = basis_signal_from_z(df["basis_z"])
    sig_C = sig_A | sig_B
    return sig_A, sig_B, sig_C


def main():
    repo = RemoteDataRepository.from_settings()
    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    tw0050 = repo.read_ohlcv(symbol="0050", market="tw_stock", interval="1d", start=START, end=END)
    if tx.empty or tw0050.empty:
        raise RuntimeError("Missing data")

    tx = normalize_taipei_daily_index(tx)
    tw0050 = normalize_taipei_daily_index(tw0050)

    df = pd.DataFrame(
        {
            "tx_open": tx["open"],
            "tx_close": tx["close"],
            "etf_close_raw": tw0050["close"],
        }
    ).dropna()
    df["prev_tx_close"] = df["tx_close"].shift(1)
    df["intraday_ret"] = (df["tx_close"] - df["tx_open"]) / df["tx_open"]
    df["fullday_ret"] = (df["tx_close"] - df["prev_tx_close"]) / df["prev_tx_close"]

    print(f"Merged: {len(df)} bars  {df.index.min().date()} → {df.index.max().date()}")
    print("Source: 0050 RAW close (no dividend reinvestment leakage)\n")

    sig_A, sig_B, sig_C = build_signals(df)
    valid = sig_A.notna() & sig_B.notna() & df["intraday_ret"].notna()
    df_valid = df.loc[valid]
    sig_A_v = sig_A.loc[df_valid.index]
    sig_B_v = sig_B.loc[df_valid.index]
    sig_C_v = sig_C.loc[df_valid.index]

    print(f"First bar with all signal inputs: {df_valid.index.min().date()}  (total valid bars: {len(df_valid)})\n")

    all_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "vol_quantile_win": VOL_QUANTILE_WIN,
            "vol_quantile": VOL_QUANTILE,
            "basis_win": BASIS_WIN,
            "basis_threshold": BASIS_THRESHOLD,
            "round_trip_cost_bps": ROUND_TRIP_COST * 1e4,
            "etf_price_source": "raw close",
        },
        "runs": {},
    }
    canonical_b_returns: pd.Series | None = None

    for warmup in WARMUP_OPTIONS:
        if len(df_valid) <= warmup + 60:
            print(f"[skip] warmup={warmup}: not enough bars ({len(df_valid)})")
            continue
        oos = df_valid.iloc[warmup:].copy()
        sig_A_oos = sig_A_v.iloc[warmup:].fillna(False)
        sig_B_oos = sig_B_v.iloc[warmup:].fillna(False)
        sig_C_oos = sig_C_v.iloc[warmup:].fillna(False)
        intraday = oos["intraday_ret"]

        print(f"=== warmup={warmup} bars ===")
        print(f"  OOS window: {oos.index.min().date()} → {oos.index.max().date()}  ({len(oos)} bars)")

        # Baseline first to get reference
        base_eng = pd.Series(True, index=oos.index)
        base_net = intraday - ROUND_TRIP_COST
        base_perf = perf_full(base_net, base_eng)

        # Strategies
        nets = {}
        perfs = {}
        for name, sig in [
            ("A weak-bounce × hi-vol", sig_A_oos),
            ("B basis_z<-1 long", sig_B_oos),
            ("C stack (A | B)", sig_C_oos),
        ]:
            engaged = sig.astype(bool)
            net = intraday.where(engaged, 0.0) - engaged.astype(float) * ROUND_TRIP_COST
            nets[name] = (net, engaged)
            perfs[name] = perf_full(net, engaged)
            if warmup == 252 and name == "B basis_z<-1 long":
                canonical_b_returns = net.rename("return")

        print("  Header: sh=Sharpe  sort=Sortino  Δ vs baseline (Δsh, Δsort, Δcum pp, Δmdd pp)")
        print(fmt_row("baseline every-day intraday", base_perf))
        for name, p in perfs.items():
            print(fmt_row(name, p, base=base_perf))

        # (b) drawdown-control efficiency: dd improvement per cum return given up
        print("  --- Drawdown-control read (b) ---")
        for name, p in perfs.items():
            d_cum_pp = (p["cum"] - base_perf["cum"]) * 100  # negative usually (you give up return)
            d_mdd_pp = (p["mdd"] - base_perf["mdd"]) * 100  # positive = less negative MDD
            ratio = (d_mdd_pp / -d_cum_pp) if d_cum_pp < 0 else float("inf")
            note = (
                "good DD control"
                if d_cum_pp < 0 and d_mdd_pp > 0 and ratio >= 1.0
                else "weak: less return AND not better DD"
                if d_cum_pp < 0 and d_mdd_pp <= 0
                else "filter beats baseline outright"
                if d_cum_pp >= 0
                else "tradeoff dominated"
            )
            print(
                f"  {name:30}  Δcum={d_cum_pp:+6.2f}pp  Δmdd={d_mdd_pp:+6.2f}pp  "
                f"DDsave/RetGiveup={ratio:+5.2f}  → {note}"
            )

        all_payload["runs"][f"warmup_{warmup}"] = {
            "oos_start": str(oos.index.min().date()),
            "oos_end": str(oos.index.max().date()),
            "oos_bars": len(oos),
            "baseline": base_perf,
            "strategies": perfs,
        }
        print()

    out_path = Path("scripts/output/tx_walkforward_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_payload, f, indent=2, default=str)
    if canonical_b_returns is None:
        raise RuntimeError("warmup_252 canonical B returns were not produced")
    canonical_b_returns.to_frame().to_parquet(out_path.with_name("tx_walkforward_v2_basis_b_returns.parquet"))
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
