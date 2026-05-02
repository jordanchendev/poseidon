#!/usr/bin/env python3
"""TX intraday alpha — walk-forward / out-of-sample validation.

In-sample test (test_tx_basis_vol.py) found two candidate signals on TX 1d:
  A. weak-bounce × high-vol regime    (yest<sma20 AND vol20>p67)  — net Sh 1.23
  B. basis_z<-1 → long next intraday                              — engaged Sh 1.47

Both used full-sample fit hyperparameters (vol p67, basis 60d window with
adj_close-leaked dividend drift). This script makes the signals point-in-time
clean and reports OOS performance:

  A. vol_p67 replaced with **rolling 252-bar 67th-percentile** of past
     vol20 (pure point-in-time, no peek).
  B. basis built from **raw 0050_close** (not adj_close), removing the
     ~3-4%/yr structural drift from dividend reinvestment that contaminated
     the basis_z distribution. 60d rolling z-score is already PIT-clean.
  C. A OR B (engage if either signal fires).

Decision (signal at t uses info available at t-1):
    A_t = (yest_close_{t} < yest_sma20_{t}) & (vol20_{t-1} > vol_p67_rolling_{t-1})
    B_t = (basis_z_{t-1} < -1.0)
    C_t = A_t | B_t

Position: long today's TX intraday (open→close) if signal fires, else flat.
Cost: 3.2 bps round-trip per engaged day.

Warmup: 504 bars (~2yr) skipped before OOS measurement so rolling stats
have stabilised history.

Pass/fail per session protocol:
    OOS Sh ≥ 0.7  → real alpha, promote to strategy class
    OOS Sh 0.3-0.7 → partial alpha, tune & re-test
    OOS Sh < 0.3  → in-sample overfit, find new direction

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_walkforward.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.data.remote_repository import RemoteDataRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START = datetime(2021, 3, 22)
END = datetime.now()
BARS_PER_YEAR = 240
ROUND_TRIP_COST = 0.00032

WARMUP_BARS = 504  # ~2yr — must precede OOS window
VOL_QUANTILE_WIN = 252  # rolling 1yr to estimate vol p67 PIT
VOL_QUANTILE = 0.67
BASIS_WIN = 60  # 60d rolling for basis z-score
BASIS_THRESHOLD = -1.0
SMA_WIN = 20
VOL_WIN = 20


def annualised_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(BARS_PER_YEAR))


def perf_block(net: pd.Series, engaged: pd.Series) -> dict:
    """Compute Sharpe / cum / MDD / Calmar / freq for a net-return series."""
    eq = (1.0 + net).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    cum = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    sh_full = annualised_sharpe(net)
    n_engaged = int(engaged.sum())
    n_total = len(net)
    freq = n_engaged / n_total if n_total else 0.0
    # Sharpe over engaged bars only (more comparable when engagement is sparse)
    engaged_net = net[engaged.astype(bool)]
    sh_eng = annualised_sharpe(engaged_net) if len(engaged_net) > 1 else 0.0
    calmar = cum / abs(mdd) if mdd < 0 else float("inf") if cum > 0 else 0.0
    return {
        "n_total": n_total,
        "n_engaged": n_engaged,
        "freq": freq,
        "sh_full": sh_full,
        "sh_engaged_only": sh_eng,
        "cum": cum,
        "mdd": mdd,
        "calmar": calmar,
    }


def fmt_perf(name: str, p: dict) -> str:
    return (
        f"  {name:30}  freq={p['freq'] * 100:5.1f}%  n_eng={p['n_engaged']:4}  "
        f"sh_full={p['sh_full']:+5.2f}  sh_eng={p['sh_engaged_only']:+5.2f}  "
        f"cum={p['cum'] * 100:+7.2f}%  mdd={p['mdd'] * 100:+6.2f}%  calmar={p['calmar']:+5.2f}"
    )


def main():
    repo = RemoteDataRepository.from_settings()

    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    if tx.empty:
        raise RuntimeError("No TX data")
    tw0050 = repo.read_ohlcv(symbol="0050", market="tw_stock", interval="1d", start=START, end=END)
    if tw0050.empty:
        raise RuntimeError("No 0050 data")

    # Use RAW close for 0050 (NOT adj_close) — adj_close contaminates basis with
    # cumulative dividend reinvestment, ~3-4%/yr structural drift.
    if "close" not in tw0050.columns:
        raise RuntimeError(f"0050 missing 'close' column. Have: {list(tw0050.columns)}")

    print(f"TX:   {len(tx)} bars  {tx.index.min().date()} → {tx.index.max().date()}")
    print(f"0050: {len(tw0050)} bars  {tw0050.index.min().date()} → {tw0050.index.max().date()}")
    print("Using 0050 RAW close (not adj_close) — basis built from undividended price.\n")

    # Align (both TZ-naive, normalised to date)
    tx_idx = tx.index.tz_localize(None) if tx.index.tz is not None else tx.index
    e_idx = tw0050.index.tz_localize(None) if tw0050.index.tz is not None else tw0050.index
    tx = tx.copy()
    tw0050 = tw0050.copy()
    tx.index = pd.to_datetime(tx_idx).normalize()
    tw0050.index = pd.to_datetime(e_idx).normalize()
    tx = tx[~tx.index.duplicated(keep="last")]
    tw0050 = tw0050[~tw0050.index.duplicated(keep="last")]

    df = pd.DataFrame(
        {
            "tx_open": tx["open"],
            "tx_close": tx["close"],
            "etf_close_raw": tw0050["close"],
        }
    ).dropna()

    print(f"Merged: {len(df)} bars  {df.index.min().date()} → {df.index.max().date()}\n")

    # === Returns ===
    df["prev_tx_close"] = df["tx_close"].shift(1)
    df["intraday_ret"] = (df["tx_close"] - df["tx_open"]) / df["tx_open"]
    df["fullday_ret"] = (df["tx_close"] - df["prev_tx_close"]) / df["prev_tx_close"]

    # === Signal A inputs (point-in-time) ===
    df["sma20"] = df["tx_close"].rolling(SMA_WIN).mean()
    df["vol20"] = df["fullday_ret"].rolling(VOL_WIN).std() * np.sqrt(BARS_PER_YEAR)

    # PIT vol p67: at bar t, use rolling 252-bar quantile over vol20[t-251..t].
    # We then SHIFT by 1 so the bar's signal uses only past info.
    df["vol_p67_rolling"] = df["vol20"].rolling(VOL_QUANTILE_WIN, min_periods=VOL_QUANTILE_WIN).quantile(VOL_QUANTILE)

    # === Signal B inputs (point-in-time) ===
    df["basis_raw"] = np.log(df["tx_close"]) - np.log(df["etf_close_raw"])
    df["basis_ma60"] = df["basis_raw"].rolling(BASIS_WIN).mean()
    df["basis_std60"] = df["basis_raw"].rolling(BASIS_WIN).std()
    df["basis_z"] = (df["basis_raw"] - df["basis_ma60"]) / df["basis_std60"]

    # === Build signals using yesterday's info only ===
    yest_close = df["tx_close"].shift(1)
    yest_sma20 = df["sma20"].shift(1)
    yest_vol20 = df["vol20"].shift(1)
    yest_vol_p67 = df["vol_p67_rolling"].shift(1)
    yest_basis_z = df["basis_z"].shift(1)

    sig_A = (yest_close < yest_sma20) & (yest_vol20 > yest_vol_p67)
    sig_B = yest_basis_z < BASIS_THRESHOLD
    sig_C = sig_A | sig_B

    # Drop rows where any inputs missing (warmup) and apply WARMUP_BARS skip
    valid_mask = (
        sig_A.notna() & sig_B.notna() & df["intraday_ret"].notna() & yest_vol_p67.notna() & yest_basis_z.notna()
    )
    first_valid = valid_mask.idxmax() if valid_mask.any() else None
    print(f"First bar with all signal inputs: {first_valid.date() if first_valid is not None else 'NONE'}")

    df_oos = df.loc[valid_mask].iloc[WARMUP_BARS:].copy() if valid_mask.any() else df.iloc[0:0]
    if len(df_oos) < 60:
        raise RuntimeError(
            f"Not enough OOS bars after warmup: {len(df_oos)}. Need ≥60. Total valid bars: {int(valid_mask.sum())}."
        )

    sig_A_oos = sig_A.loc[df_oos.index].fillna(False)
    sig_B_oos = sig_B.loc[df_oos.index].fillna(False)
    sig_C_oos = sig_C.loc[df_oos.index].fillna(False)
    intraday_oos = df_oos["intraday_ret"]

    print(
        f"\n=== OOS window ===\n"
        f"  {df_oos.index.min().date()} → {df_oos.index.max().date()}  "
        f"({len(df_oos)} bars after {WARMUP_BARS}-bar warmup)\n"
    )

    # === Strategy returns (long-only when engaged) ===
    def strategy_net(sig: pd.Series) -> tuple[pd.Series, pd.Series]:
        engaged = sig.astype(bool)
        gross = intraday_oos.where(engaged, 0.0)
        cost = engaged.astype(float) * ROUND_TRIP_COST
        net = gross - cost
        return net, engaged

    nets = {}
    perfs = {}
    for name, sig in [
        ("A weak-bounce × hi-vol", sig_A_oos),
        ("B basis_z<-1 long", sig_B_oos),
        ("C stack (A | B)", sig_C_oos),
    ]:
        net, engaged = strategy_net(sig)
        nets[name] = net
        perfs[name] = perf_block(net, engaged)
        print(fmt_perf(name, perfs[name]))

    # Baseline: every-day intraday (no filter)
    base_engaged = pd.Series(True, index=df_oos.index)
    base_net = intraday_oos - ROUND_TRIP_COST
    perfs["baseline every-day intraday"] = perf_block(base_net, base_engaged)
    print(fmt_perf("baseline every-day intraday", perfs["baseline every-day intraday"]))

    # === Sub-period stability: split OOS into 3 thirds ===
    print("\n=== OOS sub-period stability (thirds) ===")
    n = len(df_oos)
    cuts = [0, n // 3, 2 * n // 3, n]
    sub_results = {}
    for i in range(3):
        lo, hi = cuts[i], cuts[i + 1]
        idx = df_oos.index[lo:hi]
        period_label = f"P{i + 1} {idx.min().date()} → {idx.max().date()}"
        print(f"  {period_label}  ({hi - lo} bars)")
        sub_results[period_label] = {}
        sig_lookup = {
            "A weak-bounce × hi-vol": sig_A_oos,
            "B basis_z<-1 long": sig_B_oos,
            "C stack (A | B)": sig_C_oos,
        }
        for name, net in nets.items():
            sub_net = net.loc[idx]
            sub_eng = sig_lookup[name].loc[idx].astype(bool)
            p = perf_block(sub_net, sub_eng)
            sub_results[period_label][name] = p
            print(fmt_perf(name, p))

    # === Pass/fail verdict ===
    print("\n=== Verdict (per session protocol) ===")

    def verdict(sh: float) -> str:
        if sh >= 0.7:
            return "PASS — real alpha, promote to strategy class"
        if sh >= 0.3:
            return "PARTIAL — tune & re-test"
        return "FAIL — in-sample overfit"

    for name, p in perfs.items():
        if name == "baseline every-day intraday":
            continue
        print(f"  {name:30}  OOS sh_full={p['sh_full']:+5.2f}  →  {verdict(p['sh_full'])}")

    # === Persist JSON ===
    out_path = Path("scripts/output/tx_walkforward_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "warmup_bars": WARMUP_BARS,
            "vol_quantile_win": VOL_QUANTILE_WIN,
            "vol_quantile": VOL_QUANTILE,
            "basis_win": BASIS_WIN,
            "basis_threshold": BASIS_THRESHOLD,
            "round_trip_cost_bps": ROUND_TRIP_COST * 1e4,
            "etf_price_source": "raw close (not adj_close)",
        },
        "oos_window": {
            "start": str(df_oos.index.min().date()),
            "end": str(df_oos.index.max().date()),
            "bars": len(df_oos),
        },
        "overall": {k: v for k, v in perfs.items()},
        "sub_periods": sub_results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
