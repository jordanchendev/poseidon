#!/usr/bin/env python3
"""TX/0050 basis + vol regime thesis test.

Two sub-theses:

  C1. **Basis mean-reversion:** TX (futures) and 0050 (TAIEX ETF) should
      track tightly. Define basis_dev = log(TX) - log(0050_adj) minus its
      60d rolling mean. When basis_dev is extreme (futures rich vs spot,
      or futures cheap vs spot), does next-day TX move predictably?

      Test: bin days by basis_dev z-score, report next-day intraday return
      and full-day return per bin. Look for monotonic pattern — extreme
      positive basis → next-day TX underperformance, etc.

  C2. **Vol regime conditioning:** Compute 20d realized vol of TX. Classify
      days as low/mid/high vol regime (33/67th percentile cuts). Re-test the
      winning intraday rule from test_tx_intraday_costed.py
      (mean-rev: yest_close < yest_sma20) within each regime. Does the
      alpha concentrate in one regime?

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_basis_vol.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.research.tx_basis_signal import BASIS_WIN, compute_basis_z

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START = datetime(2021, 3, 22)
END = datetime.now()
BARS_PER_YEAR = 240
ROUND_TRIP_COST = 0.00032


def annualised_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0 or daily_returns.empty:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(BARS_PER_YEAR))


def fmt_bps(x: float) -> str:
    return f"{x * 1e4:+7.2f}bps"


def main():
    repo = RemoteDataRepository.from_settings()

    # --- Fetch TX and 0050 ---
    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    if tx.empty:
        raise RuntimeError("No TX data")
    tw0050 = repo.read_ohlcv(symbol="0050", market="tw_stock", interval="1d", start=START, end=END)
    if tw0050.empty:
        raise RuntimeError("No 0050 data")

    print(f"TX: {len(tx)} bars  {tx.index.min().date()} → {tx.index.max().date()}")
    print(f"0050: {len(tw0050)} bars  {tw0050.index.min().date()} → {tw0050.index.max().date()}")
    print(f"0050 columns: {list(tw0050.columns)}")

    adj_col = "adj_close" if "adj_close" in tw0050.columns else "close"
    print(f"Using 0050 column: {adj_col}\n")

    # --- Align on date (both TZ-naive) ---
    tx_idx = tx.index.tz_localize(None) if tx.index.tz is not None else tx.index
    e_idx = tw0050.index.tz_localize(None) if tw0050.index.tz is not None else tw0050.index
    tx = tx.copy()
    tw0050 = tw0050.copy()
    tx.index = tx_idx
    tw0050.index = e_idx
    tx.index = pd.to_datetime(tx.index).normalize()
    tw0050.index = pd.to_datetime(tw0050.index).normalize()
    # Dedupe (defensive: if any duplicate dates after normalize, keep last)
    tx = tx[~tx.index.duplicated(keep="last")]
    tw0050 = tw0050[~tw0050.index.duplicated(keep="last")]

    # Merge
    df = pd.DataFrame(
        {
            "tx_close": tx["close"],
            "tx_open": tx["open"],
            "etf_adj": tw0050[adj_col],
        }
    ).dropna()

    print(f"Merged: {len(df)} bars  {df.index.min().date()} → {df.index.max().date()}\n")

    # D-06: trigger-day math comes from the shared module.
    # `compute_basis_z` does its own date-index alignment and dedupe; we
    # re-derive the intermediate columns here only for diagnostics
    # (basis / basis_dev / basis_std60 still printed in the sanity block
    # below). The authoritative `basis_z` is what `compute_basis_z` returns.
    df["log_tx"] = np.log(df["tx_close"])
    df["log_etf"] = np.log(df["etf_adj"])
    df["basis"] = df["log_tx"] - df["log_etf"]
    df["basis_ma60"] = df["basis"].rolling(BASIS_WIN).mean()
    df["basis_dev"] = df["basis"] - df["basis_ma60"]
    df["basis_std60"] = df["basis"].rolling(BASIS_WIN).std()
    df["basis_z"] = compute_basis_z(tx, tw0050, adj_col=adj_col).reindex(df.index)

    # Daily returns on TX
    df["prev_tx_close"] = df["tx_close"].shift(1)
    df["overnight_ret"] = (df["tx_open"] - df["prev_tx_close"]) / df["prev_tx_close"]
    df["intraday_ret"] = (df["tx_close"] - df["tx_open"]) / df["tx_open"]
    df["fullday_ret"] = (df["tx_close"] - df["prev_tx_close"]) / df["prev_tx_close"]

    df["sma20"] = df["tx_close"].rolling(20).mean()
    df["yest_close"] = df["tx_close"].shift(1)
    df["yest_sma20"] = df["sma20"].shift(1)
    df["realized_vol20"] = df["fullday_ret"].rolling(20).std() * np.sqrt(BARS_PER_YEAR)

    df = df.dropna(subset=["basis_z", "intraday_ret", "fullday_ret", "yest_sma20", "realized_vol20"])

    print("=== Basis sanity ===")
    print(
        f"  basis (log TX - log 0050_adj):  mean={df['basis'].mean():+.4f}  "
        f"std={df['basis'].std():.4f}  min={df['basis'].min():+.4f}  max={df['basis'].max():+.4f}"
    )
    print(f"  basis_dev (vs 60d MA):          mean={df['basis_dev'].mean():+.6f}  std={df['basis_dev'].std():.4f}")
    print(f"  basis_z:                        std={df['basis_z'].std():.3f}  (should be ~1)")

    # === C1. Basis mean-reversion: bin by basis_z, look at NEXT-day TX moves ===
    print("\n=== C1. Next-day TX returns by basis_z bin ===")
    df["next_intraday"] = df["intraday_ret"].shift(-1)
    df["next_overnight"] = df["overnight_ret"].shift(-1)
    df["next_fullday"] = df["fullday_ret"].shift(-1)

    bins = [-np.inf, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, np.inf]
    labels = ["z<-2", "-2..-1", "-1..-.3", "-.3..+.3", "+.3..+1", "+1..+2", "z>+2"]
    df["bz_bin"] = pd.cut(df["basis_z"], bins=bins, labels=labels)

    grouped = (
        df.dropna(subset=["next_intraday", "next_fullday"])
        .groupby("bz_bin", observed=True)
        .agg(
            n=("next_intraday", "count"),
            next_intra_mean=("next_intraday", "mean"),
            next_intra_t=(
                "next_intraday",
                lambda s: s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else 0,
            ),
            next_full_mean=("next_fullday", "mean"),
            next_on_mean=("next_overnight", "mean"),
        )
    )
    print(
        grouped.to_string(
            formatters={
                "next_intra_mean": lambda x: f"{x * 1e4:+7.2f}bps",
                "next_intra_t": lambda x: f"t={x:+5.2f}",
                "next_full_mean": lambda x: f"{x * 1e4:+7.2f}bps",
                "next_on_mean": lambda x: f"{x * 1e4:+7.2f}bps",
            }
        )
    )

    # === C1b. Simple basis mean-rev strategy ===
    # When basis_z > +1: next day fade (short next intraday) — futures rich
    # When basis_z < -1: next day catch-up (long next intraday) — futures cheap
    print("\n=== C1b. Basis mean-rev rule on next-day intraday ===")
    next_intra = df["next_intraday"]
    rules = {
        "long when basis_z<-1, short when basis_z>+1": np.where(
            df["basis_z"] < -1,
            next_intra,
            np.where(df["basis_z"] > 1, -next_intra, np.nan),
        ),
        "long when basis_z<-1.5, short when basis_z>+1.5": np.where(
            df["basis_z"] < -1.5,
            next_intra,
            np.where(df["basis_z"] > 1.5, -next_intra, np.nan),
        ),
        "long when basis_z<-1 (one-sided)": np.where(df["basis_z"] < -1, next_intra, np.nan),
        "short when basis_z>+1 (one-sided)": np.where(df["basis_z"] > 1, -next_intra, np.nan),
    }
    for name, arr in rules.items():
        s = pd.Series(arr, index=df.index).dropna()
        n = len(s)
        if n < 20:
            continue
        gross = s
        net = s - ROUND_TRIP_COST
        sh_gross = annualised_sharpe(gross)
        sh_net = annualised_sharpe(net)
        cum_net = (1.0 + net).prod() - 1.0
        # Sharpe on full-series basis (zero on non-engaged) for fair compare
        full = pd.Series(0.0, index=df.index)
        full.loc[s.index] = net
        sh_net_full = annualised_sharpe(full)
        print(
            f"  {name:60}  n={n:4}  gross_sh={sh_gross:+5.2f}  net_sh={sh_net:+5.2f}  "
            f"net_full={sh_net_full:+5.2f}  cum_net={cum_net * 100:+6.2f}%"
        )

    # === C2. Vol regime conditioning of mean-rev rule ===
    print("\n=== C2. Mean-rev (yest<sma20) intraday rule by vol regime ===")
    df["vol_pct"] = df["realized_vol20"].rank(pct=True)
    df["vol_regime"] = pd.cut(df["vol_pct"], bins=[0, 0.33, 0.67, 1.01], labels=["low", "mid", "high"])
    mr_mask = df["yest_close"] < df["yest_sma20"]
    print(
        f"  Vol regime breakpoints (annualised): "
        f"33%={df['realized_vol20'].quantile(0.33) * 100:.1f}%  "
        f"67%={df['realized_vol20'].quantile(0.67) * 100:.1f}%"
    )
    for regime in ["low", "mid", "high"]:
        regime_mask = df["vol_regime"] == regime
        mask = regime_mask & mr_mask
        n = int(mask.sum())
        if n < 20:
            continue
        engaged = df["intraday_ret"].where(mask, 0.0).fillna(0.0)
        net = engaged - mask.astype(float) * ROUND_TRIP_COST
        sh = annualised_sharpe(net)
        cum = (1.0 + net).prod() - 1.0
        eq = (1.0 + net).cumprod()
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        mdd = float(dd.min())
        gross_engaged = df.loc[mask, "intraday_ret"].mean()
        print(
            f"  vol={regime:5}  engaged={n:4}  "
            f"avg_intra_engaged={fmt_bps(gross_engaged)}  "
            f"net_sh_full={sh:+5.2f}  net_cum={cum * 100:+6.2f}%  mdd={mdd * 100:6.2f}%"
        )

    # === C2b. Pure vol regime check on baseline intraday ===
    print("\n=== C2b. Baseline intraday (every day) by vol regime ===")
    for regime in ["low", "mid", "high"]:
        mask = df["vol_regime"] == regime
        n = int(mask.sum())
        if n < 20:
            continue
        engaged = df["intraday_ret"].where(mask, 0.0).fillna(0.0)
        net = engaged - mask.astype(float) * ROUND_TRIP_COST
        sh = annualised_sharpe(net)
        cum = (1.0 + net).prod() - 1.0
        avg = df.loc[mask, "intraday_ret"].mean()
        print(
            f"  vol={regime:5}  engaged={n:4}  "
            f"avg_intra={fmt_bps(avg)}  "
            f"net_sh_full={sh:+5.2f}  net_cum={cum * 100:+6.2f}%"
        )

    # === C3. Combine: mean-rev rule + only-low-vol or only-high-vol ===
    print("\n=== C3. Stacked filters: mean-rev × vol regime ===")
    for regime, regime_label in [
        ("low", "vol=low"),
        ("high", "vol=high"),
        ("mid_high", "vol=mid+high"),
        ("low_mid", "vol=low+mid"),
    ]:
        if regime == "mid_high":
            regime_mask = df["vol_regime"].isin(["mid", "high"])
        elif regime == "low_mid":
            regime_mask = df["vol_regime"].isin(["low", "mid"])
        else:
            regime_mask = df["vol_regime"] == regime
        mask = regime_mask & mr_mask
        n = int(mask.sum())
        if n < 20:
            continue
        engaged = df["intraday_ret"].where(mask, 0.0).fillna(0.0)
        net = engaged - mask.astype(float) * ROUND_TRIP_COST
        sh = annualised_sharpe(net)
        cum = (1.0 + net).prod() - 1.0
        eq = (1.0 + net).cumprod()
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        mdd = float(dd.min())
        calmar = cum / abs(mdd) if mdd < 0 else float("inf")
        freq = n / len(df)
        print(
            f"  mean-rev & {regime_label:13}  freq={freq * 100:5.1f}%  n={n:4}  "
            f"net_sh={sh:+5.2f}  cum={cum * 100:+6.2f}%  mdd={mdd * 100:6.2f}%  calmar={calmar:+5.2f}"
        )

    out_path = Path("scripts/output/tx_basis_vol_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "bars": len(df),
                "basis_stats": {
                    "mean": float(df["basis"].mean()),
                    "std": float(df["basis"].std()),
                },
                "vol_regime_breakpoints_pct": {
                    "p33": float(df["realized_vol20"].quantile(0.33)),
                    "p67": float(df["realized_vol20"].quantile(0.67)),
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
