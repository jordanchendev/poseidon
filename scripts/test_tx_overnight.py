#!/usr/bin/env python3
"""TX overnight effect thesis test.

For TXF daily bars (FinLab "TX一般", Asia/Taipei 08:45–13:45):
    overnight_return = (today_open - prev_close) / prev_close
    intraday_return  = (today_close - today_open) / today_open
    full_day_return  = (today_close - prev_close) / prev_close

Thesis: in US equities, all the long-term equity premium comes from holding
overnight (close → next open). Intraday is roughly flat. Test whether TX shows
the same pattern, and whether overnight-only holding has higher Sharpe than
buy-and-hold.

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_overnight.py
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
BARS_PER_YEAR = 240  # TW trading days


def annualised_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0 or daily_returns.empty:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(BARS_PER_YEAR))


def summarise(name: str, returns: pd.Series) -> dict:
    cum = (1.0 + returns).prod() - 1.0
    mean_d = returns.mean()
    std_d = returns.std()
    sharpe = annualised_sharpe(returns)
    win_rate = float((returns > 0).mean())
    n = int(returns.notna().sum())

    # Compute max drawdown of cumulative equity curve
    eq = (1.0 + returns.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())

    print(
        f"  {name:18}  n={n:4}  cum={cum * 100:7.2f}%  "
        f"mean/d={mean_d * 1e4:6.2f}bps  std/d={std_d * 100:5.3f}%  "
        f"sharpe={sharpe:5.2f}  win={win_rate * 100:5.1f}%  maxdd={max_dd * 100:6.2f}%"
    )
    return {
        "name": name,
        "n": n,
        "cum_return": cum,
        "mean_per_day": mean_d,
        "std_per_day": std_d,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_dd": max_dd,
    }


def main():
    repo = RemoteDataRepository.from_settings()
    df = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    if df.empty:
        raise RuntimeError("No TX data")

    # Ensure sorted by time, drop NaNs
    df = df.sort_index().dropna(subset=["open", "high", "low", "close"]).copy()

    # Decompose returns
    df["prev_close"] = df["close"].shift(1)
    df["overnight_ret"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    df["fullday_ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    df = df.dropna(subset=["overnight_ret", "intraday_ret", "fullday_ret"])

    print(f"\nTX {df.index.min().date()} → {df.index.max().date()}, {len(df)} bars\n")

    print("=== Decomposition (long-only, no costs) ===")
    res = {}
    res["fullday"] = summarise("buy-and-hold", df["fullday_ret"])
    res["overnight"] = summarise("overnight-only", df["overnight_ret"])
    res["intraday"] = summarise("intraday-only", df["intraday_ret"])

    # Sanity check: overnight + intraday + cross-term ≈ fullday
    df["sum_check"] = (1 + df["overnight_ret"]) * (1 + df["intraday_ret"]) - 1
    diff = (df["sum_check"] - df["fullday_ret"]).abs().max()
    print(f"\n  sanity: max|reconstruct - fullday| = {diff:.2e}  (should be ~0)")

    # Statistical significance: is overnight mean different from zero?
    from scipy import stats

    on = df["overnight_ret"].dropna()
    intr = df["intraday_ret"].dropna()
    t_on, p_on = stats.ttest_1samp(on, 0)
    t_in, p_in = stats.ttest_1samp(intr, 0)
    print("\n=== t-test: mean per-day return ≠ 0 ===")
    print(f"  overnight: t={t_on:.3f}  p={p_on:.4f}")
    print(f"  intraday:  t={t_in:.3f}  p={p_in:.4f}")

    # Same effect over different sub-periods
    print("\n=== Sub-period sharpe (overnight) ===")
    for label, mask in [
        ("2021-2022", (df.index < "2023-01-01")),
        ("2023-2024", (df.index >= "2023-01-01") & (df.index < "2025-01-01")),
        ("2025-2026", (df.index >= "2025-01-01")),
    ]:
        sub = df.loc[mask]
        if len(sub) > 20:
            print(f"  {label}  ({len(sub):3} bars):", end="")
            for col in ("fullday_ret", "overnight_ret", "intraday_ret"):
                s = annualised_sharpe(sub[col].dropna())
                print(f"  {col[:8]}={s:+.2f}", end="")
            print()

    # Conditional: does prior overnight_ret sign predict next overnight_ret?
    df["prev_on"] = df["overnight_ret"].shift(1)
    print("\n=== Cond: today overnight | yesterday overnight sign ===")
    for label, mask in [
        ("yesterday on > 0", df["prev_on"] > 0),
        ("yesterday on < 0", df["prev_on"] < 0),
    ]:
        sub = df.loc[mask, "overnight_ret"]
        if len(sub) > 20:
            print(
                f"  {label:25}  n={len(sub):4}  mean={sub.mean() * 1e4:+6.2f}bps  sharpe={annualised_sharpe(sub):+.2f}"
            )

    out_path = Path("scripts/output/tx_overnight_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "bars": len(df),
                "decomposition": res,
                "ttest": {
                    "overnight": {"t": float(t_on), "p": float(p_on)},
                    "intraday": {"t": float(t_in), "p": float(p_in)},
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
