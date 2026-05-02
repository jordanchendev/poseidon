#!/usr/bin/env python3
"""TX intraday with transaction costs + conditional rules.

Baseline (from test_tx_overnight.py): intraday-only gross Sharpe ≈ 1.00,
t-test p=0.023, max DD -12.55%, Calmar 7.20. But it round-trips every day
(open→close), so 3.2bps round-trip × ~240 days/year = ~7.7%/yr cost drag.

This driver:
  A. Apply 3.2bps round-trip cost to baseline intraday and to each variant.
  B. Test conditional / sparse rules to cut trade frequency while keeping
     risk-adjusted return:
       - only-after-positive-overnight  (gap > 0)
       - only-after-negative-overnight  (gap < 0)
       - only-on-large-gap-day          (|gap| > 0.5%)
       - only-on-vol-spike-day          (|overnight z20| > 1)
       - trend-filter (yesterday close > 20d SMA)
       - mean-rev (yesterday close < 20d SMA)

For each variant: signal frequency, gross/net Sharpe, cum return, maxDD,
Calmar. Net = gross - cost on engaged days.

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_intraday_costed.py
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
ROUND_TRIP_COST = 0.00032  # 3.2 bps per engaged day (open→close round-trip)


def annualised_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0 or daily_returns.empty:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(BARS_PER_YEAR))


def summarise(name: str, returns: pd.Series, mask: pd.Series | None = None) -> dict:
    """Apply mask (engaged days only), apply cost, compute metrics on full series.

    On non-engaged days return = 0 (flat / out of market).
    On engaged days return = intraday return - cost.
    """
    if mask is None:
        engaged = returns.copy()
        n_engaged = int(returns.notna().sum())
    else:
        engaged = returns.where(mask, 0.0)
        n_engaged = int(mask.fillna(False).sum())

    n_total = int(returns.notna().sum())
    freq = n_engaged / n_total if n_total else 0.0

    # Gross
    gross = engaged.fillna(0.0)
    gross_cum = (1.0 + gross).prod() - 1.0
    gross_sharpe = annualised_sharpe(gross[gross != 0]) if mask is not None else annualised_sharpe(gross)
    # When masked, sharpe over engaged subset only is fairer; full-series sharpe
    # dilutes by zero days. Report engaged-only sharpe.
    gross_sharpe_full = annualised_sharpe(gross)

    # Net = gross - cost on engaged days
    if mask is not None:
        cost_series = mask.astype(float) * ROUND_TRIP_COST
    else:
        cost_series = pd.Series(ROUND_TRIP_COST, index=returns.index)
    net = gross - cost_series.fillna(0.0)
    net_cum = (1.0 + net).prod() - 1.0
    net_sharpe_full = annualised_sharpe(net)
    net_sharpe_engaged = annualised_sharpe(net[net != 0]) if mask is not None else annualised_sharpe(net)

    # MDD on net
    eq = (1.0 + net).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())
    calmar = (net_cum / abs(max_dd)) if max_dd < 0 else float("inf")

    print(
        f"  {name:32}  freq={freq * 100:5.1f}%  n={n_engaged:4}  "
        f"gross_sh={gross_sharpe_full:+5.2f}  net_sh={net_sharpe_full:+5.2f}  "
        f"net_cum={net_cum * 100:+7.2f}%  mdd={max_dd * 100:6.2f}%  calmar={calmar:+5.2f}"
    )
    return {
        "name": name,
        "n_engaged": n_engaged,
        "n_total": n_total,
        "frequency": freq,
        "gross_cum_return": gross_cum,
        "gross_sharpe_full": gross_sharpe_full,
        "gross_sharpe_engaged": gross_sharpe,
        "net_cum_return": net_cum,
        "net_sharpe_full": net_sharpe_full,
        "net_sharpe_engaged": net_sharpe_engaged,
        "net_max_dd": max_dd,
        "net_calmar": calmar,
    }


def main():
    repo = RemoteDataRepository.from_settings()
    df = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=END)
    if df.empty:
        raise RuntimeError("No TX data")

    df = df.sort_index().dropna(subset=["open", "high", "low", "close"]).copy()
    df["prev_close"] = df["close"].shift(1)
    df["overnight_ret"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    df["fullday_ret"] = (df["close"] - df["prev_close"]) / df["prev_close"]
    df = df.dropna(subset=["overnight_ret", "intraday_ret", "fullday_ret"])

    # Conditional features (use yesterday's info to avoid lookahead;
    # overnight_ret is observed at today's open BEFORE intraday position
    # opens, so it is fair to condition on)
    df["abs_on"] = df["overnight_ret"].abs()
    df["on_z20"] = (df["overnight_ret"] - df["overnight_ret"].rolling(20).mean()) / df["overnight_ret"].rolling(
        20
    ).std()
    df["sma20"] = df["close"].rolling(20).mean()
    df["yest_close"] = df["close"].shift(1)
    df["yest_sma20"] = df["sma20"].shift(1)

    print(f"\nTX {df.index.min().date()} → {df.index.max().date()}, {len(df)} bars")
    print(f"Cost model: {ROUND_TRIP_COST * 1e4:.1f} bps round-trip on engaged days\n")

    print("=== A. Cost impact on baseline intraday ===")
    res = {}
    res["intraday_baseline_costed"] = summarise("intraday-every-day (cost)", df["intraday_ret"])

    print("\n=== B. Conditional / sparse intraday rules (long-only) ===")
    masks = {
        "only-after-positive-on (>0)": df["overnight_ret"] > 0,
        "only-after-negative-on (<0)": df["overnight_ret"] < 0,
        "only-large-gap (|on|>0.5%)": df["overnight_ret"].abs() > 0.005,
        "only-large-gap (|on|>1.0%)": df["overnight_ret"].abs() > 0.010,
        "only-vol-spike (|z20|>1)": df["on_z20"].abs() > 1.0,
        "only-vol-spike (|z20|>1.5)": df["on_z20"].abs() > 1.5,
        "trend-up (yest>sma20)": df["yest_close"] > df["yest_sma20"],
        "mean-rev (yest<sma20)": df["yest_close"] < df["yest_sma20"],
        "trend-up & pos-on": (df["yest_close"] > df["yest_sma20"]) & (df["overnight_ret"] > 0),
        "trend-up & neg-on": (df["yest_close"] > df["yest_sma20"]) & (df["overnight_ret"] < 0),
    }
    for name, mask in masks.items():
        res[name] = summarise(name, df["intraday_ret"], mask)

    print("\n=== Sign-flipped variants (short intraday on signal) ===")
    # If a rule has neg gross sharpe, flipping sign should flip it
    for name, mask in [
        ("SHORT only-after-pos-on", df["overnight_ret"] > 0),
        ("SHORT mean-rev (yest<sma20)", df["yest_close"] < df["yest_sma20"]),
    ]:
        res[name] = summarise(name, -df["intraday_ret"], mask)

    print("\n=== Sub-period stability for top candidates ===")
    # Pick the 3 best by net_sharpe_full
    ranked = sorted(
        [(k, v) for k, v in res.items() if v.get("frequency", 0) > 0.05],
        key=lambda kv: kv[1]["net_sharpe_full"],
        reverse=True,
    )[:3]
    print(f"  Top 3 by net Sharpe: {[r[0] for r in ranked]}\n")
    for top_name, _ in ranked:
        print(f"  --- {top_name} ---")
        for label, period_mask in [
            ("2021-22", df.index < "2023-01-01"),
            ("2023-24", (df.index >= "2023-01-01") & (df.index < "2025-01-01")),
            ("2025-26", df.index >= "2025-01-01"),
        ]:
            sub = df.loc[period_mask].copy()
            if len(sub) < 30:
                continue
            sub_sma = sub["close"].rolling(20).mean()
            sub_yc = sub["close"].shift(1)
            sub_ysma = sub_sma.shift(1)
            sub_z = (sub["overnight_ret"] - sub["overnight_ret"].rolling(20).mean()) / sub["overnight_ret"].rolling(
                20
            ).std()
            mask_lookup = {
                "only-after-positive-on (>0)": sub["overnight_ret"] > 0,
                "only-after-negative-on (<0)": sub["overnight_ret"] < 0,
                "only-large-gap (|on|>0.5%)": sub["overnight_ret"].abs() > 0.005,
                "only-large-gap (|on|>1.0%)": sub["overnight_ret"].abs() > 0.010,
                "only-vol-spike (|z20|>1)": sub_z.abs() > 1.0,
                "only-vol-spike (|z20|>1.5)": sub_z.abs() > 1.5,
                "trend-up (yest>sma20)": sub_yc > sub_ysma,
                "mean-rev (yest<sma20)": sub_yc < sub_ysma,
                "trend-up & pos-on": (sub_yc > sub_ysma) & (sub["overnight_ret"] > 0),
                "trend-up & neg-on": (sub_yc > sub_ysma) & (sub["overnight_ret"] < 0),
                "intraday-every-day (cost)": pd.Series(True, index=sub.index),
                "SHORT only-after-pos-on": sub["overnight_ret"] > 0,
                "SHORT mean-rev (yest<sma20)": sub_yc < sub_ysma,
            }
            mk = mask_lookup.get(top_name)
            if mk is None:
                continue
            ret_col = sub["intraday_ret"]
            if top_name.startswith("SHORT"):
                ret_col = -ret_col
            engaged = ret_col.where(mk, 0.0)
            cost = mk.astype(float) * ROUND_TRIP_COST
            net = engaged.fillna(0.0) - cost.fillna(0.0)
            sh = annualised_sharpe(net)
            cum = (1.0 + net).prod() - 1.0
            n_e = int(mk.fillna(False).sum())
            print(f"    {label:8}  n={len(sub):3}  engaged={n_e:3}  net_sh={sh:+5.2f}  net_cum={cum * 100:+7.2f}%")

    out_path = Path("scripts/output/tx_intraday_costed_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "bars": len(df),
                "cost_round_trip_bps": ROUND_TRIP_COST * 1e4,
                "results": res,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
