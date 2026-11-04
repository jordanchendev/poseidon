"""Run #16's 00631L / 0050 rotation: daily switching (as run) vs ``rebalance.monthly`` (last trading day).

Engine-level, no pipeline run and no LLM: run #16's spec template (``tests/pipeline/fixtures/
spec_v3_run16_rotation.json``) at its best ``ma_days`` (175, optimize) and its default (200), each with
``rebalance.on_signal_change`` on both sleeves (what the run tested) and with ``rebalance.monthly``
(``last_trading_day``), over the run's train window. Each run is repeated with zero costs; cost drag is
the CAGR lost to costs. Buy-and-hold lines: 0050 and 00631L total return from the first to the last
session of the window (``daily_gates.stock_underlying_daily``, the benchmark gate's series), no costs.

Usage (stormtrooper container, refdata mounted read-only):
  POSEIDON_PIPELINE_REFDATA_ROOT=/refdata python scripts/pipeline_v3_monthly_comparison.py \\
      tests/pipeline/fixtures/spec_v3_run16_rotation.json /out/monthly_comparison.json
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.pipeline.daily_gates import stock_underlying_daily
from poseidon.pipeline.daily_stages import daily_metrics, scaled_costs
from poseidon.pipeline.gates.daily import from_engine, moments
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec, resolve

TRADING_DAYS_PER_YEAR = 252.0
DAILY = {"component": "rebalance.on_signal_change", "params": {}}
MONTHLY = {"component": "rebalance.monthly", "params": {"day": "last_trading_day"}}
# name -> (rebalance on both sleeves, allocation band_pct override or None = the spec's 2 %)
VARIANTS = {
    "daily switch (as run)": (DAILY, None),
    "monthly, last trading day": (MONTHLY, None),
    "monthly, last trading day, band 0": (MONTHLY, 0.0),
}


def variant(template: dict, ref: dict, band: float | None) -> dict:
    doc = copy.deepcopy(template)
    for sleeve in doc["sleeves"]:
        sleeve["rebalance"] = copy.deepcopy(ref)
    if band is not None:
        doc["allocation"]["params"]["band_pct"] = band
    return doc


def held_sleeve(result) -> pd.Series:
    """Per session: the instrument alias holding units (``cash`` when none)."""

    held = result.positions_daily.pivot(index="date", columns="instrument", values="held_units").fillna(0)
    return held.apply(lambda row: next((k for k, v in row.items() if v > 0), "cash"), axis=1)


def switches(result) -> int:
    states = [s for s in held_sleeve(result) if s != "cash"]
    return sum(1 for a, b in itertools.pairwise(states) if a != b)


def drawdown_days(equity: pd.Series) -> dict:
    trough = (equity / equity.cummax()).idxmin()
    return {"drawdown_peak": equity[:trough].idxmax().date().isoformat(), "drawdown_trough": trough.date().isoformat()}


def strategy_row(store, doc: dict, ma_days: int, first: date, last: date) -> dict:
    resolved = resolve(doc, {"ma_days": ma_days}, choose_start=True)
    compiled = compile_spec(resolved, start=first, end=last, store=store)
    result = compiled.run()
    metrics = daily_metrics(from_engine(result), "return")
    free = dataclasses.replace(
        compiled,
        futures={k: dataclasses.replace(f, spec=scaled_costs(f.spec, 0.0)) for k, f in compiled.futures.items()},
    )
    free_metrics = daily_metrics(from_engine(free.run()), "return")
    costs = metrics["costs"]
    return {
        "cagr": metrics["cagr"],
        "max_drawdown": metrics["max_drawdown_pct"],
        "sharpe_annualised": metrics["sharpe_annualised"],
        "sharpe_daily": metrics["sharpe_daily"],
        "switches": switches(result),
        "fill_days": metrics["exposure_changes"],
        "costs_twd": costs["total_twd"],
        "cost_drag_cagr": free_metrics["cagr"] - metrics["cagr"],
        "cagr_without_costs": free_metrics["cagr"],
        "total_return": metrics["total_return"],
        "first_day": metrics["first_day"],
        "last_day": metrics["last_day"],
        "n_days": metrics["n_days"],
        **drawdown_days(result.daily_pnl.set_index("date")["equity"]),
        "fill_dates": sorted({pd.Timestamp(d).date().isoformat() for d in result.fills["date"]}),
    }


def buy_and_hold(store, symbol: str, first: date, last: date) -> dict:
    close = stock_underlying_daily(store, symbol, through=last)["close"]
    close = close[[first <= day <= last for day in close.index]]
    equity = close.to_numpy(float) / float(close.iloc[0])
    returns = equity[1:] / equity[:-1] - 1.0
    stats = moments(returns)
    drawdown = float(np.max(1.0 - equity / np.maximum.accumulate(equity)))
    series = pd.Series(equity, index=pd.DatetimeIndex(close.index))
    return {
        **drawdown_days(series),
        "cagr": float(equity[-1] ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1.0),
        "max_drawdown": drawdown,
        "sharpe_annualised": stats["sharpe_annualised"],
        "sharpe_daily": stats["sharpe_daily"],
        "total_return": float(equity[-1] - 1.0),
        "first_day": close.index[0].isoformat(),
        "last_day": close.index[-1].isoformat(),
        "n_days": len(returns),
    }


def main(fixture: str, out: str) -> None:
    run16 = json.loads(Path(fixture).read_text())
    template = run16["template"]
    first, last = (date.fromisoformat(run16["window"][k]) for k in ("first_day", "last_day"))
    store = RefdataStore()
    rows = {}
    for ma_days, label in ((run16["best_params"]["ma_days"], "best"), (200, "default")):
        for name, (ref, band) in VARIANTS.items():
            doc = variant(template, ref, band)
            rows[f"ma {ma_days} ({label}), {name}"] = strategy_row(store, doc, ma_days, first, last)
    rows["0050 total return, buy and hold"] = buy_and_hold(store, "0050", first, last)
    rows["00631L total return, buy and hold"] = buy_and_hold(store, "00631L", first, last)
    report = {
        "window": {"first_day": first.isoformat(), "last_day": last.isoformat()},
        "refdata": {name: store.content_sha256(name) for name in ("thalassa_tw_stock_daily", "tw_corporate_actions")},
        "rows": rows,
    }
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'':52} {'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'switch':>6} {'fills':>5} {'costs':>9} {'drag':>7}  trough")
    for name, row in rows.items():
        cost = f"{row['costs_twd']:9,.0f}" if "costs_twd" in row else f"{'-':>9}"
        drag = f"{row['cost_drag_cagr'] * 100:6.2f}%" if "cost_drag_cagr" in row else f"{'-':>7}"
        print(
            f"{name:52} {row['cagr'] * 100:6.2f}% {row['max_drawdown'] * 100:6.2f}% {row['sharpe_annualised']:7.3f} "
            f"{row.get('switches', '-'):>6} {row.get('fill_days', '-'):>5} {cost} {drag}  "
            f"{row['drawdown_peak']}..{row['drawdown_trough']}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
