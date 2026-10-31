"""M4 acceptance report: the full leveraged strategy with the protective put (design 2026-10-24 §10 M4, §5.7).

1. ``expiries``: twelve monthly expiries where a put is held to expiry (an ATM put rolled on the expiry day
   itself, engine policy roll_days 0). Each settlement must equal (K - final settlement)+ x 50 x units, with
   the final settlement price read from the TX futures table (``taifex_final_settlement``), not the TXO one.
2. ``unavailable``: the M4 spec over the whole TXO window; the days its hedge was unavailable in the run, and
   a scan that applies the same selection rule to every day.
3. ``variants``: no put / put / put + vol cap (and vol cap without put) from 2014-11-03, under the pipeline
   defaults (next_open, floor, TAIEX up to t) and the offline levtx conventions (close fill, TAIEX up to t-1,
   nearest), with CAGR / MaxDD / Sharpe and the put's own PnL.
4. ``sensitivity``: moneyness x roll days on put + vol cap (pipeline defaults).

Usage (stormtrooper container, refdata mounted):
  POSEIDON_PIPELINE_REFDATA_ROOT=/refdata python scripts/pipeline_v3_m4_report.py /out/m4_report.json
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys

import pandas as pd

from poseidon.pipeline.engine import hedge_coverage
from poseidon.pipeline.engine.options import OptionChain
from poseidon.pipeline.refdata.datasets import FSP
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec, resolve
from poseidon.pipeline.v3.examples import PROTECTIVE_PUT, tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.runtime import Conventions

START = "2014-11-03"
TXO_START = "2002-01-02"  # first trading day after the TXO listing (2001-12-24) with a full front month
SAMPLED = [
    "200206",
    "200403",
    "200609",
    "200812",
    "201006",
    "201203",
    "201409",
    "201612",
    "201806",
    "202003",
    "202209",
    "202506",
]  # the data spike's sanity-check months: four before 2008-12 (next-day settlement)
CONVENTIONS = {
    "pipeline (next_open, floor, TAIEX to t)": ("floor", Conventions()),
    "levtx (close fill, nearest, TAIEX to t-1)": ("nearest", Conventions(execution="signal_close", signal_lag=1)),
}
VARIANTS = {  # name -> (vol cap, put)
    "3x + 200DMA (no put)": (False, False),
    "3x + 200DMA + put": (False, True),
    "3x + 200DMA + put + vol cap": (True, True),
    "3x + 200DMA + vol cap (no put, M1)": (True, False),
}
MONEYNESS = (0.0, 0.025, 0.05, 0.075, 0.10)
ROLL_DAYS = (1, 3, 5, 10)
EPISODES = {"2024-07-11..08-05": ("2024-07-11", "2024-08-05"), "2026-06-22..07-30": ("2026-06-22", "2026-07-30")}


def spec(*, vol_cap: bool, put: bool, rounding: str = "floor", base_only: bool = False) -> dict:
    doc = resolve(tmf_dynamic_leverage_3x())
    sleeve = doc["sleeves"][0]
    sleeve["sizing"]["params"]["rounding"] = rounding
    if base_only:
        sleeve["leverage"] = sleeve["leverage"][:1]
    elif not vol_cap:
        del sleeve["leverage"][2]
    if put:
        sleeve["hedges"] = [copy.deepcopy(PROTECTIVE_PUT)]
    return doc


def stats(equity: pd.Series, capital: float) -> dict:
    days = (equity.index[-1] - equity.index[0]).days or 1
    returns = pd.concat([pd.Series([capital]), equity.reset_index(drop=True)]).pct_change().dropna()
    drawdown = equity / equity.cummax() - 1
    trough = drawdown.idxmin()
    return {
        "final_equity": round(float(equity.iloc[-1])),
        "cagr_pct": round(((equity.iloc[-1] / capital) ** (365.25 / days) - 1) * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "max_drawdown_window": [str(equity.loc[:trough].idxmax().date()), str(trough.date())],
        "sharpe": round(float(returns.mean() / returns.std() * math.sqrt(252)), 3),
        "episodes_drawdown_pct": {
            name: round(float((equity.loc[a:b] / equity.loc[a:b].cummax() - 1).min() * 100), 2)
            for name, (a, b) in EPISODES.items()
        },
    }


def put_stats(res, name: str) -> dict:
    fills = res.fills[res.fills["instrument"] == name]
    if fills.empty:
        return {}
    cash = -(fills["units"] * fills["price"] * 50 + fills["tax"] + fills["commission"]).sum()
    rows = res.positions_daily[res.positions_daily["instrument"] == name]
    last = rows.iloc[-1]
    open_value = float(last["held_units"]) * (float(last["close"]) if last["held_units"] else 0.0) * 50
    bought = fills[fills["units"] > 0]
    cover = hedge_coverage(res.positions_daily, res.events, [name])[name]
    return {
        "put_pnl_twd": round(float(cash + open_value)),
        "premium_paid_twd": round(float((bought["units"] * bought["price"] * 50).sum())),
        "costs_twd": round(float(fills["tax"].sum() + fills["commission"].sum() + fills["slippage_twd"].sum())),
        "fills": len(fills),
        "rolls": int((fills["reason"] == "roll").sum() // 2),
        "take_profit_sales": int(((fills["reason"] == "take_profit") & (fills["units"] < 0)).sum()),
        "expiries": int((fills["reason"] == "expiry").sum()),
        "avg_puts_held": round(float(rows["held_units"].astype(float).mean()), 2),
        "unavailable_days": cover["unavailable_days"],
        "hedged_days": cover["hedged_days"],
    }


def expiries_check(store: RefdataStore) -> dict:
    base = compile_spec(spec(vol_cap=False, put=True, base_only=True), start=TXO_START, store=store)
    (policy,) = base.hedges
    held_to_expiry = dataclasses.replace(base, hedges=(dataclasses.replace(policy, moneyness=0.0, roll_days=0),))
    res = held_to_expiry.engine_run()
    fsp = store.read(FSP).set_index("contract_month")
    rows = []
    settled = res.fills[(res.fills["reason"] == "expiry")]
    for month in SAMPLED:
        fill = settled[settled["contract"].str.startswith(f"{month}P")]
        if len(fill) != 1:
            rows.append({"month": month, "error": f"{len(fill)} expiry fills"})
            continue
        fill = fill.iloc[0]
        strike = float(fill["contract"][len(month) + 1 :])
        final_day, final_price = fsp.at[month, "final_settlement_date"], float(fsp.at[month, "final_settlement_price"])
        units = -int(fill["units"])
        expected = max(strike - final_price, 0.0) * 50 * units
        paid = units * float(fill["price"]) * 50
        rows.append(
            {
                "month": month,
                "settled_on": str(fill["date"].date()),
                "tx_final_settlement_day": str(pd.Timestamp(final_day).date()),
                "strike": strike,
                "tx_final_settlement": final_price,
                "units": units,
                "settlement_twd": paid,
                "expected_twd": expected,
                "equal": paid == expected and pd.Timestamp(final_day) == fill["date"],
            }
        )
    return {"policy": "ATM front-month put held to expiry (roll_days 0), 3x TMF, from " + TXO_START, "rows": rows}


def unavailable_report(store: RefdataStore) -> dict:
    compiled = compile_spec(spec(vol_cap=True, put=True), start=TXO_START, store=store)
    res = compiled.engine_run()
    (policy,) = compiled.hedges
    cover = hedge_coverage(res.positions_daily, res.events, [policy.name])[policy.name]
    notes = res.events[res.events["kind"] == "hedge_unavailable"][["date", "note"]]
    chain = OptionChain("TXO", compiled.options["TXO"], compiled.calendar)
    closes = res.positions_daily[res.positions_daily["instrument"] == "core"].set_index("date")["close"]
    scan = []
    for day, close in closes.items():
        series, why = chain.select(day, policy, float(close))
        if series is None:
            scan.append({"date": str(day.date()), "why": why})
    return {
        "window": [TXO_START, str(closes.index[-1].date())],
        "policy": dataclasses.asdict(policy),
        "run": {**cover, "notes": {str(d.date()): n for d, n in zip(notes["date"], notes["note"], strict=True)}},
        "every_day_scan": {"days": len(closes), "unavailable": len(scan), "list": scan},
    }


def main(out_path: str) -> None:
    store = RefdataStore()
    report: dict = {"expiries": expiries_check(store)}
    print("expiries", [(r["month"], r.get("equal")) for r in report["expiries"]["rows"]], flush=True)
    report["unavailable"] = unavailable_report(store)
    print("unavailable", report["unavailable"]["run"]["unavailable_days"], flush=True)

    report["variants"] = {}
    for label, (rounding, conventions) in CONVENTIONS.items():
        table = {}
        for name, (vol_cap, put) in VARIANTS.items():
            compiled = compile_spec(
                spec(vol_cap=vol_cap, put=put, rounding=rounding), start=START, store=store, conventions=conventions
            )
            res = compiled.run()
            row = stats(res.equity, compiled.config.initial_capital)
            row["end"] = str(res.equity.index[-1].date())
            if put:
                row["put"] = put_stats(res, compiled.hedges[0].name)
                row["inputs"] = {k: v[:12] for k, v in compiled.inputs.items()}
            table[name] = row
            print(label, name, row["cagr_pct"], row["max_drawdown_pct"], row["sharpe"], flush=True)
        report["variants"][label] = table

    full = compile_spec(spec(vol_cap=True, put=True), start=START, store=store)
    (policy,) = full.hedges
    report["sensitivity"] = {"base": dataclasses.asdict(policy), "grid": []}
    for moneyness in MONEYNESS:
        for roll_days in ROLL_DAYS:
            variant = dataclasses.replace(
                full, hedges=(dataclasses.replace(policy, moneyness=moneyness, roll_days=roll_days),)
            )
            res = variant.run()
            row = {"moneyness": moneyness, "roll_days": roll_days, **stats(res.equity, full.config.initial_capital)}
            row["put"] = put_stats(res, policy.name)
            report["sensitivity"]["grid"].append(row)
            print("sens", moneyness, roll_days, row["cagr_pct"], row["max_drawdown_pct"], row["sharpe"], flush=True)

    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m4_report.json")
