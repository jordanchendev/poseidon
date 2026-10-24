"""Acceptance report for the compiled §3.2 spec (design 2026-10-24 §10 step 3).

Runs the acceptance spec from refdata through the engine under the §12 defaults and under the
levtx conventions, walks from one to the other one convention at a time, and attributes the
fill-timing effect to the overnight gap between the decision close and the next open.

Usage (stormtrooper container, refdata mounted):
  POSEIDON_PIPELINE_REFDATA_ROOT=/refdata python scripts/pipeline_v3_acceptance_report.py /out/report.json
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import date

import pandas as pd

from poseidon.pipeline.refdata.datasets import FUTURES
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec, resolve
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.runtime import Conventions
from poseidon.pipeline.v3.validation import FeasibilityContext, validate_spec_v3

WINDOWS = {
    "2014-11..now": ("2014-11-03", None),
    "2000..2014": ("2000-01-04", "2014-10-31"),
    "2020..now": ("2020-01-02", None),
    "2008..now": ("2008-01-02", None),
}
TAIEX_LAG1 = {"signal_aux": {"core:adjusted": "taiex"}, "signal_lag": 1}
# name -> (rounding, conventions, bars, keep the vol cap): A -> B changes one convention per step
VARIANTS = {
    "A levtx (close fill, TAIEX t-1, nearest)": (
        "nearest",
        dict(execution="signal_close", **TAIEX_LAG1),
        "night",
        True,
    ),
    "A1 + floor": ("floor", dict(execution="signal_close", **TAIEX_LAG1), "night", True),
    "A2 + TX adjusted signal (t-1)": ("floor", dict(execution="signal_close", signal_lag=1), "night", True),
    "A3 + signal up to t (close fill)": ("floor", dict(execution="signal_close"), "night", True),
    "B §12 defaults (next_open)": ("floor", dict(), "night", True),
    "B' next_open, 08:45 day open only": ("floor", dict(), "day", True),
    "C earlier harness: levtx + next_open, day open": ("nearest", dict(**TAIEX_LAG1), "day", True),
    "C' levtx + next_open, night open": ("nearest", dict(**TAIEX_LAG1), "night", True),
    # controls for the signal-lag step
    "L1 A1 with TAIEX up to t": (
        "floor",
        dict(execution="signal_close", signal_aux={"core:adjusted": "taiex"}),
        "night",
        True,
    ),
    "L2 A2 without vol cap": ("floor", dict(execution="signal_close", signal_lag=1), "night", False),
    "L3 A3 without vol cap": ("floor", dict(execution="signal_close"), "night", False),
}
AFTER_HOURS = pd.Timestamp("2017-05-16")  # first TAIFEX after-hours row date (session opened 2017-05-15 15:00)


def stats(equity: pd.Series, capital: float) -> dict:
    days = (equity.index[-1] - equity.index[0]).days or 1
    peak = equity.cummax()
    return {
        "final_equity": round(float(equity.iloc[-1])),
        "cagr_pct": round(((equity.iloc[-1] / capital) ** (365.25 / days) - 1) * 100, 2),
        "max_drawdown_pct": round(float((equity / peak - 1).min() * 100), 2),
    }


def yearly(equity: pd.Series, capital: float) -> dict:
    year_end = equity.groupby(equity.index.year).last()
    previous = year_end.shift(1).fillna(capital)
    return {
        str(year): {"equity": round(float(value)), "return_pct": round(float(value / previous[year] - 1) * 100, 2)}
        for year, value in year_end.items()
    }


def day_open_bars(compiled, store: RefdataStore):
    """The same inputs with ``open`` = the 08:45 day-session open (the earlier harness's bars)."""

    quotes = store.read(FUTURES, start=compiled.calendar[0], end=compiled.config.end)
    day = quotes[(quotes["session"] == "day") & (quotes["product"] == "TX")]
    opens = day.set_index(["trade_date", "contract_month"])["open"]
    futures = {}
    for alias, fut in compiled.futures.items():
        bars = fut.bars.copy()
        bars["open"] = opens.reindex(pd.MultiIndex.from_arrays([bars["date"], bars["contract"]])).to_numpy()
        futures[alias] = dataclasses.replace(fut, bars=bars)
    return dataclasses.replace(compiled, futures=futures)


def gap_attribution(res, compiled) -> dict:
    """Next-open fills vs filling the same orders at the decision close: sum of
    units x (open(t+1) - close(t)) x point value over rebalance fills, also as a share of equity."""

    fills = res.fills[(res.fills["session"] == "open") & (res.fills["reason"] == "rebalance")].copy()
    if fills.empty:
        return {}
    fut = compiled.futures["core"]
    closes = fut.bars.assign(date=pd.to_datetime(fut.bars["date"])).set_index(["date", "contract"])["close"]
    calendar = compiled.calendar
    fills["decision_day"] = calendar[calendar.get_indexer(pd.to_datetime(fills["date"])) - 1]
    fills["decision_close"] = closes.reindex(
        pd.MultiIndex.from_arrays([fills["decision_day"], fills["contract"]])
    ).to_numpy()
    point = fut.spec.point_value
    fills["gap_cost_twd"] = fills["units"] * (fills["ref_price"] - fills["decision_close"]) * point
    equity = res.equity.shift(1).reindex(pd.to_datetime(fills["date"])).to_numpy()
    fills["gap_cost_frac"] = fills["gap_cost_twd"] / equity
    # regime-switch fills: the order is large relative to the position (>= 40 % of it)
    held_before = res.positions_daily.set_index("date")["held_units"].shift(1).reindex(pd.to_datetime(fills["date"]))
    fills["switch"] = fills["units"].abs() >= 0.4 * held_before.abs().clip(lower=1).to_numpy()
    out = {}
    for label, mask in (
        ("all", slice(None)),
        ("before after-hours (08:45 open)", pd.to_datetime(fills["date"]) < AFTER_HOURS),
        ("after-hours era (15:00 open)", pd.to_datetime(fills["date"]) >= AFTER_HOURS),
    ):
        part = fills[mask]
        if part.empty:
            continue
        out[label] = {
            "fills": len(part),
            "gap_cost_twd": round(float(part["gap_cost_twd"].sum())),
            "sum_gap_cost_frac_pct": round(float(part["gap_cost_frac"].sum() * 100), 2),
            "switch_fills": int(part["switch"].sum()),
            "switch_sum_gap_cost_frac_pct": round(float(part.loc[part["switch"], "gap_cost_frac"].sum() * 100), 2),
            "routine_sum_gap_cost_frac_pct": round(float(part.loc[~part["switch"], "gap_cost_frac"].sum() * 100), 2),
            "mean_abs_gap_points": round(float((part["ref_price"] - part["decision_close"]).abs().mean()), 1),
        }
    worst = fills.reindex(fills["gap_cost_frac"].abs().sort_values(ascending=False).index).head(8)
    out["largest"] = [
        {
            "decision_day": str(row.decision_day.date()),
            "units": int(row.units),
            "close": float(row.decision_close),
            "next_open": float(row.ref_price),
            "gap_cost_pct_equity": round(float(row.gap_cost_frac) * 100, 2),
        }
        for row in worst.itertuples()
    ]
    return out


def gap_autocorrelation(compiled) -> dict:
    """corr(close-to-close return of day t, gap close(t) -> open(t+1)) on the held contract: the sign of the
    gap cost of constant-leverage rebalancing (buy after up days, sell after down days)."""

    fut = compiled.futures["core"]
    bars = fut.bars.assign(date=pd.to_datetime(fut.bars["date"])).set_index(["date", "contract"]).sort_index()
    rows = []
    days = compiled.calendar[: len(compiled.calendar) - 10]
    for prev, today, nxt in zip(days[:-2], days[1:-1], days[2:], strict=True):
        for contract in bars.loc[today].index:
            try:
                c0, c1, o2 = (
                    bars.at[(prev, contract), "close"],
                    bars.at[(today, contract), "close"],
                    bars.at[(nxt, contract), "open"],
                )
            except KeyError:
                continue
            rows.append((today, c1 / c0 - 1, o2 / c1 - 1))
            break  # nearest listed contract
    frame = pd.DataFrame(rows, columns=["date", "ret", "gap"]).set_index("date")
    out = {}
    for label, part in (
        ("before after-hours", frame[frame.index < AFTER_HOURS - pd.Timedelta(days=1)]),
        ("after-hours era", frame[frame.index >= AFTER_HOURS]),
    ):
        if len(part) > 30:
            out[label] = {
                "days": len(part),
                "corr_ret_next_gap": round(float(part["ret"].corr(part["gap"])), 3),
                "gap_std_pct": round(float(part["gap"].std() * 100), 3),
                "ret_std_pct": round(float(part["ret"].std() * 100), 3),
            }
    return out


def main(out_path: str) -> None:
    store = RefdataStore()
    template = tmf_dynamic_leverage_3x()
    report: dict = {"validation": {}, "windows": {}}

    for name, (start, end) in WINDOWS.items():
        base = compile_spec(resolve(template), start=start, end=end, store=store)
        result = base.run()
        closes = result.positions_daily["close"].astype(float)
        feasibility = FeasibilityContext(
            start=date.fromisoformat(start),
            end=(pd.Timestamp(end) if end else result.equity.index[-1]).date(),
            prices={"TX": (float(closes.iloc[0]), float(closes.max()))},
        )
        checked = validate_spec_v3(template, feasibility=feasibility)
        tx = tmf_dynamic_leverage_3x()
        tx["instruments"]["core"]["symbol"] = "TX"
        tx_checked = validate_spec_v3(tx, feasibility=feasibility)
        report["validation"][name] = {
            "tmf_ok": checked.ok,
            "tmf_errors": [i.text() for i in checked.errors],
            "notes": list(checked.notes),
            "tx_errors": [i.text() for i in tx_checked.errors],
        }

        window = {"compile_notes": list(base.notes), "inputs": base.inputs, "variants": {}}
        for label, (rounding, conventions, bars, vol_cap) in VARIANTS.items():
            doc = resolve(template)
            doc["sleeves"][0]["sizing"]["params"]["rounding"] = rounding
            if not vol_cap:
                del doc["sleeves"][0]["leverage"][2]
            compiled = compile_spec(doc, start=start, end=end, store=store, conventions=Conventions(**conventions))
            if bars == "day":
                compiled = day_open_bars(compiled, store)
            strategy = compiled.strategy()
            res = compiled.run(strategy)
            row = stats(res.equity, compiled.config.initial_capital)
            row.update(
                end=str(res.equity.index[-1].date()),
                trades=len(res.fills),
                fees_twd=round(float(res.fills["tax"].sum() + res.fills["commission"].sum())),
                slippage_twd=round(float(res.fills["slippage_twd"].sum())),
                regime_switches=len(strategy.state("main", "leverage.1")["switches"]),
                events=res.events["kind"].value_counts().to_dict(),
            )
            if label.startswith(("A levtx", "B §12")):
                row["yearly"] = yearly(res.equity, compiled.config.initial_capital)
            if compiled.config.execution == "next_open":
                row["gap_attribution"] = gap_attribution(res, compiled)
            window["variants"][label] = row
        window["gap_autocorrelation"] = gap_autocorrelation(base)
        ref = window["variants"]["A levtx (close fill, TAIEX t-1, nearest)"]["final_equity"]
        for row in window["variants"].values():
            row["vs_levtx_pct"] = round((row["final_equity"] / ref - 1) * 100, 2)
        report["windows"][name] = window
        print(name, {k: (v["final_equity"], v["vs_levtx_pct"]) for k, v in window["variants"].items()}, flush=True)

    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "acceptance_report.json")
