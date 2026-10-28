"""The author's protective put (3x take-profit, next month) against the M4 put, at the engine level.

The author's text (owner teaching series, 2026-10-24): buy next month's TXO put about 5 % below the price,
one put per five TMF, sell it at 3x its purchase price and buy a new one; "with put 50.8 % / -48 %, without
49.3 % / -55 %". "Next month" leaves the roll open, so two readings run: (a) roll 5 trading days before the
put's own expiry, (b) roll at the front month's expiry (a next-month put at all times). Every put row runs
with and without the take-profit, and every row with and without the vol cap, under both conventions of
``pipeline_v3_m4_report``. ``gap`` re-runs the author shapes with each remaining assumption changed alone:
puts filled at the bare settlement (no slippage, tax or commission), and other take-profit multiples. ``price_scale``
is a counterfactual, not a reading of the text: every TXO settlement x f (cash settlement at expiry stays the
intrinsic value), bare fills, to measure how cheap the puts would have to be for the author's 50.8 %.

Usage (stormtrooper container, refdata with taifex_txo_daily mounted):
  POSEIDON_PIPELINE_REFDATA_ROOT=/refdata python scripts/pipeline_v3_author_put_report.py /out/author_put.json
"""

from __future__ import annotations

import dataclasses
import json
import sys

from pipeline_v3_m4_report import CONVENTIONS, put_stats, spec, stats

from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec

START, END = "2014-11-03", "2026-10-22"
BASE = {"moneyness_pct": 0.05, "cover_ratio": 1.0}
PUTS = {  # name -> hedge.protective_put@2 params (None: no put)
    "no put": None,
    "M4 put (front, roll 5)": {**BASE, "tenor": "front", "roll_days_before_expiry": 5},
    "author (a) next, roll 5 before own expiry, TP 3x": {
        **BASE,
        "tenor": "next",
        "roll_days_before_expiry": 5,
        "take_profit_multiple": 3.0,
    },
    "author (b) next, roll at front expiry, TP 3x": {
        **BASE,
        "tenor": "next",
        "roll_days_before_expiry": 0,
        "roll_when": "front_expiry",
        "take_profit_multiple": 3.0,
    },
    "(a) without TP": {**BASE, "tenor": "next", "roll_days_before_expiry": 5},
    "(b) without TP": {**BASE, "tenor": "next", "roll_days_before_expiry": 0, "roll_when": "front_expiry"},
    "M4 put + TP 3x": {**BASE, "tenor": "front", "roll_days_before_expiry": 5, "take_profit_multiple": 3.0},
}
AUTHOR = {
    "with put": {"cagr_pct": 50.8, "max_drawdown_pct": -48.0},
    "without put": {"cagr_pct": 49.3, "max_drawdown_pct": -55.0},
    "with put + vol cap": {"cagr_pct": 48.7, "max_drawdown_pct": -41.0},
}
REFERENCE_00631L = {"cagr_pct": 35.7, "max_drawdown_pct": -55.1, "note": "1->22 split corrected, offline report"}
TP_MULTIPLES = (2.0, 4.0, 5.0)
PRICE_SCALES = (0.75, 0.5, 0.25)


def put_spec(params: dict | None, *, vol_cap: bool, rounding: str) -> dict:
    doc = spec(vol_cap=vol_cap, put=params is not None, rounding=rounding)
    if params is not None:
        doc["sleeves"][0]["hedges"] = [{"component": "hedge.protective_put", "params": params}]
    return doc


def bare_settlement(compiled):
    """The same run with puts filled at the settlement price itself: no slippage, tax or commission."""

    chain = compiled.options["TXO"]
    free = dataclasses.replace(chain.spec, slippage_ticks=0.0, slippage_pct=0.0, tax_rate=0.0, commission_per_unit=0.0)
    return dataclasses.replace(compiled, options={"TXO": dataclasses.replace(chain, spec=free)})


def price_scale(compiled, factor: float):
    chain = compiled.options["TXO"]
    quotes = chain.quotes.assign(settlement=chain.quotes["settlement"] * factor)
    return dataclasses.replace(compiled, options={"TXO": dataclasses.replace(chain, quotes=quotes)})


def row(compiled) -> dict:
    res = compiled.run()
    out = stats(res.equity, compiled.config.initial_capital)
    out["end"] = str(res.equity.index[-1].date())
    if compiled.hedges:
        out["put"] = put_stats(res, compiled.hedges[0].name)
        out["policy"] = dataclasses.asdict(compiled.hedges[0])
    return out


def main(out_path: str) -> None:
    store = RefdataStore()
    report: dict = {
        "window": [START, END],
        "initial_capital_twd": 1_000_000,
        "author": AUTHOR,
        "reference_00631L": REFERENCE_00631L,
        "rows": [],
        "gap": [],
        "price_scale": [],
    }
    compiled_by = {}
    for label, (rounding, conventions) in CONVENTIONS.items():
        for vol_cap in (False, True):
            for name, params in PUTS.items():
                doc = put_spec(params, vol_cap=vol_cap, rounding=rounding)
                compiled = compile_spec(doc, start=START, end=END, store=store, conventions=conventions)
                compiled_by[(label, vol_cap, name)] = compiled
                result = {"convention": label, "vol_cap": vol_cap, "variant": name, **row(compiled)}
                report["rows"].append(result)
                print(
                    label[:8],
                    vol_cap,
                    name,
                    result["cagr_pct"],
                    result["max_drawdown_pct"],
                    result["sharpe"],
                    flush=True,
                )

    for label in CONVENTIONS:
        for vol_cap in (False, True):
            for name in ("M4 put (front, roll 5)", *(n for n in PUTS if n.startswith("author"))):
                compiled = bare_settlement(compiled_by[(label, vol_cap, name)])
                result = {"convention": label, "vol_cap": vol_cap, "variant": name, "change": "bare settlement fills"}
                report["gap"].append({**result, **row(compiled)})
                print("bare", label[:8], vol_cap, name, report["gap"][-1]["cagr_pct"], flush=True)
    label = next(iter(CONVENTIONS))
    for name in (n for n in PUTS if n.startswith("author")):
        compiled = compiled_by[(label, False, name)]
        (policy,) = compiled.hedges
        for multiple in TP_MULTIPLES:
            variant = dataclasses.replace(compiled, hedges=(dataclasses.replace(policy, take_profit=multiple),))
            for bare in (False, True):
                result = {
                    "convention": label,
                    "vol_cap": False,
                    "variant": name,
                    "change": f"take-profit {multiple:g}x" + (" + bare settlement fills" if bare else ""),
                }
                report["gap"].append({**result, **row(bare_settlement(variant) if bare else variant)})
                print("tp", multiple, bare, name, report["gap"][-1]["cagr_pct"], flush=True)

    for name in (n for n in PUTS if n.startswith("author")):
        for factor in PRICE_SCALES:
            compiled = bare_settlement(price_scale(compiled_by[(label, False, name)], factor))
            result = {"convention": label, "vol_cap": False, "variant": name, "change": f"price_scale {factor:g}"}
            report["price_scale"].append({**result, **row(compiled)})
            print("scale", factor, name, report["price_scale"][-1]["cagr_pct"], flush=True)

    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "author_put.json")
