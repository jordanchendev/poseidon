"""The protective put on the real TXO refdata (design 2026-10-24 §10 step 9 acceptance).

Runs where the store has TXO (stormtrooper: refdata mounted read-only, ``POSEIDON_PIPELINE_REFDATA_ROOT``).
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import date

import pandas as pd
import pytest

from poseidon.pipeline.engine import hedge_coverage
from poseidon.pipeline.refdata.datasets import FSP
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET
from poseidon.pipeline.v3.compiler import compile_spec, resolve
from poseidon.pipeline.v3.examples import PROTECTIVE_PUT, tmf_dynamic_leverage_3x, tmf_dynamic_leverage_3x_put

STORE = RefdataStore()
pytestmark = pytest.mark.skipif(not STORE.exists(DAILY_DATASET), reason="TXO refdata not available")
# The data spike's sanity-check months: four settle the trading day after expiry (before 2008-12).
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
]


def test_twelve_puts_held_to_expiry_settle_at_strike_minus_final_settlement_times_fifty():
    doc = resolve(tmf_dynamic_leverage_3x())
    doc["sleeves"][0]["leverage"] = doc["sleeves"][0]["leverage"][:1]  # constant 3x: puts on every month
    doc["sleeves"][0]["hedges"] = [copy.deepcopy(PROTECTIVE_PUT)]
    compiled = compile_spec(doc, start="2002-01-02", store=STORE)
    (policy,) = compiled.hedges
    # an ATM put rolled on the expiry day itself (engine policy roll_days 0): every month is held to settlement
    held = dataclasses.replace(compiled, hedges=(dataclasses.replace(policy, moneyness=0.0, roll_days=0),))
    res = held.engine_run()
    settled = res.fills[res.fills["reason"] == "expiry"].set_index("contract")
    final = STORE.read(FSP).set_index("contract_month")  # TX futures final settlement, not the TXO table
    in_the_money = 0
    for month in SAMPLED:
        (key,) = [key for key in settled.index if key.startswith(f"{month}P")]
        fill = settled.loc[key]
        strike, units = float(key[len(month) + 1 :]), -int(fill["units"])
        price, day = (
            float(final.at[month, "final_settlement_price"]),
            pd.Timestamp(final.at[month, "final_settlement_date"]),
        )
        assert units > 0 and fill["date"] == day, (month, units, fill["date"], day)
        assert units * float(fill["price"]) * 50 == units * max(strike - price, 0.0) * 50, month
        in_the_money += strike > price
    assert 0 < in_the_money < len(SAMPLED)


def test_the_m4_spec_keeps_its_hedge_available_on_more_than_95_percent_of_days():
    compiled = compile_spec(
        resolve(tmf_dynamic_leverage_3x_put(), start=date(2002, 1, 2)), start="2002-01-02", store=STORE
    )
    res = compiled.engine_run()
    (policy,) = compiled.hedges
    cover = hedge_coverage(res.positions_daily, res.events, [policy.name])[policy.name]
    days = res.daily_pnl["date"].nunique()
    assert cover["hedged_days"] > 0.8 * days and cover["unavailable_share"] <= 0.05, cover
