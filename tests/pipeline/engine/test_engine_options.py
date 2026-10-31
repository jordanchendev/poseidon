"""Hand-computed cases for bought options in the engine (design 2026-10-24 §5.7).

TXO: 50 TWD/pt, 25 TWD/contract/side, tax 0.1 % of premium value, slippage max(1 tick, 5 % of the
settlement price). TMF (no proxy): 10 TWD/pt, so cover 1 = 1 put per 5 TMF.
"""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from poseidon.pipeline.engine import (
    EngineConfig,
    HedgePolicy,
    OptionInput,
    OptionSpec,
    hedge_coverage,
    run,
    truncation_audit,
)
from poseidon.pipeline.v3.instruments import INSTRUMENTS

from .support import Scripted, day, flat_bars, tmf

TXO = OptionSpec.from_instrument(INSTRUMENTS["TXO"])
E0 = 1_000_000.0
PUT = HedgePolicy(name="TMF.put", underlying="TMF", chain="TXO", moneyness=0.05, roll_days=2, cover_ratio=1.0)


def quotes(rows: list[tuple]) -> pd.DataFrame:
    """rows: (day index, contract month, strike, settlement); puts only."""

    return pd.DataFrame(
        [(day(i), m, k, "P", s) for i, m, k, s in rows],
        columns=["date", "contract_month", "strike", "cp", "settlement"],
    )


def chain(rows: list[tuple], final: dict | None = None, expiries: dict | None = None) -> dict[str, OptionInput]:
    expiries = expiries or {"M1": day(6), "M2": day(26)}
    return {"TXO": OptionInput(TXO, quotes(rows), expiries, final or {})}


def ladder(months: dict[str, tuple[int, int, float]], strikes=(18500.0, 19000.0, 19500.0)) -> list[tuple]:
    """Every strike of each month quoted at one settlement price on days [first, last]."""

    return [(i, m, k, s) for m, (first, last, s) in months.items() for i in range(first, last + 1) for k in strikes]


def hedged_run(rows, *, n_days=12, targets=None, policy=PUT, config=None, final=None, expiries=None):
    strategy = Scripted(targets or {0: 10})
    return run(
        strategy,
        tmf(flat_bars([20000.0] * n_days)),
        config or EngineConfig(),
        options=chain(rows, final, expiries),
        hedges=[policy],
    )


def option_fills(res):
    return res.fills[res.fills["instrument"] == "TMF.put"].reset_index(drop=True)


def hedge_rows(res):
    return res.positions_daily[res.positions_daily["instrument"] == "TMF.put"].set_index("date")


def test_option_spec_comes_from_the_instrument_table():
    assert (TXO.point_value, TXO.commission_per_unit, TXO.tax_rate, TXO.slippage_pct) == (50.0, 25.0, 0.001, 0.05)
    assert [TXO.tick(p) for p in (9.9, 10, 49.5, 50, 499, 500, 999, 1000, 5000)] == [
        0.1,
        0.5,
        0.5,
        1.0,
        1.0,
        5.0,
        5.0,
        10.0,
        10.0,
    ]
    assert TXO.fill_price(2, 100.0) == 105.0  # 5 % beats one 1-point tick
    assert TXO.fill_price(2, 12.0) == pytest.approx(12.6)
    assert TXO.fill_price(2, 6.0) == pytest.approx(6.3)  # 5 % = 0.3 beats the 0.1 tick
    assert TXO.fill_price(2, 1.0) == pytest.approx(1.1)  # the 0.1 tick beats 5 % = 0.05
    assert TXO.fill_price(-2, 0.1) == 0.0  # a sale never fills below 0
    assert TXO.leg_fees(2, 105.0) == (pytest.approx(10.5), 50.0)


def test_buys_one_put_per_five_tmf_at_the_next_settlement_and_marks_it():
    res = hedged_run(ladder({"M1": (0, 6, 100.0), "M2": (0, 11, 300.0)}))
    fills = option_fills(res)
    first = fills.iloc[0]
    # decided at day 0 (close 20000, target strike 19000, M1 has 6 > 2 days left), filled at day 1's settlement
    assert (first["date"], first["contract"], first["units"], first["reason"]) == (day(1), "M1P19000", 2, "rebalance")
    assert (first["price"], first["ref_price"], first["tax"], first["commission"]) == (105.0, 100.0, 10.5, 50.0)
    assert first["slippage_twd"] == 2 * 5 * 50
    # day 1 equity: futures 10 TMF bought at 20000 + 1 tick (-100, tax 2e-5 x 10 x 20001 x 10, fees 200),
    # options: premium 2 x 105 x 50 paid, value 2 x 100 x 50 held, tax 10.5, commission 50
    futures_cost = 10 * 1 * 10 + 2e-5 * 10 * 20001 * 10 + 200
    options_cost = 2 * 105 * 50 - 2 * 100 * 50 + 10.5 + 50
    assert res.equity[day(1)] == pytest.approx(E0 - futures_cost - options_cost, abs=1e-6)
    rows = hedge_rows(res)
    assert (rows.loc[day(1), "held_units"], rows.loc[day(1), "close"], rows.loc[day(1), "contract"]) == (
        2,
        100.0,
        "M1P19000",
    )
    assert rows.loc[day(0), "target_units"] == 2 and rows.loc[day(0), "held_units"] == 0


def test_rolls_roll_days_before_expiry_into_the_next_month():
    res = hedged_run(ladder({"M1": (0, 6, 100.0), "M2": (0, 11, 300.0)}))
    fills = option_fills(res)
    roll = fills[fills["reason"] == "roll"]
    # day 4 is the first decision with <= 2 trading days left on M1 (expiry day 6): filled at day 5
    assert list(roll["date"]) == [day(5), day(5)]
    assert list(zip(roll["contract"], roll["units"], roll["price"], strict=True)) == [
        ("M1P19000", -2, 95.0),
        ("M2P19000", 2, 315.0),
    ]
    assert hedge_rows(res).loc[day(5) :, "contract"].eq("M2P19000").all()
    assert len(fills) == 3  # no re-trading while the target is unchanged


def test_a_put_held_to_expiry_settles_at_intrinsic_value_times_fifty():
    hold = dataclasses.replace(PUT, roll_days=0)
    rows = ladder({"M1": (0, 5, 100.0), "M2": (0, 11, 300.0)})  # M1 has no settlement on its expiry day
    res = hedged_run(rows, policy=hold, final={"M1": (day(6), 18800.0)})
    expiry = option_fills(res).query("reason == 'expiry'").iloc[0]
    assert (expiry["date"], expiry["contract"], expiry["units"], expiry["price"]) == (day(6), "M1P19000", -2, 200.0)
    assert (expiry["tax"], expiry["commission"]) == (pytest.approx(0.001 * 2 * 200 * 50), 50.0)
    event = res.events.query("kind == 'option_expiry'").iloc[0]
    assert event["value"] == (19000 - 18800) * 50 * 2
    # day 6 PnL: the futures are flat, so it is the settlement minus the day-5 mark (100) minus fees
    pnl = res.daily_pnl.set_index("date")["pnl_twd"]
    assert pnl[day(6)] == pytest.approx(2 * (200 - 100) * 50 - 20 - 50, abs=1e-6)
    # the expiry-day decision buys the next month, filled the day after
    fills = option_fills(res)
    after = fills[(fills["reason"] == "rebalance") & (fills["date"] > day(6))].iloc[0]
    assert (after["date"], after["contract"]) == (day(7), "M2P19000")


def test_a_put_expiring_worthless_costs_nothing_and_a_late_final_settlement_waits_for_its_day():
    hold = dataclasses.replace(PUT, roll_days=0)
    rows = ladder({"M1": (0, 6, 100.0), "M2": (0, 11, 300.0)})
    # before 2008-12 the final settlement day was the trading day after the last trading day
    res = hedged_run(rows, policy=hold, final={"M1": (day(7), 19500.0)})
    expiry = option_fills(res).query("reason == 'expiry'").iloc[0]
    assert (expiry["date"], expiry["price"], expiry["tax"], expiry["commission"]) == (day(7), 0.0, 0.0, 0.0)
    assert hedge_rows(res).loc[day(6), "held_units"] == 2  # still held (marked at 100) on the last trading day


def test_signal_close_fills_at_the_decision_day_settlement():
    res = hedged_run(ladder({"M1": (0, 6, 100.0), "M2": (0, 11, 300.0)}), config=EngineConfig(execution="signal_close"))
    first = option_fills(res).iloc[0]
    assert (first["date"], first["price"]) == (day(0), 105.0)


def test_flat_underlying_sells_the_hedge_and_small_positions_buy_none():
    res = hedged_run(ladder({"M1": (0, 6, 100.0), "M2": (0, 11, 300.0)}), targets={0: 10, 2: 0})
    fills = option_fills(res)
    assert list(zip(fills["date"], fills["units"], fills["reason"], strict=True)) == [
        (day(1), 2, "rebalance"),
        (day(3), -2, "rebalance"),
    ]
    assert hedge_rows(res).loc[day(3) :, "held_units"].eq(0).all()
    four = hedged_run(ladder({"M1": (0, 6, 100.0)}), targets={0: 4})
    assert option_fills(four).empty and hedge_rows(four)["target_units"].dropna().eq(0).all()


def test_no_strike_within_five_percent_is_hedge_unavailable():
    far = [(i, "M1", 17000.0, 50.0) for i in range(7)] + [(i, "M2", 17000.0, 80.0) for i in range(12)]
    res = hedged_run(far)
    assert option_fills(res).empty
    unavailable = res.events.query("kind == 'hedge_unavailable'")
    assert len(unavailable) == 12 and "10.5% from the target 19,000" in unavailable["note"].iloc[0]
    cover = hedge_coverage(res.positions_daily, res.events, ["TMF.put"])["TMF.put"]
    assert (cover["hedged_days"], cover["unavailable_days"], cover["unavailable_share"]) == (12, 12, 1.0)
    # a strike 4.7 % from the target still counts
    edge = [(i, "M1", 18100.0, 50.0) for i in range(7)] + [(i, "M2", 18100.0, 80.0) for i in range(12)]
    assert len(option_fills(hedged_run(edge))) > 0


def test_equity_includes_the_marked_put():
    rows = [(i, "M1", 19000.0, 100.0 + 10 * i) for i in range(7)] + [(i, "M2", 19000.0, 300.0) for i in range(12)]
    res = hedged_run(rows)
    equity = res.equity
    # futures are flat at 20000 after day 1 and nothing trades on days 2-3: equity moves only with the put mark
    assert equity[day(3)] - equity[day(2)] == pytest.approx(2 * 10 * 50)
    assert hedge_rows(res).loc[day(3), "close"] == 130.0


def test_truncation_audit_covers_the_hedge():
    rows = [
        (i, m, k, 100.0 + i + (k - 18500) / 100)
        for m, last in (("M1", 6), ("M2", 25))
        for i in range(last + 1)
        for k in (18500.0, 19000.0, 19500.0)
    ]
    closes = [20000.0 + 150 * math.sin(i / 3) for i in range(24)]
    report = truncation_audit(
        lambda: Scripted({0: 10, 5: 15, 9: 0, 12: 12}),
        tmf(flat_bars(closes)),
        EngineConfig(),
        n_cuts=10,
        options=chain(rows, {"M1": (day(6), 18900.0)}),
        hedges=[PUT],
    )
    assert report.ok, report.mismatches


def test_hedge_names_and_references_are_checked():
    with pytest.raises(ValueError, match="unknown underlying"):
        hedged_run([], policy=dataclasses.replace(PUT, underlying="TX"))
    with pytest.raises(ValueError, match="differ from instrument names"):
        hedged_run([], policy=dataclasses.replace(PUT, name="TMF"))


# --------------------------------------------------------------------------- take-profit and roll timing

TP = dataclasses.replace(PUT, take_profit=3.0)


def drop_run(rows, *, closes, policy=TP, expiries=None, final=None):
    return run(
        Scripted({0: 10}),
        tmf(flat_bars(closes)),
        EngineConfig(),
        options=chain(rows, final, expiries),
        hedges=[policy],
    )


def tp_rows(path: dict[int, float], *, n=8, months=("M1",)) -> list[tuple]:
    """P19000 settles at ``path`` (by day, carried forward), P18000 at 50, for each month in ``months``."""

    rows, price = [], 100.0
    for i in range(n):
        price = path.get(i, price)
        for m in months:
            rows += [(i, m, 19000.0, price), (i, m, 18000.0, 50.0)]
    return rows


def test_take_profit_sells_at_the_first_settlement_of_three_times_the_fill_and_buys_by_the_rule():
    # bought at day 1 for 105 (settlement 100 + 5 %): day 2's 300 is 3x the settlement but not the fill
    rows = tp_rows({2: 300.0, 3: 315.0, 4: 400.0})
    closes = [20000.0] * 3 + [19000.0] * 5  # day 3: target strike 19000 x 0.95 = 18050 -> 18000
    res = drop_run(rows, closes=closes, expiries={"M1": day(12), "M2": day(30)})
    fills = option_fills(res)
    assert list(zip(fills["date"], fills["contract"], fills["units"], fills["reason"], strict=True)) == [
        (day(1), "M1P19000", 2, "rebalance"),
        (day(4), "M1P19000", -2, "take_profit"),
        (day(4), "M1P18000", 2, "take_profit"),
    ]
    sale, buy = fills.iloc[1], fills.iloc[2]
    assert (sale["price"], buy["price"]) == (380.0, 52.5)  # day 4's settlements -/+ 5 % slippage
    assert hedge_rows(res).loc[day(4) :, "contract"].eq("M1P18000").all()


def test_take_profit_does_not_fire_below_the_multiple_or_when_off():
    rows = tp_rows({2: 300.0, 3: 314.9})
    closes = [20000.0] * 3 + [19000.0] * 5
    for policy in (TP, PUT):
        fills = option_fills(drop_run(rows, closes=closes, policy=policy, expiries={"M1": day(12), "M2": day(30)}))
        assert list(fills["reason"]) == ["rebalance"]
    off = option_fills(drop_run(tp_rows({3: 400.0}), closes=closes, policy=PUT, expiries={"M1": day(12)}))
    assert list(off["reason"]) == ["rebalance"]


def test_take_profit_on_a_roll_day_sells_as_take_profit_into_the_next_month():
    # day 4: M1 (expiry day 6) has 2 <= roll_days left and its put is at 3x: one replacement, reason take_profit
    rows = tp_rows({4: 400.0}, n=12, months=("M1", "M2"))
    res = drop_run(rows, closes=[20000.0] * 12)
    fills = option_fills(res)
    assert list(zip(fills["date"], fills["contract"], fills["units"], fills["reason"], strict=True)) == [
        (day(1), "M1P19000", 2, "rebalance"),
        (day(5), "M1P19000", -2, "take_profit"),
        (day(5), "M2P19000", 2, "take_profit"),
    ]


def test_expiry_settles_before_a_take_profit_order_due_the_same_day():
    # held to expiry (roll_days 0). Day 5 decides a take-profit into M1P18000; day 6 is M1's final settlement:
    # the held put is cash-settled first, M1 has no quote that day so the new put is not bought, and the day-6
    # decision buys the next month instead.
    hold = dataclasses.replace(TP, roll_days=0)
    rows = [r for r in tp_rows({5: 400.0}, n=12, months=("M1", "M2")) if not (r[1] == "M1" and r[0] >= 6)]
    closes = [20000.0] * 5 + [19000.0] * 7
    res = drop_run(rows, closes=closes, policy=hold, final={"M1": (day(6), 18900.0)})
    fills = option_fills(res)
    assert list(zip(fills["date"], fills["contract"], fills["units"], fills["reason"], strict=True)) == [
        (day(1), "M1P19000", 2, "rebalance"),
        (day(6), "M1P19000", -2, "expiry"),
        (day(7), "M2P18000", 2, "rebalance"),
    ]
    assert fills.iloc[1]["price"] == 100.0  # intrinsic 19000 - 18900
    missing = res.events.query("kind == 'hedge_fill_missing'")
    assert list(missing["date"]) == [day(6)] and "M1P18000" in missing["note"].iloc[0]


def test_a_next_month_put_rolls_at_its_own_expiry_or_when_the_front_expires():
    expiries = {"M1": day(6), "M2": day(26), "M3": day(46)}
    rows = ladder({"M1": (0, 6, 100.0), "M2": (0, 11, 300.0), "M3": (0, 11, 500.0)})
    own = dataclasses.replace(PUT, tenor=1, roll_days=5)
    fills = option_fills(hedged_run(rows, policy=own, expiries=expiries))
    assert list(zip(fills["date"], fills["contract"], fills["reason"], strict=True)) == [
        (day(1), "M2P19000", "rebalance")  # M2 has 20 days left at the end: no roll in the window
    ]
    front = dataclasses.replace(PUT, tenor=1, roll_days=0, roll_with_tenor=True)
    res = hedged_run(rows, policy=front, expiries=expiries)
    fills = option_fills(res)
    # day 6 is M1's expiry: M2 becomes the front, so the day-6 decision rolls into M3
    assert list(zip(fills["date"], fills["contract"], fills["units"], fills["reason"], strict=True)) == [
        (day(1), "M2P19000", 2, "rebalance"),
        (day(7), "M2P19000", -2, "roll"),
        (day(7), "M3P19000", 2, "roll"),
    ]
    held = hedge_rows(res)["contract"]
    assert held.loc[: day(6)].iloc[1:].eq("M2P19000").all() and held.loc[day(7) :].eq("M3P19000").all()


def test_truncation_audit_covers_take_profit_and_the_front_expiry_roll():
    rows = [
        (i, m, k, (100.0 * 1.3**i if m == "M2" else 100.0 + i) + (k - 18500) / 100)
        for m, last in (("M1", 6), ("M2", 25), ("M3", 25))
        for i in range(last + 1)
        for k in (18000.0, 18500.0, 19000.0, 19500.0)
    ]
    closes = [20000.0 - 60 * i for i in range(24)]  # the target strike drifts down: a take-profit buys a new one
    for policy in (TP, dataclasses.replace(TP, tenor=1, roll_days=0, roll_with_tenor=True)):
        report = truncation_audit(
            lambda: Scripted({0: 10, 5: 15, 9: 0, 12: 12}),
            tmf(flat_bars(closes)),
            EngineConfig(),
            n_cuts=10,
            options=chain(rows, {"M1": (day(6), 18900.0)}, {"M1": day(6), "M2": day(26), "M3": day(46)}),
            hedges=[policy],
        )
        assert report.ok, report.mismatches
    res = run(
        Scripted({0: 10, 5: 15, 9: 0, 12: 12}),
        tmf(flat_bars(closes)),
        EngineConfig(),
        options=chain(rows, {"M1": (day(6), 18900.0)}, {"M1": day(6), "M2": day(26), "M3": day(46)}),
        hedges=[TP],
    )
    assert (option_fills(res)["reason"] == "take_profit").any()  # the audit covers a run that takes profit
