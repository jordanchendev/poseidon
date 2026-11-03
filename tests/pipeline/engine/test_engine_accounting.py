"""Hand-computed ledger cases for the target-position engine (design §5.3-5.6, §12 defaults).

TMF: 10 TWD/pt, tick 1, 20 TWD/contract/side, tax 2e-5 x notional per side, 1 tick slippage.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import EngineConfig, run
from poseidon.pipeline.v3.instruments import INSTRUMENTS, contract_spec

from .support import FAR, TMF, TMF_PLAIN, Scripted, bars, day, flat_bars, tmf

E0 = 1_000_000.0


def approx(x):
    return pytest.approx(x, abs=1e-6)


# ----------------------------------------------------------------------------- sizing and fees


def test_contract_spec_comes_from_the_instrument_table():
    tx = contract_spec(INSTRUMENTS["TX"])
    assert (tx.point_value, tx.tick_size, tx.commission_per_unit, tx.tax_rate, tx.slippage_ticks) == (
        200.0,
        1.0,
        50.0,
        2e-5,
        1.0,
    )
    assert (TMF.point_value, TMF.commission_per_unit, TMF.price_proxy, str(TMF.listed_since)) == (
        10.0,
        20.0,
        "TX",
        "2024-07-29",
    )
    assert (TMF.initial_margin_ratio, TMF.maintenance_margin_ratio) == (0.07, 0.054)
    assert contract_spec(INSTRUMENTS["MTX"]).commission_per_unit == 25.0
    with pytest.raises(ValueError, match="no ledger for option"):
        contract_spec(INSTRUMENTS["TXO"])
    tax_on_exit_only = dataclasses.replace(
        INSTRUMENTS["TX"], cost_model=dataclasses.replace(INSTRUMENTS["TX"].cost_model, tax_on_entry=False)
    )
    with pytest.raises(ValueError, match="tax_on_entry"):
        contract_spec(tax_on_exit_only)
    assert TMF.fill_price(15, 20000) == 20001 and TMF.fill_price(-15, 20000) == 19999
    tax, commission = TMF.leg_fees(15, 20001)
    assert tax == approx(2e-5 * 15 * 20001 * 10) and commission == 300.0
    with pytest.raises(ValueError, match="maintenance"):
        dataclasses.replace(TMF, maintenance_margin_ratio=0.08)


# ----------------------------------------------------------------------------- fills and daily settlement


def test_next_open_fill_and_daily_settlement():
    frame = bars(
        [
            (0, "A", 20000, 20100, 19900, 20000),
            (1, "A", 20050, 20200, 20000, 20100),
            (2, "A", 20100, 20150, 19950, 20000),
        ]
    )
    res = run(Scripted({0: 15}), tmf(frame), EngineConfig(execution="next_open"))

    fill = res.fills.iloc[0]
    assert (fill["date"], fill["session"], fill["units"], fill["price"], fill["reason"]) == (
        day(1),
        "open",
        15,
        20051.0,
        "rebalance",
    )
    assert fill["tax"] == approx(2e-5 * 15 * 20051 * 10) and fill["commission"] == 300.0
    assert fill["slippage_twd"] == approx(150.0)
    # day 1: 15 x (20100 - 20051) x 10 = 7,350 minus tax 60.153 and commission 300
    eq = res.daily_pnl.set_index("date")["equity"]
    assert eq[day(0)] == E0
    assert eq[day(1)] == approx(E0 + 7350 - 60.153 - 300)
    assert eq[day(2)] == approx(E0 + 7350 - 60.153 - 300 - 15000)
    held = res.held_units("TMF")
    assert list(held) == [0, 15, 15]  # next_open: the day-0 order is held from day 1
    assert res.daily_pnl["pnl_twd"].iloc[2] == approx(-15000)
    assert res.daily_pnl["daily_return"].iloc[2] == approx(-15000 / eq[day(1)])
    assert res.daily_pnl["fees_twd"].iloc[1] == approx(360.153)
    lev = res.daily_pnl["leverage"].iloc[2]
    assert lev == approx(15 * 20000 * 10 / eq[day(2)])
    assert res.unfilled == {}


def test_signal_close_fills_at_the_decision_close():
    frame = bars([(0, "A", 20000, 20100, 19900, 20000), (1, "A", 20050, 20200, 20000, 20100)])
    res = run(Scripted({0: 15}), tmf(frame), EngineConfig(execution="signal_close"))
    assert res.fills.iloc[0][["session", "price"]].tolist() == ["close", 20001.0]
    eq = res.equity
    assert eq[day(0)] == approx(E0 - 150 - 60.003 - 300)
    assert eq[day(1)] == approx(E0 - 150 - 60.003 - 300 + 15000)


def test_last_day_order_is_reported_unfilled():
    res = run(Scripted({1: 4}), tmf(flat_bars([20000, 20000])), EngineConfig())
    assert res.fills.empty and res.unfilled == {"TMF": 4}


# ----------------------------------------------------------------------------- roll


def roll_frame():
    # A expires on day 2, so with days_before=1 the roll is at the close of day 1.
    return bars(
        [
            (0, "A", 20000, 20000, 20000, 20000),
            (0, "B", 19950, 19950, 19950, 19950),
            (1, "A", 20050, 20150, 20000, 20100),
            (1, "B", 20000, 20080, 19950, 20040),
            (2, "A", 20150, 20250, 20100, 20200),
            (2, "B", 20060, 20180, 20000, 20150),
            (3, "B", 20150, 20160, 20100, 20120),
        ]
    )


ROLL_EXPIRIES = {"A": day(2), "B": FAR}


def test_roll_before_settlement_never_books_the_roll_gap():
    res = run(Scripted({0: 10}), tmf(roll_frame(), ROLL_EXPIRIES), EngineConfig(execution="next_open"))
    f = res.fills
    assert f[["date", "session", "contract", "units", "price", "reason"]].values.tolist() == [
        [day(1), "open", "A", 10, 20051.0, "rebalance"],
        [day(1), "close", "A", -10, 20099.0, "roll"],
        [day(1), "close", "B", 10, 20041.0, "roll"],
    ]
    fees = (2e-5 * 10 * 10) * (20051 + 20099 + 20041) + 3 * 200
    # PnL: A 20051 -> 20099 on 10 contracts, then B 20041 -> 20150 on day 2. The A/B gap (-60) is not PnL.
    eq = res.equity
    assert eq[day(1)] == approx(E0 + 10 * 10 * (20099 - 20051) + 10 * 10 * (20040 - 20041) - fees)  # B marked at 20040
    assert eq[day(2)] == approx(E0 + 10 * 10 * (20099 - 20051) - fees + 10 * 10 * (20150 - 20041))
    pos = res.positions_daily.set_index("date")
    assert pos.loc[day(0), "contract"] == "A" and pos.loc[day(1), "contract"] == "B"
    rolls = res.events[res.events["kind"] == "roll"]
    assert rolls["note"].tolist() == ["A -> B (10 -> 10 units)"]


def test_signal_close_nets_the_roll_with_the_rebalance():
    res = run(Scripted({0: 10, 1: 12}), tmf(roll_frame(), ROLL_EXPIRIES), EngineConfig(execution="signal_close"))
    f = res.fills[res.fills["date"] == day(1)]
    assert f[["contract", "units", "reason"]].values.tolist() == [["A", -10, "roll"], ["B", 12, "roll"]]


def test_roll_at_settlement_day():
    frame = roll_frame()
    res = run(Scripted({0: 1}), tmf(frame, ROLL_EXPIRIES, roll_days_before=0), EngineConfig(execution="signal_close"))
    rolls = res.fills[res.fills["reason"] == "roll"]
    assert rolls["date"].unique().tolist() == [day(2)]
    assert rolls["contract"].tolist() == ["A", "B"]


def test_adjusted_close_compounds_held_contract_returns():
    seen = {}

    class Peek(Scripted):
        def on_bar(self, view, book):
            seen[book.date] = (view["TMF"].close[-1], view["TMF"].adjusted_close.copy(), view["TMF"].contract[-1])
            return super().on_bar(view, book)

    run(Peek({}), tmf(roll_frame(), ROLL_EXPIRIES), EngineConfig())
    assert seen[day(1)][0] == 20040 and seen[day(1)][2] == "B"  # on the roll day the view shows the new contract
    adj = seen[day(3)][1]
    assert adj[0] == 20000
    assert adj[1] == approx(20000 * 20100 / 20000)  # day 1 return on A (held over day 1)
    assert adj[2] == approx(adj[1] * 20150 / 20040)  # day 2 return on B
    assert adj[3] == approx(adj[2] * 20120 / 20150)


# ----------------------------------------------------------------------------- margin


def crash_frame(low: float, close: float) -> pd.DataFrame:
    return bars(
        [(0, "A", 20000, 20000, 20000, 20000), (1, "A", 19000, 19000, low, close), (2, "A", close, close, close, close)]
    )


LOOSE = EngineConfig(execution="signal_close", leverage_cap=20.0)
OPEN_COST = 25 * 10 * 1 + 2e-5 * 25 * 20001 * 10 + 25 * 20  # slippage + tax + commission on the day-0 buy


def test_margin_call_forces_reduction_to_initial_margin():
    # -16% close, intraday low -16.5%: risk ratio 0.596 (> 0.25) so no liquidation; equity < maintenance.
    res = run(Scripted({0: 25, 1: 25}), tmf(crash_frame(16700, 16800)), LOOSE)
    equity = E0 - OPEN_COST - 25 * 10 * 3200
    mm, im = 25 * 16800 * 10 * 0.054, 25 * 16800 * 10 * 0.07
    calls = res.events[res.events["kind"] == "margin_call"]
    assert len(calls) == 1
    assert calls.iloc[0]["value"] == approx(equity) and calls.iloc[0]["threshold"] == approx(mm)
    cap = int(25 * equity / im)  # 16
    assert cap == 16
    f = res.fills.iloc[1]
    assert (f["date"], f["units"], f["reason"]) == (day(1), cap - 25, "margin_call")
    sell = 9 * 10 * 1 + 2e-5 * 9 * 16799 * 10 + 9 * 20
    assert res.equity[day(1)] == approx(equity - sell)
    assert res.positions_daily.set_index("date").loc[day(1), "target_units"] == 25  # strategy still asked for 25
    assert res.ruined is None and "forced_liquidation" not in set(res.events["kind"])


def test_margin_call_overrides_an_omitted_target():
    res = run(Scripted({0: 25}), tmf(crash_frame(16700, 16800)), LOOSE)
    assert res.positions_daily.set_index("date").loc[day(1), "target_units"] is pd.NA
    assert res.fills.iloc[1][["units", "reason"]].tolist() == [-9, "margin_call"]


def test_margin_call_cut_waits_for_the_next_open():
    frame = bars(
        [
            (0, "A", 20000, 20000, 20000, 20000),
            (1, "A", 20000, 20000, 20000, 20000),
            (2, "A", 20000, 20000, 16700, 16800),
            (3, "A", 16900, 16900, 16900, 16900),
        ]
    )
    res = run(Scripted({0: 25}), tmf(frame), dataclasses.replace(LOOSE, execution="next_open"))
    assert "margin_call" in set(res.events["kind"])
    cut = res.fills[res.fills["reason"] == "margin_call"]
    assert cut[["date", "session", "units"]].values.tolist() == [[day(3), "open", -9]]


def test_intraday_forced_liquidation_at_the_worst_price():
    # low -19%: equity at the low 49,149.995 vs initial margin 283,500 -> ratio 0.173 < 0.25
    res = run(Scripted({0: 25}), tmf(crash_frame(16200, 16900)), LOOSE)
    liq = res.events[res.events["kind"] == "forced_liquidation"]
    equity_low = E0 - OPEN_COST - 25 * 10 * 3800
    assert liq.iloc[0]["value"] == approx(equity_low / (25 * 16200 * 10 * 0.07))
    f = res.fills.iloc[1]
    assert (f["session"], f["units"], f["price"], f["reason"]) == ("intraday", -25, 16199.0, "forced_liquidation")
    after = equity_low - 25 * 10 - 2e-5 * 25 * 16199 * 10 - 25 * 20
    assert res.equity[day(1)] == approx(after)
    assert res.equity[day(2)] == approx(after)  # flat: the strategy omitted a target, so it stays flat
    assert res.ruined is None


def test_ruin_stops_the_run():
    res = run(Scripted({0: 25}), tmf(crash_frame(15000, 15500)), LOOSE)
    assert res.ruined == day(1)
    assert res.daily_pnl["date"].tolist() == [day(0), day(1)]
    assert res.daily_pnl["equity"].iloc[-1] < 0
    assert set(res.events["kind"]) >= {"forced_liquidation", "ruin"}


def test_gap_at_the_open_liquidates_before_the_fill():
    frame = bars(
        [
            (0, "A", 20000, 20000, 20000, 20000),
            (1, "A", 20000, 20000, 20000, 20000),
            (2, "A", 16000, 16500, 16000, 16400),
            (3, "A", 16400, 16400, 16400, 16400),
        ]
    )
    res = run(Scripted({0: 25, 1: 30}), tmf(frame), dataclasses.replace(LOOSE, execution="next_open"))
    day2 = res.fills[res.fills["date"] == day(2)]
    assert day2[["session", "units", "reason"]].values.tolist() == [["open", -25, "forced_liquidation"]]


# ----------------------------------------------------------------------------- caps, band, reporting


def test_leverage_cap_clips_the_order():
    res = run(Scripted({0: 30}), tmf(flat_bars([20000, 20000])), EngineConfig(execution="signal_close"))
    assert res.fills.iloc[0]["units"] == 25  # 6x requested, 5x allowed
    clip = res.events[res.events["kind"] == "clip_leverage"].iloc[0]
    assert clip["value"] == approx(6.0) and clip["threshold"] == 5.0
    assert res.positions_daily.iloc[0][["target_units", "order_units"]].tolist() == [30, 25]


def test_margin_utilisation_cap():
    cfg = EngineConfig(execution="signal_close", max_margin_utilisation=0.2)
    res = run(Scripted({0: 25}), tmf(flat_bars([20000, 20000])), cfg)
    assert res.fills.iloc[0]["units"] == 14  # 350,000 IM vs 200,000 allowed -> trunc(25 x 0.5714)
    assert "clip_margin" in set(res.events["kind"])


def test_min_change_band_skips_small_rebalances():
    cfg = EngineConfig(execution="signal_close", min_change_units=2)
    res = run(Scripted({0: 10, 1: 11, 2: 13}), tmf(flat_bars([20000] * 3)), cfg)
    assert res.fills["units"].tolist() == [10, 3]


def test_positions_daily_reports_rounding_and_margin():
    res = run(
        Scripted({0: 14}, leverage={0: 3.0}), tmf(flat_bars([20000, 20000])), EngineConfig(execution="signal_close")
    )
    row = res.positions_daily.iloc[0]
    assert row["implied_target_leverage"] == approx(14 * 20000 * 10 / E0)
    assert row["rounding_error"] == approx(3.0 - 2.8)
    eq = res.equity.iloc[0]
    assert row["actual_leverage"] == approx(14 * 20000 * 10 / eq)
    assert row["margin_utilisation"] == approx(14 * 20000 * 10 * 0.07 / eq)
    assert not row["proxy"]


def test_proxy_flag_from_listing_date_and_price_proxy():
    spec = dataclasses.replace(TMF_PLAIN, listed_since=day(1).date())
    res = run(Scripted({}), tmf(flat_bars([20000] * 3), spec=spec), EngineConfig())
    assert res.positions_daily["proxy"].tolist() == [True, False, False]
    res = run(Scripted({}), tmf(flat_bars([20000] * 2), spec=TMF), EngineConfig())
    assert res.positions_daily["proxy"].all()  # TMF is priced from TX for the whole window


def test_warm_up_bars_before_start_are_visible_but_not_traded():
    seen = []

    class Peek(Scripted):
        def on_bar(self, view, book):
            seen.append(len(view["TMF"]))
            return super().on_bar(view, book)

    res = run(Peek({0: 1}), tmf(flat_bars([20000] * 5)), EngineConfig(start=day(3)))
    assert seen == [4, 5]
    assert res.daily_pnl["date"].tolist() == [day(3), day(4)]


def test_view_is_append_only_and_read_only():
    checks = []

    class Probe(Scripted):
        def on_bar(self, view, book):
            close = view["TMF"].close
            checks.append((len(close), bool(np.isnan(close.base[len(close) :]).all())))
            with pytest.raises(ValueError):
                close[0] = 1.0
            return super().on_bar(view, book)

    run(Probe({}), tmf(flat_bars([20000, 20100, 20200])), EngineConfig())
    assert checks == [(1, True), (2, True), (3, True)]


def test_bad_targets_and_inputs_are_rejected():
    with pytest.raises(TypeError, match="integer"):
        run(Scripted({0: 1.5}), tmf(flat_bars([20000])), EngineConfig())
    with pytest.raises(ValueError, match="unknown instrument"):
        run(Scripted({0: 1}, instrument="TX"), tmf(flat_bars([20000])), EngineConfig())
    with pytest.raises(ValueError, match="no expiry"):
        run(Scripted({}), tmf(flat_bars([20000]), {"Z": "2100-01-01"}), EngineConfig())
    with pytest.raises(ValueError, match="duplicate"):
        run(Scripted({}), tmf(pd.concat([flat_bars([20000])] * 2)), EngineConfig())
