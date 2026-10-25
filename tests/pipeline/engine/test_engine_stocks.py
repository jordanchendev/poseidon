"""Stocks in the target-position engine: paid in full, long only, TW costs, corporate actions."""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest
from pytest import approx

from poseidon.pipeline.engine import ContractSpec, Decision, EngineConfig, FuturesInput, run
from poseidon.pipeline.engine.data import FOREVER, stock_bars, stock_input
from poseidon.pipeline.engine.specs import KIND_POLICIES
from poseidon.pipeline.v3.instruments import ETF_TICKS, INSTRUMENTS, STOCK_TICKS
from tests.pipeline.engine.support import FAR, TMF_PLAIN, Scripted, day, flat_bars

ETF = ContractSpec.from_instrument(INSTRUMENTS["0050"])  # one unit = a board lot of 1,000 shares
ODD = ContractSpec.from_instrument(dataclasses.replace(INSTRUMENTS["0050"], lot=1))
STOCK = ContractSpec.from_instrument(INSTRUMENTS["2330"])
RATE = 0.001425


def ohlc(closes: list[float], opens: list[float] | None = None, traded: list[bool] | None = None) -> pd.DataFrame:
    opens = opens or closes
    frame = pd.DataFrame(
        {
            "date": [day(i) for i in range(len(closes))],
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) for o, c in zip(opens, closes, strict=True)],
            "close": closes,
        }
    )
    if traded is not None:
        frame["traded"] = traded
    return frame


def action(date_index: int, kind: str, prev: float, ref: float, cash: float | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": day(date_index),
                "action_type": kind,
                "prev_close": prev,
                "reference_price": ref,
                "cash_dividend": cash,
            }
        ]
    )


def etf(closes, opens=None, traded=None, actions=None, spec=ETF):
    return {"S": stock_input(spec, ohlc(closes, opens, traded), actions)}


# ----------------------------------------------------------------------------- the kind table


def test_stock_spec_comes_from_the_instrument_table():
    assert (ETF.kind, ETF.point_value, ETF.commission_per_unit, ETF.commission_rate, ETF.tax_rate) == (
        "stock",
        1000.0,
        0.0,
        RATE,
        0.001,
    )
    assert (ETF.tax_on_buy, ETF.initial_margin_ratio, ETF.maintenance_margin_ratio, ETF.tick_ladder) == (
        False,
        0.0,
        0.0,
        ETF_TICKS,
    )
    assert (STOCK.tax_rate, STOCK.tick_ladder, str(STOCK.listed_since)) == (0.003, STOCK_TICKS, "2021-01-04")
    assert ODD.point_value == 1.0 and ODD.commission_rate == RATE
    assert ETF.policy == KIND_POLICIES["stock"] and not ETF.policy.short and not ETF.policy.margined
    assert TMF_PLAIN.policy.short and TMF_PLAIN.policy.margined and TMF_PLAIN.policy.rolls
    with pytest.raises(ValueError, match="paid in full"):
        dataclasses.replace(ETF, initial_margin_ratio=0.07)
    with pytest.raises(ValueError, match="unknown instrument kind"):
        dataclasses.replace(ETF, kind="bond")
    with pytest.raises(ValueError, match="inf"):
        dataclasses.replace(ETF, tick_ladder=((50.0, 0.01),))


def test_tick_ladders_and_fees():
    assert ETF.fill_price(1, 49.99) == approx(50.0) and ETF.fill_price(1, 50.0) == approx(50.05)
    assert ETF.fill_price(-1, 50.0) == approx(49.95)
    assert [STOCK.tick(p) for p in (9.99, 10, 49.99, 50, 99.9, 100, 499.5, 500, 999, 1000, 2385)] == [
        0.01,
        0.05,
        0.05,
        0.1,
        0.1,
        0.5,
        0.5,
        1.0,
        1.0,
        5.0,
        5.0,
    ]
    assert TMF_PLAIN.tick(20000) == TMF_PLAIN.tick_size == 1.0
    # buy: commission only; sell: commission + transaction tax on the notional
    assert ETF.leg_fees(3, 100.0) == (0.0, approx(RATE * 300_000))
    assert ETF.leg_fees(-3, 100.0) == (approx(0.001 * 300_000), approx(RATE * 300_000))
    assert STOCK.leg_fees(-1, 1000.0) == (approx(0.003 * 1_000_000), approx(RATE * 1_000_000))
    assert ODD.leg_fees(-7, 100.0) == (approx(0.001 * 700), approx(RATE * 700))


def test_stock_input_is_one_contract_held_forever():
    inp = stock_input(ETF, ohlc([100, 101]))
    assert set(inp.bars["contract"]) == {"0050"} and inp.expiries == {"0050": FOREVER}
    assert list(inp.bars["factor"]) == [1.0, 1.0] and list(inp.bars["traded"]) == [True, True]
    with pytest.raises(ValueError, match="roll schedule"):
        stock_input(TMF_PLAIN, ohlc([100]))


# ----------------------------------------------------------------------------- the ledger


def test_buy_and_hold_pays_in_full_and_sells_with_tax():
    inputs = etf([100, 110, 120], opens=[100, 100, 120])
    result = run(Scripted({0: 5, 1: 0}, "S"), inputs, EngineConfig(execution="next_open"))
    buy, sell = result.fills.itertuples(index=False)
    assert (buy.date, buy.units, buy.price, buy.contract) == (day(1), 5, approx(100.05), "0050")
    cost = 5 * 1000 * 100.05
    cash_after_buy = 1_000_000 - cost - RATE * cost
    assert buy.commission == approx(RATE * cost) and buy.tax == 0.0
    assert result.daily_pnl["equity"].iloc[:2].tolist() == approx([1_000_000, cash_after_buy + 5 * 1000 * 110])
    assert (sell.date, sell.units, sell.price) == (day(2), -5, approx(119.95))
    proceeds = 5 * 1000 * 119.95
    assert sell.tax == approx(0.001 * proceeds) and sell.commission == approx(RATE * proceeds)
    assert result.final_equity == approx(cash_after_buy + proceeds - 0.001 * proceeds - RATE * proceeds)
    daily = result.daily_pnl
    assert daily["margin_required_twd"].tolist() == [0.0, 0.0, 0.0]
    assert daily["gross_exposure_twd"].tolist() == approx([0.0, 5 * 1000 * 110, 0.0])
    assert daily["leverage"].iloc[1] == approx(5 * 1000 * 110 / daily["equity"].iloc[1])
    assert result.events.empty


def test_signal_close_fills_at_the_close_with_slippage():
    result = run(Scripted({0: 2}, "S"), etf([100, 110]), EngineConfig(execution="signal_close"))
    fill = result.fills.iloc[0]
    assert (fill.date, fill.session, fill.price) == (day(0), "close", approx(100.05))
    assert result.final_equity == approx(1_000_000 - 2 * 1000 * 0.05 - RATE * 2 * 1000 * 100.05 + 2 * 1000 * 10)


def test_odd_lots_trade_single_shares():
    result = run(Scripted({0: 7}, "S"), etf([100, 110], spec=ODD), EngineConfig(execution="signal_close"))
    assert result.fills.iloc[0].units == 7
    assert result.final_equity == approx(1_000_000 - 7 * 0.05 - RATE * 7 * 100.05 + 7 * 10)


def test_short_targets_are_rejected_for_stocks():
    with pytest.raises(ValueError, match="cannot be sold short"):
        run(Scripted({0: -1}, "S"), etf([100, 110]), EngineConfig(execution="signal_close"))


def test_buys_are_clipped_to_cash_at_the_decision_and_trimmed_at_the_fill():
    per_lot = 1000 * 100.05 * (1 + RATE)  # fill at close + 1 tick, plus commission
    assert 9 * per_lot < 1_000_000 < 10 * per_lot
    result = run(Scripted({0: 20}, "S"), etf([100, 100, 100]), EngineConfig(execution="next_open", leverage_cap=50))
    clip = result.events.iloc[0]
    assert (clip.kind, clip.value, clip.threshold) == ("clip_cash", approx(20 * per_lot), 1_000_000)
    assert result.positions_daily.iloc[0][["target_units", "order_units"]].tolist() == [20, 9]
    assert result.fills.iloc[0].units == 9 and result.daily_pnl["equity"].iloc[-1] < 1_000_000

    gap = run(Scripted({0: 20}, "S"), etf([100, 112, 112], opens=[100, 112, 112]), EngineConfig(execution="next_open"))
    trim = gap.events[gap.events["kind"] == "trim_cash"].iloc[0]
    assert math.floor(1_000_000 / (1000 * 112.05 * (1 + RATE))) == 8
    assert (trim.date, trim.note) == (day(1), "open: 9 -> 8 units at 112.05")
    assert gap.fills.iloc[0].units == 8 and gap.fills.iloc[0].price == approx(112.05)
    assert gap.daily_pnl["equity"].min() > 0


def test_sells_fund_buys_in_the_same_batch():
    """Rotating out of one ETF into another on one open: the sale fills first and pays for the buy."""

    inputs = {
        "A": stock_input(ETF, ohlc([100, 100, 100])),
        "B": stock_input(dataclasses.replace(ETF, symbol="B"), ohlc([200, 200, 200])),
    }

    class Rotate:
        def on_bar(self, view, book):
            return Decision({"A": 9} if len(view["A"]) == 1 else {"A": 0, "B": 4})

    result = run(Rotate(), inputs, EngineConfig(execution="next_open"))
    day2 = result.fills[result.fills["date"] == day(2)]
    assert day2["instrument"].tolist() == ["A", "B"] and day2["units"].tolist() == [-9, 4]
    assert result.events[result.events["kind"].isin(["clip_cash", "trim_cash"])].empty


def test_a_split_multiplies_the_units_and_keeps_equity_and_adjusted_close():
    split = action(2, "split", prev=100.0, ref=25.0)
    frame = stock_bars(ohlc([100, 100, 25, 26]), split)
    assert frame["factor"].tolist() == [1.0, 1.0, 0.25, 1.0] and frame["unit_ratio"].tolist() == [1.0, 1.0, 4.0, 1.0]
    strategy = Scripted({0: 3}, "S")
    result = run(strategy, etf([100, 100, 25, 26], actions=split), EngineConfig(execution="next_open"))
    event = result.events[result.events["kind"] == "corporate_action"].iloc[0]
    assert (event.date, event.instrument, event.value, event.threshold) == (day(2), "S", 4.0, 0.0)
    assert result.held_units("S").tolist() == [0, 3, 12, 12]
    equity = result.daily_pnl["equity"]
    assert equity.iloc[2] == approx(equity.iloc[1]) and equity.iloc[3] == approx(equity.iloc[2] + 12 * 1000 * 1)
    assert strategy.books[-1].positions["S"] == 12


def test_adjusted_close_is_continuous_across_actions():
    seen = {}

    class Peek:
        def on_bar(self, view, book):
            seen[view.date] = (view["S"].close[-1], view["S"].adjusted_close[-1])
            return Decision({})

    dividend = action(2, "cash_dividend", prev=100.0, ref=98.0, cash=2.0)
    run(Peek(), etf([100, 100, 98, 99], actions=dividend), EngineConfig(execution="signal_close"))
    closes = [seen[day(i)][0] for i in range(4)]
    adjusted = [seen[day(i)][1] for i in range(4)]
    assert closes == [100, 100, 98, 99]
    assert adjusted == approx([100, 100, 100, 100 * 99 / 98])


def test_a_cash_dividend_pays_cash_and_a_fraction_is_paid_at_the_reference_price():
    dividend = action(2, "cash_dividend", prev=100.0, ref=98.0, cash=2.0)
    result = run(Scripted({0: 3}, "S"), etf([100, 100, 98, 98], actions=dividend), EngineConfig(execution="next_open"))
    assert result.held_units("S").tolist() == [0, 3, 3, 3]
    equity = result.daily_pnl["equity"]
    assert equity.iloc[2] == approx(equity.iloc[1])  # 3 lots x 2 TWD/share = 6,000 cash replaces the price drop
    assert result.events.iloc[0].note == "3 -> 3 units, 2 TWD/share, reference 98"

    # an untyped old-layout cash dividend row pays prev - reference
    untyped = action(2, "cash_dividend", prev=100.0, ref=98.0, cash=None)
    same = run(Scripted({0: 3}, "S"), etf([100, 100, 98, 98], actions=untyped), EngineConfig(execution="next_open"))
    assert same.daily_pnl["equity"].tolist() == approx(equity.tolist())

    reverse = action(2, "reverse_split", prev=100.0, ref=200.0)
    result = run(Scripted({0: 3}, "S"), etf([100, 100, 200, 200], actions=reverse), EngineConfig(execution="next_open"))
    assert result.held_units("S").tolist() == [0, 3, 1, 1]
    assert result.daily_pnl["equity"].iloc[2] == approx(result.daily_pnl["equity"].iloc[1])  # 0.5 lot x 200 x 1000 cash


def test_pending_orders_are_rescaled_by_a_split():
    split = action(2, "split", prev=100.0, ref=25.0)
    result = run(
        Scripted({0: 3, 1: 4}, "S"), etf([100, 100, 25, 25], actions=split), EngineConfig(execution="next_open")
    )
    fills = result.fills
    assert fills["units"].tolist() == [3, 4]  # the day-1 order for 4 old lots fills as 16 new lots: 12 -> 16
    assert result.held_units("S").tolist() == [0, 3, 16, 16]


def test_untraded_days_fill_nothing_and_the_decision_replaces_the_queue():
    traded = [True, False, True, True]
    # the strategy asks again on the untraded day (as a re-sized spec would); its order fills at the next session
    result = run(
        Scripted({0: 2, 1: 2}, "S"), etf([100, 100, 105, 105], traded=traded), EngineConfig(execution="next_open")
    )
    assert result.fills["date"].tolist() == [day(2)] and result.fills.iloc[0].price == approx(105.05)
    assert result.held_units("S").tolist() == [0, 0, 2, 2]
    now = run(Scripted({1: 2}, "S"), etf([100, 100, 105, 105], traded=traded), EngineConfig(execution="signal_close"))
    assert now.fills.empty and now.held_units("S").tolist() == [0, 0, 0, 0]


def test_an_action_on_a_halted_ex_date_lands_on_the_next_traded_bar():
    split = action(2, "split", prev=100.0, ref=25.0)
    traded = [True, True, False, True, True]
    frame = stock_bars(ohlc([100, 100, 100, 25, 26], traded=traded), split)  # the carried bar keeps 100
    assert frame["unit_ratio"].tolist() == [1.0, 1.0, 1.0, 4.0, 1.0] and frame["factor"].tolist()[3] == 0.25
    result = run(Scripted({0: 3}, "S"), etf([100, 100, 100, 25, 26], traded=traded, actions=split), EngineConfig())
    assert result.held_units("S").tolist() == [0, 3, 3, 12, 12]
    equity = result.daily_pnl["equity"].tolist()
    assert equity[2] == approx(equity[1]) and equity[3] == approx(equity[2])  # no 4x spike on the halted day
    halted_through_the_end = stock_bars(ohlc([100, 100, 100], traded=[True, True, False]), split)
    assert halted_through_the_end["unit_ratio"].tolist() == [1.0, 1.0, 1.0]  # lands on a later bar, once there is one


def test_an_action_without_prices_is_refused():
    broken = action(2, "split", prev=100.0, ref=25.0)
    broken.loc[0, "reference_price"] = None
    with pytest.raises(ValueError, match="no prev_close / reference_price"):
        stock_bars(ohlc([100, 100, 25, 26]), broken)


def test_stock_buys_leave_cash_for_the_futures_margin():
    inputs = {
        "S": stock_input(ETF, ohlc([100, 100, 100])),
        "TMF": FuturesInput(spec=TMF_PLAIN, bars=flat_bars([20000, 20000, 20000]), expiries={"A": FAR}),
    }

    class Both:
        def on_bar(self, view, book):
            return Decision({"S": 9, "TMF": 10}) if len(view["S"]) == 1 else Decision({})

    result = run(Both(), inputs, EngineConfig(execution="next_open"))
    im = 10 * 20000 * 10 * 0.07  # 140,000 initial margin on 10 TMF
    per_lot = 1000 * 100.05 * (1 + RATE)
    futures_fees = 10 * 20 + 2e-5 * 10 * 20001 * 10 + 10 * 1 * 10  # commission, tax, 1-tick slippage
    affordable = math.floor((1_000_000 - futures_fees - im) / per_lot)
    assert affordable == 8
    clip = result.events[result.events["kind"] == "clip_cash"].iloc[0]
    assert clip.date == day(0) and "'S': 8" in clip.note
    fills = result.fills[result.fills["date"] == day(1)]
    assert fills["instrument"].tolist() == ["TMF", "S"] and fills["units"].tolist() == [10, 8]
    daily = result.daily_pnl.iloc[1]
    assert daily["margin_required_twd"] == approx(im)
    cash = daily["equity"] - 8 * 1000 * 100  # TMF is marked at its close, the ETF carries units x close
    assert cash >= im


def test_equity_based_ruin_and_margin_call_see_stock_value():
    """A stock-only book never margin-calls, never force-liquidates, and is ruined only at zero equity."""

    result = run(Scripted({0: 9}, "S"), etf([100, 100, 1, 1]), EngineConfig(execution="next_open"))
    assert result.ruined is None
    kinds = set(result.events["kind"])
    assert not kinds & {"margin_call", "forced_liquidation", "ruin"}
    assert result.daily_pnl["equity"].iloc[-1] == approx(result.daily_pnl["equity"].iloc[1] - 9 * 1000 * 99)


def test_futures_only_runs_are_unchanged_by_the_stock_paths():
    """Two instruments of the futures kind fill in input order and see no cash events."""

    flat = flat_bars([20000, 20000, 20000])
    inputs = {
        "X": FuturesInput(spec=TMF_PLAIN, bars=flat, expiries={"A": FAR}),
        "Y": FuturesInput(spec=dataclasses.replace(TMF_PLAIN, symbol="Y"), bars=flat, expiries={"A": FAR}),
    }

    class Two:
        def on_bar(self, view, book):
            return Decision({"Y": 3, "X": 2}) if len(view["X"]) == 1 else Decision({})

    result = run(Two(), inputs, EngineConfig(execution="next_open"))
    assert result.fills["instrument"].tolist() == ["X", "Y"]
    assert result.events.empty
