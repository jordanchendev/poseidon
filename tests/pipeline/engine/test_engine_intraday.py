"""Intraday clock of the target-position engine: R1 contract schedule, synthetic roll, protective orders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import BookView, ContractSpec, Decision, MarketView, run
from poseidon.pipeline.engine.intraday import IntradayConfig, IntradayInput, r1_contracts, run_intraday
from poseidon.pipeline.engine.view import Protect
from tests.dsl_v2_support import thalassa_stamps
from tests.pipeline.engine.support import flat_bars, tmf

SPEC = ContractSpec("X", point_value=10.0, tick_size=1.0, commission_per_unit=20.0, tax_rate=0.0, slippage_ticks=1.0)
EXPIRIES = {"202609": "2026-09-16", "202610": "2026-10-21", "202611": "2026-11-18"}
CAPITAL = 1_000_000.0


def at(index: pd.DatetimeIndex, local: str) -> int:
    return int(index.get_loc(pd.Timestamp(local, tz="Asia/Taipei").tz_convert("UTC")))


def frame(index: pd.DatetimeIndex, o, h, lo, c) -> pd.DataFrame:
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=index, dtype=float)


def flat(index: pd.DatetimeIndex, prices) -> pd.DataFrame:
    prices = np.asarray(prices, dtype=float)
    return frame(index, prices, prices, prices, prices)


class Script:
    """Decisions by bar number; bars not listed return an empty Decision (keep everything)."""

    def __init__(self, decisions: dict[int, Decision]):
        self.decisions = decisions
        self.books: list[BookView] = []

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        self.books.append(book)
        return self.decisions.get(len(self.books) - 1, Decision())


def simulate(bars: pd.DataFrame, decisions: dict[int, Decision], contracts=None, **config):
    script = Script(decisions)
    contracts = r1_contracts(bars.index, EXPIRIES) if contracts is None else contracts
    result = run_intraday(script, "X", IntradayInput(SPEC, bars, contracts), IntradayConfig(**config))
    return result, script


def test_r1_contract_of_each_bar_switches_at_1330_on_the_last_trading_day():
    index = thalassa_stamps(["2026-10-20", "2026-10-21", "2026-10-22"])
    contracts = pd.Series(r1_contracts(index, EXPIRIES), index=index.tz_convert("Asia/Taipei").tz_localize(None))
    assert contracts[pd.Timestamp("2026-10-20 13:45")] == "202610"
    assert contracts[pd.Timestamp("2026-10-20 15:00")] == "202610"  # night session of the last trading day
    assert contracts[pd.Timestamp("2026-10-21 04:55")] == "202610"
    assert contracts[pd.Timestamp("2026-10-21 13:25")] == "202610"
    assert contracts[pd.Timestamp("2026-10-21 13:30")] == "202611"
    assert contracts[pd.Timestamp("2026-10-21 15:00")] == "202611"
    assert (contracts != contracts.shift()).sum() == 2  # the first bar and one switch
    with pytest.raises(ValueError, match="settlement calendar ends"):
        r1_contracts(index, {"202609": "2026-09-16"})


def test_a_position_rolls_at_the_switch_and_never_books_the_gap_between_months():
    index = thalassa_stamps(["2026-10-21"], night=False)
    switch = at(index, "2026-10-21 13:30")
    prices = np.where(np.arange(len(index)) < switch, 100.0, 150.0)
    result, _ = simulate(flat(index, prices), {0: Decision({"X": 1})})

    fills = result.fills
    assert list(fills["reason"]) == ["rebalance", "roll", "roll"]
    assert list(fills["contract"]) == ["202610", "202610", "202611"]
    assert list(fills["price"]) == [101.0, 99.0, 151.0]  # both roll legs slip one tick
    assert list(fills["session"]) == ["close", "close", "open"]
    assert fills["date"].iloc[1] == index[switch - 1] and fills["date"].iloc[2] == index[switch]
    # entry 101 -> 99 old month, 151 -> 150 new month; three legs of 20 TWD; the 50-point gap is not PnL
    assert result.equity.iloc[-1] == pytest.approx(CAPITAL + (99 - 101) * 10 + (150 - 151) * 10 - 3 * 20)
    assert list(result.events["kind"]) == ["roll"]
    trip = result.round_trips().iloc[0]
    assert pd.isna(trip["exit_time"]) and trip["legs"] == 3 and trip["units"] == 1


def test_a_flat_book_does_not_roll_and_a_closed_round_trip_excludes_the_gap():
    index = thalassa_stamps(["2026-10-21"], night=False)
    switch = at(index, "2026-10-21 13:30")
    prices = np.where(np.arange(len(index)) < switch, 100.0, 150.0)
    result, _ = simulate(flat(index, prices), {0: Decision({"X": -1}), switch + 1: Decision({"X": 0})})
    trips = result.round_trips()
    assert len(trips) == 1 and trips["legs"].iloc[0] == 4 and trips["side"].iloc[0] == "short"
    # short 99 -> cover 101 (old month), short 149 -> cover 151 (new month)
    assert trips["gross_pnl"].iloc[0] == pytest.approx((99 - 101) * 10 + (149 - 151) * 10)
    assert trips["pnl"].iloc[0] == pytest.approx(trips["gross_pnl"].iloc[0] - 4 * 20)
    assert result.equity.iloc[-1] == pytest.approx(CAPITAL + trips["pnl"].iloc[0])

    untouched, _ = simulate(flat(index, prices), {0: Decision({"X": 1}), switch - 3: Decision({"X": 0})})
    assert "roll" not in set(untouched.fills["reason"]) and untouched.events.empty


def _two_bars(bar1: tuple[float, float, float, float]) -> pd.DataFrame:
    index = thalassa_stamps(["2026-10-22"], night=False)[:3]
    o, h, lo, c = zip((100, 100, 100, 100), bar1, (100, 100, 100, 100), strict=True)
    return frame(index, o, h, lo, c)


@pytest.mark.parametrize(
    ("bar1", "config", "fill", "reason"),
    [
        ((99, 101, 94, 96), {}, 95.0, "stop"),  # trades through the stop: fills at the stop
        ((90, 92, 88, 91), {}, 90.0, "stop"),  # opens through it: gap-aware fill at the open
        ((90, 92, 88, 91), {"gap_aware_stops": False}, 95.0, "stop"),
        ((90, 92, 88, 91), {"stop_slippage_ticks": 2}, 88.0, "stop"),
        ((101, 112, 94, 100), {}, 95.0, "stop"),  # touches both: the stop wins
        ((115, 116, 114, 115), {}, 110.0, "take_profit"),  # gaps over the target: still at the target
    ],
)
def test_protective_orders_fill_like_the_v1_runner(bar1, config, fill, reason):
    entry = Decision({"X": 1}, protective={"X": Protect(stop=95.0, take_profit=110.0)})
    result, script = simulate(_two_bars(bar1), {0: entry}, **config)
    exit_fill = result.fills.iloc[1]
    assert (exit_fill["price"], exit_fill["reason"], exit_fill["session"]) == (fill, reason, "intraday")
    assert exit_fill["slippage_twd"] == 0.0 and exit_fill["commission"] == 20.0
    book = script.books[1]
    assert book.positions["X"] == 0 and dict(book.protective) == {}
    assert [f.reason for f in book.fills_today] == ["rebalance", reason]  # every fill since the last decision


def test_a_new_position_never_inherits_the_previous_orders_and_an_update_replaces_them():
    index = thalassa_stamps(["2026-10-22"], night=False)[:4]
    bars = frame(index, [100, 100, 100, 100], [100, 100, 100, 200], [100, 100, 100, 100], [100, 100, 100, 100])
    decisions = {0: Decision({"X": 1}, protective={"X": Protect(stop=95.0)}), 1: Decision({"X": -1})}
    result, script = simulate(bars, decisions)
    assert "stop" not in set(result.fills["reason"])  # the short is not stopped by the long's 95
    assert dict(script.books[2].protective) == {}

    moved = {0: Decision({"X": 1}, protective={"X": Protect(stop=95.0)}), 1: Decision(protective={"X": Protect(99.0)})}
    bars = frame(index, [100] * 4, [100] * 4, [100, 100, 100, 98], [100] * 4)
    result, script = simulate(bars, moved)
    assert script.books[2].protective["X"] == Protect(99.0)
    assert result.fills.iloc[-1]["price"] == 99.0 and result.fills.iloc[-1]["reason"] == "stop"


def test_the_daily_engine_refuses_protective_orders():
    class Protected:
        def on_bar(self, view, book):
            return Decision({"TMF": 1}, protective={"TMF": Protect(stop=1.0)})

    with pytest.raises(NotImplementedError, match="intraday engine"):
        run(Protected(), tmf(flat_bars([100.0, 101.0])))
