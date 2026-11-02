"""Step 8: stocks / ETFs through the daily stages' data path and the daily shadow paper (no DB).

The synthetic store of ``daily_paper_support`` (TX, TAIEX, the real 2026 schedule) plus 0050 and
00631L bars on the same sessions, a 00631L 1 -> 4 split, and the corporate-actions dataset.
"""

from __future__ import annotations

import dataclasses
import shutil
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline.daily_stages import (
    STOCK_UNDERLYING,
    WindowStore,
    instrument_kinds,
    price_points,
    scaled_costs,
    underlying_daily,
    underlying_product,
)
from poseidon.pipeline.engine import BookView, ContractSpec, Decision, EngineConfig, MarketView
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, MARKET_CALENDAR, TAIEX
from poseidon.pipeline.refdata.store import RefdataStore, frame_sha256
from poseidon.pipeline.refdata.tw_corporate_actions import ACTION_COLUMNS
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.refdata.tw_stocks import COLUMNS
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.shadow_paper_daily import CompiledStrategy, PaperInstrument
from poseidon.pipeline.stages import UpstreamResult
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.v3.compiler import NotCompilable
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050, tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.instruments import INSTRUMENTS
from tests.pipeline.daily_paper_support import AsOf, DailyDriver, _by_month, build_store, morning_after, trading_days

DAYS = trading_days(last="2026-10-21")
SPLIT_DAY = pd.Timestamp("2026-03-31")  # 00631L 1 -> 4 (the real one was 1 -> 22)
SPLIT_RATIO = 4.0
PAPER_START = pd.Timestamp("2026-09-01")


def _commit(store, dataset, frame, column):
    store.commit(
        dataset,
        schema_version=1,
        date_column=column,
        partitions=_by_month(frame, column),
        sources=[],
        code_version="t",
        full=True,
    )


@pytest.fixture(scope="module")
def stock_store(tmp_path_factory):
    store = build_store(tmp_path_factory.mktemp("refdata"), last="2026-10-21")
    rng = np.random.default_rng(8)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(DAYS))))
    lev = 40.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, len(DAYS))))
    lev_raw = np.where(DAYS >= SPLIT_DAY, lev / SPLIT_RATIO, lev)
    rows = []
    for symbol, closes in (("0050", base), ("00631L", lev_raw)):
        frame = pd.DataFrame({"trade_date": DAYS, "symbol": symbol, "close": closes})
        frame["open"] = frame["close"] * 0.998
        frame["high"] = frame["close"] * 1.01
        frame["low"] = frame["close"] * 0.99
        frame["volume"] = 1000.0
        if symbol == "00631L":
            frame = frame[frame["trade_date"] != DAYS[200]]  # one session without a bar (a halt)
        rows.append(frame[COLUMNS])
    _commit(store, STOCKS, pd.concat(rows, ignore_index=True), "trade_date")
    prev = float(lev[DAYS.get_loc(SPLIT_DAY) - 1])
    action = dict.fromkeys(ACTION_COLUMNS)
    action.update(
        date=SPLIT_DAY,
        market="twse",
        symbol="00631L",
        action_type="split",
        prev_close=prev,
        reference_price=prev / SPLIT_RATIO,
        factor=1 / SPLIT_RATIO,
        source="test",
    )
    _commit(store, ACTIONS, pd.DataFrame([action]), "date")
    return store


# ----------------------------------------------------------------------------- window identity


def test_window_identity_is_unchanged_for_a_futures_spec_and_covers_stock_data_for_a_stock_spec(stock_store):
    upper = date(2026, 6, 1)
    futures_only = WindowStore(stock_store, upper)  # the default: as before step 8
    assert (
        futures_only.data_sha256()
        == WindowStore(stock_store, upper, instrument_kinds(tmf_dynamic_leverage_3x())).data_sha256()
    )

    def identity(names):
        frames = {name: frame_sha256(stock_store.read(name, end=upper if name != CALENDAR else None)) for name in names}
        return content_sha256({"upper": upper.isoformat(), "frames": frames})

    assert futures_only.data_sha256() == identity([FUTURES, CALENDAR, TAIEX])
    stocks = WindowStore(stock_store, upper, instrument_kinds(etf_rotation_00631l_0050()))
    assert stocks.kinds == frozenset({"stock"}) and stocks.data_sha256() == identity([TAIEX, STOCKS, ACTIONS])
    assert stocks.exists(STOCKS) and not stocks.exists("nope")
    assert instrument_kinds({"instruments": {"a": {"symbol": "TMF"}, "b": {"symbol": "0050"}}}) == {"future", "stock"}
    assert instrument_kinds({"instruments": "junk"}) == frozenset() and instrument_kinds(None) == frozenset()


def test_price_points_of_a_stock_are_its_own_first_and_max_close(stock_store):
    store = WindowStore(stock_store, date(2026, 6, 1), frozenset({"stock"}))
    first, last = date(2025, 3, 3), date(2025, 12, 31)
    closes = store.read(STOCKS, start=first, end=last).query("symbol == '0050'")["close"]
    assert price_points(store, INSTRUMENTS["0050"], first, last) == (float(closes.iloc[0]), float(closes.max()))
    assert price_points(store, INSTRUMENTS["0050"], date(2027, 1, 1), date(2027, 2, 1)) is None


def test_scaled_costs_scale_the_stock_commission_rate_too():
    spec = ContractSpec.from_instrument(INSTRUMENTS["0050"])
    doubled = scaled_costs(spec, 2.0)
    assert (doubled.commission_rate, doubled.tax_rate, doubled.slippage_ticks) == approx(
        (2 * spec.commission_rate, 2 * spec.tax_rate, 2 * spec.slippage_ticks)
    )
    assert scaled_costs(spec, 1.0) is spec


# ----------------------------------------------------------------------------- the 1x underlying


def _ctx(spec: dict):
    report = {"template": spec}
    return type(
        "Ctx", (), {"upstream": {"spec_validate": UpstreamResult("spec_validate", 1, "0" * 64, "1" * 64, report)}}
    )()


def test_underlying_is_0050_for_a_stock_spec_tx_for_the_tx_family_and_undefined_for_a_mix():
    assert underlying_product(_ctx(etf_rotation_00631l_0050())) == STOCK_UNDERLYING == "0050"
    assert underlying_product(_ctx(tmf_dynamic_leverage_3x())) == "TX"
    mixed = tmf_dynamic_leverage_3x()
    mixed["instruments"]["etf"] = {"symbol": "0050"}
    with pytest.raises(NotCompilable, match="not for a mix"):
        underlying_product(_ctx(mixed))


def test_stock_underlying_is_total_return_adjusted_across_the_split(stock_store):
    through = date(2026, 6, 1)
    store = WindowStore(stock_store, through, frozenset({"stock"}))
    daily = underlying_daily(store, "00631L", through=through)
    raw = stock_store.read(STOCKS, end=through).query("symbol == '00631L'").set_index("trade_date")["close"]
    assert list(daily.columns) == ["open", "high", "low", "close"] and daily.index[-1] == raw.index[-1].date()
    assert daily["close"].iloc[-1] == approx(float(raw.iloc[-1]))  # the latest bar keeps its price
    before, after = SPLIT_DAY - pd.Timedelta(days=1), SPLIT_DAY
    adjusted_step = daily["close"][after.date()] / daily["close"][raw[:before].index[-1].date()]
    raw_step = raw[after] * SPLIT_RATIO / raw[:before].iloc[-1]
    assert adjusted_step == approx(raw_step)  # no jump on the split day
    assert daily["close"][raw[:before].index[-1].date()] == approx(float(raw[:before].iloc[-1]) / SPLIT_RATIO)
    assert (daily["high"] >= daily["close"]).all() and (daily["low"] <= daily["close"]).all()
    sessions = DAYS[: DAYS.searchsorted(pd.Timestamp(through), side="right")]
    missing = sessions[200]  # the session 00631L lacks a bar for in the fixture: carried forward
    assert len(daily) == len(sessions) and daily["close"][missing.date()] == approx(
        daily["close"][sessions[199].date()]
    )
    base = underlying_daily(store, "0050", through=through)  # no action: the raw bars
    raw_base = stock_store.read(STOCKS, end=through).query("symbol == '0050'")["close"].to_numpy()
    assert base["close"].to_numpy() == approx(raw_base)


def test_a_stock_underlying_without_the_actions_dataset_is_not_compilable(stock_store, tmp_path):
    root = tmp_path / "refdata"
    shutil.copytree(stock_store.root, root, ignore=lambda _dir, names: [n for n in names if n == ACTIONS])
    through = date(2026, 6, 1)
    store = WindowStore(RefdataStore(root), through, frozenset({"stock"}))
    with pytest.raises(NotCompilable, match=f"benchmark underlying: {ACTIONS} is missing"):
        underlying_daily(store, "00631L", through=through)


# ----------------------------------------------------------------------------- the daily paper


class HoldBoth:
    """Half the equity in each ETF, re-sized every day (the paper's consistency check needs fills)."""

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        units = {}
        for alias in ("lev", "base"):
            price = float(view[alias].close[-1]) * 1000.0
            units[alias] = int(book.equity * 0.5 // price) if book.equity > 0 else 0
        return Decision(units, {"lev": 0.5, "base": 0.5})


def stock_strategy(aux=None) -> CompiledStrategy:
    return CompiledStrategy(
        factory=HoldBoth,
        instruments={
            alias: PaperInstrument(symbol, ContractSpec.from_instrument(INSTRUMENTS[symbol]), roll_days_before=0)
            for alias, symbol in (("lev", "00631L"), ("base", "0050"))
        },
        engine=EngineConfig(initial_capital=1_000_000.0, execution="next_open", start=pd.Timestamp("2025-04-01")),
        data_start=date(2025, 1, 2),
        train_end=date(2026, 6, 1),
        fingerprint="f" * 64,
        aux=aux or {},
    )


def test_load_inputs_reads_stock_bars_and_books_the_split(stock_store):
    as_of = DAYS[DAYS <= "2026-09-10"][-1]
    inputs = spd.load_inputs(
        AsOf(stock_store, as_of), stock_strategy({"taiex": TAIEX}), spd.DEFAULT_DAILY_SHADOW_PAPER, PAPER_START.date()
    )
    assert inputs.cutoff == as_of and len(inputs.closures) == 0
    assert set(inputs.sources) == {MARKET_CALENDAR, TAIEX, STOCKS, ACTIONS}
    assert set(inputs.last_dates) == {f"{STOCKS}:00631L", f"{STOCKS}:0050", "aux:taiex", f"{ACTIONS}:complete_through"}
    assert all(day == as_of.date().isoformat() for day in inputs.last_dates.values())
    past = inputs.calendar[inputs.calendar <= as_of]
    assert past.equals(DAYS[: len(past)]) and past[-1] == as_of and inputs.calendar[-1] > as_of
    lev = inputs.futures["lev"]
    assert lev.spec.kind == "stock" and lev.roll_days_before == 0 and list(lev.bars["date"]) == list(past)
    booked = lev.bars.set_index("date")
    assert booked.loc[SPLIT_DAY, "unit_ratio"] == approx(SPLIT_RATIO)
    assert (booked.drop(SPLIT_DAY)["unit_ratio"] == 1.0).all() and (
        inputs.futures["base"].bars["unit_ratio"] == 1.0
    ).all()
    assert (~booked["traded"]).sum() == 1 and not booked.loc[DAYS[200], "traded"]  # carried forward, no fill
    # the split is part of that day's input hash: a revised action is a revision of the day
    without = dataclasses.replace(lev, bars=lev.bars.assign(unit_ratio=1.0))
    hashes, _history = spd.input_hashes({"lev": without, "base": inputs.futures["base"]}, inputs.aux, date(2026, 3, 1))
    with_split, _ = spd.input_hashes(inputs.futures, inputs.aux, date(2026, 3, 1))
    differ = sorted(day for day in with_split if with_split[day] != hashes[day])
    assert differ == [SPLIT_DAY.date().isoformat()]


def test_the_cutoff_waits_for_the_slowest_stock(stock_store):
    class Late(AsOf):
        def read(self, dataset, start=None, end=None):
            frame = super().read(dataset, start=start, end=end)
            if dataset == STOCKS:
                keep = (frame["symbol"] != "0050") | (frame["trade_date"] <= pd.Timestamp("2026-09-03"))
                return frame[keep].reset_index(drop=True)
            return frame

    inputs = spd.load_inputs(
        Late(stock_store, "2026-09-10"), stock_strategy(), spd.DEFAULT_DAILY_SHADOW_PAPER, PAPER_START.date()
    )
    assert inputs.cutoff == pd.Timestamp("2026-09-03")
    assert inputs.last_dates == {
        f"{STOCKS}:00631L": "2026-09-10",
        f"{STOCKS}:0050": "2026-09-03",
        f"{ACTIONS}:complete_through": "2026-09-10",
    }


def test_the_daily_paper_books_stock_days_and_its_replay_matches(stock_store, tmp_path):
    driver = DailyDriver(tmp_path, None, strategy=stock_strategy())
    days = DAYS[(DAYS >= PAPER_START) & (DAYS <= "2026-09-12")]
    results = []
    for day in days:
        driver.reader = AsOf(stock_store, day)
        results.append(driver.wake(morning_after(day)))
    assert all(r.verdict is Verdict.PASS and r.is_waiting for r in results)
    ledger = driver.ledger()
    assert [d["trading_day"] for d in ledger["days"]] == list(days.strftime("%Y-%m-%d"))
    report = results[-1].report
    assert report["consistency"]["ratio"] == 1.0 and report["consistency"]["compared"] == len(days)
    assert report["engine"]["instruments"]["lev"] == {"product": "00631L", "symbol": "00631L", "roll_days_before": 0}
    fills = [fill for day in ledger["days"] for fill in day["fills"]]
    assert fills and all(fill["tax"] == 0 for fill in fills if fill["units"] > 0)  # the stock tax is on sells only
    assert any(fill["commission"] > 0 for fill in fills)
