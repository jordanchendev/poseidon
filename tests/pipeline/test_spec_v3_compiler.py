"""The spec v3 compiler on a synthetic refdata store (layout and columns of the real datasets)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import ContractSpec
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, TAIEX, TAIEX_TR
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3 import compiler
from poseidon.pipeline.v3.compiler import NotCompilable, compile_spec, futures_bars, plan, resolve
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.instruments import INSTRUMENTS
from poseidon.pipeline.v3.runtime import Conventions

from .engine.support import synthetic_contracts
from .spec_support import resolved_spec

NIGHT_FROM = pd.Timestamp("2024-10-01")  # synthetic "after-hours session" start (inside the warm-up)
START = pd.Timestamp("2024-11-01")


def _commit(store: RefdataStore, dataset: str, frame: pd.DataFrame, column: str) -> None:
    parts = {key: group.reset_index(drop=True) for key, group in frame.groupby(frame[column].dt.strftime("%Y-%m"))}
    store.commit(dataset, schema_version=1, date_column=column, partitions=parts, sources=[], code_version="test")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    frame, expiries, index = synthetic_contracts(n_days=480)
    day = frame.rename(columns={"date": "trade_date", "contract": "contract_month"}).assign(
        product="TX", session="day", settlement=lambda f: f["close"]
    )
    night = day[day["trade_date"] >= NIGHT_FROM].assign(
        session="night", open=lambda f: f["open"] - 7, high=lambda f: f["high"] + 3, low=lambda f: f["low"] - 3
    )
    weekly = day[day["contract_month"] == day["contract_month"].iloc[0]].assign(
        contract_month=lambda f: f["contract_month"] + "W2"
    )
    mtx = day.assign(product="MTX", close=1.0)
    quotes = pd.concat([day, night, weekly, mtx], ignore_index=True).sort_values(
        ["trade_date", "product", "contract_month"]
    )
    last = day["trade_date"].max()
    calendar = pd.DataFrame(
        {
            "last_trading_day": list(expiries.values()),
            "product": "TX",
            "contract_month": list(expiries),
            "status": ["settled" if e <= last else "active" for e in expiries.values()],
        }
    )
    taiex = index.rename_axis("trade_date").reset_index().assign(open=np.nan, high=np.nan, low=np.nan)
    root = tmp_path_factory.mktemp("refdata")
    store = RefdataStore(root)
    _commit(store, FUTURES, quotes, "trade_date")
    _commit(store, CALENDAR, calendar, "last_trading_day")
    _commit(store, TAIEX, taiex, "trade_date")
    _commit(store, TAIEX_TR, taiex[["trade_date", "close"]].assign(close=lambda f: f["close"] * 1.5), "trade_date")
    return store


def spec(**kwargs):
    return resolved_spec(ma_days=50, **kwargs)


# ----------------------------------------------------------------------------- refdata -> bars


def test_futures_bars_merge_the_after_hours_session_into_the_next_trading_day():
    d1, d2, d3 = pd.Timestamp("2026-10-21"), pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-23")
    quotes = pd.DataFrame(
        [
            (d1, "TX", "202611", "day", 100.0, 110.0, 95.0, 105.0, 105.0),
            (d2, "TX", "202611", "night", 104.0, 120.0, 101.0, 118.0, np.nan),  # opened 15:00 on d1
            (d2, "TX", "202611", "day", 117.0, 119.0, 90.0, 99.0, 99.5),
            (d2, "TX", "202611W4", "day", 1.0, 1.0, 1.0, 1.0, 1.0),
            (d2, "MTX", "202611", "day", 1.0, 1.0, 1.0, 1.0, 1.0),
            (d3, "TX", "202611", "night", 98.0, 99.0, 97.0, 98.0, np.nan),  # no day session yet
        ],
        columns=["trade_date", "product", "contract_month", "session", "open", "high", "low", "close", "settlement"],
    )
    got = futures_bars(quotes, "TX")
    assert got.to_dict("records") == [
        {"date": d1, "contract": "202611", "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "settle": 105.0},
        {"date": d2, "contract": "202611", "open": 104.0, "high": 120.0, "low": 90.0, "close": 99.0, "settle": 99.5},
    ]


# ----------------------------------------------------------------------------- spec -> engine


def test_compile_wires_the_spec_and_refdata_into_engine_inputs(store):
    compiled = compile_spec(spec(days_before=2), start=START, store=store)
    fut = compiled.futures["core"]
    assert fut.spec == ContractSpec.from_instrument(INSTRUMENTS["TMF"])
    assert fut.roll_days_before == 2
    assert set(fut.bars["contract"]) <= set(fut.expiries) and all(len(c) == 6 for c in fut.bars["contract"])
    assert (fut.bars["close"] != 1.0).all()  # TX rows only (TMF is priced from TX), MTX ignored
    quotes = store.read(FUTURES)
    day_open = quotes[(quotes["session"] == "day") & (quotes["product"] == "TX")].set_index(
        ["trade_date", "contract_month"]
    )["open"]
    expected_open = day_open.reindex(pd.MultiIndex.from_frame(fut.bars[["date", "contract"]])).to_numpy()
    after_hours = (fut.bars["date"] >= NIGHT_FROM).to_numpy()
    assert after_hours.any() and (~after_hours).any()
    np.testing.assert_array_equal(fut.bars["open"].to_numpy()[after_hours], expected_open[after_hours] - 7)
    np.testing.assert_array_equal(fut.bars["open"].to_numpy()[~after_hours], expected_open[~after_hours])
    cfg = compiled.config
    assert (cfg.execution, cfg.initial_capital, cfg.leverage_cap, cfg.max_margin_utilisation, cfg.min_change_units) == (
        "next_open",
        1_000_000.0,
        5.0,
        0.8,
        1,
    )
    assert cfg.start == START and cfg.end is None
    data_days = pd.DatetimeIndex(sorted(fut.bars["date"].unique()))
    assert compiled.calendar[: len(data_days)].equals(data_days) and len(compiled.calendar) == len(data_days) + 10
    assert compiled.inputs == {name: store.content_sha256(name) for name in (FUTURES, CALENDAR, TAIEX, TAIEX_TR)}
    assert set(compiled.aux) == {"taiex", "taiex_tr"}
    notes = " ".join(compiled.notes)
    assert "TMF: priced with TX prices" in notes and "projected by the third-Wednesday rule" in notes
    assert "08:45" not in notes  # the window starts after the synthetic after-hours session existed
    assert compiled.plan.warmup_bars == 50  # max(regime.ma_band 50, risk.vol_cap 21)


def test_next_open_fills_at_the_after_hours_open(store):
    compiled = compile_spec(spec(), start=START, store=store)
    res = compiled.run()
    fills = res.fills[res.fills["session"] == "open"].iloc[0]
    bar = compiled.futures["core"].bars.set_index(["date", "contract"]).loc[(fills["date"], fills["contract"])]
    assert fills["ref_price"] == bar["open"]


def test_more_history_than_the_warm_up_does_not_change_the_run(store, monkeypatch):
    base = compile_spec(spec(), start=START, store=store)
    monkeypatch.setattr(compiler, "warmup_calendar_days", lambda bars, frequency: 3 * bars * 365 // 245)
    longer = compile_spec(spec(), start=START, store=store)
    assert longer.calendar[0] < base.calendar[0]
    a, b = base.run(), longer.run()
    pd.testing.assert_frame_equal(a.daily_pnl, b.daily_pnl)
    pd.testing.assert_frame_equal(a.fills, b.fills)


def test_missing_warm_up_is_not_compilable(store):
    first = store.read(FUTURES)["trade_date"].min()
    with pytest.raises(NotCompilable, match="warm-up"):
        compile_spec(spec(), start=first + pd.Timedelta(days=20), store=store)


def test_conventions_override_execution_and_route_the_signal(store):
    levtx = Conventions(execution="signal_close", signal_aux={"core:adjusted": "taiex"}, signal_lag=1)
    compiled = compile_spec(spec(rounding="nearest"), start=START, store=store, conventions=levtx)
    assert compiled.config.execution == "signal_close"
    strategy = compiled.strategy()
    res = compiled.run(strategy)
    assert res.fills["session"].eq("close").all()
    assert strategy.state("main", "leverage.1")["rows"] == len(compiled.aux["taiex"].loc[: res.equity.index[-1]]) - 1
    with pytest.raises(NotCompilable, match="aux"):
        compile_spec(spec(), start=START, store=store, conventions=Conventions(signal_aux={"core:adjusted": "nope"}))


def test_compiled_spec_passes_the_truncation_audit(store):
    report = compile_spec(spec(), start=START, store=store).audit(n_cuts=10, seed=3)
    assert len(report.cuts) == 10 and report.ok, report.mismatches


# ----------------------------------------------------------------------------- what does not compile


def test_unsupported_shapes_are_not_compilable():
    with pytest.raises(ValueError, match=r"\$param"):
        plan(tmf_dynamic_leverage_3x())
    intraday = spec()
    intraday["frequency"] = "5m"
    with pytest.raises(NotCompilable, match="intraday"):
        plan(intraday)
    with_exit = spec()
    with_exit["sleeves"][0]["exits"] = [{"component": "exit.stop_loss", "params": {}}]
    with pytest.raises(NotCompilable, match="exits"):
        plan(with_exit)
    twice = spec()
    twice["sleeves"].append(copy.deepcopy(twice["sleeves"][0]) | {"id": "second"})
    with pytest.raises(NotCompilable, match="two sleeves"):
        plan(twice)


def test_resolve_validates_and_binds_params():
    doc = resolve(tmf_dynamic_leverage_3x(), {"ma_days": 225, "vol_k": 0.5})
    links = doc["sleeves"][0]["leverage"]
    assert (links[1]["params"]["ma_days"], links[1]["params"]["band_pct"], links[2]["params"]["k"]) == (225, 0.02, 0.5)
    with pytest.raises(ValueError, match="off their declared grid"):
        resolve(tmf_dynamic_leverage_3x(), {"ma_days": 210})
    with pytest.raises(ValueError, match="unknown parameters"):
        resolve(tmf_dynamic_leverage_3x(), {"nope": 1})
    broken = tmf_dynamic_leverage_3x()
    broken["sleeves"][0]["sizing"]["component"] = "sizing.nope"
    with pytest.raises(ValueError, match="does not validate"):
        resolve(broken)
