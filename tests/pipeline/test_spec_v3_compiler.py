"""The spec v3 compiler on a synthetic refdata store (layout and columns of the real datasets)."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import ContractSpec
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, TAIEX, TAIEX_TR
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET as OPTION_QUOTES
from poseidon.pipeline.refdata.taifex_options import FSP_DATASET as OPTION_FINAL
from poseidon.pipeline.v3 import compiler
from poseidon.pipeline.v3.compiler import (
    OPTION_DATASETS,
    HedgeCoverageError,
    NotCompilable,
    compile_spec,
    futures_bars,
    plan,
    resolve,
)
from poseidon.pipeline.v3.examples import PROTECTIVE_PUT, tmf_dynamic_leverage_3x
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


def test_index_signal_reads_the_aux_series_under_levtx_conventions(store):
    levtx = Conventions(execution="signal_close", signal_lag=1)
    compiled = compile_spec(spec(rounding="nearest", signal="index:taiex"), start=START, store=store, conventions=levtx)
    assert compiled.plan.signal_indices == ("taiex",)
    assert compiled.config.execution == "signal_close"
    strategy = compiled.strategy()
    res = compiled.run(strategy)
    assert res.fills["session"].eq("close").all()
    assert strategy.state("main", "leverage.1")["rows"] == len(compiled.aux["taiex"].loc[: res.equity.index[-1]]) - 1
    assert compile_spec(spec(), start=START, store=store).plan.signal_indices == ()
    assert compiler.AUX_DATASETS == {"taiex": TAIEX, "taiex_tr": TAIEX_TR, "vix": "cboe_vix_daily"}
    assert compiler.ALWAYS_ATTACHED == ("taiex", "taiex_tr")  # the VIX is attached only when a signal reads it


def test_index_signal_needs_its_own_warm_up_rows(store, tmp_path):
    late = RefdataStore(tmp_path)
    _commit(late, FUTURES, store.read(FUTURES), "trade_date")
    _commit(late, CALENDAR, store.read(CALENDAR), "last_trading_day")
    _commit(late, TAIEX_TR, store.read(TAIEX_TR), "trade_date")
    taiex = store.read(TAIEX)
    _commit(late, TAIEX, taiex[taiex["trade_date"] >= START - pd.Timedelta(days=30)], "trade_date")
    with pytest.raises(NotCompilable, match=r"signal index:taiex: \d+ rows .* need 50 for warm-up"):
        compile_spec(spec(signal="index:taiex"), start=START, store=late)
    lagged = Conventions(signal_lag=1)
    assert compile_spec(spec(), start=START, store=late, conventions=lagged).plan.signal_indices == ()


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


# ----------------------------------------------------------------------------- hedges (TXO, M4)


def _hedged(**kwargs) -> dict:
    doc = spec(**kwargs)
    doc["sleeves"][0]["hedges"] = [copy.deepcopy(PROTECTIVE_PUT)]
    return doc


@pytest.fixture(scope="module")
def hedged_store(store, tmp_path_factory):
    """The futures store plus TXO puts on every monthly expiry: strikes every 100 points around the index,
    settlement = intrinsic + 40 (one row per strike per day), final settlement = the index on the expiry day."""

    _frame, expiries, index = synthetic_contracts(n_days=480)
    rows, finals = [], []
    for month, expiry in expiries.items():
        for t, level in index["close"].items():
            if t > expiry or (expiry - t).days > 95:
                continue
            for strike in np.arange(round(level / 100) * 100 - 1500, round(level / 100) * 100 + 600, 100.0):
                settle = np.nan if t == expiry else max(strike - level, 0.0) + 40.0
                rows.append((t, "day", month, "monthly", expiry, strike, "P", settle))
                rows.append((t, "night", month, "monthly", expiry, strike, "P", np.nan))
                rows.append((t, "day", month, "monthly", expiry, strike, "C", 1.0))
        if expiry <= index.index[-1]:
            finals.append((expiry, month, "monthly", float(index["close"][expiry])))
    quotes = pd.DataFrame(
        rows,
        columns=["trade_date", "session", "contract_month", "expiry_kind", "expiry_date", "strike", "cp", "settlement"],
    )
    final = pd.DataFrame(
        finals, columns=["final_settlement_date", "contract_month", "expiry_kind", "final_settlement_price"]
    )
    hedged = RefdataStore(tmp_path_factory.mktemp("refdata_txo"))
    for dataset in (FUTURES, CALENDAR, TAIEX, TAIEX_TR):
        _commit(hedged, dataset, store.read(dataset), store.manifest(dataset)["date_column"])
    _commit(hedged, OPTION_QUOTES, quotes, "trade_date")
    _commit(hedged, OPTION_FINAL, final, "final_settlement_date")
    return hedged


def test_a_hedged_spec_compiles_to_an_option_chain_and_a_put_policy(hedged_store):
    compiled = compile_spec(_hedged(), start=START, store=hedged_store)
    (policy,) = compiled.hedges
    assert (policy.name, policy.underlying, policy.chain, policy.moneyness, policy.roll_days, policy.cover_ratio) == (
        "core.protective_put",
        "core",
        "TXO",
        0.05,
        5,
        1.0,
    )
    chain = compiled.options["TXO"]
    assert set(chain.quotes["cp"]) == {"P"} and chain.quotes["settlement"].notna().any()
    assert set(OPTION_DATASETS) <= set(compiled.inputs) and any(note.startswith("TXO:") for note in compiled.notes)
    res = compiled.run()
    fills = res.fills[res.fills["instrument"] == "core.protective_put"]
    assert len(fills) and set(fills["reason"]) <= {"rebalance", "roll"} and (fills["session"] == "close").all()
    # one put per five TMF of the engine's final futures order at each decision
    rows = res.positions_daily.set_index(["instrument", "date"])
    assert (rows.loc["core.protective_put", "target_units"] == rows.loc["core", "order_units"] // 5).all()
    plain = compile_spec(spec(), start=START, store=hedged_store)
    assert not plain.hedges and not plain.options and not set(OPTION_DATASETS) & set(plain.inputs)


def test_a_hedged_spec_passes_the_truncation_audit(hedged_store):
    report = compile_spec(_hedged(), start=START, store=hedged_store).audit(n_cuts=8, seed=5)
    assert len(report.cuts) == 8 and report.ok, report.mismatches


def test_more_than_five_percent_unavailable_hedge_days_fail_the_run(hedged_store):
    compiled = compile_spec(_hedged(), start=START, store=hedged_store)
    chain = compiled.options["TXO"]
    far = dataclasses.replace(chain, quotes=chain.quotes.assign(strike=chain.quotes["strike"] - 5000.0))
    broken = dataclasses.replace(compiled, options={"TXO": far})
    with pytest.raises(HedgeCoverageError, match=r"hedge unavailable on \d+ of \d+ hedged days"):
        broken.run()
    assert "hedge_unavailable" in set(broken.engine_run().events["kind"])  # the report path does not raise


def test_the_window_store_clips_the_chain_and_hashes_it_only_for_hedged_runs(hedged_store):
    from poseidon.pipeline.daily_stages import WindowStore

    upper = START.date()
    clipped = WindowStore(hedged_store, upper, extra=OPTION_DATASETS)
    rows = clipped.read(OPTION_QUOTES)
    assert (
        rows["trade_date"].max() <= START
        and set(rows["cp"]) == {"P"}
        and set(rows.columns) == set(compiler.OPTION_QUOTE_FIELDS)
    )
    plain = WindowStore(hedged_store, upper)
    assert plain.data_sha256() != clipped.data_sha256()
    assert plain.data_sha256() == WindowStore(hedged_store, upper).data_sha256()


def test_daily_shadow_paper_refuses_a_hedged_spec_instead_of_dropping_the_hedge():
    from types import SimpleNamespace

    from poseidon.pipeline.daily_stages import compile_strategy
    from poseidon.pipeline.shadow_paper_daily import StrategyNotCompiled

    ctx = SimpleNamespace(
        config={"components": {}}, upstream={"optimize": SimpleNamespace(report={"best_spec": _hedged()})}
    )
    with pytest.raises(StrategyNotCompiled, match="does not run hedges"):
        compile_strategy(ctx)
