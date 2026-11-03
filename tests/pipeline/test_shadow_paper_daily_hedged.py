"""Daily shadow paper with a protective put (``data.option_chain``): the chain is read with the futures,
the cutoff waits for it, the put legs are priced from the TXO rows of the booked days.

Synthetic refdata from ``daily_paper_support`` (real 2026 TWSE schedule, the 2026-10-23 typhoon) plus a
synthetic TXO chain (``write_txo``); ``AsOf`` replays what the store held on each morning.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline.daily_stages import compile_strategy
from poseidon.pipeline.engine import HedgePolicy, run
from poseidon.pipeline.plan import effective_config
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET as OPTION_QUOTES
from poseidon.pipeline.refdata.taifex_options import FSP_DATASET as OPTION_FINAL
from poseidon.pipeline.stage_api import StageContext, UpstreamResult
from poseidon.pipeline.v3.config import daily_run_config, idea_run_config, option_chain_record
from tests.pipeline.daily_paper_support import (
    AsOf,
    DailyDriver,
    build_store,
    compiled,
    daily_config,
    local,
    morning_after,
    write_txo,
)

FIXTURES = Path(__file__).parent / "fixtures"
RUN17 = {
    **json.loads((FIXTURES / "v3_terminal_runs.json").read_text())["runs"]["17"],
    **json.loads((FIXTURES / "run17_hedged.json").read_text()),
}
PUT = HedgePolicy(
    name="core.protective_put", underlying="core", chain="TXO", moneyness=0.05, roll_days=5, cover_ratio=1.0
)
WAKES = ("2026-09-10", "2026-09-18", "2026-10-22")  # across the 202609 expiry (09-16) and 202610 (10-21)


def hedged():
    return dataclasses.replace(compiled(), hedges=(PUT,))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    store = build_store(tmp_path_factory.mktemp("refdata"))
    puts = write_txo(store)
    return store, puts.set_index(["trade_date", "contract_month", "strike"])["settlement"]


def _settlement(puts, day: str, contract: str) -> float:
    month, strike = contract.split("P")
    return float(puts[(pd.Timestamp(day), month, float(strike))])


def test_put_legs_are_booked_from_the_txo_rows_of_each_wake_and_replay_matches(world, tmp_path):
    store, puts = world
    long_paper = daily_config(paper_min_trading_days=200)
    driver = DailyDriver(tmp_path, store, strategy=hedged(), config=long_paper, started_at=local(2026, 9, 1, 10))
    for day in WAKES:
        driver.reader = AsOf(store, day)
        result = driver.wake(morning_after(day))
        assert result.is_waiting and result.report["cutoff"] == day
        assert result.report["consistency"]["ratio"] == 1.0
        assert result.report["snapshot"]["hedge_coverage"]["core.protective_put"]["unavailable_days"] == 0
    ledger = driver.ledger()
    assert ledger["days"][0]["trading_day"] == "2026-09-01" and ledger["days"][-1]["trading_day"] == "2026-10-22"
    assert {d["booked_at_cutoff"] for d in ledger["days"]} == set(WAKES)

    reasons = set()
    for day in ledger["days"]:
        put = day["positions"]["core.protective_put"]
        assert put["target_units"] == day["positions"]["core"]["order_units"] // 5  # 1 TXO (50/pt) per 5 TMF
        for fill in day["fills"]:
            if fill["instrument"] != "core.protective_put":
                continue
            reasons.add(fill["reason"])
            if fill["reason"] != "expiry":
                assert fill["ref_price"] == _settlement(puts, day["trading_day"], fill["contract"])
                assert fill["price"] > fill["ref_price"] if fill["units"] > 0 else fill["price"] < fill["ref_price"]
        if put["contract"] and not day["fills"]:
            assert put["close"] == _settlement(puts, day["trading_day"], put["contract"])  # marked at the day's row
    assert {"rebalance", "roll"} <= reasons

    inputs = spd.load_inputs(
        AsOf(store, WAKES[-1]), driver.strategy, spd.DEFAULT_DAILY_SHADOW_PAPER, ledger_start(ledger)
    )
    config = dataclasses.replace(driver.strategy.engine, end=inputs.cutoff)
    backtest = run(
        driver.strategy.factory(),
        inputs.futures,
        config,
        aux=inputs.aux,
        calendar=inputs.calendar,
        options=inputs.options,
        hedges=(PUT,),
    )
    expected = spd.day_rows(backtest, ledger_start(ledger))
    assert [{k: v for k, v in d.items() if k != "booked_at_cutoff"} for d in ledger["days"]] == list(expected.values())
    assert {OPTION_QUOTES, OPTION_FINAL} <= set(driver.results[-1].report["snapshot"]["data"]["sources"])


def ledger_start(ledger):
    return pd.Timestamp(ledger["paper_start_day"]).date()


class TxoBehind(AsOf):
    """Futures and TAIEX through ``as_of``, TXO only through ``txo_through``."""

    def __init__(self, store, as_of, txo_through):
        super().__init__(store, as_of)
        self.txo = AsOf(store, txo_through)

    def read(self, dataset, start=None, end=None, rows=None):
        if dataset in (OPTION_QUOTES, OPTION_FINAL):
            return self.txo.read(dataset, start=start, end=end, rows=rows)
        return super().read(dataset, start=start, end=end, rows=rows)


def test_the_paper_waits_for_a_late_txo_day_instead_of_booking_an_unhedged_one(world, tmp_path):
    store, _puts = world
    driver = DailyDriver(tmp_path, store, strategy=hedged(), started_at=local(2026, 10, 12, 10))
    driver.reader = TxoBehind(store, "2026-10-22", "2026-10-20")
    late = driver.wake(morning_after("2026-10-22"))
    data = late.report["snapshot"]["data"]
    assert late.is_waiting and late.report["cutoff"] == "2026-10-20"
    assert data["waiting_on"] == [OPTION_QUOTES] and data["lag_trading_days"] == 2
    assert data["last_dates"][OPTION_QUOTES] == "2026-10-20"
    assert driver.ledger()["days"][-1]["trading_day"] == "2026-10-20"

    driver.reader = AsOf(store, "2026-10-22")
    caught_up = driver.wake(morning_after("2026-10-22") + pd.Timedelta(hours=8))
    assert caught_up.report["cutoff"] == "2026-10-22" and caught_up.report["consistency"]["ratio"] == 1.0
    assert [d["trading_day"] for d in driver.ledger()["days"]][-2:] == ["2026-10-21", "2026-10-22"]


@pytest.mark.parametrize(
    ("damage", "cutoff"),
    [
        ({"drop_days": ["2026-10-14"]}, "2026-10-13"),  # a trading day without a TXO row
        ({"unsettled": ["202610"]}, "2026-10-20"),  # the 10-21 expiry without its final settlement price
    ],
)
def test_a_chain_that_cannot_price_a_paper_day_stops_the_cutoff_before_it(tmp_path, damage, cutoff):
    store = build_store(tmp_path / "refdata")
    write_txo(store, **damage)
    driver = DailyDriver(tmp_path / "art", store, strategy=hedged(), started_at=local(2026, 10, 1, 10))
    driver.reader = AsOf(store, "2026-10-22")
    result = driver.wake(morning_after("2026-10-22"))
    assert result.is_waiting and result.report["cutoff"] == cutoff
    assert result.report["snapshot"]["data"]["waiting_on"] == [OPTION_QUOTES]
    assert driver.ledger()["days"][-1]["trading_day"] == cutoff


def test_a_revised_txo_settlement_is_reported_as_a_revision_of_its_day(world, tmp_path):
    store, _puts = world
    driver = DailyDriver(tmp_path, store, strategy=hedged(), started_at=local(2026, 10, 12, 10))
    driver.reader = AsOf(store, "2026-10-20")
    driver.wake(morning_after("2026-10-20"))

    class Revised(AsOf):
        def read(self, dataset, start=None, end=None, rows=None):
            frame = super().read(dataset, start=start, end=end, rows=rows)
            if dataset == OPTION_QUOTES:
                hit = frame["trade_date"] == pd.Timestamp("2026-10-15")
                frame.loc[hit, "settlement"] = frame.loc[hit, "settlement"] + 1.0
            return frame

    driver.reader = Revised(store, "2026-10-21")
    result = driver.wake(morning_after("2026-10-21"))
    revisions = result.report["revisions"]["recent"]
    assert [(r["trading_day"], r["kind"]) for r in revisions] == [("2026-10-15", "changed")]


def test_an_unhedged_strategy_reads_no_option_dataset(world, tmp_path):
    store, _puts = world

    class NoOptions(AsOf):
        def read(self, dataset, start=None, end=None, rows=None):
            assert dataset not in (OPTION_QUOTES, OPTION_FINAL)
            return super().read(dataset, start=start, end=end, rows=rows)

    driver = DailyDriver(tmp_path, store, strategy=compiled(), started_at=local(2026, 10, 12, 10))
    driver.reader = NoOptions(store, "2026-10-22")
    result = driver.wake(morning_after("2026-10-22"))
    assert result.report["cutoff"] == "2026-10-22" and "hedge_coverage" not in result.report["snapshot"]
    assert "hedges" not in result.report["engine"]
    assert not set(result.report["snapshot"]["data"]["sources"]) & {OPTION_QUOTES, OPTION_FINAL}


# ----------------------------------------------------------------- frozen key (run #17, the live hedged run)


def _paper_context(config: dict) -> StageContext:
    optimize = UpstreamResult("optimize", 1, "0" * 64, "0" * 64, {"best_spec": RUN17["best_spec"]})
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="",
        input_spec_json=None,
        config=config,
        config_sha256=RUN17["config_sha256"],
        stage="shadow_paper",
        attempt=1,
        upstream={"optimize": optimize},
    )


def test_run_17_keeps_its_hash_and_unsupported_paper_and_the_same_run_frozen_today_hedges():
    assert content_sha256(RUN17["config_json"]) == RUN17["config_sha256"]
    frozen = effective_config(RUN17["config_json"], RUN17["route"])
    assert "option_chain" not in frozen["data"]
    with pytest.raises(spd.StrategyNotCompiled, match=r"data\.option_chain"):
        compile_strategy(_paper_context(frozen))

    new = copy.deepcopy(frozen)
    new["data"]["option_chain"] = option_chain_record()
    paper = compile_strategy(_paper_context(new))
    (put,) = paper.hedges
    assert (put.name, put.underlying, put.chain, put.moneyness, put.roll_days, put.cover_ratio, put.tenor) == (
        "core.protective_put",
        "core",
        "TXO",
        0.1,
        5,
        1.0,
        0,
    )
    assert paper.instruments["core"].product == "TX" and paper.aux == {"taiex": "twse_taiex_daily"}


def test_new_daily_configs_freeze_the_option_chain_rule_and_routing_keeps_it():
    config = daily_run_config(RUN17["best_spec"], as_of=date(2026, 10, 28), brief_text="brief", refdata={})
    assert config["data"]["option_chain"] == option_chain_record()
    routed = idea_run_config(as_of=date(2026, 10, 28), brief_text="brief", refdata={})
    assert routed["by_class"]["daily_equity"]["data"]["option_chain"] == option_chain_record()
    variant = effective_config(routed, RUN17["route"])  # spec_generate's window replaces data, the rule stays
    assert variant["data"]["option_chain"] == option_chain_record()
    assert variant["data"]["start"] == RUN17["route"]["data"]["start"]
