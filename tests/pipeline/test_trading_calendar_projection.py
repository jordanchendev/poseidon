"""The days ``compile_spec`` projects past a data end, where a ``roll.before_settlement`` roll is placed.

The engine holds the first contract that expires after the *next* trading day, so the day after the data end
decides whether a roll due on the last day happens. Runs frozen without ``data.trading_calendar`` project
weekdays: on 255 of the 5351 TX data ends since 2005 a holiday weekday sat before an expiry inside the
projection, and a roll due on the last day was placed later. New runs project TAIFEX trading days.

The store is the synthetic one of the daily paper tests: the real 2026 TWSE schedule (Lunar New Year: no trading
2026-02-12 .. 02-20, the February contract expires 02-23) and the 2026-10-23 typhoon closure.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline.plan import effective_config
from poseidon.pipeline.refdata.datasets import TAIEX, TAIEX_TR
from poseidon.pipeline.v3.compiler import compile_spec
from poseidon.pipeline.v3.config import daily_run_config, trading_calendar_record
from tests.pipeline.daily_paper_support import AsOf, build_store
from tests.pipeline.spec_support import resolved_spec

SPEC = resolved_spec(ma_days=50, days_before=1)
START = "2025-06-02"
BEFORE_NEW_YEAR = pd.Timestamp("2026-02-11")  # the last session before the break; the next one is 02-23
TERMINAL = json.loads((Path(__file__).parent / "fixtures" / "v3_terminal_runs.json").read_text())["runs"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store = build_store(tmp_path_factory.mktemp("refdata"))
    taiex = store.read(TAIEX)
    parts = {key: rows.reset_index(drop=True) for key, rows in taiex.groupby(taiex["trade_date"].dt.strftime("%Y-%m"))}
    store.commit(TAIEX_TR, schema_version=1, date_column="trade_date", partitions=parts, sources=[], code_version="t")
    return store


def held_on(compiled, day: pd.Timestamp) -> str:
    rows = compiled.run().positions_daily
    return rows.loc[pd.to_datetime(rows["date"]) == day, "contract"].item()


def test_a_new_run_rolls_on_the_last_session_before_the_new_year_expiry_as_the_full_replay_does(store):
    assert held_on(compile_spec(SPEC, start=START, store=store), BEFORE_NEW_YEAR) == "202603"
    new = compile_spec(SPEC, start=START, end=BEFORE_NEW_YEAR, store=store, calendar_rule=trading_calendar_record())
    assert new.calendar[new.calendar > BEFORE_NEW_YEAR][0] == pd.Timestamp("2026-02-23")
    assert held_on(new, BEFORE_NEW_YEAR) == "202603"


def test_a_run_frozen_without_the_rule_keeps_projecting_weekdays(store):
    legacy = compile_spec(SPEC, start=START, end=BEFORE_NEW_YEAR, store=store)
    assert legacy.calendar[legacy.calendar > BEFORE_NEW_YEAR][0] == pd.Timestamp("2026-02-12")
    assert held_on(legacy, BEFORE_NEW_YEAR) == "202602"


def test_the_projection_skips_an_unscheduled_closure_once_taifex_dates_the_night_session_past_it(store):
    morning_after_typhoon = AsOf(store, "2026-10-23")  # 10-22's night session is dated 10-27; 10-26 is a holiday
    compiled = compile_spec(SPEC, start=START, store=morning_after_typhoon, calendar_rule=trading_calendar_record())
    assert list(compiled.calendar[compiled.calendar > pd.Timestamp("2026-10-22")][:2]) == [
        pd.Timestamp("2026-10-27"),
        pd.Timestamp("2026-10-28"),
    ]


def test_new_configs_freeze_the_projection_rule_and_terminal_runs_do_not():
    config = daily_run_config(SPEC, as_of=date(2026, 10, 28), brief_text="brief", refdata={})
    assert config["data"]["trading_calendar"] == trading_calendar_record()
    for run in TERMINAL.values():
        assert "trading_calendar" not in effective_config(run["config_json"], run["route"])["data"]
