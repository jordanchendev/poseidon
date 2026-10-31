"""The DSL shadow paper across the 2026-10-22 -> 10-27 closure, wake by wake, against one engine run.

A run frozen with ``backtest.night_session = next_day_session`` books Thu 10-22's night into Tue 10-27 (10-23
typhoon closure, 10-26 holiday). Whatever the wakes saw on the way, the paper ledger must hold what one engine run
over the final bars holds: the same trading days with the same PnL, the same trades, and no revision when no bar
changed. The strategy holds a long through the night, so a night booked twice moves the PnL.

``first_new_day`` and ``detect_revisions``' ``evaluated_before`` take the weekday label of the previous cutoff
(Thu 10-22 15:00 -> 10-23) where the run books 10-27. That label only bounds a filter over the evaluation's day
labels, and no day of the bars lies between the two labels, so it cannot change what is booked: pinned by the
full-delivery case. What did go wrong is the cutoff: a day without a day session whose 13:40 had passed counted as
complete, so a partial delivery of Tuesday booked the dangling Thursday night as a pseudo day 10-23, booked it again
in 10-27, and reported 10-23 as a removed day. A partial Monday does the same to Friday's night over any weekend.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
from poseidon.pipeline import shadow_paper as sp
from poseidon.pipeline.backtesting import run_backtest
from poseidon.pipeline.plan import default_config
from tests.dsl_v2_support import random_walk_bars
from tests.pipeline.daily_paper_support import trading_days as taifex_days
from tests.pipeline.gate_support import overnight_long_spec
from tests.pipeline.paper_support import PaperDriver, PaperSource, local, paper_upstream

RUN6 = json.loads((Path(__file__).parent / "fixtures" / "run6_frozen_v4.json").read_text())["config_json"]
SPEC = overnight_long_spec()
TUE_WAKE = local(2026, 10, 28, 9, 30)  # the first wake that can see Tuesday's day session
WAKES = [
    local(2026, 10, 22, 20, 0),
    *(local(2026, 10, day, 9, 30) for day in (23, 24, 27)),
    TUE_WAKE,
    local(2026, 10, 29, 9, 30),
]
PARTIAL_TUESDAY = (TUE_WAKE, local(2026, 10, 27, 11, 5))  # Thalassa holds Tuesday's day session only up to 11:00
STARTS = {"paper_start_thu_1500": local(2026, 10, 22, 10, 0), "paper_start_tue_1500": local(2026, 10, 20, 10, 0)}


def new_config() -> dict:
    return default_config(stage_plan=None, as_of=date(2026, 10, 22))


def wake_through(tmp_path, config: dict, started_at, wakes=WAKES, partial=None):
    """Every wake sees Thalassa's usual lag, except ``partial`` = (wake, delivered up to)."""

    source = PaperSource()
    source.frame = random_walk_bars([day.strftime("%Y-%m-%d") for day in taifex_days("2026-09-01", "2026-12-31")])
    upstream = paper_upstream(tmp_path, spec=SPEC)
    driver = PaperDriver(tmp_path, source, config=config, upstream=upstream, started_at=started_at)
    for now in wakes:
        source.at(now)
        if partial is not None and now == partial[0]:
            source.available_until = partial[1]
        driver.wake(now, thalassa_lag=False)
    return driver, source


def booked(driver) -> tuple[list, list]:
    ledger = driver.ledger()
    days = [(row["trading_day"], round(row["pnl_twd"], 6)) for row in ledger["daily"]]
    trades = [(t["entry_time"], t["exit_time"], t["side"], round(t["pnl_twd"], 6)) for t in ledger["trades"]]
    return days, trades


def one_engine_run(driver, source, config: dict) -> tuple[list, list]:
    ledger = driver.ledger()
    outcome = run_backtest(
        DslV2StrategyFactory(spec=SPEC, bar_label=config["backtest"]["bar_label"]).build(),
        source.frame[source.frame.index < pd.Timestamp(ledger["cutoff_utc"])],
        config["backtest"],
        trade_from=pd.Timestamp(ledger["paper_start_utc"]),
    )
    days = [
        (pd.Timestamp(day).date().isoformat(), round(float(row["pnl_twd"]), 6)) for day, row in outcome.daily.iterrows()
    ]
    trades = [(t["entry_time"], t["exit_time"], t["side"], round(t["pnl_twd"], 6)) for t in outcome.trades]
    return days, trades


@pytest.mark.parametrize("started_at", STARTS.values(), ids=STARTS.keys())
def test_the_weekday_label_of_the_previous_cutoff_cannot_change_what_the_paper_books(tmp_path, started_at):
    driver, source = wake_through(tmp_path, new_config(), started_at)
    days, trades = booked(driver)
    assert (days, trades) == one_engine_run(driver, source, new_config())
    assert driver.ledger()["revisions"] == []
    # the 10-28 wake resumed from the 10-22 15:00 cutoff, labelled 10-23 by the weekday rule, and booked 10-27
    assert sp.trading_day(local(2026, 10, 22, 15, 0)) == date(2026, 10, 23)
    tuesday = driver.results[WAKES.index(TUE_WAKE)].report["snapshot"]
    assert tuesday["new_trading_days"] == ["2026-10-27"] and "2026-10-23" not in [day for day, _ in days]


@pytest.mark.parametrize("started_at", STARTS.values(), ids=STARTS.keys())
def test_a_partial_delivery_after_the_closure_books_nothing_until_tuesday_is_complete(tmp_path, started_at):
    driver, source = wake_through(tmp_path, new_config(), started_at, partial=PARTIAL_TUESDAY)
    tuesday = driver.results[WAKES.index(TUE_WAKE)].report
    assert tuesday["snapshot"]["data"]["last_bar_utc"] == local(2026, 10, 27, 11, 0).isoformat()
    assert tuesday["cutoff_utc"] == local(2026, 10, 22, 15, 0).isoformat()  # Thursday's night is not a day yet
    assert tuesday["snapshot"]["new_trading_days"] == []
    assert booked(driver) == one_engine_run(driver, source, new_config())
    assert driver.ledger()["revisions"] == []


def test_a_partial_monday_delivery_books_fridays_night_once(tmp_path):
    monday_wake = local(2026, 11, 3, 9, 30)
    wakes = [*(local(2026, 10, day, 9, 30) for day in (30, 31)), monday_wake, local(2026, 11, 4, 9, 30)]
    partial = (monday_wake, local(2026, 11, 2, 11, 5))
    driver, source = wake_through(tmp_path, new_config(), local(2026, 10, 28, 10, 0), wakes, partial)
    assert driver.results[wakes.index(monday_wake)].report["snapshot"]["new_trading_days"] == []
    assert booked(driver) == one_engine_run(driver, source, new_config())
    assert driver.ledger()["revisions"] == []


# Run #6's shape (no backtest.night_session) through the same partial delivery, recorded at e74c021: the weekday
# rule makes 10-23 a trading day of its own, so the paper keeps booking it exactly as before.
RUN6_LEDGER_SHA256 = "fcd680d32669ea59f5c0ad0bd9b058ac85661863e2d33806ae80b67490da1631"
RUN6_INPUT_SHA256 = "7da961d379a27112f92688c7e8d3676c4273f0f61ddd83ddd34162ac2aafccca"
RUN6_DAY_COUNTS = [1, 2, 2, 2, 3, 5]  # the 10-28 wake books 10-23 on the partial delivery


def test_run_6s_shape_books_the_partial_delivery_byte_identically(tmp_path):
    assert "night_session" not in RUN6["backtest"]
    driver, source = wake_through(tmp_path, RUN6, STARTS["paper_start_tue_1500"], partial=PARTIAL_TUESDAY)
    assert booked(driver) == one_engine_run(driver, source, RUN6)
    assert "2026-10-23" in [day for day, _ in booked(driver)[0]]
    assert [result.report["trading_days"] for result in driver.results] == RUN6_DAY_COUNTS
    assert hashlib.sha256(driver.file(sp.LEDGER_FILE)).hexdigest() == RUN6_LEDGER_SHA256
    assert driver.resume.input_sha256 == RUN6_INPUT_SHA256
    assert sp.PAPER_VERSION == 1
