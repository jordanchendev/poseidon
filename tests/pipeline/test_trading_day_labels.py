"""Which trading day a TX night session belongs to (``backtesting.trading_days``).

TAIFEX books a night session (15:00-05:00 Asia/Taipei) into the next trading day. DSL runs frozen without
``backtest.night_session`` (run #6 among them) roll it to the next weekday, so the night before a weekday closure
became a night-only trading day of its own: Thu 2026-10-22's night was labelled 10-23 (typhoon closure) while
TAIFEX dates it 10-27 (10-26 was a holiday). New runs book it into the next day session in the bars.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from poseidon.pipeline.backtesting import run_backtest
from poseidon.pipeline.plan import default_config
from tests.dsl_v2_support import random_walk_bars
from tests.pipeline.daily_paper_support import trading_days as taifex_days
from tests.pipeline.gate_support import RandomEntryFactory, random_entry_spec
from tests.pipeline.paper_support import PaperDriver, PaperSource, local, paper_upstream

RUN6 = json.loads((Path(__file__).parent / "fixtures" / "run6_frozen_v4.json").read_text())["config_json"]
DAYS = ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-27", "2026-10-28"]  # 10-23 typhoon, 10-26 holiday


def new_config() -> dict:
    return default_config(stage_plan=None, as_of=date(2026, 10, 22))


def backtest_days(backtest_config: dict) -> list[str]:
    bars = random_walk_bars(DAYS, seed=4)
    outcome = run_backtest(RandomEntryFactory(spec=random_entry_spec()).build(), bars, backtest_config)
    return [day.date().isoformat() for day in outcome.daily.index]


def test_a_new_run_books_the_night_before_a_closure_into_the_next_session():
    assert backtest_days(new_config()["backtest"]) == [
        "2026-10-20",
        "2026-10-21",
        "2026-10-22",
        "2026-10-27",
        "2026-10-28",
        "2026-10-29",  # the last night: no day session after it yet, the next weekday
    ]


def test_run_6_keeps_its_weekday_labels():
    assert backtest_days(RUN6["backtest"]) == [
        "2026-10-20",
        "2026-10-21",
        "2026-10-22",
        "2026-10-23",
        "2026-10-27",
        "2026-10-28",
        "2026-10-29",
    ]


def paper_days(tmp_path, config: dict) -> list[str]:
    source = PaperSource()
    source.frame = random_walk_bars([day.strftime("%Y-%m-%d") for day in taifex_days("2026-09-01", "2026-12-31")])
    driver = PaperDriver(tmp_path, source, config=config, upstream=paper_upstream(tmp_path))
    driver.wake(local(2026, 10, 28, 9, 30))  # 10-27's day session has arrived: the cutoff is 10-27 15:00
    return [row["trading_day"] for row in driver.ledger()["daily"]]


def test_a_new_paper_counts_thursdays_night_and_tuesday_as_one_trading_day(tmp_path):
    assert paper_days(tmp_path, new_config()) == ["2026-10-27"]


def test_run_6s_paper_keeps_the_night_only_day(tmp_path):
    assert paper_days(tmp_path, RUN6) == ["2026-10-23", "2026-10-27"]
