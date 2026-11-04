"""Step 5 verification on the real refdata (stormtrooper): replay 10 recorded trading days.

Runs only with ``PV2_PAPER_REFDATA`` = a refdata root that has ``twse_market_calendar`` and the
schema-2 ``taifex_settlement_calendar`` (mount it read-only). ``PV2_PAPER_EVIDENCE`` (optional) =
a writable directory for ``daily_paper_replay.json``.

The paper is woken once per recorded day with the store cut at that day (``AsOf``: what the store
held the next morning, including the night session TAIFEX books into the next trading day). The
ledger it books must equal one backtest over the same days row for row (targets, orders, held units,
fills with prices and costs, equity, PnL), a re-wake on the last day must change nothing, and every
in-stage replay must match 100%.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline import shadow_paper_inputs
from poseidon.pipeline.engine import run
from poseidon.pipeline.refdata.datasets import FUTURES, TAIEX
from poseidon.pipeline.refdata.engine_input import day_session_dates, product_quotes
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.state import Verdict
from tests.pipeline.daily_paper_support import AsOf, DailyDriver, LeverageBand, compiled, local, morning_after

ROOT = os.environ.get("PV2_PAPER_REFDATA")
EVIDENCE = os.environ.get("PV2_PAPER_EVIDENCE")
pytestmark = pytest.mark.skipif(not ROOT, reason="PV2_PAPER_REFDATA not set")
EVIDENCE_DATA: dict = {}


def _strategy():
    # 3x / 0.5x TMF (priced from TX) on a 200-day TAIEX band: the acceptance strategy's shape (§3.2)
    return compiled(
        lambda: LeverageBand(ma=200, band=0.02, on=3.0, off=0.5, signal="taiex"),
        start="2016-01-04",
        train_end="2025-01-02",
        data_start="2015-01-05",
        aux={"taiex": TAIEX},
    )


@pytest.fixture(scope="module")
def store():
    return RefdataStore(ROOT)


@pytest.fixture(scope="module")
def recorded_days(store):
    quotes = store.read(FUTURES, start="2026-09-01")
    days = day_session_dates(product_quotes(quotes, "TX"))
    return days[-10:]


def _strip(days):
    return [{k: v for k, v in d.items() if k != "booked_at_cutoff"} for d in days]


def test_ten_recorded_days_ledger_equals_the_backtest(store, recorded_days, tmp_path):
    strategy = _strategy()
    first = recorded_days[0]
    driver = DailyDriver(tmp_path, store, strategy=strategy, started_at=local(first.year, first.month, first.day, 10))
    for day in recorded_days:
        driver.reader = AsOf(store, day)
        result = driver.wake(morning_after(day))
        assert result.verdict is Verdict.PASS and result.is_waiting
        assert result.report["cutoff"] == day.date().isoformat()
        assert result.report["consistency"]["ratio"] == 1.0
    ledger = driver.ledger()
    booked = [d["trading_day"] for d in ledger["days"]]
    assert booked == list(recorded_days.strftime("%Y-%m-%d"))
    assert [d["booked_at_cutoff"] for d in ledger["days"]] == booked  # one new day per wake

    before = {name: driver.file(name) for name in (spd.LEDGER_FILE, *spd.CARRIED_FILES)}
    driver.wake(morning_after(recorded_days[-1]) + pd.Timedelta(hours=3))
    assert {name: driver.file(name) for name in before} == before  # idempotent re-wake

    inputs = shadow_paper_inputs.load_inputs(
        AsOf(store, recorded_days[-1]), strategy, spd.DEFAULT_DAILY_SHADOW_PAPER, first.date()
    )
    config = dataclasses.replace(strategy.engine, end=recorded_days[-1])
    backtest = run(strategy.factory(), inputs.futures, config, aux=inputs.aux, calendar=inputs.calendar)
    expected = spd.day_rows(backtest, first.date())
    assert _strip(ledger["days"]) == [expected[d] for d in booked]

    fills = [dict(f, trading_day=d["trading_day"]) for d in ledger["days"] for f in d["fills"]]
    EVIDENCE_DATA["ten_day_replay"] = {
        "days": booked,
        "ledger_equals_backtest": True,
        "rewake_identical_files": sorted(before),
        "in_stage_replay_ratio_each_wake": [r.report["consistency"]["ratio"] for r in driver.results[:10]],
        "rows": [
            {
                "trading_day": d["trading_day"],
                "target_units": d["positions"]["core"]["target_units"],
                "order_units": d["positions"]["core"]["order_units"],
                "held_units": d["positions"]["core"]["held_units"],
                "contract": d["positions"]["core"]["contract"],
                "equity": round(d["equity"], 2),
                "pnl_twd": round(d["pnl_twd"], 2),
                "fills": len(d["fills"]),
            }
            for d in ledger["days"]
        ],
        "fills": fills,
        "last_summary": driver.results[-1].report["summary"],
        "sources": driver.results[-1].report["snapshot"]["data"]["sources"],
    }


def test_first_real_wake_reports(store, tmp_path):
    """What the stage reports on its first real wake-ups (2026-10-24 09:30 Taipei).

    The store is cut at 2026-10-22, what it held on that morning, so the result does not depend on when the
    refdata copy was taken.
    """

    store = AsOf(store, "2026-10-22")
    strategy = _strategy()
    fresh = DailyDriver(tmp_path / "a", store, strategy=strategy, started_at=local(2026, 10, 24, 9, 0))
    started_now = fresh.wake(local(2026, 10, 24, 9, 30))
    assert started_now.is_waiting and started_now.report["paper_start_day"] == "2026-10-27"
    assert started_now.report["snapshot"]["kind"] == "not_started" and started_now.report["cutoff"] == "2026-10-22"
    assert started_now.report["summary"] == "shadow_paper ⏳ paper starts 2026-10-27 · data through 2026-10-22"
    assert started_now.report["next_evaluation_utc"] == local(2026, 10, 25, 9, 30).isoformat()

    week = DailyDriver(tmp_path / "b", store, strategy=strategy, started_at=local(2026, 10, 19, 9, 0))
    result = week.wake(local(2026, 10, 24, 9, 30))
    report = result.report
    assert [d["trading_day"] for d in week.ledger()["days"]] == ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
    assert report["snapshot"]["data"]["unscheduled_closures"] == ["2026-10-23"]
    assert report["snapshot"]["data"]["lag_trading_days"] == 0 and not report["snapshot"]["data"]["stale"]
    rolls = [f for d in week.ledger()["days"] for f in d["fills"] if f["reason"] == "roll"]
    assert {f["contract"] for f in rolls} == {"202610", "202611"}  # rolled at the 10-20 close
    EVIDENCE_DATA["first_wake"] = {
        "started_now": {
            k: started_now.report[k] for k in ("summary", "paper_start_day", "cutoff", "next_evaluation_utc")
        },
        "started_10_19": {
            "summary": report["summary"],
            "cutoff": report["cutoff"],
            "snapshot": report["snapshot"],
            "paper": report["paper"],
            "protection": report["protection"],
            "consistency": {k: report["consistency"][k] for k in ("compared", "matched", "ratio")},
            "rolls": rolls,
        },
    }
    if EVIDENCE:
        Path(EVIDENCE, "daily_paper_replay.json").write_text(json.dumps(EVIDENCE_DATA, indent=1, default=str))
