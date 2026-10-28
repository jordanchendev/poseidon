"""The DSL shadow paper names the days the market was closed while it waits (report only).

Run #6 sat at "day 0/20" through the 2026-10-23 typhoon closure and the 10-26 holiday with nothing in its
summary saying why. The waiting summary and the snapshot now name the closed days and the next session, and
the stale metric counts TAIFEX trading days, from refdata's calendar (``refdata.engine_input.TaifexCalendar``).
None of it reaches the ledger, a day count, an input hash or PAPER_VERSION, so run #6 resumes as before
(its input hashes: ``test_pipeline_frozen_v4``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poseidon.pipeline.shadow_paper import PAPER_VERSION
from tests.pipeline.daily_paper_support import AsOf, build_store
from tests.pipeline.paper_support import PaperDriver, PaperSource, local, paper_upstream

RUN6 = json.loads((Path(__file__).parent / "fixtures" / "run6_frozen_v4.json").read_text())
CLOSED = ["2026-10-23", "2026-10-26"]  # typhoon; make-up holiday for Sunday 10-25
WAKES = [local(2026, 10, 24, 9, 30), local(2026, 10, 27, 9, 30)]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("refdata"))


def run6_paper(tmp_path, refdata=None):
    driver = PaperDriver(
        tmp_path,
        PaperSource(holidays=CLOSED),
        config=RUN6["config_json"],
        upstream=paper_upstream(tmp_path),
        refdata=refdata,
    )
    return driver, [driver.wake(now) for now in WAKES]


def test_run_6s_waiting_summary_names_the_closed_days_and_the_next_session(tmp_path, store):
    _driver, results = run6_paper(tmp_path, lambda: AsOf(store, "2026-10-23"))
    for result in results:
        assert result.is_waiting and result.reason.startswith("shadow_paper ⏳ day 0/20 · 0/10 trades")
        assert result.reason.endswith(" · 休市：10-23 臨時休市、10-26 補假 · 下一交易日 10-27")
        assert result.report["snapshot"]["market_calendar"] == {
            "closed": [{"date": "2026-10-23", "reason": "臨時休市"}, {"date": "2026-10-26", "reason": "補假"}],
            "next_session": "2026-10-27",
        }
    data = results[-1].report["snapshot"]["data"]  # Tue 09:30: no bar after Thursday's close
    assert (data["trading_days_without_new_bars"], data["stale"]) == (0, False)


def test_the_closure_report_leaves_run_6s_ledger_and_day_count_alone(tmp_path, store):
    with_calendar, named = run6_paper(tmp_path / "a", lambda: AsOf(store, "2026-10-23"))
    without, plain = run6_paper(tmp_path / "b")
    assert with_calendar.file("paper_ledger.json") == without.file("paper_ledger.json")
    for a, b in zip(named, plain, strict=True):
        assert (a.report["trading_days"], a.report["paper_version"]) == (b.report["trading_days"], PAPER_VERSION)
        assert a.report["summary"].startswith(b.report["summary"])
