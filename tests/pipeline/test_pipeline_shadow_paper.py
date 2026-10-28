"""Step 7 shadow_paper: calendar, cutoff, ledger, replay comparison and every exit path (no DB).

The stage is driven directly (``paper_support.PaperDriver``) with an injected
clock and a synthetic Thalassa that reproduces its observed lag.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline import shadow_paper as sp
from poseidon.pipeline.plan import DEFAULT_SHADOW_PAPER, default_config
from poseidon.pipeline.shadow_paper import SHADOW_CONFIG_MISSING
from poseidon.pipeline.state import Verdict
from tests.pipeline.gate_support import RandomEntryFactory, random_entry_spec
from tests.pipeline.paper_support import (
    AS_OF,
    PAPER_START,
    STAGE_START,
    PaperDriver,
    PaperSource,
    local,
    paper_config,
    paper_upstream,
    wake_times,
)

SETTINGS = DEFAULT_SHADOW_PAPER


def _ledger_view(ledger: dict) -> dict:
    """The ledger without when-it-was-booked fields (catch-up vs daily must agree on the rest)."""

    def strip(row):
        return {k: v for k, v in row.items() if k != "booked_at_cutoff_utc"}

    return {"trades": [strip(t) for t in ledger["trades"]], "daily": [strip(d) for d in ledger["daily"]]}


# ----------------------------------------------------------------- calendar


def test_paper_starts_at_the_next_trading_day_boundary():
    assert sp.paper_start_for(STAGE_START) == PAPER_START  # Thu 10:00 -> Thu 15:00
    assert sp.paper_start_for(local(2026, 10, 22, 15, 0)) == PAPER_START
    assert sp.paper_start_for(local(2026, 10, 22, 15, 1)) == local(2026, 10, 23, 15, 0)
    assert sp.latest_boundary(local(2026, 10, 23, 9, 30)) == PAPER_START
    assert sp.latest_boundary(local(2026, 10, 23, 15, 0)) == local(2026, 10, 23, 15, 0)
    # the boundary is where backtesting.trading_days starts the next trading day
    assert sp.trading_day(PAPER_START - timedelta(minutes=5)).isoformat() == "2026-10-22"
    assert sp.trading_day(PAPER_START).isoformat() == "2026-10-23"
    assert sp.trading_day(local(2026, 10, 23, 15, 0)).isoformat() == "2026-10-26"  # Fri night -> Monday


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (local(2026, 10, 23, 10, 0), local(2026, 10, 24, 9, 30)),  # Fri day session -> Sat morning
        (local(2026, 10, 23, 9, 30), local(2026, 10, 24, 9, 30)),  # exactly at the wake time -> next day
        (local(2026, 10, 23, 9, 29), local(2026, 10, 23, 9, 30)),
        (local(2026, 10, 23, 23, 0), local(2026, 10, 24, 9, 30)),  # night session
        (local(2026, 10, 24, 12, 0), local(2026, 10, 25, 9, 30)),  # weekend wake-ups are cheap no-ops
    ],
)
def test_next_wake_is_the_next_0930_taipei(now, expected):
    assert sp.next_wake(now, SETTINGS) == expected


def test_next_wake_honours_the_weekday_list():
    tue_to_sat = {**SETTINGS, "wake_weekdays": [1, 2, 3, 4, 5]}
    assert sp.next_wake(local(2026, 10, 24, 10, 0), tue_to_sat) == local(2026, 10, 27, 9, 30)  # Sat -> Tue
    with pytest.raises(ValueError):
        sp.next_wake(local(2026, 10, 24, 10, 0), {**SETTINGS, "wake_weekdays": []})


@pytest.mark.parametrize(
    ("last_bar", "upper", "expected"),
    [
        # Fri 09:30: Thalassa holds up to Fri 05:00 -> Thursday is complete
        (local(2026, 10, 30, 5, 0), local(2026, 10, 29, 15, 0), local(2026, 10, 29, 15, 0)),
        # Thursday's day session only half delivered -> Wednesday
        (local(2026, 10, 29, 11, 0), local(2026, 10, 29, 15, 0), local(2026, 10, 28, 15, 0)),
        # the closing bar is enough
        (local(2026, 10, 29, 13, 40), local(2026, 10, 29, 15, 0), local(2026, 10, 29, 15, 0)),
        # Sunday 15:00 boundary: no Saturday/Sunday session -> Friday
        (local(2026, 10, 31, 5, 0), local(2026, 11, 1, 15, 0), local(2026, 10, 30, 15, 0)),
        # Wednesday holiday, nothing after Tuesday's close yet -> Tuesday
        (local(2026, 10, 27, 13, 40), local(2026, 10, 28, 15, 0), local(2026, 10, 27, 15, 0)),
        # nothing at all after the paper start
        (None, local(2026, 10, 29, 15, 0), PAPER_START),
        (local(2026, 10, 22, 13, 40), local(2026, 10, 29, 15, 0), PAPER_START),
    ],
)
def test_settled_cutoff_only_counts_complete_trading_days(last_bar, upper, expected):
    assert sp.settled_cutoff(last_bar, PAPER_START, upper) == expected


def test_weekday_gap_and_last_complete_weekday():
    assert sp.last_complete_weekday(local(2026, 11, 1, 15, 0)).isoformat() == "2026-10-30"  # Sun -> Fri
    assert sp.weekdays_after(datetime(2026, 10, 23).date(), datetime(2026, 10, 30).date()) == 5
    assert sp.weekdays_after(datetime(2026, 10, 30).date(), datetime(2026, 10, 30).date()) == 0
    assert sp.weekdays_after(datetime(2026, 10, 30).date(), datetime(2026, 11, 2).date()) == 1


def test_trade_comparison_and_drawdown():
    paper = [
        {"entry_time": "a", "side": "long", "exit_time": "b", "pnl_twd": 1000.0, "quantity": 1.0},
        {"entry_time": "c", "side": "short", "exit_time": "d", "pnl_twd": -400.0, "quantity": 1.0},
        {"entry_time": "e", "side": "long", "exit_time": "f", "pnl_twd": 50.0, "quantity": 1.0},
        {"entry_time": "g", "side": "long", "exit_time": "h", "pnl_twd": 50.0, "quantity": 1.0},
    ]
    replay = [
        {**paper[0], "pnl_twd": 1200.0},  # within 1 tick (200 TWD)
        {**paper[1], "pnl_twd": -601.0},  # beyond
        {**paper[2], "exit_time": "z"},
        {"entry_time": "x", "side": "short", "exit_time": "y", "pnl_twd": 0.0, "quantity": 1.0},
    ]
    out = sp.compare_trades(paper, replay, 200.0)
    assert out["by_status"] == {"exit_differs": 1, "match": 1, "paper_only": 1, "pnl_differs": 1, "replay_only": 1}
    assert (out["matched"], out["compared"], out["ratio"]) == (1, 5, 0.2)
    assert sp.trade_max_drawdown(paper) == 400.0 and sp.trade_max_drawdown([]) == 0.0


# ----------------------------------------------------------------- the stage, wake by wake


def test_first_wake_before_the_paper_start_just_schedules(tmp_path):
    driver = PaperDriver(tmp_path, PaperSource())
    result = driver.wake(STAGE_START)
    assert result.is_waiting and result.wait_until == local(2026, 10, 23, 9, 30)
    assert result.report["status"] == "waiting" and result.report["paper_start_utc"] == PAPER_START.isoformat()
    assert result.report["snapshot"]["kind"] == "not_started"
    assert driver.source.calls == []  # nothing to fetch yet
    assert result.reason.startswith("shadow_paper ⏳ day 0/20 · 0/10 trades")


def test_day_night_weekend_and_holiday_wakes(tmp_path):
    """Thalassa's lag, a weekend and a Wednesday holiday: only complete trading days are booked."""

    source = PaperSource(holidays=["2026-10-28"])
    driver = PaperDriver(tmp_path, source)
    driver.wake(STAGE_START)
    cutoffs = {}
    for now in wake_times("2026-10-23", "2026-10-31"):
        result = driver.wake(now)
        assert result.is_waiting and result.wait_until == now + timedelta(days=1)
        cutoffs[now.astimezone(sp.MARKET_TZ).strftime("%a %d")] = result.report["cutoff_utc"]
    expect = {
        "Fri 23": None,  # Thalassa holds up to Fri 05:00: Friday's trading day is not complete
        "Sat 24": local(2026, 10, 23, 15, 0).isoformat(),  # Fri complete (night Thu + day Fri)
        "Sun 25": local(2026, 10, 23, 15, 0).isoformat(),  # weekend: nothing new
        "Mon 26": local(2026, 10, 23, 15, 0).isoformat(),
        "Tue 27": local(2026, 10, 26, 15, 0).isoformat(),  # Monday
        "Wed 28": local(2026, 10, 27, 15, 0).isoformat(),  # Tuesday
        "Thu 29": local(2026, 10, 27, 15, 0).isoformat(),  # Wednesday holiday: no bars, no new trading day
        "Fri 30": local(2026, 10, 29, 15, 0).isoformat(),  # Thursday (Wednesday skipped)
        "Sat 31": local(2026, 10, 30, 15, 0).isoformat(),
    }
    assert {day: (c or None) for day, c in cutoffs.items()} == expect
    ledger = driver.ledger()
    assert [row["trading_day"] for row in ledger["daily"]] == [
        "2026-10-23",
        "2026-10-26",
        "2026-10-27",
        "2026-10-29",
        "2026-10-30",
    ]
    snapshots = driver.snapshots()
    # one snapshot per cutoff: the Sun/Mon repeats of Friday's cutoff replaced it
    friday = [s for s in snapshots if s.get("cutoff_utc") == local(2026, 10, 23, 15, 0).isoformat()]
    assert len(friday) == 1 and friday[0]["evaluations_at_cutoff"] == 3
    assert driver.results[-1].report["trading_days"] == 5


def test_catch_up_after_missed_days_matches_daily_evaluation(tmp_path):
    daily = PaperDriver(tmp_path / "daily", PaperSource())
    for now in wake_times("2026-10-23", "2026-11-07"):
        daily.wake(now)
    caught_up = PaperDriver(tmp_path / "catchup", PaperSource())
    caught_up.wake(STAGE_START)
    caught_up.wake(local(2026, 10, 27, 9, 30))  # then the runner is down until Sat 11-07
    result = caught_up.wake(local(2026, 11, 7, 9, 30))
    assert (
        result.report["cutoff_utc"] == daily.results[-1].report["cutoff_utc"] == local(2026, 11, 6, 15, 0).isoformat()
    )
    assert _ledger_view(caught_up.ledger()) == _ledger_view(daily.ledger())
    assert len(daily.ledger()["trades"]) > 0
    assert result.report["consistency"]["ratio"] == 1.0
    assert result.report["snapshot"]["new_trading_days"][0] == "2026-10-27"


def test_a_repeated_wake_is_idempotent(tmp_path):
    driver = PaperDriver(tmp_path, PaperSource())
    first = driver.wake(local(2026, 10, 31, 9, 30))
    ledger, snapshots = driver.ledger(), driver.snapshots()
    again = driver.wake(local(2026, 10, 31, 9, 30))
    assert _ledger_view(driver.ledger()) == _ledger_view(ledger)
    assert driver.ledger()["trades"] == ledger["trades"]  # nothing re-booked
    assert len(driver.snapshots()) == len(snapshots) and driver.snapshots()[-1]["evaluations_at_cutoff"] == 2
    assert again.report["paper"] == first.report["paper"] and again.wait_until == first.wait_until


def test_a_data_revision_is_reported_and_shows_up_as_a_mismatch(tmp_path):
    source = PaperSource()
    driver = PaperDriver(tmp_path, source)
    driver.wake(local(2026, 10, 31, 9, 30))  # books Fri 10-23 .. Fri 10-30
    booked = driver.ledger()
    revised_day = next(t for t in booked["trades"] if t["entry_time"] >= local(2026, 10, 27).isoformat())
    day = sp.trading_day(revised_day["entry_time"]).isoformat()
    source.revise(day)  # Thalassa back-fills / edits that day afterwards
    result = driver.wake(local(2026, 11, 1, 9, 30))
    revisions = result.report["revisions"]["recent"]
    assert [(r["trading_day"], r["kind"]) for r in revisions] == [(day, "changed")]
    assert result.report["snapshot"]["revisions_found"] == revisions
    mismatches = result.report["consistency"]["mismatches"]
    assert mismatches and all(day in row["revised_days"] for row in mismatches)
    assert result.report["consistency"]["ratio"] < 1.0
    # the paper keeps what it booked at the time
    assert driver.ledger()["trades"] == booked["trades"]


def test_a_warmup_revision_is_reported(tmp_path):
    source = PaperSource()
    driver = PaperDriver(tmp_path, source)
    driver.wake(local(2026, 10, 27, 9, 30))
    source.revise("2026-10-15")  # inside the 14-day warm-up
    result = driver.wake(local(2026, 10, 28, 9, 30))
    assert ("warmup", "changed") in [(r["trading_day"], r["kind"]) for r in result.report["revisions"]["recent"]]


def test_passes_after_twenty_trading_days_with_ten_trades(tmp_path):
    driver = PaperDriver(tmp_path, PaperSource())
    driver.wake(STAGE_START)
    early = driver.wake(local(2026, 11, 19, 9, 30))  # 19 trading days booked (10-23 .. 11-18)
    assert early.is_waiting and early.report["trading_days"] == 19
    result = driver.wake(local(2026, 11, 20, 9, 30))
    assert result.verdict == Verdict.PASS and not result.is_waiting
    report = result.report
    assert report["trading_days"] == 20 and report["paper"]["n_trades"] >= 10
    assert report["consistency"]["ratio"] == 1.0 and report["protection"]["tripped"] is None
    assert report["summary"].startswith("shadow_paper ✅ 20 trading days")
    assert set(result.files) == {
        "paper_ledger.json",
        "snapshots.jsonl",
        "paper_trades.csv",
        "replay_trades.csv",
        "replay_daily_pnl.csv",
        "comparison.json",
    }


def test_protection_trip_on_a_daily_loss_fails_at_once(tmp_path):
    # train backtest's p95 daily loss is 10 TWD -> limit 30 TWD: the first real losing day trips
    upstream = paper_upstream(tmp_path, train_daily=np.r_[np.full(50, -10.0), np.full(50, 20.0)])
    driver = PaperDriver(tmp_path, PaperSource(), upstream=upstream)
    result = driver.wake(local(2026, 10, 31, 9, 30))
    assert result.verdict == Verdict.FAIL and result.report["reason_code"] == "paper_protection_trip"
    trip = result.report["protection"]["tripped"]
    assert trip["kind"] == "daily_loss" and trip["pnl_twd"] < -30.0 and trip["limit_twd"] == pytest.approx(30.0)
    assert result.reason.startswith("protection trip: ")


def test_protection_trip_on_max_drawdown(tmp_path):
    upstream = paper_upstream(tmp_path, train_max_dd=100.0, train_daily=np.full(30, 5.0))  # no losing train day
    driver = PaperDriver(tmp_path, PaperSource(), upstream=upstream)
    result = driver.wake(local(2026, 10, 31, 9, 30))
    report = result.report
    assert report["protection"]["daily_loss_limit_twd"] is None  # not armed: no losing day in train
    assert report["protection"]["train_daily_loss_basis"] == "no_losing_day"
    assert result.verdict == Verdict.FAIL and report["protection"]["tripped"]["kind"] == "max_drawdown"
    assert report["protection"]["tripped"]["limit_twd"] == 150.0


def test_insufficient_trades_keeps_waiting_until_sixty_trading_days(tmp_path):
    driver = PaperDriver(tmp_path, PaperSource(), config=paper_config(paper_min_trades=10_000))
    driver.wake(STAGE_START)
    middle = driver.wake(local(2026, 12, 5, 9, 30))  # >= 20 trading days, far below the trade minimum
    assert middle.is_waiting and middle.report["trading_days"] > 20
    before = driver.wake(local(2027, 1, 14, 9, 30))
    assert before.is_waiting and before.report["trading_days"] == 59
    result = driver.wake(local(2027, 1, 15, 9, 30))
    assert result.verdict == Verdict.FAIL and result.report["reason_code"] == "insufficient_paper_trades"
    assert result.report["trading_days"] == 60
    assert result.reason.startswith("insufficient_paper_trades: ")


class ReseedingRandomEntryFactory(RandomEntryFactory):
    """Every build gets a new seed: the paper and its replay disagree (non-determinism)."""

    seeds = itertools.count(1)

    def build(self, params=None):
        from tests.pipeline.gate_support import RandomEntryStrategy

        values = self.defaults if params is None else {**self.defaults, **params}
        return RandomEntryStrategy(
            seed=next(self.seeds),
            entry_probability=0.2,
            config=self.template,
            param_values=values,
            bar_label=self._bar_label,
        )


def test_non_deterministic_replay_fails_the_consistency_check(tmp_path):
    spec = random_entry_spec()
    driver = PaperDriver(
        tmp_path, PaperSource(), upstream=paper_upstream(tmp_path, spec=spec), factory=ReseedingRandomEntryFactory
    )
    result = driver.wake(local(2026, 11, 21, 9, 30))
    report = result.report
    assert report["trading_days"] >= 20 and report["paper"]["n_trades"] >= 10, report["paper"]
    assert result.verdict == Verdict.FAIL and report["reason_code"] == "paper_replay_inconsistent"
    assert report["consistency"]["ratio"] < 0.9


def test_thalassa_outage_waits_and_retries_within_the_hour(tmp_path):
    source = PaperSource()
    driver = PaperDriver(tmp_path, source)
    driver.wake(local(2026, 10, 28, 9, 30))
    booked = driver.ledger()
    source.fail = 1
    now = local(2026, 10, 29, 9, 30)
    result = driver.wake(now)
    assert result.is_waiting and result.wait_until == now + timedelta(minutes=60)
    assert result.report["snapshot"]["kind"] == "source_error"
    assert driver.ledger() == booked and "paper_trades.csv" in result.files  # carried over unchanged
    recovered = driver.wake(now + timedelta(minutes=60))
    assert recovered.report["snapshot"]["kind"] == "evaluation"
    assert recovered.report["cutoff_utc"] == local(2026, 10, 28, 15, 0).isoformat()


def test_no_new_bars_for_more_than_three_weekdays_is_reported_not_failed(tmp_path):
    source = PaperSource()
    driver = PaperDriver(tmp_path, source)
    driver.wake(local(2026, 10, 28, 9, 30))  # through Tue 10-27
    source.at(local(2026, 10, 28, 9, 30))
    frozen = source.available_until
    gaps = []
    for now in wake_times("2026-10-29", "2026-11-04"):
        source.available_until = frozen  # Thalassa stopped ingesting after Wed 05:00
        result = driver.wake(now, thalassa_lag=False)
        gaps.append(
            (
                result.report["snapshot"]["data"]["trading_days_without_new_bars"],
                result.report["snapshot"]["data"]["stale"],
            )
        )
    # last bar Wed 10-28 05:00 (Wednesday's trading day); missing: Thu, Fri, (weekend), Mon, Tue
    assert gaps == [(0, False), (1, False), (2, False), (2, False), (2, False), (3, False), (4, True)]
    assert result.is_waiting and result.report["cutoff_utc"] == local(2026, 10, 27, 15, 0).isoformat()


def test_fewer_bars_than_before_keeps_the_ledger(tmp_path):
    source = PaperSource()
    driver = PaperDriver(tmp_path, source)
    driver.wake(local(2026, 10, 31, 9, 30))
    booked = driver.ledger()
    source.available_until = local(2026, 10, 27, 5, 5)
    result = driver.wake(local(2026, 11, 1, 9, 30), thalassa_lag=False)
    assert result.report["snapshot"]["kind"] == "data_regression" and result.is_waiting
    assert driver.ledger()["trades"] == booked["trades"] and driver.ledger()["cutoff_utc"] == booked["cutoff_utc"]


def test_resume_ignores_an_artifact_of_another_spec(tmp_path):
    driver = PaperDriver(tmp_path, PaperSource())
    driver.wake(local(2026, 10, 28, 9, 30))
    other = paper_upstream(tmp_path / "other", spec=random_entry_spec())
    driver.upstream = other
    driver.stage.strategy_factory = RandomEntryFactory
    result = driver.wake(local(2026, 10, 29, 9, 30))
    assert result.report["resume"]["why"] == "different paper version or spec"


def test_runs_frozen_before_step_7_are_unsupported(tmp_path):
    config = default_config(stage_plan=None, as_of=AS_OF)
    config["version"] = 3
    config.pop("shadow_paper")
    result = PaperDriver(tmp_path, PaperSource(), config=config).wake(local(2026, 10, 28, 9, 30))
    assert (result.verdict, result.reason) == (Verdict.UNSUPPORTED, SHADOW_CONFIG_MISSING)


def test_wake_up_cost_stays_small(tmp_path):
    """60 trading days + warm-up in one evaluation: both runs together well under a minute."""

    import time

    driver = PaperDriver(tmp_path, PaperSource(), config=paper_config(paper_min_trades=10_000))
    started = time.monotonic()
    driver.wake(local(2027, 1, 15, 9, 30))
    elapsed = time.monotonic() - started
    assert driver.results[-1].report["trading_days"] == 60
    print(f"one 60-trading-day wake-up: {elapsed:.1f}s")
    assert elapsed < 60


def test_wake_times_helper_is_taipei_local():
    assert wake_times("2026-10-23", "2026-10-23") == [datetime(2026, 10, 23, 1, 30, tzinfo=UTC)]


# ----------------------------------------------------------------- revised_days labelling


def _closure_bars() -> pd.DatetimeIndex:
    """Thu 10-22 day + night, then 10-23 (typhoon) and 10-26 (holiday) closed, Tue 10-27 day session."""

    return pd.DatetimeIndex(
        [local(2026, 10, 22, 9, 0), local(2026, 10, 22, 20, 0), local(2026, 10, 27, 9, 0)], tz="UTC"
    )


def test_revised_day_labels_follow_the_night_session_rule_of_the_config():
    night = local(2026, 10, 22, 20, 0).isoformat()
    day = local(2026, 10, 27, 9, 0).isoformat()
    moments = {night, day}
    weekday = sp.trading_day_labels(moments, _closure_bars(), None)
    assert weekday == {night: sp.trading_day(night), day: sp.trading_day(day)}  # absent key: the old label
    assert weekday == sp.trading_day_labels(moments, _closure_bars(), "next_weekday")
    assert weekday[night].isoformat() == "2026-10-23"  # the pseudo day of a closed Friday
    booked = sp.trading_day_labels(moments, _closure_bars(), "next_day_session")
    assert booked[night].isoformat() == "2026-10-27" and booked[day].isoformat() == "2026-10-27"


def test_a_run_frozen_without_night_session_reports_revised_days_exactly_as_before(tmp_path):
    def mismatches(config, root):
        source = PaperSource()
        driver = PaperDriver(root, source, config=config)
        driver.wake(local(2026, 10, 31, 9, 30))
        revised = next(t for t in driver.ledger()["trades"] if t["entry_time"] >= local(2026, 10, 27).isoformat())
        source.revise(sp.trading_day(revised["entry_time"]).isoformat())
        return driver.wake(local(2026, 11, 1, 9, 30)).report["consistency"]["mismatches"]

    legacy = paper_config()
    del legacy["backtest"]["night_session"]  # run #6's shape
    old_rows = mismatches(legacy, tmp_path / "legacy")
    new_rows = mismatches(paper_config(), tmp_path / "new")
    assert old_rows and all(row["revised_days"] for row in old_rows)
    assert old_rows == new_rows  # no closure in this window: both labellings agree
