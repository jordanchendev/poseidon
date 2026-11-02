"""Daily target-position shadow paper (design §7, step 5): cutoff, ledger, replay, exits, seams (no DB).

Synthetic refdata (real 2026 TWSE holiday schedule, the 2026-10-23 typhoon closure) in a tmp
store; ``AsOf`` replays what the store held on each morning.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline.engine import run
from poseidon.pipeline.plan import default_config
from poseidon.pipeline.refdata.datasets import FUTURES, TAIEX
from poseidon.pipeline.refdata.engine_input import futures_bars
from poseidon.pipeline.refdata.store import RefdataError
from poseidon.pipeline.shadow_paper import PAPER_VERSION, ShadowPaperStage
from poseidon.pipeline.stages import StageContext
from poseidon.pipeline.state import Verdict
from tests.pipeline.daily_paper_support import (
    SCHEDULE,
    AsOf,
    DailyDriver,
    Flaky,
    LeverageBand,
    build_store,
    compiled,
    daily_config,
    local,
    morning_after,
    stub_upstream,
    trading_days,
    write_futures,
)

DAYS = trading_days()
PAPER_DAYS = DAYS[(DAYS >= "2026-09-01") & (DAYS <= "2026-12-31")]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("refdata"))


def _strip(days: list[dict]) -> list[dict]:
    return [{k: v for k, v in d.items() if k != "booked_at_cutoff"} for d in days]


def _daily(driver: DailyDriver, store, days):
    out = []
    for day in days:
        driver.reader = AsOf(store, day)
        out.append(driver.wake(morning_after(day)))
    return out


# ----------------------------------------------------------------- calendar and data


def test_paper_start_is_the_first_decision_after_the_stage_started():
    assert spd.paper_start_day(local(2026, 9, 1, 10, 0), SCHEDULE).isoformat() == "2026-09-01"
    assert spd.paper_start_day(local(2026, 9, 1, 13, 46), SCHEDULE).isoformat() == "2026-09-02"
    assert (
        spd.paper_start_day(local(2026, 10, 24, 9, 0), SCHEDULE).isoformat() == "2026-10-27"
    )  # Sat; Mon 10-26 holiday


def test_cutoff_is_the_last_day_session_and_a_night_only_date_is_not_a_trading_day(store):
    reader = AsOf(store, "2026-10-22")  # morning of Sat 10-24: night row dated 10-27 is already out
    quotes = reader.read(FUTURES)
    assert set(quotes.loc[quotes["trade_date"] > "2026-10-22", "session"]) == {"night"}
    assert set(quotes.loc[quotes["trade_date"] > "2026-10-22", "trade_date"]) == {pd.Timestamp("2026-10-27")}
    inputs = spd.load_inputs(reader, compiled(), spd.DEFAULT_DAILY_SHADOW_PAPER, pd.Timestamp("2026-09-01").date())
    assert inputs.cutoff == pd.Timestamp("2026-10-22")
    assert list(inputs.closures.strftime("%Y-%m-%d")) == ["2026-10-23"]  # night session booked into 10-27
    assert inputs.futures["core"].bars["date"].max() == pd.Timestamp("2026-10-22")
    # after the cutoff the calendar is the official schedule: 10-23 (typhoon, unknown ahead) and 10-27
    after = inputs.calendar[inputs.calendar > inputs.cutoff][:3]
    assert list(after.strftime("%Y-%m-%d")) == ["2026-10-23", "2026-10-27", "2026-10-28"]


def test_bar_open_is_the_night_session_taifex_books_into_the_day(store):
    quotes = store.read(FUTURES, start="2026-10-01", end="2026-10-02")
    bars = futures_bars(quotes[quotes["contract_month"] == "202610"]).set_index("date")
    night = quotes[(quotes["session"] == "night") & (quotes["contract_month"] == "202610")].set_index("trade_date")
    day = quotes[(quotes["session"] == "day") & (quotes["contract_month"] == "202610")].set_index("trade_date")
    t = pd.Timestamp("2026-10-02")
    assert bars.loc[t, "open"] == night.loc[t, "open"]
    assert bars.loc[t, "close"] == day.loc[t, "close"]
    assert bars.loc[t, "high"] == max(day.loc[t, "high"], night.loc[t, "high"])


# ----------------------------------------------------------------- ledger and replay


def test_each_wake_books_the_new_day_and_the_ledger_equals_a_backtest(store, tmp_path):
    driver = DailyDriver(tmp_path, store)
    days = PAPER_DAYS[:8]
    results = _daily(driver, store, days)
    assert all(r.verdict is Verdict.PASS and r.is_waiting for r in results)
    ledger = driver.ledger()
    assert [d["trading_day"] for d in ledger["days"]] == list(days.strftime("%Y-%m-%d"))
    assert [d["booked_at_cutoff"] for d in ledger["days"]] == list(days.strftime("%Y-%m-%d"))
    report = results[-1].report
    assert report["consistency"]["ratio"] == 1.0 and report["consistency"]["compared"] == 8
    assert report["summary"].startswith("shadow_paper ⏳ day 8/20 · target match 100%")
    # the same strategy backtested once over the whole span books exactly the same days
    inputs = spd.load_inputs(AsOf(store, days[-1]), driver.strategy, spd.DEFAULT_DAILY_SHADOW_PAPER, days[0].date())
    config = dataclasses.replace(driver.strategy.engine, end=days[-1])
    backtest = run(driver.strategy.factory(), inputs.futures, config, calendar=inputs.calendar)
    expected = spd.day_rows(backtest, days[0].date())
    assert _strip(ledger["days"]) == [expected[d] for d in days.strftime("%Y-%m-%d")]
    assert sum(len(d["fills"]) for d in ledger["days"]) > 0  # the sizing trades daily


def test_re_wake_on_the_same_day_changes_nothing(store, tmp_path):
    driver = DailyDriver(tmp_path, store)
    _daily(driver, store, PAPER_DAYS[:3])
    before = driver.file("paper_ledger.json")
    files = {name: driver.file(name) for name in spd.CARRIED_FILES}
    again = driver.wake(morning_after(PAPER_DAYS[2]) + pd.Timedelta(minutes=5))
    assert driver.file("paper_ledger.json") == before
    assert {name: driver.file(name) for name in spd.CARRIED_FILES} == files
    snapshots = driver.snapshots()
    assert snapshots[-1]["kind"] == "no_new_day" and again.report["snapshot"]["new_trading_days"] == []
    driver.wake(morning_after(PAPER_DAYS[2]) + pd.Timedelta(minutes=10))
    assert driver.snapshots()[-1]["evaluations_at_cutoff"] == 2 and len(driver.snapshots()) == len(snapshots)


def test_missed_wakes_catch_up_to_the_same_ledger(store, tmp_path):
    daily = DailyDriver(tmp_path / "a", store)
    _daily(daily, store, PAPER_DAYS[:6])
    late = DailyDriver(tmp_path / "b", store)
    _daily(late, store, [PAPER_DAYS[0], PAPER_DAYS[5]])
    assert _strip(late.ledger()["days"]) == _strip(daily.ledger()["days"])
    assert [d["booked_at_cutoff"] for d in late.ledger()["days"]][-1] == PAPER_DAYS[5].strftime("%Y-%m-%d")


def test_no_new_day_waits_and_never_fails(store, tmp_path):
    driver = DailyDriver(tmp_path, store)
    _daily(driver, store, PAPER_DAYS[:2])
    driver.reader = AsOf(store, PAPER_DAYS[1])  # the refdata job did not run for a week
    for offset in range(1, 8):
        result = driver.wake(morning_after(PAPER_DAYS[1]) + pd.Timedelta(days=offset))
        assert result.is_waiting and result.verdict is Verdict.PASS
    snapshot = driver.snapshots()[-1]
    assert snapshot["kind"] == "no_new_day" and snapshot["data"]["stale"] is True
    assert len(driver.ledger()["days"]) == 2


def test_typhoon_and_holiday_gap_waits_then_resumes(store, tmp_path):
    driver = DailyDriver(tmp_path, store, started_at=local(2026, 10, 20, 9, 0))
    for as_of, now in [
        ("2026-10-22", local(2026, 10, 23, 9, 30)),
        ("2026-10-22", local(2026, 10, 24, 9, 30)),
        ("2026-10-22", local(2026, 10, 27, 9, 30)),
        ("2026-10-27", local(2026, 10, 28, 9, 30)),
    ]:
        driver.reader = AsOf(store, as_of)
        driver.wake(now)
    days = [d["trading_day"] for d in driver.ledger()["days"]]
    assert days == ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-27"]
    assert driver.results[-1].report["consistency"]["ratio"] == 1.0  # roll at the 10-20 close matches the replay


def test_pass_after_min_trading_days(store, tmp_path):
    driver = DailyDriver(tmp_path, store, config=daily_config(paper_min_trading_days=5))
    results = _daily(driver, store, PAPER_DAYS[:5])
    assert results[-1].verdict is Verdict.PASS and not results[-1].is_waiting
    assert results[-1].report["reason_code"] == "paper_passed"
    assert results[-1].report["summary"].startswith("shadow_paper ✅ 5 trading days · target match 100%")


def test_non_deterministic_strategy_fails_the_replay_check(store, tmp_path):
    config = daily_config(paper_min_trading_days=4)
    config["shadow_paper"]["protection"].update(daily_loss_multiple=1e6, max_drawdown_multiple=1e6)
    driver = DailyDriver(tmp_path, store, strategy=compiled(Flaky), config=config)
    results = _daily(driver, store, PAPER_DAYS[:4])
    assert results[-1].verdict is Verdict.FAIL
    assert results[-1].report["reason_code"] == "paper_replay_inconsistent"
    assert results[-1].report["consistency"]["by_status"].get("target_differs", 0) >= 1


def test_a_revised_day_is_reported_and_compared(tmp_path):
    store = build_store(tmp_path / "rd")
    driver = DailyDriver(tmp_path, store)
    _daily(driver, store, PAPER_DAYS[:4])
    quotes = store.read(FUTURES)
    revised_day = PAPER_DAYS[1]
    mask = (quotes["trade_date"] == revised_day) & (quotes["session"] == "day")
    quotes.loc[mask, ["close", "settlement"]] = quotes.loc[mask, ["close", "settlement"]] * 0.97
    write_futures(store, quotes)
    driver.reader = AsOf(store, PAPER_DAYS[4])
    result = driver.wake(morning_after(PAPER_DAYS[4]))
    revisions = driver.ledger()["revisions"]
    assert [r["trading_day"] for r in revisions] == [revised_day.strftime("%Y-%m-%d")]
    mismatch = {row["trading_day"]: row for row in result.report["consistency"]["mismatches"]}
    assert revised_day.strftime("%Y-%m-%d") in mismatch and mismatch[revised_day.strftime("%Y-%m-%d")]["revised"]
    assert result.report["revisions"]["count"] == 1


def test_forced_liquidation_on_a_paper_day_fails(tmp_path):
    crash = PAPER_DAYS[3]
    store = build_store(tmp_path / "crash", crash=crash.strftime("%Y-%m-%d"))
    driver = DailyDriver(tmp_path, store, strategy=compiled(lambda: LeverageBand(on=5.0, off=5.0)))
    results = _daily(driver, store, PAPER_DAYS[:5])
    final = next(r for r in results if not r.is_waiting)
    assert final.verdict is Verdict.FAIL and final.report["reason_code"] == "paper_protection_trip"
    assert final.report["protection"]["tripped"] == {
        "kind": "forced_liquidation",
        "trading_day": crash.strftime("%Y-%m-%d"),
        "note": final.report["protection"]["tripped"]["note"],
    }


def test_daily_loss_beyond_the_train_quantile_trips(store, tmp_path):
    config = daily_config()
    config["shadow_paper"]["protection"]["daily_loss_multiple"] = 0.01
    driver = DailyDriver(tmp_path, store, config=config)
    results = _daily(driver, store, PAPER_DAYS[:6])
    final = next(r for r in results if not r.is_waiting)
    assert final.verdict is Verdict.FAIL and final.report["protection"]["tripped"]["kind"] == "daily_loss"
    assert final.reason.startswith("protection trip:")


def test_stalled_data_fails_after_max_calendar_days(store, tmp_path):
    driver = DailyDriver(tmp_path, store)
    driver.reader = AsOf(store, PAPER_DAYS[1])
    result = None
    for offset in (1, 30, 91, 95):
        result = driver.wake(morning_after(PAPER_DAYS[0]) + pd.Timedelta(days=offset))
    assert result.verdict is Verdict.FAIL and result.report["reason_code"] == "paper_stalled"


class Broken:
    def read(self, dataset, start=None, end=None):
        raise RefdataError(f"{dataset} partition does not match its manifest")

    def content_sha256(self, dataset):
        return ""


def test_unreadable_refdata_retries_soon(tmp_path):
    driver = DailyDriver(tmp_path, build_store(tmp_path / "x"))
    driver.reader = Broken()
    now = local(2026, 9, 2, 9, 30)
    result = driver.wake(now)
    assert result.is_waiting and result.verdict is Verdict.PASS
    assert result.wait_until == now + pd.Timedelta(minutes=60)


def test_resume_ignores_an_artifact_of_another_strategy(store, tmp_path):
    driver = DailyDriver(tmp_path, store)
    _daily(driver, store, PAPER_DAYS[:2])
    driver.strategy = compiled(fingerprint="e" * 64)
    result = driver.wake(morning_after(PAPER_DAYS[1]))
    assert result.report["resume"]["why"].startswith("different daily paper version or strategy")


def test_without_the_step_4_seam_the_stage_is_unsupported(store, tmp_path):
    stage = ShadowPaperStage(refdata=lambda: store)
    ctx = _ctx(daily_config())
    result = stage.run(ctx)
    assert result.verdict is Verdict.UNSUPPORTED and "compile_strategy" in result.reason
    config = daily_config()
    config["thresholds"].pop("paper_target_match_min")
    result = ShadowPaperStage(refdata=lambda: store, compile_strategy=lambda ctx: compiled()).run(_ctx(config))
    assert result.verdict is Verdict.UNSUPPORTED and "thresholds.paper_target_match_min" in result.reason


# ----------------------------------------------------------------- frozen runs


def _ctx(config) -> StageContext:
    import uuid

    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="",
        input_spec_json=None,
        config=config,
        config_sha256=content_sha256(config),
        stage="shadow_paper",
        attempt=1,
        upstream=stub_upstream(),
        now=local(2026, 9, 2, 9, 30),
        stage_started_at=local(2026, 9, 1),
    )


def test_daily_runs_hash_their_paper_version_and_older_runs_do_not():
    stage = ShadowPaperStage()
    daily = stage.input_payload(_ctx(daily_config()))
    assert daily["paper_version"] == PAPER_VERSION and daily["daily_paper_version"] == spd.DAILY_PAPER_VERSION
    v5 = default_config(stage_plan=None, as_of=pd.Timestamp("2026-10-01").date())
    v5["shadow_paper"]["mode"] = spd.DAILY_MODE  # the mode key alone never switches a v<=5 run
    assert not spd.is_daily_mode(v5)
    assert "daily_paper_version" not in stage.input_payload(_ctx(v5))


def test_run_6_shadow_paper_input_hash_is_unchanged():
    from pathlib import Path

    from tests.pipeline.test_pipeline_frozen_v4 import FIXTURE, ROWS
    from tests.pipeline.test_pipeline_frozen_v4 import _ctx as frozen_ctx

    config = FIXTURE["config_json"]
    assert config["version"] == 4 and not spd.is_daily_mode(config)
    ctx = frozen_ctx("shadow_paper", config, FIXTURE["config_sha256"], ROWS)
    assert content_sha256(ShadowPaperStage().input_payload(ctx)) == ROWS["shadow_paper"]["input_sha256"]
    assert Path(__file__).with_name("fixtures").joinpath("run6_frozen_v4.json").exists()


def test_aux_series_bound_the_cutoff(store):
    strategy = compiled(aux={"taiex": TAIEX})
    reader = AsOf(store, "2026-10-21")

    class LaggingTaiex(AsOf):
        def read(self, dataset, start=None, end=None):
            frame = super().read(dataset, start, end)
            return frame[frame["trade_date"] <= "2026-10-20"] if dataset == TAIEX else frame

    assert spd.load_inputs(
        reader, strategy, spd.DEFAULT_DAILY_SHADOW_PAPER, PAPER_DAYS[0].date()
    ).cutoff == pd.Timestamp("2026-10-21")
    lagging = LaggingTaiex(store, "2026-10-21")
    assert spd.load_inputs(
        lagging, strategy, spd.DEFAULT_DAILY_SHADOW_PAPER, PAPER_DAYS[0].date()
    ).cutoff == pd.Timestamp("2026-10-20")


def test_ledger_is_canonical_json(store, tmp_path):
    driver = DailyDriver(tmp_path, store)
    _daily(driver, store, PAPER_DAYS[:2])
    ledger = driver.ledger()
    assert ledger["daily_paper_version"] == spd.DAILY_PAPER_VERSION and ledger["basis"]["train_days"] > 200
    assert json.dumps(ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() == driver.file(
        "paper_ledger.json"
    )
