"""Official last trading days (TWSE holiday schedule), the daily refdata job and its freshness check."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline.refdata import daily, taifex, twse
from poseidon.pipeline.refdata import datasets as D
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.taifex import MarketSchedule, official_last_trading_day
from poseidon.pipeline.window import MARKET_TZ

FIXTURES = Path(__file__).parent / "fixtures" / "refdata"


def local(year, month, day, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ).astimezone(UTC)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def schedule_2026() -> MarketSchedule:
    return MarketSchedule.from_frame(twse.parse_holidays(fixture("twse_holidays_2026.json")))


def quotes_2026() -> pd.DataFrame:
    frames = [taifex.parse_futures_csv(fixture(f"taifex_TX_{m}.csv")) for m in ("202609", "202610")]
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- the official schedule


def test_holiday_schedule_marks_closed_days_and_open_markers():
    frame = twse.parse_holidays(fixture("twse_holidays_2026.json")).set_index("date")
    assert frame.loc["2026-10-26", "closed"] and frame.loc["2026-02-12", "closed"]  # holiday; settlement-only day
    assert not frame.loc["2026-01-02", "closed"] and not frame.loc["2026-02-11", "closed"]  # first/last trading day
    assert pd.Timestamp("2026-10-23") not in frame.index  # the typhoon closure was never scheduled
    assert twse.parse_holidays(fixture("twse_holidays_2027.json")).empty  # not published yet on 2026-10-24
    assert not twse.is_holiday_json(b"<html>blocked</html>")


@pytest.mark.parametrize(
    ("contract", "expected", "source"),
    [
        ("202602", "2026-02-23", "twse_schedule"),  # 3rd Wed 02-18 is Lunar New Year -> next trading day
        ("202610", "2026-10-21", "twse_schedule"),
        ("202611", "2026-11-18", "twse_schedule"),
        ("202701", "2027-01-20", "rule_unpublished_year"),
    ],
)
def test_official_last_trading_day(contract, expected, source):
    assert official_last_trading_day(contract, schedule_2026()) == (pd.Timestamp(expected), source)


def test_calendar_uses_the_schedule_for_unexpired_contracts_and_checks_settled_ones():
    market = twse.parse_holidays(fixture("twse_holidays_2026.json"))
    fsp = taifex.parse_fsp_html(fixture("taifex_fsp_sample.html"))
    calendar = taifex.settlement_calendar(quotes_2026(), fsp, market).set_index("contract_month")
    assert calendar.loc["202609", "ltd_source"] == "traded" and calendar.loc["202609", "ltd_check"]
    assert (
        calendar.loc["202610", "last_trading_day"] == pd.Timestamp("2026-10-21") and calendar.loc["202610", "ltd_check"]
    )
    active = calendar.loc["202611"]
    assert (active["status"], active["ltd_source"]) == ("active", "twse_schedule")
    assert active["last_trading_day"] == pd.Timestamp("2026-11-18") == active["scheduled_last_trading_day"]


def test_an_unscheduled_closure_on_the_last_trading_day_shifts_it():
    """A contract still trading after its scheduled day (typhoon on that day) expires on the next trading day."""

    market = twse.parse_holidays(fixture("twse_holidays_2026.json"))
    quotes = quotes_2026()
    rows = quotes[(quotes["contract_month"] == "202610") & (quotes["trade_date"] == "2026-10-20")].copy()
    rows["trade_date"] = pd.Timestamp("2026-10-22")  # 202610 still trades on 10-22: 10-21 was closed
    quotes = pd.concat([quotes[quotes["trade_date"] != "2026-10-21"], rows], ignore_index=True)
    empty_fsp = taifex.parse_fsp_html(fixture("taifex_fsp_sample.html")).iloc[0:0]
    calendar = taifex.settlement_calendar(quotes, empty_fsp, market).set_index("contract_month")
    row = calendar.loc["202610"]
    assert (row["status"], row["ltd_source"]) == ("active", "shifted_after_closure")
    # the next scheduled trading day after the latest trade date (the schedule cannot know a later closure)
    assert row["last_trading_day"] == pd.Timestamp("2026-10-23")


# ----------------------------------------------------------------- the daily job


class FakeSources:
    """TX 2026-09/10 CSVs, FSP page, TWSE 2026-09 index months (later months: no data), 2026/2027 schedules."""

    def __init__(self):
        self.calls: list[str] = []

    def fetch(self, url, *, data=None, validate=None):
        if url == taifex.FSP_URL:
            key, body = "fsp", fixture("taifex_fsp_sample.html")
        elif url == taifex.FUT_DAILY_URL:
            key = data["commodity_id"] + "_" + data["queryStartDate"][:7].replace("/", "")
            body = fixture(f"taifex_{key}.csv")
        elif url.startswith(twse.HOLIDAY_URL):
            year = url.split("date=")[1][:4]
            key, body = f"holidays_{year}", fixture(f"twse_holidays_{year}.json")
        else:
            kind = "MI_5MINS_HIST" if "MI_5MINS_HIST" in url else "MFI94U"
            month = url.split("date=")[1][:6]
            key = f"{kind}_{month}"
            body = fixture(f"twse_{kind}_{month}.json") if month == "202609" else fixture("twse_no_data.json")
        self.calls.append(key)
        assert validate is None or validate(body)
        return body


@pytest.fixture
def small_world(monkeypatch):
    """Every dataset starts in 2026-09 (fixtures), TX only, schedules from 2026."""

    tx = dataclasses.replace(D._taifex_source("TX"), first_month=(2026, 9))
    futures = dataclasses.replace(D.MONTHLY[D.FUTURES], sources=(tx,))
    taiex = dataclasses.replace(
        D.MONTHLY[D.TAIEX], sources=(dataclasses.replace(D.MONTHLY[D.TAIEX].sources[0], first_month=(2026, 9)),)
    )
    total = dataclasses.replace(
        D.MONTHLY[D.TAIEX_TR], sources=(dataclasses.replace(D.MONTHLY[D.TAIEX_TR].sources[0], first_month=(2026, 9)),)
    )
    monkeypatch.setattr(D, "MONTHLY", {D.FUTURES: futures, D.TAIEX: taiex, D.TAIEX_TR: total})
    monkeypatch.setattr(twse, "HOLIDAY_FIRST_YEAR", 2026)
    monkeypatch.setattr(daily, "PRODUCTS", ("TX",))
    sources = FakeSources()
    return {"taifex": sources, "twse": sources}


def test_daily_job_is_idempotent_and_reports_freshness(tmp_path, small_world):
    store = RefdataStore(tmp_path)
    now = local(2026, 10, 24, 17, 0)  # Saturday after the 10-23 typhoon closure
    first = daily.run_daily(store, now=now, fetchers=small_world, code_version="t")
    assert first["errors"] == [] and D.MARKET_CALENDAR in first["changed"] and D.CALENDAR in first["changed"]
    fresh = first["freshness"]
    assert fresh["expected_latest_trading_day"] == "2026-10-23"  # scheduled open; closed by the typhoon
    # TAIFEX booked the 10-22 night session into 10-27: 10-23 was closed (typhoon), nothing is missing
    assert fresh["unscheduled_closures"] == ["2026-10-23"]
    assert fresh["datasets"][f"{D.FUTURES}:TX"] == {"last": "2026-10-22", "lag_trading_days": 0, "fresh": True}
    assert fresh["datasets"][D.TAIEX]["last"] == "2026-09-30" and fresh["status"] == "stale"
    calendar = store.read(D.CALENDAR).set_index("contract_month")
    assert calendar.loc["202611", "ltd_source"] == "twse_schedule"
    assert not any("202611" in warning for warning in fresh["warnings"])  # on the official schedule

    manifests = {name: (tmp_path / name / "manifest.json").read_bytes() for name in D.GROUPS["futures"]}
    small_world["taifex"].calls.clear()
    second = daily.run_daily(store, now=now, fetchers=small_world, code_version="t")
    assert second["changed"] == [] and second["errors"] == []
    assert {name: (tmp_path / name / "manifest.json").read_bytes() for name in manifests} == manifests
    calls = small_world["taifex"].calls
    assert [c for c in calls if c.startswith("holidays_")] == ["holidays_2026", "holidays_2027"]  # current + next


def test_a_blocked_source_stops_the_job_without_raising(tmp_path, small_world):
    from poseidon.pipeline.refdata.fetch import SourceBlockedError

    class Blocked:
        def fetch(self, url, *, data=None, validate=None):
            raise SourceBlockedError(f"{url}: HTTP 429 on every attempt; stopping")

    result = daily.run_daily(
        RefdataStore(tmp_path),
        now=local(2026, 10, 27, 17, 0),
        fetchers={"taifex": Blocked(), "twse": Blocked()},
        code_version="t",
    )
    assert result["errors"][0]["kind"] == "blocked" and result["freshness"]["status"] == "missing"


def test_extra_updaters_run_and_their_failure_is_contained(tmp_path, small_world, monkeypatch):
    seen = []

    def txo(store, *, today, code_version):
        seen.append(today)
        raise RuntimeError("txo source down")

    monkeypatch.setitem(daily.EXTRA_UPDATERS, "txo", txo)
    result = daily.run_daily(
        RefdataStore(tmp_path), now=local(2026, 10, 27, 17, 0), fetchers=small_world, code_version="t"
    )
    assert seen == [date(2026, 10, 27)]
    assert {"group": "txo", "kind": "RuntimeError", "error": "txo source down"} in result["errors"]
    assert result["updates"]  # the core datasets were still updated


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (local(2026, 10, 22, 15, 59), "2026-10-21"),  # before publication: yesterday
        (local(2026, 10, 22, 16, 0), "2026-10-22"),
        (local(2026, 10, 27, 9, 30), "2026-10-23"),  # Mon 10-26 holiday; 10-23 was scheduled open
        (local(2026, 2, 23, 9, 30), "2026-02-11"),  # across Lunar New Year (02-12/13 settlement only)
    ],
)
def test_expected_latest_trading_day(now, expected):
    assert daily.expected_latest_trading_day(now, schedule_2026()).date().isoformat() == expected


def test_beat_entry_is_17_00_taipei_outside_the_heavy_window():
    entry = daily.beat_entry()
    assert entry["task"] == "poseidon.pipeline.tasks.update_refdata_daily"
    assert entry["schedule"].hour == {9} and entry["schedule"].minute == {0}
    assert not 6 <= daily.DAILY_RUN_UTC.hour + daily.DAILY_RUN_UTC.minute / 60 <= 8 + 40 / 60
    # not wired into the live runner's beat here: test_pipeline_unit checks it still schedules only the sweeper
