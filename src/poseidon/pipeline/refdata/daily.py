"""The daily refdata job: idempotent incremental update + freshness check (design §9.2).

Schedule: once a day at **09:00Z (17:00 Asia/Taipei)**, every day of the week.

* TAIFEX publishes the day session's daily quotes after the 13:45 close (the expiring contract
  stops at 13:30) and TWSE the TAIEX / total-return rows after its 13:30 close; both are out by
  about 15:00-16:00 Taipei. The night session TAIFEX books into a trading day ended at 05:00
  that morning, so at 17:00 trading day D is complete in both sources.
* 09:00Z is right after the 06:00-08:40Z heavy-CPU window and the whole run is ~15 polite
  requests (2 months x 3 products + FSP page at >= 1.5 s, 2 TWSE months x 2 + 2 schedule years at
  >= 3 s): well under a minute.
* The daily shadow paper wakes at 09:30 Taipei (01:30Z) the next morning and only evaluates days
  the data already has, so a missed or late job just delays the paper by a day (it waits, it
  never fails on stale data). Weekend runs are no-ops that catch up a failed Friday.

Freshness: the latest trading day each dataset should hold by ``now`` is the last scheduled
trading day (official TWSE schedule) on or before today, today itself only from
``PUBLISHED_BY_LOCAL`` (16:00 Taipei) on. Each dataset's last date (for futures: last date with a
day session, per product) is compared with it; ``lag_trading_days`` counts the scheduled trading
days missing. An unscheduled closure (typhoon) is not missing data: when TAIFEX has already booked
the night session after the last day session into a later date, the scheduled days in between
were closed (``engine_input.unscheduled_closures``); they are reported and left out of the lag.
Next year's schedule not yet published is a warning from ``SCHEDULE_WARN_FROM`` (1 December) on,
and so is a contract expiring within ``LTD_WARN_DAYS`` whose last trading day is not on the
official schedule.

``EXTRA_UPDATERS`` is the hook for later datasets (TXO quotes, corporate actions): each entry
``name -> fn(store, today=..., code_version=...) -> list[UpdateReport]`` runs after the futures
and TWSE groups, and its failure is recorded without stopping the others.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, time

import pandas as pd

from poseidon.pipeline.refdata.datasets import (
    CALENDAR,
    FUTURES,
    MARKET_CALENDAR,
    PRODUCTS,
    TAIEX,
    TAIEX_TR,
    UpdateReport,
    update_all,
)
from poseidon.pipeline.refdata.engine_input import last_day_session, product_quotes, unscheduled_closures
from poseidon.pipeline.refdata.fetch import SourceBlockedError, SourceError
from poseidon.pipeline.refdata.store import RefdataError, RefdataStore
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.window import MARKET_TZ

logger = logging.getLogger(__name__)

DAILY_RUN_UTC = time(9, 0)  # 17:00 Asia/Taipei
PUBLISHED_BY_LOCAL = time(16, 0)
SCHEDULE_WARN_FROM = (12, 1)
LTD_WARN_DAYS = 62  # front and next monthly contract
DAILY_TASK = "poseidon.pipeline.tasks.update_refdata_daily"
UpdaterHook = Callable[..., list[UpdateReport]]
EXTRA_UPDATERS: dict[str, UpdaterHook] = {}


def beat_entry() -> dict:
    """The pipeline-beat schedule entry (``celery_app.beat_schedule["update-refdata-daily"]``)."""

    from celery.schedules import crontab

    return {"task": DAILY_TASK, "schedule": crontab(hour=DAILY_RUN_UTC.hour, minute=DAILY_RUN_UTC.minute)}


def expected_latest_trading_day(now: datetime, schedule: MarketSchedule) -> pd.Timestamp:
    local = now.astimezone(MARKET_TZ)
    day = pd.Timestamp(local.date())
    if not (schedule.is_open(day) and local.time() >= PUBLISHED_BY_LOCAL):
        day -= pd.Timedelta(days=1)
    for _ in range(30):
        if schedule.is_open(day):
            return day
        day -= pd.Timedelta(days=1)
    raise ValueError("no scheduled trading day in the last 30 days")


def _entry(last: pd.Timestamp | None, expected: pd.Timestamp, schedule: MarketSchedule, closed) -> dict:
    if last is None:
        return {"last": None, "lag_trading_days": None, "fresh": False}
    missing = schedule.open_days(last, expected) if last < expected else pd.DatetimeIndex([])
    lag = int((~missing.isin(closed)).sum())
    return {"last": last.date().isoformat(), "lag_trading_days": lag, "fresh": lag == 0}


def freshness(store: RefdataStore, now: datetime) -> dict:
    schedule = MarketSchedule.from_frame(store.read(MARKET_CALENDAR)) if store.exists(MARKET_CALENDAR) else None
    if schedule is None:
        return {"status": "missing", "missing": [MARKET_CALENDAR]}
    expected = expected_latest_trading_day(now, schedule)
    since = expected - pd.Timedelta(days=45)
    quotes = store.read(FUTURES, start=since)
    closed = unscheduled_closures(product_quotes(quotes, "TX"), schedule)
    datasets = {
        f"{FUTURES}:{p}": _entry(last, expected, schedule, closed)
        for p, last in last_day_session(quotes, PRODUCTS).items()
    }
    for name in (TAIEX, TAIEX_TR):
        frame = store.read(name, start=since)
        last = pd.Timestamp(frame["trade_date"].max()) if len(frame) else None
        datasets[name] = _entry(last, expected, schedule, closed)
    warnings = []
    local = now.astimezone(MARKET_TZ).date()
    if (local.month, local.day) >= SCHEDULE_WARN_FROM and local.year + 1 not in schedule.published_years:
        warnings.append(f"TWSE has not published the {local.year + 1} holiday schedule yet")
    calendar = store.read(CALENDAR, start=pd.Timestamp(local))
    soon = calendar["last_trading_day"] <= pd.Timestamp(local) + pd.Timedelta(days=LTD_WARN_DAYS)
    upcoming = calendar[(calendar["status"] == "active") & (calendar["ltd_source"] != "twse_schedule") & soon]
    for row in upcoming.sort_values("last_trading_day").itertuples():
        warnings.append(
            f"{row.product} {row.contract_month}: last trading day {row.last_trading_day.date()} is {row.ltd_source}"
        )
    stale = sorted(name for name, entry in datasets.items() if not entry["fresh"])
    return {
        "status": "fresh" if not stale else "stale",
        "expected_latest_trading_day": expected.date().isoformat(),
        "unscheduled_closures": [day.date().isoformat() for day in closed],
        "datasets": datasets,
        "stale": stale,
        "warnings": warnings,
    }


def run_daily(
    store: RefdataStore | None = None,
    *,
    now: datetime | None = None,
    fetchers=None,
    code_version: str = "unknown",
) -> dict:
    """Update every daily dataset, then check freshness. Never raises for stale or blocked data."""

    store = store or RefdataStore()
    now = now or datetime.now(UTC)
    today = now.astimezone(MARKET_TZ).date()
    result: dict = {"run_at_utc": now.isoformat(), "updates": [], "errors": []}
    try:
        reports = update_all(store, refresh_months=2, code_version=code_version, fetchers=fetchers, today=today)
        result["updates"] = [asdict(report) for report in reports]
    except SourceBlockedError as error:  # stop at once; tomorrow's run fills the gap
        result["errors"].append({"group": "core", "kind": "blocked", "error": str(error)[:500]})
    except (SourceError, RefdataError) as error:
        result["errors"].append({"group": "core", "kind": type(error).__name__, "error": str(error)[:500]})
    for name, hook in EXTRA_UPDATERS.items():
        try:
            result["updates"].extend(asdict(r) for r in hook(store, today=today, code_version=code_version))
        except Exception as error:  # a later dataset must never take the core ones down
            result["errors"].append({"group": name, "kind": type(error).__name__, "error": str(error)[:500]})
    try:
        result["freshness"] = freshness(store, now)
    except RefdataError as error:
        result["freshness"] = {"status": "error", "error": str(error)[:500]}
    result["changed"] = sorted(u["dataset"] for u in result["updates"] if u["changed"])
    level = logging.INFO if not result["errors"] and result["freshness"].get("status") == "fresh" else logging.WARNING
    logger.log(
        level,
        "refdata daily: changed=%s freshness=%s errors=%s",
        result["changed"],
        result["freshness"].get("status"),
        result["errors"],
    )
    return result
