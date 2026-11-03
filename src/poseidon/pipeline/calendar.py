"""Trading-day and session calendar: which trading day a bar or a moment belongs to (Asia/Taipei, TAIFEX sessions)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from poseidon.pipeline.window import MARKET_TZ

if TYPE_CHECKING:
    from poseidon.pipeline.refdata.engine_input import TaifexCalendar

MARKET_TZ_NAME = "Asia/Taipei"  # the pandas spelling of MARKET_TZ
TRADING_DAY_SHIFT = pd.Timedelta(hours=9)
DAY_SESSION_HOURS = (pd.Timedelta(hours=8), pd.Timedelta(hours=15))  # local open labels of day-session bars
NEXT_WEEKDAY = "next_weekday"
NEXT_DAY_SESSION = "next_day_session"
NIGHT_SESSION_DAY = {
    NEXT_WEEKDAY: "night session -> next trading day; Asia/Taipei + 9h, weekend rolled to Monday",
    NEXT_DAY_SESSION: "night session -> the next day with a day session in the bars, the day TAIFEX books it to",
}


def night_session_rule(backtest_config: Mapping) -> str:
    """The run's frozen ``backtest.night_session``; :data:`NEXT_WEEKDAY` for a run frozen without it."""

    rule = backtest_config.get("night_session", NEXT_WEEKDAY)
    if rule not in NIGHT_SESSION_DAY:
        raise ValueError(f"unknown backtest.night_session {rule!r}")
    return rule


def trading_days(index: pd.DatetimeIndex, night_session: str = NEXT_WEEKDAY) -> pd.DatetimeIndex:
    """Trading-day label (midnight, naive) of each UTC bar timestamp.

    A day-session bar belongs to its own date. A night-session bar (15:00-05:00 Asia/Taipei) belongs to the
    next trading day: ``next_weekday`` takes the next weekday, so the night before a weekday closure is a
    night-only day of its own; ``next_day_session`` takes the next day with a day session in ``index``, as
    TAIFEX does (Thu 2026-10-22's night -> Tue 10-27 after the 10-23 typhoon closure and the 10-26 holiday).
    A night after the last day session in ``index`` keeps the next weekday.
    """

    local = index.tz_convert(MARKET_TZ_NAME).tz_localize(None)
    days = (local + TRADING_DAY_SHIFT).normalize()
    weekday = days.weekday
    roll = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
    labels = days + pd.to_timedelta(roll, unit="D")
    if night_session == NEXT_WEEKDAY:
        return labels
    sessions = day_sessions(index)
    if not len(sessions):
        return labels
    position = sessions.searchsorted(days)
    inside = position < len(sessions)
    return pd.DatetimeIndex(np.where(inside, sessions.values[np.minimum(position, len(sessions) - 1)], labels.values))


def day_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates (midnight, naive) with a day-session bar in ``index`` (UTC bar timestamps)."""

    local = index.tz_convert(MARKET_TZ_NAME).tz_localize(None)
    clock = local - local.normalize()
    return pd.DatetimeIndex(
        np.unique(local.normalize()[(clock >= DAY_SESSION_HOURS[0]) & (clock < DAY_SESSION_HOURS[1])])
    )


BOUNDARY_LOCAL = time(15, 0)  # a TX trading day starts with the night session at 15:00


DAY_CLOSE_BAR_LOCAL = time(13, 40)  # open label of the last regular day-session 5m bar


def _boundary_on(day: date) -> datetime:
    return datetime.combine(day, BOUNDARY_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)


def paper_start_for(started_at: datetime) -> datetime:
    """First trading-day boundary (15:00 Asia/Taipei) at or after ``started_at``."""

    candidate = _boundary_on(started_at.astimezone(MARKET_TZ).date())
    if candidate < started_at:
        candidate = _boundary_on(started_at.astimezone(MARKET_TZ).date() + timedelta(days=1))
    return candidate


def latest_boundary(now: datetime) -> datetime:
    """Latest trading-day boundary at or before ``now``."""

    candidate = _boundary_on(now.astimezone(MARKET_TZ).date())
    if candidate > now:
        candidate = _boundary_on(now.astimezone(MARKET_TZ).date() - timedelta(days=1))
    return candidate


def settled_cutoff(
    last_bar: datetime | None, paper_start: datetime, upper: datetime, session_days: set[date] | None = None
) -> datetime:
    """Latest boundary in ``(paper_start, upper]`` whose day session's closing bar has arrived.

    Bars arrive in time order, so every trading day before that boundary is
    complete. ``paper_start`` when none is. With ``session_days`` (runs frozen
    with ``next_day_session``) only a day with a day session ends a trading
    day: the night before a closure belongs to the next session, so a boundary
    on the closed day would book that night before its trading day is complete.
    """

    if last_bar is None:
        return paper_start
    boundary = upper
    while boundary > paper_start:
        local_day = boundary.astimezone(MARKET_TZ).date()
        close_bar = datetime.combine(local_day, DAY_CLOSE_BAR_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)
        if last_bar >= close_bar and (session_days is None or local_day in session_days):
            return boundary
        boundary = _boundary_on(local_day - timedelta(days=1))
    return paper_start


def next_wake(now: datetime, settings: dict) -> datetime:
    """Next ``wake_local_time`` (Asia/Taipei) strictly after ``now`` on a ``wake_weekdays`` day."""

    hour, minute = (int(part) for part in settings["wake_local_time"].split(":"))
    weekdays = set(settings["wake_weekdays"])
    if not weekdays:
        raise ValueError("shadow_paper.wake_weekdays is empty")
    local_day = now.astimezone(MARKET_TZ).date()
    for offset in range(8):
        day = local_day + timedelta(days=offset)
        candidate = datetime.combine(day, time(hour, minute), tzinfo=MARKET_TZ).astimezone(UTC)
        if candidate > now and day.weekday() in weekdays:
            return candidate
    raise AssertionError("unreachable: a weekday recurs within 8 days")


def trading_day(moment) -> date:
    stamp = pd.Timestamp(moment)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return trading_days(pd.DatetimeIndex([stamp]))[0].date()


def trading_day_labels(moments, bars_index: pd.DatetimeIndex, night_session: str | None = None) -> dict:
    """``{moment: trading-day date}``. Under ``next_day_session`` a night bar belongs to the next day with a day
    session in ``bars_index`` (the rule :func:`day_hashes` and the backtest use); otherwise the weekday label."""

    if night_session in (None, NEXT_WEEKDAY):
        return {moment: trading_day(moment) for moment in moments}
    stamps = {}
    for moment in moments:
        stamp = pd.Timestamp(moment)
        stamps[moment] = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    bars = pd.DatetimeIndex(bars_index)
    bars = bars.tz_localize("UTC") if bars.tz is None else bars.tz_convert("UTC")
    nanos = np.union1d(bars.asi8, [stamp.value for stamp in stamps.values()])
    index = pd.DatetimeIndex(nanos.astype("datetime64[ns]")).tz_localize("UTC")
    label_of = dict(zip(nanos.tolist(), trading_days(index, night_session), strict=True))
    return {moment: label_of[stamp.value].date() for moment, stamp in stamps.items()}


def last_complete_weekday(upper: datetime) -> date:
    """The trading day that ends at boundary ``upper`` (rolled back to Friday on weekends)."""

    day = upper.astimezone(MARKET_TZ).date()
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def weekdays_after(after: date, through: date) -> int:
    """Weekdays ``d`` with ``after < d <= through``."""

    if through <= after:
        return 0
    return int(np.busday_count(after + timedelta(days=1), through + timedelta(days=1)))


def trading_days_after(after: date, through: date, calendar: TaifexCalendar | None) -> int:
    """TAIFEX trading days ``d`` with ``after < d <= through``; weekdays without a calendar."""

    if calendar is None:
        return weekdays_after(after, through)
    return len(calendar.open_days(pd.Timestamp(after), pd.Timestamp(through))) if through > after else 0


def closure_report(calendar: TaifexCalendar, cutoff: datetime) -> dict:
    """The next session after the trading day ending at ``cutoff`` and the closed weekdays before it."""

    day = pd.Timestamp(cutoff.astimezone(MARKET_TZ).date())
    next_session = calendar.after(day, 1)[0]
    return {
        "closed": [
            {"date": closed.date().isoformat(), "reason": why}
            for closed, why in calendar.closed_between(day, next_session)
        ],
        "next_session": next_session.date().isoformat(),
    }


def closure_text(report: dict) -> str:
    """`` · 休市：10-23 臨時休市、10-26 補假 · 下一交易日 10-27``, or "" when no day was closed."""

    if not report.get("closed"):
        return ""
    days = "、".join(f"{row['date'][5:]} {row['reason']}" for row in report["closed"])
    return f" · 休市：{days} · 下一交易日 {report['next_session'][5:]}"
