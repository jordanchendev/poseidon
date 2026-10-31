"""Refdata -> target-engine inputs: per-contract daily bars, expiries, the trading calendar, aux series.

Bars (``engine.data`` contract) for one TAIFEX product, monthly contracts only, one row per
(trading day, contract) that has a **day-session** row:

* ``open``  = the night-session open when TAIFEX books a night row into that trading day (the
  15:00 session of the previous trading day, see ``refdata.taifex``), else the day open;
* ``high`` / ``low`` = extremes over both sessions;
* ``close`` = day-session close, ``settle`` = daily settlement (fills a missing close).

A trade date that has only a night row (the session after the last day session, booked into a
trading day whose day session has not happened yet, e.g. 2026-10-27 seen on 2026-10-24 after the
10-23 typhoon closure and the 10-26 holiday) is not a trading day yet and is left out, so the data
cutoff is always the last date with a day session.

Expiries come from ``taifex_settlement_calendar`` (traded last day for settled contracts, the
official schedule for unexpired ones). The trading calendar is the product's day-session dates up
to ``through`` followed by the official schedule (``twse_market_calendar``) for ``horizon_days``:
the engine needs the next trading day to place a ``roll.before_settlement`` roll at the right
close, and it must be the same day a later full replay will see.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import ContractSpec, FuturesInput
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, MARKET_CALENDAR, TAIEX
from poseidon.pipeline.refdata.taifex import MarketSchedule, official_last_trading_day

MONTHLY = r"\d{6}"
CALENDAR_HORIZON_DAYS = 120
MAX_STOCK_GAP_DAYS = 10  # sessions a stock may miss (carried forward) before its data counts as ended
UNSCHEDULED_CLOSURE = "臨時休市"  # a scheduled day the market did not open (typhoon); refdata does not say why
CLOSURE_LOOKBACK_DAYS = 31  # futures rows read before the last session to find the closures after it


class StockDataGap(ValueError):
    """A stock's bars do not cover the run's calendar (listed later, delisted, or a long hole)."""


class RefdataReader(Protocol):
    """What the daily paper reads: ``RefdataStore`` or a test view of it."""

    def read(self, dataset: str, start=None, end=None, rows=None) -> pd.DataFrame: ...

    def content_sha256(self, dataset: str) -> str: ...

    def exists(self, dataset: str) -> bool: ...

    def manifest(self, dataset: str) -> dict: ...


def product_quotes(quotes: pd.DataFrame, product: str) -> pd.DataFrame:
    rows = quotes[(quotes["product"] == product) & quotes["contract_month"].astype(str).str.fullmatch(MONTHLY)]
    return rows.reset_index(drop=True)


def day_session_dates(quotes: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(quotes.loc[quotes["session"] == "day", "trade_date"].unique()))


def futures_bars(quotes: pd.DataFrame) -> pd.DataFrame:
    """Engine bars from one product's monthly quotes (both sessions)."""

    day = quotes[quotes["session"] == "day"].set_index(["trade_date", "contract_month"])
    night = quotes[quotes["session"] == "night"].set_index(["trade_date", "contract_month"])
    night = night.reindex(day.index)
    out = pd.DataFrame(
        {
            "open": night["open"].where(night["open"].notna(), day["open"]),
            "high": np.fmax(day["high"].to_numpy(float), night["high"].to_numpy(float)),
            "low": np.fmin(day["low"].to_numpy(float), night["low"].to_numpy(float)),
            "close": day["close"],
            "settle": day["settlement"],
        },
        index=day.index,
    ).reset_index()
    out = out.rename(columns={"trade_date": "date", "contract_month": "contract"})
    out["contract"] = out["contract"].astype(str)
    return out.sort_values(["date", "contract"], kind="mergesort").reset_index(drop=True)


def expiries_for(calendar: pd.DataFrame, product: str, contracts, schedule: MarketSchedule) -> dict[str, pd.Timestamp]:
    """Last trading day per contract; a contract the calendar does not list yet gets the official one."""

    rows = calendar[calendar["product"] == product]
    known = dict(zip(rows["contract_month"].astype(str), pd.to_datetime(rows["last_trading_day"]), strict=True))
    return {c: known[c] if c in known else official_last_trading_day(c, schedule)[0] for c in contracts}


def trading_calendar(
    day_dates: pd.DatetimeIndex, schedule: MarketSchedule, through: pd.Timestamp, horizon_days: int
) -> pd.DatetimeIndex:
    past = day_dates[day_dates <= through]
    future = schedule.open_days(through, through + pd.Timedelta(days=horizon_days))
    return past.append(future)


def futures_input(
    quotes: pd.DataFrame,
    calendar: pd.DataFrame,
    schedule: MarketSchedule,
    *,
    product: str,
    spec: ContractSpec,
    through: pd.Timestamp,
    roll_days_before: int,
) -> FuturesInput:
    bars = futures_bars(product_quotes(quotes, product))
    bars = bars[bars["date"] <= through].reset_index(drop=True)
    expiries = expiries_for(calendar, product, bars["contract"].unique(), schedule)
    return FuturesInput(spec=spec, bars=bars, expiries=expiries, roll_days_before=roll_days_before)


def align_stock_bars(bars: pd.DataFrame, days: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    """``bars`` (``date``, OHLC, optional ``volume``) on exactly the trading ``days``: a day without
    a bar carries the previous close (open = high = low = close) with ``traded = False``, so the
    engine fills nothing there. More than ``MAX_STOCK_GAP_DAYS`` missing in a row, bars starting
    after the first day or ending before the last one raise :class:`StockDataGap`."""

    frame = bars.assign(date=pd.to_datetime(bars["date"]).dt.normalize()).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise StockDataGap(f"{name}: no bars")
    if frame.index[0] > days[0]:
        raise StockDataGap(f"{name}: bars start {frame.index[0].date()}, after the window's first day {days[0].date()}")
    on_calendar = frame.index.isin(days)
    if not on_calendar.all():
        extra = frame.index[~on_calendar][:3]
        raise StockDataGap(f"{name}: bars on days outside the trading calendar, e.g. {[d.date() for d in extra]}")
    present = pd.Series(days.isin(frame.index), index=days)
    missing = ~present
    run = missing.groupby((~missing).cumsum()).cumsum()
    if run.max() > MAX_STOCK_GAP_DAYS:
        first = run[run > MAX_STOCK_GAP_DAYS].index[0]
        raise StockDataGap(
            f"{name}: more than {MAX_STOCK_GAP_DAYS} sessions without a bar before {first.date()} "
            f"(last bar {frame.index[-1].date()})"
        )
    out = frame.reindex(days)
    out["close"] = out["close"].ffill()
    for column in ("open", "high", "low"):
        out[column] = out[column].where(present, out["close"])
    if "volume" in out.columns:
        out["volume"] = out["volume"].where(present, 0.0)
    out["traded"] = present.to_numpy()
    return out.rename_axis("date").reset_index()


def aux_frame(frame: pd.DataFrame, date_column: str = "trade_date") -> pd.DataFrame:
    """A refdata daily series as an engine aux frame: DatetimeIndex, numeric columns only."""

    indexed = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame[date_column])).normalize())
    numeric = indexed.drop(columns=[date_column]).select_dtypes("number")
    return numeric.sort_index().astype(float)


def read_schedule(reader: RefdataReader) -> MarketSchedule:
    return MarketSchedule.from_frame(reader.read(MARKET_CALENDAR))


def read_quotes(reader: RefdataReader, start) -> pd.DataFrame:
    return reader.read(FUTURES, start=start)


def read_calendar(reader: RefdataReader) -> pd.DataFrame:
    return reader.read(CALENDAR)


def last_day_session(quotes: pd.DataFrame, products: Mapping[str, str] | list[str]) -> dict[str, pd.Timestamp | None]:
    out = {}
    for product in products:
        dates = day_session_dates(product_quotes(quotes, product))
        out[product] = dates[-1] if len(dates) else None
    return out


def unscheduled_closures(quotes: pd.DataFrame, schedule: MarketSchedule) -> pd.DatetimeIndex:
    """Scheduled trading days TAIFEX skipped right after the last day session.

    A night row dated D after the last day session means TAIFEX booked that session into D, so every
    scheduled trading day strictly between was closed (2026-10-23, typhoon: the 10-22 15:00 session is
    dated 10-27). Undetectable when the night session was closed too.
    """

    days = day_session_dates(quotes)
    if not len(days):
        return pd.DatetimeIndex([])
    later = quotes.loc[(quotes["session"] == "night") & (quotes["trade_date"] > days[-1]), "trade_date"]
    if not len(later):
        return pd.DatetimeIndex([])
    return schedule.open_days(days[-1], pd.Timestamp(later.min()) - pd.Timedelta(days=1))


@dataclass(frozen=True)
class TaifexCalendar:
    """Which days TX trades: the day sessions held (``twse_taiex_daily``, the same days as TX's day session,
    typhoon closures included), then the official schedule (``twse_market_calendar``) minus the unscheduled
    closures detected after the last held session (:func:`unscheduled_closures`).

    ``holidays``: why a scheduled holiday is closed (the schedule's name, ``補假`` for a make-up day).
    ``sources``: content_sha256 of the datasets read.
    """

    sessions: pd.DatetimeIndex
    schedule: MarketSchedule
    closures: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    holidays: Mapping[pd.Timestamp, str] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)

    def open_days(self, after: pd.Timestamp, through: pd.Timestamp) -> pd.DatetimeIndex:
        """Trading days in ``(after, through]``."""

        after, through = pd.Timestamp(after).normalize(), pd.Timestamp(through).normalize()
        held = self.sessions[(self.sessions > after) & (self.sessions <= through)]
        last = self.sessions[-1] if len(self.sessions) else after
        scheduled = self.schedule.open_days(max(after, last), through)
        return held.append(scheduled[~scheduled.isin(self.closures)])

    def after(self, day: pd.Timestamp, count: int) -> pd.DatetimeIndex:
        """The ``count`` trading days after ``day``."""

        through = pd.Timestamp(day).normalize()
        days = pd.DatetimeIndex([])
        while len(days) < count:
            through += pd.Timedelta(days=31)
            days = self.open_days(day, through)
        return days[:count]

    def closed_between(self, after: pd.Timestamp, before: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
        """Weekdays strictly between ``after`` and ``before`` with no session, and why."""

        weekdays = pd.bdate_range(pd.Timestamp(after).normalize() + pd.Timedelta(days=1), before, inclusive="left")
        open_ = self.open_days(after, before)
        return [(day, self.holidays.get(day, UNSCHEDULED_CLOSURE)) for day in weekdays if day not in open_]


def holiday_label(day: pd.Timestamp, name: str, note: str) -> str:
    """``補假`` when the schedule's note makes ``day`` the make-up day of a weekend holiday, else its name."""

    for month, dom in re.findall(r"於(\d+)月(\d+)日[（(]星期.[）)]補假", note or ""):
        if (int(month), int(dom)) == (day.month, day.day):
            return "補假"
    return name


def read_taifex_calendar(reader: RefdataReader, since: pd.Timestamp, product: str = "TX") -> TaifexCalendar:
    """The :class:`TaifexCalendar` from ``since`` on, read from ``reader``."""

    taiex = reader.read(TAIEX, start=since)
    column = reader.manifest(TAIEX)["date_column"]
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(taiex[column]).dt.normalize().unique()))
    frame = reader.read(MARKET_CALENDAR)
    schedule = MarketSchedule.from_frame(frame)
    closed = frame[frame["closed"].astype(bool)]
    holidays = {
        pd.Timestamp(day).normalize(): holiday_label(pd.Timestamp(day), name, note)
        for day, name, note in zip(closed["date"], closed["name"], closed["note"], strict=True)
    }
    last = sessions[-1] if len(sessions) else pd.Timestamp(since)
    quotes = product_quotes(reader.read(FUTURES, start=last - pd.Timedelta(days=CLOSURE_LOOKBACK_DAYS)), product)
    return TaifexCalendar(
        sessions=sessions,
        schedule=schedule,
        closures=unscheduled_closures(quotes, schedule),
        holidays=holidays,
        sources={name: reader.content_sha256(name) for name in (TAIEX, MARKET_CALENDAR, FUTURES)},
    )
