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

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import pandas as pd

from poseidon.pipeline.engine import ContractSpec, FuturesInput
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, MARKET_CALENDAR
from poseidon.pipeline.refdata.taifex import MarketSchedule, official_last_trading_day

MONTHLY = r"\d{6}"
CALENDAR_HORIZON_DAYS = 120


class RefdataReader(Protocol):
    """What the daily paper reads: ``RefdataStore`` or a test view of it."""

    def read(self, dataset: str, start=None, end=None) -> pd.DataFrame: ...

    def content_sha256(self, dataset: str) -> str: ...


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
