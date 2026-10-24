"""TAIFEX sources: daily futures quotes and the index-futures final settlement table.

Daily quotes (``futDataDown``)
    One POST per product per calendar month returns a cp950 CSV of every
    contract month, both sessions. Back-fill reaches the first trading day of
    each product (TX 1998-07-21, MTX 2001-04-09, TMF 2024-07-29); a month
    before listing returns the header only.

    Rows kept: single contracts (``YYYYMM`` and weekly ``YYYYMMWn``); calendar
    spreads (``YYYYMM/YYYYMM``) are dropped. Prices of ``-`` or <= 0 become
    NaN (the expiring contract reports settlement 0 on its last trading day).

    **Session attribution.** ``session = "night"`` rows (TAIFEX 盤後, from
    2017-05-16) carry TAIFEX's trade date: the night row dated D is the
    after-hours session that opened at 15:00 on the trading day *before* D and
    closed at 05:00 the next morning; TAIFEX books it into trading day D
    (its change column is measured from the previous day's settlement). So on
    any date D the night row precedes the day row. This is the opposite of
    ``pipeline.data._session_keys``, which dates a Thalassa night session by
    the day it opens.

Final settlement prices (``futIndxFSP``)
    One POST returns an HTML table of every TX/MTX/TMF final settlement (date,
    contract month incl. weekly, price) since 1998-09. Until the 2008-11
    contract the final settlement day was the trading day *after* the last
    trading day (the price came from the next morning's index); from 2008-12
    they coincide.
"""

from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FUT_DAILY_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
FSP_URL = "https://www.taifex.com.tw/cht/5/futIndxFSP"

FIRST_TRADING_DAY = {"TX": "1998-07-21", "MTX": "2001-04-09", "TMF": "2024-07-29"}

_HEADER_PREFIX = "交易日期,契約,到期月份(週別)"
_RENAME = {
    "交易日期": "trade_date",
    "契約": "product",
    "到期月份(週別)": "contract_month",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
    "成交量": "volume",
    "結算價": "settlement",
    "未沖銷契約數": "open_interest",
    "最後最佳買價": "best_bid",
    "最後最佳賣價": "best_ask",
    "交易時段": "session",
}
_SESSIONS = {"一般": "day", "盤後": "night"}
PRICE_COLUMNS = ["open", "high", "low", "close", "settlement", "best_bid", "best_ask"]
FUTURES_COLUMNS = [
    "trade_date",
    "product",
    "contract_month",
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "settlement",
    "open_interest",
    "best_bid",
    "best_ask",
]
FUTURES_SORT = ["trade_date", "product", "contract_month", "session"]
FSP_COLUMNS = ["final_settlement_date", "contract_month", "final_settlement_price"]


class TaifexParseError(ValueError):
    """A TAIFEX response is not in the expected format."""


@dataclass(frozen=True)
class Request:
    url: str
    params: dict  # POST form fields


def futures_request(product: str, year: int, month: int) -> Request:
    last = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).day
    return Request(
        FUT_DAILY_URL,
        {
            "down_type": "1",
            "commodity_id": product,
            "commodity_id2": "",
            "queryStartDate": f"{year}/{month:02d}/01",
            "queryEndDate": f"{year}/{month:02d}/{last:02d}",
        },
    )


def fsp_request(last_year: int) -> Request:
    """Every TX/MTX/TMF final settlement (commodity group 1) up to ``last_year``."""

    return Request(
        FSP_URL,
        {
            "commodityIds": "1",
            "_all": "on",
            "start_year": "1998",
            "start_month": "01",
            "end_year": str(last_year),
            "end_month": "12",
        },
    )


def is_futures_csv(body: bytes) -> bool:
    try:
        return body.decode("cp950").lstrip("﻿").startswith(_HEADER_PREFIX)
    except UnicodeDecodeError:
        return False


def _number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned.where(~cleaned.isin(["-", "", "nan"])), errors="coerce")


def parse_futures_csv(body: bytes) -> pd.DataFrame:
    """Rows of one ``futDataDown`` response (any product, any period)."""

    if not is_futures_csv(body):
        raise TaifexParseError(f"not a TAIFEX futures CSV: {body[:80]!r}")
    raw = pd.read_csv(io.BytesIO(body), encoding="cp950", index_col=False, dtype=str, keep_default_na=False)
    raw.columns = [str(column).strip().lstrip("﻿") for column in raw.columns]
    missing = set(_RENAME) - set(raw.columns) - {"交易時段"}
    if missing:
        raise TaifexParseError(f"TAIFEX CSV lacks columns {sorted(missing)}")
    frame = raw[[column for column in _RENAME if column in raw.columns]].rename(columns=_RENAME)
    if "session" not in frame:
        frame["session"] = "一般"
    for column in ("product", "contract_month", "session"):
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[frame["contract_month"].str.fullmatch(r"\d{6}(W\d)?")].copy()
    unknown = set(frame["session"]) - set(_SESSIONS)
    if unknown:
        raise TaifexParseError(f"unknown TAIFEX sessions {sorted(unknown)}")
    frame["session"] = frame["session"].map(_SESSIONS)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].str.strip(), format="%Y/%m/%d")
    for column in PRICE_COLUMNS:
        values = _number(frame[column]).astype("float64")
        frame[column] = values.where(values > 0)
    frame["volume"] = _number(frame["volume"]).fillna(0).astype("int64")
    frame["open_interest"] = _number(frame["open_interest"]).astype("float64")
    frame = frame[FUTURES_COLUMNS].sort_values(FUTURES_SORT, kind="mergesort")
    if frame.duplicated(FUTURES_SORT).any():
        raise TaifexParseError("duplicate (trade_date, product, contract_month, session) rows")
    for column in ("product", "contract_month", "session"):
        frame[column] = frame[column].astype(object)
    return frame.reset_index(drop=True)


def parse_fsp_html(body: bytes) -> pd.DataFrame:
    """The final settlement table: one row per expiry (monthly and weekly)."""

    text = body.decode("utf-8", errors="replace")
    start = text.find("table_f")
    if start < 0 or "結算日" not in text[start : start + 2000]:
        raise TaifexParseError("final settlement table not found")
    table = text[start : text.find("</table>", start)]
    cells = [
        re.sub(r"<[^>]+>|\s+", " ", html.unescape(cell)).strip()
        for cell in re.findall(r"<td[^>]*>(.*?)</td>", table, flags=re.S | re.I)
    ]
    if len(cells) % 3:
        raise TaifexParseError(f"final settlement table has {len(cells)} cells, not a multiple of 3")
    rows = [cells[index : index + 3] for index in range(0, len(cells), 3)]
    frame = pd.DataFrame(rows, columns=FSP_COLUMNS)
    if not frame["contract_month"].str.fullmatch(r"\d{6}(W\d)?").all():
        raise TaifexParseError("unexpected contract month in the final settlement table")
    frame["final_settlement_date"] = pd.to_datetime(frame["final_settlement_date"], format="%Y/%m/%d")
    frame["final_settlement_price"] = _number(frame["final_settlement_price"]).astype("float64")
    frame["contract_month"] = frame["contract_month"].astype(object)
    frame = frame.sort_values(["final_settlement_date", "contract_month"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated("contract_month").any():
        raise TaifexParseError("duplicate contract month in the final settlement table")
    return frame


def is_fsp_html(body: bytes) -> bool:
    try:
        parse_fsp_html(body)
    except TaifexParseError:
        return False
    return True


def same_fsp_table(old: bytes, new: bytes) -> bool:
    return parse_fsp_html(old).equals(parse_fsp_html(new))


def third_wednesday(contract_month: str) -> pd.Timestamp:
    first = pd.Timestamp(int(contract_month[:4]), int(contract_month[4:6]), 1)
    return first + pd.Timedelta(days=(2 - first.weekday()) % 7 + 14)


# --- settlement calendar -----------------------------------------------------

CALENDAR_COLUMNS = [
    "last_trading_day",
    "product",
    "contract_month",
    "status",
    "first_trade_date",
    "final_settlement_date",
    "final_settlement_price",
    "last_close",
    "fsp_minus_last_close",
    "fsd_check",
]


def settlement_calendar(quotes: pd.DataFrame, fsp: pd.DataFrame) -> pd.DataFrame:
    """Per product and monthly contract: last trading day and final settlement.

    ``status``:
      ``settled``   - the contract's last day-session row is before the latest
                      trade date of its product; ``last_trading_day`` is that row's date.
      ``active``    - still trading; ``last_trading_day`` is projected as the
                      third Wednesday of the contract month (TAIFEX rule; a
                      holiday on that day moves it to the next trading day,
                      which the projection cannot know yet).

    ``fsd_check`` (settled contracts with a published final settlement):
    True iff the final settlement date equals the last trading day (contracts
    from 2008-12) or the next trading day of the product (earlier contracts).
    """

    day = quotes[(quotes["session"] == "day") & quotes["contract_month"].str.fullmatch(r"\d{6}")]
    fsp_monthly = fsp[fsp["contract_month"].str.fullmatch(r"\d{6}")].set_index("contract_month")
    rows = []
    for product, group in day.groupby("product", sort=True):
        dates = pd.DatetimeIndex(sorted(group["trade_date"].unique()))
        latest = dates[-1]
        close = group.set_index(["contract_month", "trade_date"])["close"]
        for contract, contract_rows in group.groupby("contract_month", sort=True):
            first, last = contract_rows["trade_date"].min(), contract_rows["trade_date"].max()
            settled = last < latest
            ltd = last if settled else max(third_wednesday(contract), latest)
            fsd = fsp_monthly["final_settlement_date"].get(contract, pd.NaT) if settled else pd.NaT
            price = fsp_monthly["final_settlement_price"].get(contract, np.nan) if settled else np.nan
            last_close = close.get((contract, last), np.nan) if settled else np.nan
            check = None
            if settled and pd.notna(fsd):
                position = dates.get_loc(last)
                expected = last if contract >= "200812" else dates[position + 1] if position + 1 < len(dates) else None
                check = bool(expected is not None and fsd == expected)
            rows.append(
                {
                    "last_trading_day": ltd,
                    "product": product,
                    "contract_month": contract,
                    "status": "settled" if settled else "active",
                    "first_trade_date": first,
                    "final_settlement_date": fsd,
                    "final_settlement_price": price,
                    "last_close": last_close,
                    "fsp_minus_last_close": price - last_close,
                    "fsd_check": check,
                }
            )
    frame = pd.DataFrame(rows, columns=CALENDAR_COLUMNS)
    frame["final_settlement_date"] = pd.to_datetime(frame["final_settlement_date"])
    for column in ("final_settlement_price", "last_close", "fsp_minus_last_close"):
        frame[column] = frame[column].astype("float64")
    frame["fsd_check"] = frame["fsd_check"].astype("boolean")
    for column in ("product", "contract_month", "status"):
        frame[column] = frame[column].astype(object)
    return frame.sort_values(["last_trading_day", "product", "contract_month"], kind="mergesort").reset_index(drop=True)


# --- R1 roll table ------------------------------------------------------------

ROLL_COLUMNS = ["front_end", "product", "contract_month", "front_start", "next_contract_month", "status"]


def roll_table_r1(quotes: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """R1 front-month segments: the contract a nearest-expiry continuous series shows.

    A contract is front from the first trading day after the previous front
    contract's last trading day (or its own first listed day) through its own
    last trading day (``front_end``), inclusive. The next contract is front
    from the following trading day. Contracts that never were the nearest
    listed one are skipped.
    """

    day = quotes[(quotes["session"] == "day") & quotes["contract_month"].str.fullmatch(r"\d{6}")]
    rows = []
    for product, cal in calendar.groupby("product", sort=True):
        dates = pd.DatetimeIndex(sorted(day.loc[day["product"] == product, "trade_date"].unique()))
        cal = cal.sort_values("last_trading_day", kind="mergesort").reset_index(drop=True)
        segments = []
        boundary = None  # last trading day of the previous front contract
        for record in cal.itertuples(index=False):
            later = dates if boundary is None else dates[dates > boundary]
            # beyond the data (projected contracts) the front starts the next business day
            start = later[0] if len(later) else boundary + pd.offsets.BDay(1)
            start = max(start, record.first_trade_date)
            if start > record.last_trading_day:
                continue
            segments.append([record.last_trading_day, product, record.contract_month, start, record.status])
            boundary = record.last_trading_day
        for index, segment in enumerate(segments):
            following = segments[index + 1][2] if index + 1 < len(segments) else None
            front_end, _, contract, front_start, status = segment
            rows.append(
                {
                    "front_end": front_end,
                    "product": product,
                    "contract_month": contract,
                    "front_start": front_start,
                    "next_contract_month": following,
                    "status": status,
                }
            )
    frame = pd.DataFrame(rows, columns=ROLL_COLUMNS)
    for column in ("product", "contract_month", "next_contract_month", "status"):
        frame[column] = frame[column].astype(object)
    return frame.sort_values(["front_end", "product"], kind="mergesort").reset_index(drop=True)
