"""TWSE sources: TAIEX price index daily OHLC and the TAIEX total-return index.

Both endpoints answer one calendar month per GET as JSON with ROC dates
(``115/10/23`` = 2026-10-23) and comma-grouped numbers.

``MI_5MINS_HIST`` (發行量加權股價指數歷史資料): open/high/low/close, from 1999-01.
``MFI94U`` (發行量加權股價報酬指數): one daily value, from 2003-01-02.

A month without data answers ``stat != "OK"`` (parsed as no rows); a body
that is not JSON is what a throttled client sees and is rejected.

``holidaySchedule`` (市場開休市日期): one GET per calendar year, the official
schedule of market holidays TWSE publishes ahead of the year (the 2027
schedule was still empty on 2026-10-24; years before 2021 answer empty
too, checked the same day). TAIFEX index futures trade on TWSE trading days (TX contract spec:
本契約交易日同臺灣證券交易所交易日), so this is the official calendar the
futures last trading days are scheduled on. Rows named ``...開始交易日`` /
``...最後交易日`` mark the first/last trading day around a break and are
trading days; every other listed date is closed (holidays, and the
``市場無交易，僅辦理結算交割作業`` days before Lunar New Year). Unscheduled
closures (typhoons, e.g. 2026-10-23) never appear here.
"""

from __future__ import annotations

import json

import pandas as pd

TAIEX_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
TOTAL_RETURN_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U"
FIRST_MONTH = {"MI_5MINS_HIST": (1999, 1), "MFI94U": (2003, 1)}
TAIEX_COLUMNS = ["trade_date", "open", "high", "low", "close"]
TOTAL_RETURN_COLUMNS = ["trade_date", "close"]
HOLIDAY_URL = "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule"
HOLIDAY_FIRST_YEAR = 2021
HOLIDAY_COLUMNS = ["date", "name", "note", "closed"]
_OPEN_MARKERS = ("開始交易", "最後交易")


class TwseParseError(ValueError):
    """A TWSE response is not in the expected format."""


def month_url(base: str, year: int, month: int) -> tuple[str, dict]:
    params = {"date": f"{year}{month:02d}01", "response": "json"}
    return f"{base}?date={params['date']}&response=json", params


def is_twse_json(body: bytes) -> bool:
    try:
        return isinstance(json.loads(body), dict) and "stat" in json.loads(body)
    except ValueError:
        return False


def _roc_date(text: str) -> pd.Timestamp:
    year, month, day = (int(part) for part in text.strip().split("/"))
    return pd.Timestamp(year + 1911, month, day)


def _parse(body: bytes, columns: list[str]) -> pd.DataFrame:
    try:
        payload = json.loads(body)
    except ValueError as error:
        raise TwseParseError(f"not JSON: {body[:80]!r}") from error
    if payload.get("stat") != "OK":
        return pd.DataFrame(
            {name: pd.Series(dtype="datetime64[ns]" if name == "trade_date" else "float64") for name in columns}
        )
    values = len(columns) - 1
    rows = []
    for record in payload.get("data", []):
        if len(record) < 1 + values:
            raise TwseParseError(f"short TWSE row {record!r}")
        rows.append([_roc_date(record[0])] + [float(str(cell).replace(",", "")) for cell in record[1 : 1 + values]])
    frame = pd.DataFrame(rows, columns=columns)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if frame["trade_date"].duplicated().any():
        raise TwseParseError("duplicate trade dates")
    return frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def parse_taiex(body: bytes) -> pd.DataFrame:
    return _parse(body, TAIEX_COLUMNS)


def parse_total_return(body: bytes) -> pd.DataFrame:
    return _parse(body, TOTAL_RETURN_COLUMNS)


def holiday_url(year: int) -> tuple[str, dict]:
    params = {"date": f"{year}0101", "response": "json"}
    return f"{HOLIDAY_URL}?date={params['date']}&response=json", params


def is_holiday_json(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return isinstance(payload, dict) and str(payload.get("stat", "")).lower() == "ok" and "data" in payload


def parse_holidays(body: bytes) -> pd.DataFrame:
    """One row per listed date; ``closed`` is False for the first/last-trading-day markers."""

    if not is_holiday_json(body):
        raise TwseParseError(f"not a TWSE holiday schedule: {body[:80]!r}")
    rows = []
    for record in json.loads(body)["data"]:
        if len(record) < 2:
            raise TwseParseError(f"short holiday row {record!r}")
        name = str(record[1]).strip()
        note = str(record[2]).strip() if len(record) > 2 else ""
        rows.append([pd.Timestamp(str(record[0]).strip()), name, note, not any(m in name for m in _OPEN_MARKERS)])
    frame = pd.DataFrame(rows, columns=HOLIDAY_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    for column in ("name", "note"):
        frame[column] = frame[column].astype(object)
    frame["closed"] = frame["closed"].astype(bool)
    if frame["date"].duplicated().any():
        raise TwseParseError("duplicate dates in the holiday schedule")
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)
