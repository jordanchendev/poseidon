"""TWSE sources: TAIEX price index daily OHLC and the TAIEX total-return index.

Both endpoints answer one calendar month per GET as JSON with ROC dates
(``115/10/23`` = 2026-10-23) and comma-grouped numbers.

``MI_5MINS_HIST`` (發行量加權股價指數歷史資料): open/high/low/close, from 1999-01.
``MFI94U`` (發行量加權股價報酬指數): one daily value, from 2003-01-02.

A month without data answers ``stat != "OK"`` (parsed as no rows); a body
that is not JSON is what a throttled client sees and is rejected.
"""

from __future__ import annotations

import json

import pandas as pd

TAIEX_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
TOTAL_RETURN_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U"
FIRST_MONTH = {"MI_5MINS_HIST": (1999, 1), "MFI94U": (2003, 1)}
TAIEX_COLUMNS = ["trade_date", "open", "high", "low", "close"]
TOTAL_RETURN_COLUMNS = ["trade_date", "close"]


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
