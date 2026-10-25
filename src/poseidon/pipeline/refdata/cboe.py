"""CBOE source: the VIX daily history (one CSV, 1990-01-02 on).

The old host cdn.cboe.com answers 307 to cdn-api.cboe.com and the fetcher does not follow redirects, so the
final host is used.

``VIX_History.csv`` is the whole history in one public file (``DATE,OPEN,HIGH,LOW,CLOSE``, dates
``MM/DD/YYYY`` in US Eastern, one row per US trading day). Chosen over Thalassa's ``macro_vix`` (checked
2026-10-26): Thalassa starts 2024-01-02 (741 values), forward-fills US holidays and weekends (14 phantom
rows), and differs from CBOE by up to 2.98 points on 20 of 727 shared days; CBOE has 9,301 rows from
1990-01-02 to 2026-10-23, no gaps but the 2001-09-11 closure, strictly increasing dates, no nulls.

``trade_date`` is the US date. Aligning it with a Taiwan decision is the consumer's job
(``v3.instruments.align_signal_frame``: a row is readable from the next calendar day on).
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd

VIX_URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
VIX_COLUMNS = ["trade_date", "open", "high", "low", "close"]
HEADER = b"DATE,OPEN,HIGH,LOW,CLOSE"
MIN_ROWS = 5000  # a truncated or error body is rejected rather than stored as the history
# The VIX of US date u is final by about 16:15 US Eastern, 21:15 UTC at the latest; one hour of margin.
FINAL_AFTER = timedelta(hours=22)


class CboeParseError(ValueError):
    """The CBOE response is not the VIX history."""


def is_vix_csv(body: bytes) -> bool:
    return body.lstrip(b"\xef\xbb\xbf").startswith(HEADER) and body.count(b"\n") >= MIN_ROWS


def last_final_date(now: datetime):
    """Latest US date whose VIX close is final at ``now`` (timezone-aware)."""

    return (now - FINAL_AFTER).date()


def parse_vix(body: bytes, *, final_through=None) -> pd.DataFrame:
    """The history as ``trade_date, open, high, low, close``; rows after ``final_through`` (a date) are dropped."""

    if not is_vix_csv(body):
        raise CboeParseError(f"not the CBOE VIX history: {body[:80]!r}")
    frame = pd.read_csv(io.BytesIO(body), encoding="utf-8-sig")
    frame.columns = VIX_COLUMNS
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%m/%d/%Y").astype("datetime64[ns]")
    for column in VIX_COLUMNS[1:]:
        frame[column] = frame[column].astype(float)
    if frame[VIX_COLUMNS].isna().any().any() or (frame[VIX_COLUMNS[1:]] <= 0).any().any():
        raise CboeParseError("VIX history holds a null or non-positive value")
    if frame["trade_date"].duplicated().any():
        raise CboeParseError("duplicate trade dates in the VIX history")
    frame = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if final_through is not None:
        frame = frame[frame["trade_date"] <= pd.Timestamp(final_through)].reset_index(drop=True)
    return frame
