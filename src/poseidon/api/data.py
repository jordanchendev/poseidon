"""Data management API endpoints.

Phase 61: Trimmed to OHLCV-only. All fetch/backfill/coverage/gaps/freshness/funding
endpoints removed -- those responsibilities moved to Thalassa.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query
from pydantic import BaseModel as PydanticBase

from poseidon.data.remote_repository import RemoteDataRepository

router = APIRouter()


# --- OHLCV response schemas ---


class OHLCVPoint(PydanticBase):
    time: str  # ISO format string
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVResponse(PydanticBase):
    data: list[OHLCVPoint]
    symbol: str
    market: str
    interval: str
    count: int


# --- GET /ohlcv: OHLCV candlestick data (API-01) ---


@router.get("/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str = Query(..., description="Ticker symbol, e.g. BTCUSDT"),
    market: str = Query(..., description="Market name, e.g. crypto_perp"),
    interval: str = Query("1d", description="Candle interval, e.g. 1d, 4h, 1h"),
    start: datetime | None = Query(None, description="Start date (ISO format)"),
    end: datetime | None = Query(None, description="End date (ISO format)"),
) -> OHLCVResponse:
    """Return OHLCV candlestick data for a given symbol/market/interval/date range."""
    repo = RemoteDataRepository.from_settings()
    df = repo.read_ohlcv(symbol, market, interval, start=start, end=end)
    if df.empty:
        return OHLCVResponse(data=[], symbol=symbol, market=market, interval=interval, count=0)
    records = df.reset_index()
    data = [
        OHLCVPoint(
            time=row["time"].isoformat() if hasattr(row["time"], "isoformat") else str(row["time"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for _, row in records.iterrows()
    ]
    return OHLCVResponse(data=data, symbol=symbol, market=market, interval=interval, count=len(data))
