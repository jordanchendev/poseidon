"""Data management API endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel as PydanticBase
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    BackfillRequest,
    BackfillStatusResponse,
    FetchRequest,
    MessageResponse,
)
from poseidon.models.backfill import BackfillJob  # noqa: F401
from poseidon.models.base import get_db
from poseidon.workers.cpu_tasks import fetch_market_data, trigger_backfill

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


# --- Funding rate response schemas ---


class FundingRatePoint(PydanticBase):
    time: str
    symbol: str
    funding_rate: float
    mark_price: float | None
    index_price: float | None


class FundingRateResponse(PydanticBase):
    data: list[FundingRatePoint]
    count: int


@router.post("/fetch", response_model=MessageResponse, status_code=202)
async def trigger_fetch(request: FetchRequest):
    """Trigger data fetch for a market.

    Dispatches a Celery task to fetch latest data for all symbols in the market.
    """
    task = fetch_market_data.delay(request.market, request.interval, request.symbol)
    label = f"{request.market}/{request.interval}"
    if request.symbol:
        label += f" (symbol={request.symbol})"
    return MessageResponse(message=f"Fetch task dispatched for {label}", task_id=task.id)


@router.post("/backfill", response_model=MessageResponse, status_code=202)
async def trigger_backfill_endpoint(request: BackfillRequest):
    """Trigger historical data backfill.

    If market is specified, backfills only that market.
    If both market and symbol are None, backfills all configured symbols.
    """
    task = trigger_backfill.delay(request.market, request.symbol)
    market_label = request.market or "all markets"
    if request.symbol:
        market_label += f" (symbol={request.symbol})"
    return MessageResponse(message=f"Backfill task dispatched for {market_label}", task_id=task.id)


@router.get("/backfill/status", response_model=list[BackfillStatusResponse])
async def get_backfill_status(
    market: str | None = None,
    db: Session = Depends(get_db),
):
    """Get backfill job status for all symbols or a specific market.

    Phase 38 D-10: backfill_progress replaced by backfill_jobs. Response model
    fields not yet wired up — returns empty list until plan 38-03 rewires
    the endpoint to the new BackfillJob schema.
    """
    del market, db  # noqa: F841 — will be used after 38-03 rewrite
    return []


# --- GET /ohlcv: OHLCV candlestick data (API-01) ---


@router.get("/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str = Query(..., description="Ticker symbol, e.g. BTCUSDT"),
    market: str = Query(..., description="Market name, e.g. crypto_perp"),
    interval: str = Query("1d", description="Candle interval, e.g. 1d, 4h, 1h"),
    start: datetime | None = Query(None, description="Start date (ISO format)"),
    end: datetime | None = Query(None, description="End date (ISO format)"),
    db: Session = Depends(get_db),
) -> OHLCVResponse:
    """Return OHLCV candlestick data for a given symbol/market/interval/date range."""
    from poseidon.data.storage import read_ohlcv

    df = read_ohlcv(db, symbol, market, interval, start=start, end=end)
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


# --- GET /funding-rates: Funding rate history (API-06) ---


@router.get("/funding-rates", response_model=FundingRateResponse)
async def get_funding_rates(
    symbol: str = Query(..., description="Perp symbol, e.g. BTC/USDT:USDT"),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> FundingRateResponse:
    """Return funding rate history for a perpetual contract symbol."""
    from poseidon.models.funding_rate import FundingRateRecord

    query = db.query(FundingRateRecord).filter(FundingRateRecord.symbol == symbol)
    if start is not None:
        query = query.filter(FundingRateRecord.time >= start)
    if end is not None:
        query = query.filter(FundingRateRecord.time <= end)
    rows = query.order_by(FundingRateRecord.time.desc()).limit(limit).all()
    data = [
        FundingRatePoint(
            time=r.time.isoformat() if hasattr(r.time, "isoformat") else str(r.time),
            symbol=r.symbol,
            funding_rate=r.funding_rate,
            mark_price=r.mark_price,
            index_price=r.index_price,
        )
        for r in rows
    ]
    return FundingRateResponse(data=data, count=len(data))
