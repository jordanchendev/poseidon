"""Data management API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from poseidon.core.schemas import (
    BackfillRequest,
    BackfillStatusResponse,
    FetchRequest,
    MessageResponse,
)
from poseidon.models.backfill import BackfillProgress
from poseidon.models.base import get_db
from poseidon.workers.cpu_tasks import fetch_market_data, trigger_backfill

router = APIRouter()


@router.post("/fetch", response_model=MessageResponse, status_code=202)
async def trigger_fetch(request: FetchRequest):
    """Trigger data fetch for a market.

    Dispatches a Celery task to fetch latest data for all symbols in the market.
    """
    task = fetch_market_data.delay(request.market, request.interval)
    return MessageResponse(message=f"Fetch task dispatched for {request.market}/{request.interval}", task_id=task.id)


@router.post("/backfill", response_model=MessageResponse, status_code=202)
async def trigger_backfill_endpoint(request: BackfillRequest):
    """Trigger historical data backfill.

    If market is specified, backfills only that market.
    If both market and symbol are None, backfills all configured symbols.
    """
    task = trigger_backfill.delay(request.market)
    market_label = request.market or "all markets"
    return MessageResponse(message=f"Backfill task dispatched for {market_label}", task_id=task.id)


@router.get("/backfill/status", response_model=list[BackfillStatusResponse])
async def get_backfill_status(
    market: str | None = None,
    db: Session = Depends(get_db),
):
    """Get backfill progress status for all symbols or a specific market."""
    query = db.query(BackfillProgress)
    if market:
        query = query.filter(BackfillProgress.market == market)
    rows = query.order_by(BackfillProgress.market, BackfillProgress.symbol).all()
    return rows
