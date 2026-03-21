"""Signal list/detail API endpoints.

Provides read-only REST endpoints for querying trading signals (API-01).
Signals are created by the evaluation pipeline; this router exposes
them for monitoring and review.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel as PydanticBase
from pydantic import ConfigDict
from sqlalchemy.orm import Session

from poseidon.models.base import get_db
from poseidon.models.signal import SignalRecord

router = APIRouter()


# --------------- Pydantic schemas ---------------


class SignalResponse(PydanticBase):
    """Response model for a signal record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    strategy_id: uuid.UUID | None
    model_id: uuid.UUID | None
    symbol: str
    market: str
    instrument: str
    action: str
    confidence: float
    quantity_pct: float | None
    signal_time: datetime
    valid_until: datetime | None
    interval: str
    params: dict
    status: str
    reject_reason: str | None
    created_at: datetime


# --------------- Endpoints ---------------


@router.get("", response_model=list[SignalResponse])
async def list_signals(
    market: str | None = None,
    symbol: str | None = None,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[SignalResponse]:
    """List signals with optional filtering by market, symbol, and status.

    Results are ordered by creation time descending (newest first).
    """
    query = db.query(SignalRecord)

    if market is not None:
        query = query.filter(SignalRecord.market == market)
    if symbol is not None:
        query = query.filter(SignalRecord.symbol == symbol)
    if status is not None:
        query = query.filter(SignalRecord.status == status)

    signals = query.order_by(SignalRecord.created_at.desc()).limit(limit).all()
    return signals


@router.get("/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> SignalResponse:
    """Get a single signal by ID."""
    record = db.get(SignalRecord, signal_id)
    if not record:
        raise HTTPException(status_code=404, detail="Signal not found")
    return record
