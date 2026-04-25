"""Protection locks API -- visibility and manual override endpoints (Phase 36).

Endpoints:
    GET  /locks          Active protection locks with optional filters (D-18)
    GET  /history        Expired/released locks with pagination (D-19)
    POST /locks/{id}/override   Manual lock release (D-20)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy.orm import Session

from poseidon.models.base import get_db
from poseidon.models.protection_lock import ProtectionLockRecord

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Pydantic response schemas ---


class ProtectionLockResponse(PydanticBaseModel):
    """Single protection lock record."""

    id: int
    protection_type: str
    symbol: str | None
    market: str
    reason: str
    locked_at: datetime
    expires_at: datetime | None
    active: bool
    released_at: datetime | None
    released_by: str | None

    model_config = {"from_attributes": True}


class ProtectionLocksListResponse(PydanticBaseModel):
    """List of protection locks with total count."""

    locks: list[ProtectionLockResponse]
    total: int


class ProtectionOverrideResponse(PydanticBaseModel):
    """Result of a manual lock override."""

    success: bool
    message: str
    lock_id: int


# --- Endpoints ---


@router.get("/locks", response_model=ProtectionLocksListResponse)
def get_active_locks(
    symbol: str | None = Query(None, description="Filter by symbol"),
    market: str | None = Query(None, description="Filter by market"),
    type: str | None = Query(None, alias="type", description="Filter by protection type"),
    db: Session = Depends(get_db),
) -> ProtectionLocksListResponse:
    """Return all active protection locks with optional filters (D-18)."""
    query = db.query(ProtectionLockRecord).filter(
        ProtectionLockRecord.active.is_(True),
    )

    if symbol is not None:
        query = query.filter(ProtectionLockRecord.symbol == symbol)
    if market is not None:
        query = query.filter(ProtectionLockRecord.market == market)
    if type is not None:
        query = query.filter(ProtectionLockRecord.protection_type == type)

    query = query.order_by(ProtectionLockRecord.locked_at.desc())
    locks = query.all()

    return ProtectionLocksListResponse(
        locks=[ProtectionLockResponse.model_validate(lock) for lock in locks],
        total=len(locks),
    )


@router.get("/history", response_model=ProtectionLocksListResponse)
def get_lock_history(
    symbol: str | None = Query(None, description="Filter by symbol"),
    market: str | None = Query(None, description="Filter by market"),
    type: str | None = Query(None, alias="type", description="Filter by protection type"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> ProtectionLocksListResponse:
    """Return expired/released protection locks with pagination (D-19)."""
    query = db.query(ProtectionLockRecord).filter(
        ProtectionLockRecord.active.is_(False),
    )

    if symbol is not None:
        query = query.filter(ProtectionLockRecord.symbol == symbol)
    if market is not None:
        query = query.filter(ProtectionLockRecord.market == market)
    if type is not None:
        query = query.filter(ProtectionLockRecord.protection_type == type)

    total = query.count()
    locks = query.order_by(ProtectionLockRecord.released_at.desc()).offset(offset).limit(limit).all()

    return ProtectionLocksListResponse(
        locks=[ProtectionLockResponse.model_validate(lock) for lock in locks],
        total=total,
    )


@router.post("/locks/{lock_id}/override", response_model=ProtectionOverrideResponse)
def override_lock(
    lock_id: int,
    db: Session = Depends(get_db),
) -> ProtectionOverrideResponse:
    """Manually release an active protection lock (D-20)."""
    lock = (
        db.query(ProtectionLockRecord)
        .filter(
            ProtectionLockRecord.id == lock_id,
        )
        .first()
    )

    if lock is None:
        raise HTTPException(status_code=404, detail="Lock not found")

    if not lock.active:
        raise HTTPException(status_code=409, detail="Lock already released")

    lock.active = False
    lock.released_at = datetime.now(UTC)
    lock.released_by = "manual"
    db.commit()

    logger.info(
        "Protection lock manually released: id=%d type=%s symbol=%s market=%s",
        lock.id,
        lock.protection_type,
        lock.symbol,
        lock.market,
    )

    return ProtectionOverrideResponse(
        success=True,
        message="Lock released manually",
        lock_id=lock_id,
    )
