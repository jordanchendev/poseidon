"""Strategy CRUD API endpoints.

Provides REST endpoints for managing trading strategies (API-01).
Strategies are persisted in the database and can be activated/deactivated
to control which strategies participate in signal evaluation.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel as PydanticBase
from pydantic import ConfigDict
from sqlalchemy.orm import Session

from poseidon.models.base import get_db
from poseidon.models.strategy import StrategyRecord

router = APIRouter()

_VALID_STRATEGY_TYPES = {"model", "rule"}


# --------------- Pydantic schemas ---------------


class StrategyCreate(PydanticBase):
    """Request body for creating a new strategy."""

    name: str
    strategy_type: str
    config: dict = {}
    symbol: str
    market: str
    interval: str = "1d"
    model_version_id: uuid.UUID | None = None


class StrategyUpdate(PydanticBase):
    """Request body for updating an existing strategy.

    Only provided (non-None) fields are applied.
    """

    name: str | None = None
    config: dict | None = None
    symbol: str | None = None
    market: str | None = None
    interval: str | None = None
    model_version_id: uuid.UUID | None = None


class StrategyResponse(PydanticBase):
    """Response model for a strategy."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    strategy_type: str
    config: dict
    symbol: str
    market: str
    interval: str
    active: bool
    model_version_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


# --------------- Endpoints ---------------


@router.get("", response_model=list[StrategyResponse])
async def list_strategies(
    db: Session = Depends(get_db),
) -> list[StrategyResponse]:
    """List all strategies ordered by creation time."""
    strategies = (
        db.query(StrategyRecord)
        .order_by(StrategyRecord.created_at)
        .all()
    )
    return strategies


@router.post("", response_model=StrategyResponse, status_code=201)
async def create_strategy(
    body: StrategyCreate,
    db: Session = Depends(get_db),
) -> StrategyResponse:
    """Create a new strategy.

    Validates that ``strategy_type`` is 'model' or 'rule'
    and that ``name`` is unique.
    """
    if body.strategy_type not in _VALID_STRATEGY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy_type '{body.strategy_type}'. "
            f"Must be one of: {sorted(_VALID_STRATEGY_TYPES)}",
        )

    existing = (
        db.query(StrategyRecord)
        .filter(StrategyRecord.name == body.name)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Strategy with name '{body.name}' already exists",
        )

    record = StrategyRecord(
        id=uuid.uuid4(),
        name=body.name,
        strategy_type=body.strategy_type,
        config=body.config,
        symbol=body.symbol,
        market=body.market,
        interval=body.interval,
        model_version_id=body.model_version_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> StrategyResponse:
    """Get a single strategy by ID."""
    record = db.get(StrategyRecord, strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return record


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: uuid.UUID,
    body: StrategyUpdate,
    db: Session = Depends(get_db),
) -> StrategyResponse:
    """Update a strategy. Only non-None fields are applied."""
    record = db.get(StrategyRecord, strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail="Strategy not found")

    if body.name is not None:
        # Check uniqueness if name is changing
        if body.name != record.name:
            dup = (
                db.query(StrategyRecord)
                .filter(StrategyRecord.name == body.name)
                .first()
            )
            if dup:
                raise HTTPException(
                    status_code=409,
                    detail=f"Strategy with name '{body.name}' already exists",
                )
        record.name = body.name
    if body.config is not None:
        record.config = body.config
    if body.symbol is not None:
        record.symbol = body.symbol
    if body.market is not None:
        record.market = body.market
    if body.interval is not None:
        record.interval = body.interval
    if body.model_version_id is not None:
        record.model_version_id = body.model_version_id

    db.commit()
    db.refresh(record)
    return record


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(
    strategy_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Delete a strategy."""
    record = db.get(StrategyRecord, strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail="Strategy not found")
    db.delete(record)
    db.commit()


@router.post("/{strategy_id}/activate", response_model=StrategyResponse)
async def activate_strategy(
    strategy_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> StrategyResponse:
    """Activate a strategy (set active=True)."""
    record = db.get(StrategyRecord, strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail="Strategy not found")
    record.active = True
    db.commit()
    db.refresh(record)
    return record


@router.post("/{strategy_id}/deactivate", response_model=StrategyResponse)
async def deactivate_strategy(
    strategy_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> StrategyResponse:
    """Deactivate a strategy (set active=False)."""
    record = db.get(StrategyRecord, strategy_id)
    if not record:
        raise HTTPException(status_code=404, detail="Strategy not found")
    record.active = False
    db.commit()
    db.refresh(record)
    return record
