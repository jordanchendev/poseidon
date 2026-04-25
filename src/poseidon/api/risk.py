"""Risk rule CRUD API endpoints.

Provides REST endpoints for managing risk rules at runtime (RISK-03).
Changes take effect on the next signal evaluation cycle because
RiskEngine.load_rules_from_db() is called per-evaluation in the
SignalPipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel as PydanticBase
from pydantic import ConfigDict
from sqlalchemy.orm import Session

from poseidon.models.base import get_db
from poseidon.models.risk_rule import RiskRuleRecord
from poseidon.risk.rules import RULE_REGISTRY

router = APIRouter()


# --------------- Pydantic schemas ---------------


class RiskRuleCreate(PydanticBase):
    """Request body for creating a new risk rule."""

    rule_type: str
    name: str
    enabled: bool = True
    priority: int = 0
    params: dict = {}


class RiskRuleUpdate(PydanticBase):
    """Request body for updating an existing risk rule.

    Only provided (non-None) fields are applied.
    """

    enabled: bool | None = None
    priority: int | None = None
    params: dict | None = None


class RiskRuleResponse(PydanticBase):
    """Response model for a risk rule."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    rule_type: str
    name: str
    enabled: bool
    priority: int
    params: dict
    created_at: datetime
    updated_at: datetime


class VirtualPositionResponse(PydanticBase):
    """Response model for a virtual portfolio position."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    symbol: str
    market: str
    instrument: str
    side: str
    quantity_pct: float
    entry_time: datetime


# --------------- Endpoints ---------------


@router.get("/types", response_model=list[str])
async def list_rule_types() -> list[str]:
    """Return available risk rule types from the registry."""
    return sorted(RULE_REGISTRY.keys())


@router.get("", response_model=list[RiskRuleResponse])
async def list_rules(db: Session = Depends(get_db)) -> list[RiskRuleResponse]:
    """List all risk rules ordered by priority."""
    rules = db.query(RiskRuleRecord).order_by(RiskRuleRecord.priority).all()
    return rules


@router.post("", response_model=RiskRuleResponse, status_code=201)
async def create_rule(
    body: RiskRuleCreate,
    db: Session = Depends(get_db),
) -> RiskRuleResponse:
    """Create a new risk rule.

    Validates that ``rule_type`` is a known type in RULE_REGISTRY
    and that ``name`` is unique.
    """
    if body.rule_type not in RULE_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown rule_type '{body.rule_type}'. Available: {sorted(RULE_REGISTRY.keys())}",
        )

    existing = db.query(RiskRuleRecord).filter(RiskRuleRecord.name == body.name).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Rule with name '{body.name}' already exists",
        )

    record = RiskRuleRecord(
        id=uuid.uuid4(),
        rule_type=body.rule_type,
        name=body.name,
        enabled=body.enabled,
        priority=body.priority,
        params=body.params,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/portfolio", response_model=list[VirtualPositionResponse])
async def get_portfolio(
    market: str | None = None,
    db: Session = Depends(get_db),
) -> list[VirtualPositionResponse]:
    """Get current virtual portfolio positions from DB.

    Returns open (non-closed) positions, optionally filtered by market.
    """
    from poseidon.models.virtual_position import VirtualPositionRecord

    query = db.query(VirtualPositionRecord).filter(
        VirtualPositionRecord.closed == False  # noqa: E712
    )
    if market:
        query = query.filter(VirtualPositionRecord.market == market)
    return query.all()


@router.get("/{rule_id}", response_model=RiskRuleResponse)
async def get_rule(
    rule_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> RiskRuleResponse:
    """Get a single risk rule by ID."""
    record = db.get(RiskRuleRecord, rule_id)
    if not record:
        raise HTTPException(status_code=404, detail="Rule not found")
    return record


@router.patch("/{rule_id}", response_model=RiskRuleResponse)
async def update_rule(
    rule_id: uuid.UUID,
    body: RiskRuleUpdate,
    db: Session = Depends(get_db),
) -> RiskRuleResponse:
    """Update a risk rule's enabled status, priority, or params.

    Only fields that are not None are updated.
    """
    record = db.get(RiskRuleRecord, rule_id)
    if not record:
        raise HTTPException(status_code=404, detail="Rule not found")

    if body.enabled is not None:
        record.enabled = body.enabled
    if body.priority is not None:
        record.priority = body.priority
    if body.params is not None:
        record.params = body.params

    db.commit()
    db.refresh(record)
    return record


@router.delete("/{rule_id}", status_code=204)
async def delete_rule(
    rule_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """Delete a risk rule."""
    record = db.get(RiskRuleRecord, rule_id)
    if not record:
        raise HTTPException(status_code=404, detail="Rule not found")
    db.delete(record)
    db.commit()
