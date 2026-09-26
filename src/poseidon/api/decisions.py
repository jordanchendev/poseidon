"""Scoped HTTP routes for frozen portfolio decisions."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionNotFoundError, DecisionService
from poseidon.decision_loop.manifest import ValidationError

router = APIRouter()
PositiveRevision = Annotated[int, Field(strict=True, ge=1)]
DecisionAction = Literal["watch", "hold", "enter", "add", "reduce", "exit", "reject"]


class DecisionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_run_id: uuid.UUID
    account_scope: Annotated[str, Field(min_length=1)]
    original_json: dict
    portfolio_snapshot_json: dict
    risk_snapshot_json: dict


class DecisionApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: PositiveRevision
    final_action: DecisionAction | None = None
    override_reason: str | None = None


class DecisionRejectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: PositiveRevision
    reason: Annotated[str, Field(min_length=1)]

    @field_validator("reason")
    @classmethod
    def require_nonblank_reason(cls, value):
        if not value.strip():
            raise ValueError("reason must be non-empty")
        return value


def _decision_response(record):
    return {
        "id": record.id,
        "evaluation_run_id": record.evaluation_run_id,
        "strategy_version_id": record.strategy_version_id,
        "account_scope": record.account_scope,
        "decision_as_of": record.decision_as_of,
        "valid_until": record.valid_until,
        "status": record.status,
        "revision": record.revision,
        "creation_sha256": record.creation_sha256,
        "policy_sha256": record.policy_sha256,
        "original_json": record.original_json,
        "final_json": record.final_json,
        "portfolio_snapshot_json": record.portfolio_snapshot_json,
        "risk_snapshot_json": record.risk_snapshot_json,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _raise_api_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, DecisionNotFoundError):
        raise HTTPException(status_code=404, detail="Decision not found") from error
    if isinstance(error, DecisionConflictError):
        raise HTTPException(status_code=409, detail="Decision conflict") from error
    if isinstance(error, (ValidationError, PydanticValidationError)):
        raise HTTPException(status_code=422, detail="Invalid decision input") from error
    raise error


@router.post("/")
def create_decision(
    body: DecisionCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        record = DecisionService(db).create_decision(
            body.evaluation_run_id,
            principal=principal,
            account_scope=body.account_scope,
            original_json=body.original_json,
            portfolio_snapshot_json=body.portfolio_snapshot_json,
            risk_snapshot_json=body.risk_snapshot_json,
        )
        db.commit()
        db.refresh(record)
        return _decision_response(record)
    except Exception as error:
        db.rollback()
        _raise_api_error(error)


@router.get("/")
def list_decisions(
    status: Literal["pending_approval"] = Query(...),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        records = DecisionService(db).list_pending(principal=principal)
        return [_decision_response(record) for record in records]
    except Exception as error:
        _raise_api_error(error)


@router.get("/{decision_id}")
def get_decision(
    decision_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        record = DecisionService(db).get(decision_id, principal=principal)
        return _decision_response(record)
    except Exception as error:
        _raise_api_error(error)


@router.post("/{decision_id}/approve")
def approve_decision(
    decision_id: uuid.UUID,
    body: DecisionApprovalRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        response = DecisionService(db).approve(
            decision_id,
            body.model_dump(mode="json"),
            principal=principal,
            idempotency_key=idempotency_key,
        )
        db.commit()
        return response
    except Exception as error:
        db.rollback()
        _raise_api_error(error)


@router.post("/{decision_id}/reject")
def reject_decision(
    decision_id: uuid.UUID,
    body: DecisionRejectionRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1)],
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        response = DecisionService(db).reject(
            decision_id,
            body.model_dump(mode="json"),
            principal=principal,
            idempotency_key=idempotency_key,
        )
        db.commit()
        return response
    except Exception as error:
        db.rollback()
        _raise_api_error(error)


@router.get("/{decision_id}/trace")
def get_decision_trace(
    decision_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        return DecisionService(db).trace(decision_id, principal=principal)
    except Exception as error:
        _raise_api_error(error)
