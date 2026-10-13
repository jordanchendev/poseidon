"""Scoped HTTP routes for frozen portfolio decisions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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

    @model_validator(mode="after")
    def require_reason_for_modified_action(self):
        if self.final_action is not None and not (self.override_reason or "").strip():
            raise ValueError("override_reason must be non-empty when final_action is supplied")
        return self


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


class DecisionAPIError(BaseModel):
    detail: str | dict
    error_code: str
    status_code: int


DECISION_ERROR_RESPONSES = {
    status: {"model": DecisionAPIError}
    for status in (403, 404, 409, 422)
}


class DecisionCapabilityMetadata(BaseModel):
    can_approve: bool
    can_modify: bool
    can_reject: bool


class DecisionResponse(BaseModel):
    id: uuid.UUID
    evaluation_run_id: uuid.UUID
    strategy_version_id: uuid.UUID
    account_scope: str
    decision_as_of: datetime
    valid_until: datetime
    status: str
    revision: int
    creation_sha256: str
    policy_sha256: str
    original_json: dict
    final_json: dict
    original_action: str | None
    final_action: str | None
    portfolio_snapshot_json: dict
    risk_snapshot_json: dict
    execution_status: str
    reconciliation_status: str
    event_history: list[dict]
    action_metadata: DecisionCapabilityMetadata
    created_at: datetime
    updated_at: datetime


class DecisionMutationResponse(BaseModel):
    id: uuid.UUID
    status: str
    revision: int
    final_json: dict


def _decision_response(record):
    original = record.original_json if isinstance(record.original_json, dict) else {}
    final = record.final_json if isinstance(record.final_json, dict) else {}
    can_transition = record.status == "pending_approval"
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
        "original_action": original.get("final_action") or original.get("action"),
        "final_action": final.get("final_action") or final.get("action"),
        "execution_status": getattr(record, "execution_status", record.status),
        "reconciliation_status": getattr(record, "reconciliation_status", "not_required"),
        "event_history": list(getattr(record, "event_history", [])),
        "action_metadata": {
            "can_approve": can_transition,
            "can_modify": can_transition,
            "can_reject": can_transition,
        },
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


@router.post("/", response_model=DecisionResponse, responses=DECISION_ERROR_RESPONSES)
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


@router.get("/", response_model=list[DecisionResponse], responses=DECISION_ERROR_RESPONSES)
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


@router.get("/{decision_id}", response_model=DecisionResponse, responses=DECISION_ERROR_RESPONSES)
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


@router.post(
    "/{decision_id}/approve",
    response_model=DecisionMutationResponse,
    responses=DECISION_ERROR_RESPONSES,
)
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


@router.post(
    "/{decision_id}/reject",
    response_model=DecisionMutationResponse,
    responses=DECISION_ERROR_RESPONSES,
)
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
