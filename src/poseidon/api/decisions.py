"""Scoped HTTP routes for frozen portfolio decisions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionNotFoundError, DecisionService
from poseidon.decision_loop.manifest import ValidationError
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.order import OrderRecord

router = APIRouter()
PositiveRevision = Annotated[int, Field(strict=True, ge=1)]
DecisionAction = Literal["watch", "hold", "enter", "add", "reduce", "exit", "reject"]
DecisionStatus = Literal[
    "pending_approval",
    "approved",
    "rejected",
    "superseded",
    "expired",
    "execution_claimed",
    "risk_blocked",
]
ExecutionStatus = Literal[
    "not_started",
    "pending",
    "pending_submit",
    "submitted",
    "partially_filled",
    "filled",
    "rejected",
    "cancelled",
    "reconciliation_required",
    "mixed",
]
ReconciliationStatus = Literal["not_required", "pending", "required", "resolved", "mixed"]
ActionReason = Literal[
    "available",
    "status_not_pending",
    "decision_expired",
    "role_not_authorized",
    "account_scope_not_authorized",
]


class DecisionIntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_snapshot_id: uuid.UUID
    symbol: str
    market: Literal["tw_stock", "crypto_perp"]
    instrument: str
    action: Literal["enter", "add", "reduce", "exit"]
    side: Literal["long", "short"]
    target_weight: float
    order_type: Literal["market"] = "market"


class DecisionDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    selected_evaluation_ids: list[uuid.UUID]
    final_action: DecisionAction
    order_intents: list[DecisionIntentResponse] = Field(default_factory=list)
    override_reason: str | None = None


class DecisionRiskSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    hard_failures: list[str]
    allowed_actions: list[DecisionAction]


class DecisionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_run_id: uuid.UUID
    account_scope: Annotated[str, Field(min_length=1)]
    original_json: DecisionDocumentResponse
    portfolio_snapshot_json: dict[str, JsonValue]
    risk_snapshot_json: DecisionRiskSnapshotResponse


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
    detail: str


DECISION_ERROR_RESPONSES = {status: {"model": DecisionAPIError} for status in (403, 404, 409, 422)}


class DecisionCapabilityMetadata(BaseModel):
    can_approve: bool
    approve_reason: ActionReason
    can_modify: bool
    modify_reason: ActionReason
    can_reject: bool
    reject_reason: ActionReason


class DecisionEventHistoryResponse(BaseModel):
    id: uuid.UUID
    event_type: str
    actor_id: str
    expected_revision: int
    created_at: datetime


class DecisionResponse(BaseModel):
    id: uuid.UUID
    evaluation_run_id: uuid.UUID
    strategy_version_id: uuid.UUID
    account_scope: str
    decision_as_of: datetime
    valid_until: datetime
    status: DecisionStatus
    revision: int
    creation_sha256: str
    policy_sha256: str
    original_json: DecisionDocumentResponse
    final_json: DecisionDocumentResponse
    original_action: DecisionAction
    final_action: DecisionAction
    portfolio_snapshot_json: dict[str, JsonValue]
    risk_snapshot_json: DecisionRiskSnapshotResponse
    execution_status: ExecutionStatus
    reconciliation_status: ReconciliationStatus
    event_history: list[DecisionEventHistoryResponse]
    action_metadata: DecisionCapabilityMetadata
    created_at: datetime
    updated_at: datetime


class DecisionMutationResponse(BaseModel):
    id: uuid.UUID
    status: DecisionStatus
    revision: int
    final_json: DecisionDocumentResponse


def _action_metadata(record, principal: AuthPrincipal) -> DecisionCapabilityMetadata:
    valid_until = record.valid_until.replace(tzinfo=UTC) if record.valid_until.tzinfo is None else record.valid_until
    if record.status != "pending_approval":
        reason = "status_not_pending"
    elif valid_until <= datetime.now(UTC):
        reason = "decision_expired"
    elif record.account_scope not in principal.account_scopes:
        reason = "account_scope_not_authorized"
    elif "portfolio_manager" not in principal.roles or "decision-worker" in principal.roles:
        reason = "role_not_authorized"
    else:
        reason = "available"
    allowed = reason == "available"
    return DecisionCapabilityMetadata(
        can_approve=allowed,
        approve_reason=reason,
        can_modify=allowed,
        modify_reason=reason,
        can_reject=allowed,
        reject_reason=reason,
    )


def _decision_response(record, *, principal: AuthPrincipal, db: Session):
    original = DecisionDocumentResponse.model_validate(record.original_json)
    final = DecisionDocumentResponse.model_validate(record.final_json)
    events = db.scalars(
        select(DecisionEvent)
        .where(DecisionEvent.decision_id == record.id)
        .order_by(DecisionEvent.created_at, DecisionEvent.id)
    ).all()
    orders = db.scalars(
        select(OrderRecord).where(OrderRecord.decision_id == record.id).order_by(OrderRecord.created_at, OrderRecord.id)
    ).all()
    execution_values = sorted({order.status for order in orders})
    execution_status = (
        "not_started"
        if not execution_values
        else execution_values[0]
        if len(execution_values) == 1
        and execution_values[0]
        in {
            "pending",
            "pending_submit",
            "submitted",
            "partially_filled",
            "filled",
            "rejected",
            "cancelled",
            "reconciliation_required",
        }
        else "mixed"
    )
    reconciliation_values = sorted(
        {order.reconciliation_status for order in orders if order.reconciliation_status is not None}
    )
    reconciliation_status = (
        "not_required"
        if not reconciliation_values
        else reconciliation_values[0]
        if len(reconciliation_values) == 1 and reconciliation_values[0] in {"pending", "required", "resolved"}
        else "mixed"
    )
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
        "original_json": original,
        "final_json": final,
        "portfolio_snapshot_json": record.portfolio_snapshot_json,
        "risk_snapshot_json": record.risk_snapshot_json,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "original_action": original.final_action,
        "final_action": final.final_action,
        "execution_status": execution_status,
        "reconciliation_status": reconciliation_status,
        "event_history": [
            DecisionEventHistoryResponse(
                id=event.id,
                event_type=event.event_type,
                actor_id=event.actor_id,
                expected_revision=event.expected_revision,
                created_at=event.created_at,
            )
            for event in events
        ],
        "action_metadata": _action_metadata(record, principal),
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
            original_json=body.original_json.model_dump(mode="json"),
            portfolio_snapshot_json=body.portfolio_snapshot_json,
            risk_snapshot_json=body.risk_snapshot_json.model_dump(mode="json"),
        )
        db.commit()
        db.refresh(record)
        return _decision_response(record, principal=principal, db=db)
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
        return [_decision_response(record, principal=principal, db=db) for record in records]
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
        return _decision_response(record, principal=principal, db=db)
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
