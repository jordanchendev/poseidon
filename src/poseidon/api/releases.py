"""Typed, principal-scoped strategy release API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError
from poseidon.decision_loop.manifest import ValidationError
from poseidon.decision_loop.releases import ReleaseService
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion

router = APIRouter()
NonNegativeRevision = Annotated[int, Field(strict=True, ge=0)]
Reason = Annotated[str, Field(min_length=1, max_length=1024)]


class ReleaseAPIError(BaseModel):
    detail: str | dict
    error_code: str
    status_code: int


ERROR_RESPONSES = {status: {"model": ReleaseAPIError} for status in (403, 404, 409, 422)}


class ReleasePredicateResponse(BaseModel):
    code: str
    state: Literal["passed", "blocked", "unavailable", "inconclusive"]
    reason: str
    evidence: dict


class ReleaseGateResponse(BaseModel):
    allowed: bool
    predicates: list[ReleasePredicateResponse]


class ReleaseEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    strategy_id: uuid.UUID
    event_type: Literal["release", "pause", "rollback", "reject"]
    candidate_version_id: uuid.UUID | None
    from_version_id: uuid.UUID | None
    to_version_id: uuid.UUID | None
    source_release_event_id: uuid.UUID | None
    campaign_id: uuid.UUID | None
    review_id: uuid.UUID | None
    holdout_use_id: uuid.UUID | None
    reconciliation_id: uuid.UUID | None
    actor_id: str
    account_scope: str
    expected_revision: int
    resulting_revision: int
    reason: str
    gate_json: dict | list
    created_at: datetime


class RollbackTargetResponse(BaseModel):
    strategy_version_id: uuid.UUID
    version_no: int
    content_sha256: str
    source_release_event_id: uuid.UUID
    released_revision: int


class TransitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: NonNegativeRevision
    expected_active_version_id: uuid.UUID | None
    reason: Reason


class ReleaseRequest(TransitionRequest):
    candidate_version_id: uuid.UUID
    campaign_id: uuid.UUID
    review_id: uuid.UUID
    reconciliation_id: uuid.UUID


class PauseRequest(TransitionRequest):
    pass


class RollbackRequest(TransitionRequest):
    target_version_id: uuid.UUID
    reconciliation_id: uuid.UUID


class RejectRequest(TransitionRequest):
    candidate_version_id: uuid.UUID


class ReleaseMutationResponse(BaseModel):
    event_id: uuid.UUID
    event_type: Literal["release", "pause", "rollback", "reject"]
    strategy_id: uuid.UUID
    from_version_id: uuid.UUID | None
    to_version_id: uuid.UUID | None
    release_revision: int


def _strategy(db: Session, strategy_id: uuid.UUID) -> StrategyRecord:
    strategy = db.get(StrategyRecord, strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


def _version_scope(
    db: Session,
    strategy: StrategyRecord,
    version_id: uuid.UUID | None,
    principal: AuthPrincipal,
) -> StrategyVersion:
    if version_id is None:
        raise HTTPException(status_code=404, detail="Strategy version not found")
    version = db.get(StrategyVersion, version_id)
    if version is None or version.strategy_id != strategy.id:
        raise HTTPException(status_code=404, detail="Strategy version not found")
    account_scope = version.policy_json.get("account_scope") if isinstance(version.policy_json, dict) else None
    principal.require_account_scope(account_scope)
    return version


def _raise_release_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, ValidationError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    if isinstance(error, DecisionConflictError):
        status = 422 if "gate did not pass" in str(error) else 409
        raise HTTPException(status_code=status, detail=str(error)) from error
    raise error


def _mutate(db: Session, operation):
    try:
        response = operation(ReleaseService(db))
        db.commit()
        return response
    except Exception as error:
        db.rollback()
        _raise_release_error(error)


@router.get(
    "/{strategy_id}/release-gate",
    response_model=ReleaseGateResponse,
    responses=ERROR_RESPONSES,
)
def get_release_gate(
    strategy_id: uuid.UUID,
    candidate_version_id: uuid.UUID = Query(...),
    campaign_id: uuid.UUID = Query(...),
    review_id: uuid.UUID = Query(...),
    reconciliation_id: uuid.UUID = Query(...),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> ReleaseGateResponse:
    strategy = _strategy(db, strategy_id)
    _version_scope(db, strategy, candidate_version_id, principal)
    result = ReleaseService(db).evaluate_gate(
        strategy_id,
        candidate_version_id,
        campaign_id,
        review_id,
        reconciliation_id,
        principal=principal,
    )
    return ReleaseGateResponse.model_validate(result.as_json())


@router.get(
    "/{strategy_id}/release-events",
    response_model=list[ReleaseEventResponse],
    responses=ERROR_RESPONSES,
)
def list_release_events(
    strategy_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> list[StrategyReleaseEvent]:
    _strategy(db, strategy_id)
    events = db.scalars(
        select(StrategyReleaseEvent)
        .where(StrategyReleaseEvent.strategy_id == strategy_id)
        .order_by(StrategyReleaseEvent.resulting_revision, StrategyReleaseEvent.id)
    ).all()
    for event in events:
        principal.require_account_scope(event.account_scope)
    return events


@router.get(
    "/{strategy_id}/rollback-targets",
    response_model=list[RollbackTargetResponse],
    responses=ERROR_RESPONSES,
)
def list_rollback_targets(
    strategy_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> list[RollbackTargetResponse]:
    strategy = _strategy(db, strategy_id)
    events = db.scalars(
        select(StrategyReleaseEvent)
        .where(
            StrategyReleaseEvent.strategy_id == strategy_id,
            StrategyReleaseEvent.event_type == "release",
            StrategyReleaseEvent.to_version_id.is_not(None),
        )
        .order_by(StrategyReleaseEvent.resulting_revision.desc(), StrategyReleaseEvent.id)
    ).all()
    targets = []
    seen = set()
    for event in events:
        principal.require_account_scope(event.account_scope)
        if event.to_version_id in seen:
            continue
        version = _version_scope(db, strategy, event.to_version_id, principal)
        seen.add(event.to_version_id)
        targets.append(
            RollbackTargetResponse(
                strategy_version_id=version.id,
                version_no=version.version_no,
                content_sha256=version.content_sha256,
                source_release_event_id=event.id,
                released_revision=event.resulting_revision,
            )
        )
    return targets


@router.post(
    "/{strategy_id}/release",
    response_model=ReleaseMutationResponse,
    responses=ERROR_RESPONSES,
)
def release_strategy(
    strategy_id: uuid.UUID,
    body: ReleaseRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=256)],
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> ReleaseMutationResponse:
    def operation(service):
        principal.require_role("release_manager")
        strategy = _strategy(db, strategy_id)
        _version_scope(db, strategy, body.candidate_version_id, principal)
        return service.release(
            strategy_id,
            body.candidate_version_id,
            body.campaign_id,
            body.review_id,
            body.reconciliation_id,
            expected_revision=body.expected_revision,
            expected_active_version_id=body.expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            reason=body.reason,
        )

    return _mutate(db, operation)


@router.post("/{strategy_id}/pause", response_model=ReleaseMutationResponse, responses=ERROR_RESPONSES)
def pause_strategy(
    strategy_id: uuid.UUID,
    body: PauseRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=256)],
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> ReleaseMutationResponse:
    def operation(service):
        principal.require_role("release_manager")
        strategy = _strategy(db, strategy_id)
        _version_scope(db, strategy, strategy.active_version_id, principal)
        return service.pause(
            strategy_id,
            expected_revision=body.expected_revision,
            expected_active_version_id=body.expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            reason=body.reason,
        )

    return _mutate(db, operation)


@router.post(
    "/{strategy_id}/rollback",
    response_model=ReleaseMutationResponse,
    responses=ERROR_RESPONSES,
)
def rollback_strategy(
    strategy_id: uuid.UUID,
    body: RollbackRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=256)],
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> ReleaseMutationResponse:
    def operation(service):
        principal.require_role("release_manager")
        strategy = _strategy(db, strategy_id)
        _version_scope(db, strategy, body.target_version_id, principal)
        return service.rollback(
            strategy_id,
            body.target_version_id,
            body.reconciliation_id,
            expected_revision=body.expected_revision,
            expected_active_version_id=body.expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            reason=body.reason,
        )

    return _mutate(db, operation)


@router.post("/{strategy_id}/reject", response_model=ReleaseMutationResponse, responses=ERROR_RESPONSES)
def reject_strategy(
    strategy_id: uuid.UUID,
    body: RejectRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=256)],
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> ReleaseMutationResponse:
    def operation(service):
        principal.require_role("release_manager")
        strategy = _strategy(db, strategy_id)
        _version_scope(db, strategy, body.candidate_version_id, principal)
        return service.reject(
            strategy_id,
            body.candidate_version_id,
            expected_revision=body.expected_revision,
            expected_active_version_id=body.expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            reason=body.reason,
        )

    return _mutate(db, operation)
