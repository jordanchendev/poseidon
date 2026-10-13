"""Typed, principal-scoped strategy release API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError
from poseidon.decision_loop.manifest import ValidationError
from poseidon.decision_loop.releases import ReleaseGateRejected, ReleaseService
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion

router = APIRouter()
NonNegativeRevision = Annotated[int, Field(strict=True, ge=0)]
Reason = Annotated[str, Field(min_length=1, max_length=1024)]


class ReleaseAPIError(BaseModel):
    detail: str


class ReleasePredicateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_version_id: uuid.UUID | None = None
    strategy_version_id: uuid.UUID | None = None
    strategy_id: uuid.UUID | None = None
    active_version_id: uuid.UUID | None = None
    campaign_incumbent_version_id: uuid.UUID | None = None
    campaign_id: uuid.UUID | None = None
    review_id: uuid.UUID | None = None
    holdout_use_id: uuid.UUID | None = None
    owner_campaign_id: uuid.UUID | None = None
    reconciliation_id: uuid.UUID | None = None
    source_release_event_id: uuid.UUID | None = None
    content_sha256: str | None = None
    policy_sha256: str | None = None
    contract_sha256: str | None = None
    input_sha256: str | None = None
    result_sha256: str | None = None
    holdout_identity_sha256: str | None = None
    declared_trial_identity_sha256: str | None = None
    actual_trial_identity_sha256: str | None = None
    paired_sample_sha256: str | None = None
    paired_sample_key_sha256: str | None = None
    account_scope: str | None = None
    source_account_scope: str | None = None
    account_generation: str | None = None
    source_account_generation: str | None = None
    strategy_type: Literal["rule", "voting", "liquidity_sweep", "model"] | None = None
    symbol: str | None = None
    market: Literal["tw_stock", "crypto_spot", "crypto_perp"] | None = None
    instrument: Literal["spot", "perpetual"] | None = None
    interval: str | None = None
    strategy_market: str | None = None
    mode: Literal["human", "assisted", "automatic"] | None = None
    status: Literal["passed", "failed", "blocked", "unavailable", "inconclusive"] | None = None
    policy_state: Literal["passed", "blocked", "unavailable", "inconclusive"] | None = None
    requires_human_release: bool | None = None
    protective_exit: bool | None = None
    age_seconds: float | None = None
    broker_state_watermark: str | None = None
    internal_state_watermark: str | None = None
    actor_id: str | None = None
    trial_ids: list[uuid.UUID] | None = None
    terminal_states: list[str] | None = None
    newer_reconciliation_ids: list[uuid.UUID] | None = None
    position_lot_ids: list[uuid.UUID] | None = None
    claimed_decision_ids: list[uuid.UUID] | None = None
    active_order_ids: list[uuid.UUID] | None = None


ReleasePredicateCode = Literal[
    "candidate_identity",
    "policy_identity",
    "campaign_identity",
    "incumbent_pointer",
    "six_cell_evidence",
    "review_identity",
    "holdout_ownership",
    "reconciliation",
    "release_authority",
    "rollback_target",
    "rollback_policy",
    "rollback_holdings",
    "rollback_outstanding_work",
    "rollback_reconciliation",
]


class ReleasePredicateResponse(BaseModel):
    code: ReleasePredicateCode
    state: Literal["passed", "blocked", "unavailable", "inconclusive"]
    reason: str
    evidence: ReleasePredicateEvidence


class ReleaseGateResponse(BaseModel):
    allowed: bool
    predicates: list[ReleasePredicateResponse]


class ReleaseGateFailureDetail(BaseModel):
    code: Literal["release_gate_failed"] = "release_gate_failed"
    message: str
    gate: ReleaseGateResponse


class ReleaseValidationFailureDetail(BaseModel):
    code: Literal["validation_failed"] = "validation_failed"
    message: str
    gate: None = None


class ReleaseUnprocessableResponse(BaseModel):
    detail: ReleaseGateFailureDetail | ReleaseValidationFailureDetail


class RequestValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str
    input: JsonValue | None = None
    ctx: dict[str, JsonValue] | None = None


class RequestValidationResponse(BaseModel):
    detail: Literal["Validation error"]
    error_code: Literal["VALIDATION_ERROR"]
    status_code: Literal[422]
    errors: list[RequestValidationIssue]


ERROR_RESPONSES = {
    403: {"model": ReleaseAPIError},
    404: {"model": ReleaseAPIError},
    409: {"model": ReleaseAPIError},
    422: {"model": ReleaseUnprocessableResponse | RequestValidationResponse},
}


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
    gate_json: ReleaseGateResponse
    created_at: datetime


class RollbackTargetResponse(BaseModel):
    strategy_version_id: uuid.UUID
    version_no: int
    content_sha256: str
    source_release_event_id: uuid.UUID
    released_revision: int
    eligible: Literal[True]
    predicates: list[ReleasePredicateResponse]


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


def _authorize_strategy_read(
    db: Session,
    strategy: StrategyRecord,
    principal: AuthPrincipal,
    events: list[StrategyReleaseEvent],
) -> None:
    scopes = {event.account_scope for event in events}
    if not scopes:
        versions = db.scalars(
            select(StrategyVersion)
            .where(StrategyVersion.strategy_id == strategy.id)
            .order_by(StrategyVersion.version_no, StrategyVersion.id)
        ).all()
        scopes = {
            version.policy_json.get("account_scope")
            for version in versions
            if isinstance(version.policy_json, dict) and version.policy_json.get("account_scope")
        }
    if not scopes:
        raise HTTPException(status_code=404, detail="Strategy release scope not found")
    for account_scope in sorted(scopes):
        principal.require_account_scope(account_scope)


def _raise_release_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, ValidationError):
        detail = ReleaseValidationFailureDetail(message=str(error)).model_dump(mode="json")
        raise HTTPException(status_code=422, detail=detail) from error
    if isinstance(error, ReleaseGateRejected):
        detail = ReleaseGateFailureDetail(
            message="Release gate did not pass",
            gate=ReleaseGateResponse.model_validate(error.result.as_json()),
        ).model_dump(mode="json", exclude_none=True)
        raise HTTPException(status_code=422, detail=detail) from error
    if isinstance(error, DecisionConflictError):
        raise HTTPException(status_code=409, detail=str(error)) from error
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
    response_model_exclude_none=True,
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
    response_model_exclude_none=True,
    responses=ERROR_RESPONSES,
)
def list_release_events(
    strategy_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
) -> list[StrategyReleaseEvent]:
    strategy = _strategy(db, strategy_id)
    events = db.scalars(
        select(StrategyReleaseEvent)
        .where(StrategyReleaseEvent.strategy_id == strategy_id)
        .order_by(StrategyReleaseEvent.resulting_revision, StrategyReleaseEvent.id)
    ).all()
    _authorize_strategy_read(db, strategy, principal, events)
    return events


@router.get(
    "/{strategy_id}/rollback-targets",
    response_model=list[RollbackTargetResponse],
    response_model_exclude_none=True,
    responses=ERROR_RESPONSES,
)
def list_rollback_targets(
    strategy_id: uuid.UUID,
    reconciliation_id: uuid.UUID = Query(...),
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
    _authorize_strategy_read(db, strategy, principal, events)
    if not events:
        return []
    service = ReleaseService(db)
    authority_version = service._current_or_paused_authority_version(strategy)
    source_policy = service._policy_for_version(authority_version, strategy, principal)
    reconciliation = db.get(AccountReconciliation, reconciliation_id)
    targets = []
    seen = set()
    for event in events:
        if event.to_version_id in seen or event.to_version_id == strategy.active_version_id:
            continue
        version = _version_scope(db, strategy, event.to_version_id, principal)
        seen.add(event.to_version_id)
        gate, source, _policy = service._rollback_gate(
            strategy,
            version,
            reconciliation,
            source_policy,
            principal,
            None,
        )
        if not gate.allowed or source is None:
            continue
        targets.append(
            RollbackTargetResponse(
                strategy_version_id=version.id,
                version_no=version.version_no,
                content_sha256=version.content_sha256,
                source_release_event_id=source.id,
                released_revision=source.resulting_revision,
                eligible=True,
                predicates=[ReleasePredicateResponse.model_validate(item.as_json()) for item in gate.predicates],
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
        _strategy(db, strategy_id)
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
