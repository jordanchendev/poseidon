"""Principal-scoped manifest freeze and research revision routes.

These routes wrap the frozen-input services only. They never submit orders,
approve decisions, or touch release pointers; the acting identity always comes
from the authenticated principal, never from the request body.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal, get_auth_principal
from poseidon.core.database import get_db
from poseidon.decision_loop.manifest import ManifestService, ValidationError
from poseidon.decision_loop.research import ResearchService
from poseidon.models.data_manifest import DataManifest
from poseidon.models.research_revision import ResearchRevision

router = APIRouter()
NonEmptyText = Annotated[str, Field(min_length=1, max_length=512)]
Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
RevisionStatus = Literal["prepared", "completed", "needs_review"]
FREEZE_ROLES = ("researcher", "decision-worker")


class ResearchAPIError(BaseModel):
    detail: str


RESEARCH_ERROR_RESPONSES = {status: {"model": ResearchAPIError} for status in (403, 404, 409, 422)}


class ManifestFreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: dict[str, JsonValue]
    as_of: str | None = None


class ManifestIdentityResponse(BaseModel):
    id: uuid.UUID
    content_sha256: str
    as_of: datetime
    market: str
    interval: str
    account_scope: str


class ManifestResponse(ManifestIdentityResponse):
    capability_json: dict[str, JsonValue]
    sources_json: list[JsonValue]
    payload_json: dict[str, JsonValue]
    created_at: datetime


class RevisionPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_key: NonEmptyText
    manifest_id: uuid.UUID
    policy_version: NonEmptyText
    provider: NonEmptyText
    model: NonEmptyText
    runtime_digest: NonEmptyText
    expected_previous_revision_id: uuid.UUID | None = None


class RevisionCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research: dict[str, JsonValue]
    account_scope: NonEmptyText
    request_sha256: Digest


class RevisionProviderFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: NonEmptyText


class RevisionResponse(BaseModel):
    id: uuid.UUID
    status: RevisionStatus
    request_sha256: str
    content_sha256: str | None
    manifest_id: uuid.UUID
    scope_key: str
    previous_revision_id: uuid.UUID | None
    policy_version: str
    provider: str
    model: str
    runtime_digest: str
    research_json: dict[str, JsonValue]
    created_at: datetime


def _raise_api_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, (ValidationError, PydanticValidationError)):
        raise HTTPException(status_code=422, detail="Invalid research input") from error
    raise error


def _require_any_role(principal: AuthPrincipal, roles: tuple[str, ...]) -> None:
    for role in roles:
        try:
            principal.require_role(role)
        except HTTPException:
            continue
        return
    raise HTTPException(status_code=403, detail="Role not authorized")


def _manifest_scope(manifest: DataManifest) -> str:
    payload = manifest.payload_json if isinstance(manifest.payload_json, dict) else {}
    scope = payload.get("account_scope")
    return scope if isinstance(scope, str) else ""


def _scoped_manifest(db: Session, manifest_id, principal: AuthPrincipal) -> DataManifest:
    manifest = db.get(DataManifest, manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Manifest not found")
    principal.require_account_scope(_manifest_scope(manifest))
    return manifest


def _scoped_revision(db: Session, revision_id, principal: AuthPrincipal) -> ResearchRevision:
    revision = db.get(ResearchRevision, revision_id)
    if revision is None:
        raise HTTPException(status_code=404, detail="Research revision not found")
    _scoped_manifest(db, revision.manifest_id, principal)
    return revision


def _manifest_identity(manifest: DataManifest) -> dict:
    return {
        "id": manifest.id,
        "content_sha256": manifest.content_sha256,
        "as_of": manifest.as_of,
        "market": manifest.market,
        "interval": manifest.interval,
        "account_scope": _manifest_scope(manifest),
    }


def _manifest_response(manifest: DataManifest) -> dict:
    return {
        **_manifest_identity(manifest),
        "capability_json": manifest.capability_json,
        "sources_json": manifest.sources_json,
        "payload_json": manifest.payload_json,
        "created_at": manifest.created_at,
    }


def _revision_response(revision: ResearchRevision) -> dict:
    return {
        "id": revision.id,
        "status": revision.status,
        "request_sha256": revision.request_sha256,
        "content_sha256": revision.content_sha256,
        "manifest_id": revision.manifest_id,
        "scope_key": revision.scope_key,
        "previous_revision_id": revision.previous_revision_id,
        "policy_version": revision.policy_version,
        "provider": revision.provider,
        "model": revision.model,
        "runtime_digest": revision.runtime_digest,
        "research_json": revision.research_json,
        "created_at": revision.created_at,
    }


def _write(db: Session, operation):
    try:
        record = operation()
        db.commit()
        db.refresh(record)
        return record
    except Exception as error:
        db.rollback()
        _raise_api_error(error)


@router.post("/manifests", response_model=ManifestIdentityResponse, responses=RESEARCH_ERROR_RESPONSES)
def freeze_manifest(
    body: ManifestFreezeRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    def operation():
        _require_any_role(principal, FREEZE_ROLES)
        account_scope = body.request.get("account_scope")
        if not isinstance(account_scope, str):
            raise HTTPException(status_code=403, detail="Account scope not authorized")
        principal.require_account_scope(account_scope)
        return ManifestService(db).freeze(body.request, body.as_of)

    return _manifest_identity(_write(db, operation))


@router.get("/manifests/{manifest_id}", response_model=ManifestResponse, responses=RESEARCH_ERROR_RESPONSES)
def get_manifest(
    manifest_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        return _manifest_response(_scoped_manifest(db, manifest_id, principal))
    except Exception as error:
        _raise_api_error(error)


@router.post("/revisions", response_model=RevisionResponse, responses=RESEARCH_ERROR_RESPONSES)
def prepare_revision(
    body: RevisionPrepareRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    def operation():
        _require_any_role(principal, FREEZE_ROLES)
        _scoped_manifest(db, body.manifest_id, principal)
        options = {}
        if "expected_previous_revision_id" in body.model_fields_set:
            options["expected_previous_revision_id"] = body.expected_previous_revision_id
        return ResearchService(db).prepare(
            body.scope_key,
            body.manifest_id,
            policy_version=body.policy_version,
            provider=body.provider,
            model=body.model,
            runtime_digest=body.runtime_digest,
            **options,
        )

    return _revision_response(_write(db, operation))


@router.get("/revisions", response_model=list[RevisionResponse], responses=RESEARCH_ERROR_RESPONSES)
def list_revisions(
    scope_key: Annotated[str, Query(min_length=1, max_length=512)],
    status: RevisionStatus | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    if not principal.account_scopes:
        raise HTTPException(status_code=403, detail="Account scope not authorized")
    query = (
        select(ResearchRevision, DataManifest)
        .join(DataManifest, DataManifest.id == ResearchRevision.manifest_id)
        .where(ResearchRevision.scope_key == scope_key)
        .order_by(ResearchRevision.created_at, ResearchRevision.id)
    )
    if status is not None:
        query = query.where(ResearchRevision.status == status)
    return [
        _revision_response(revision)
        for revision, manifest in db.execute(query).all()
        if _manifest_scope(manifest) in principal.account_scopes
    ]


@router.get("/revisions/{revision_id}", response_model=RevisionResponse, responses=RESEARCH_ERROR_RESPONSES)
def get_revision(
    revision_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    try:
        return _revision_response(_scoped_revision(db, revision_id, principal))
    except Exception as error:
        _raise_api_error(error)


@router.post(
    "/revisions/{revision_id}/complete",
    response_model=RevisionResponse,
    responses=RESEARCH_ERROR_RESPONSES,
)
def complete_revision(
    revision_id: uuid.UUID,
    body: RevisionCompleteRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    def operation():
        # The service owns the researcher-only rule; the route adds no wider path.
        if db.get(ResearchRevision, revision_id) is None:
            raise HTTPException(status_code=404, detail="Research revision not found")
        return ResearchService(db).complete_human_research(
            revision_id,
            body.research,
            principal=principal,
            account_scope=body.account_scope,
            request_sha256=body.request_sha256,
        )

    return _revision_response(_write(db, operation))


@router.post(
    "/revisions/{revision_id}/provider-failed",
    response_model=RevisionResponse,
    responses=RESEARCH_ERROR_RESPONSES,
)
def mark_provider_failed(
    revision_id: uuid.UUID,
    body: RevisionProviderFailedRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    def operation():
        principal.require_role("decision-worker")
        _scoped_revision(db, revision_id, principal)
        return ResearchService(db).provider_failed(revision_id, body.reason_code)

    return _revision_response(_write(db, operation))
