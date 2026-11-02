"""Principal-scoped strategy-pipeline API (``/api/v1/pipelines``).

Writes need the ``researcher`` role (the Telegram bridge and scout use a
dedicated ``strategy-bridge`` principal holding it); reads need viewer,
researcher or release_manager. The legacy shared key carries no role and is
always 403. There is no PUT or DELETE, and thresholds are server defaults only.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import uuid
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthorizedRoute, get_auth_principal
from poseidon.core.authz import AuthPrincipal
from poseidon.core.canonical import content_sha256
from poseidon.core.database import get_db
from poseidon.pipeline.models import DRAFTERS
from poseidon.pipeline.service import (
    PipelineConflict,
    PipelineNotFound,
    PipelineRequestInvalid,
    PipelineService,
    attempt_json,
    event_json,
    run_json,
    trial_json,
)
from poseidon.pipeline.state import RunStatus

logger = logging.getLogger(__name__)
router = APIRouter(route_class=AuthorizedRoute)

WRITE_ROLE = "researcher"
READ_ROLES = frozenset({"viewer", "researcher", "release_manager"})
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=256)]
Revision = Annotated[int, Field(strict=True, ge=1)]
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024)]
StageName = Annotated[str, StringConstraints(min_length=1, max_length=32)]
ORIGINAL_TEXT_MAX_CHARS = 64 * 1024  # macminim4 reads at most 64 KiB of stdin per draft


class IdeaOriginalRequest(BaseModel):
    """The owner's material behind a Sisyphus-drafted ``idea_text`` (macminim4 ``strategy-pending``).

    ``text`` is kept byte for byte (no stripping): ``sha256`` is macminim4's hash of exactly this text.
    """

    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, StringConstraints(min_length=1, max_length=ORIGINAL_TEXT_MAX_CHARS)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    drafted_by: Literal[DRAFTERS]

    @model_validator(mode="after")
    def _sha256_matches(self):
        if not self.text.strip():
            raise ValueError("original text is blank")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("original sha256 does not match its text")
        return self


class CreatePipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    source: Literal["telegram", "scout", "api"]
    family_id: uuid.UUID | None = None
    spec_json: dict[str, JsonValue] | None = None
    stage_plan: Annotated[list[StageName], Field(min_length=1, max_length=32)] | None = None
    original: IdeaOriginalRequest | None = None


class RerunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_stage: StageName | None = None
    expected_revision: Revision
    reason: Reason


class AbortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: Revision
    reason: Reason


def get_pipeline_enqueuer() -> Callable[[uuid.UUID], None]:
    """Overridable in tests; the default publishes to the pipeline Celery queue."""

    from poseidon.pipeline.tasks import enqueue_advance

    return enqueue_advance


def _require_reader(principal: AuthPrincipal) -> None:
    if not READ_ROLES.intersection(principal.roles) or "decision-worker" in principal.roles:
        raise HTTPException(status_code=403, detail="Role not authorized")


def _raise(error: Exception):
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, PipelineNotFound):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, PipelineConflict):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, PipelineRequestInvalid):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise error


def _mutate(db: Session, enqueue, status_code: int, operation) -> JSONResponse:
    try:
        response, replayed = operation(PipelineService(db))
        db.commit()
    except Exception as error:
        db.rollback()
        _raise(error)
    if not replayed and response["status"] == RunStatus.QUEUED:
        try:
            enqueue(uuid.UUID(response["id"]))
        except Exception:  # committed already; sweep_pipelines re-enqueues idle queued runs
            logger.exception("pipeline enqueue failed for %s", response["id"])
    return JSONResponse(status_code=status_code, content=response)


def _read(operation):
    try:
        return operation()
    except Exception as error:
        _raise(error)


@router.post("", status_code=201)
def create_pipeline(
    body: CreatePipelineRequest,
    idempotency_key: IdempotencyKey,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
    enqueue=Depends(get_pipeline_enqueuer),
):
    principal.require_role(WRITE_ROLE)
    # A request without an original hashes exactly as before p004 (Idempotency-Key replays stay valid).
    payload = body.model_dump(mode="json", exclude={"original"} if body.original is None else None)
    return _mutate(
        db,
        enqueue,
        201,
        lambda service: service.create(payload, actor_id=principal.actor_id, idempotency_key=idempotency_key),
    )


@router.get("")
def list_pipelines(
    status: RunStatus | None = Query(None),
    family_id: uuid.UUID | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    _require_reader(principal)
    runs = PipelineService(db).list_runs(
        status=status.value if status else None, family_id=family_id, limit=limit, offset=offset
    )
    return {"items": [run_json(run) for run in runs], "limit": limit, "offset": offset}


@router.get("/capabilities")
def pipeline_capabilities(principal: AuthPrincipal = Depends(get_auth_principal)):
    """What a spec can express: the capability brief, the spec v3 JSON Schema and the registry version
    (design §3.9, §8), rendered from the registry (the same text a new idea-only run freezes the sha256 of)."""

    _require_reader(principal)
    return capabilities_document()


@functools.cache
def capabilities_document() -> dict:
    import hashlib

    from poseidon.pipeline.v3.brief import render_brief, spec_json_schema
    from poseidon.pipeline.v3.registry import REGISTRY_VERSION
    from poseidon.pipeline.v3.spec import SPEC_VERSION

    brief = render_brief()
    schema = spec_json_schema()
    return {
        "spec_version": SPEC_VERSION,
        "registry_version": REGISTRY_VERSION,
        "brief": brief,
        "brief_sha256": hashlib.sha256(brief.encode("utf-8")).hexdigest(),
        "schema": schema,
        "schema_sha256": content_sha256(schema),
    }


@router.get("/families/{family_id}/trials")
def family_trials(
    family_id: uuid.UUID,
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    _require_reader(principal)
    count, trials = _read(lambda: PipelineService(db).family_trials(family_id, limit=limit, offset=offset))
    return {"family_id": str(family_id), "trial_count": count, "items": [trial_json(trial) for trial in trials]}


@router.get("/{run_id}")
def get_pipeline(
    run_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    _require_reader(principal)
    return run_json(_read(lambda: PipelineService(db).get_run(run_id)))


@router.get("/{run_id}/stages")
def get_pipeline_stages(
    run_id: uuid.UUID,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    _require_reader(principal)
    return {"run_id": str(run_id), "stages": _read(lambda: PipelineService(db).stages(run_id))}


@router.get("/{run_id}/stages/{stage}/attempts/{attempt}")
def get_stage_attempt(
    run_id: uuid.UUID,
    stage: str,
    attempt: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    _require_reader(principal)
    return attempt_json(_read(lambda: PipelineService(db).attempt(run_id, stage, attempt)), full=True)


@router.get("/{run_id}/events")
def get_pipeline_events(
    run_id: uuid.UUID,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    _require_reader(principal)
    events = _read(lambda: PipelineService(db).events(run_id, limit=limit, offset=offset))
    return {"run_id": str(run_id), "items": [event_json(event) for event in events]}


@router.post("/{run_id}/rerun", status_code=202)
def rerun_pipeline(
    run_id: uuid.UUID,
    body: RerunRequest,
    idempotency_key: IdempotencyKey,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
    enqueue=Depends(get_pipeline_enqueuer),
):
    principal.require_role(WRITE_ROLE)
    payload = body.model_dump(mode="json")
    return _mutate(
        db,
        enqueue,
        202,
        lambda service: service.rerun(run_id, payload, actor_id=principal.actor_id, idempotency_key=idempotency_key),
    )


@router.post("/{run_id}/abort", status_code=202)
def abort_pipeline(
    run_id: uuid.UUID,
    body: AbortRequest,
    idempotency_key: IdempotencyKey,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Session = Depends(get_db),
):
    principal.require_role(WRITE_ROLE)
    payload = body.model_dump(mode="json")
    return _mutate(
        db,
        lambda run_id: None,
        202,
        lambda service: service.abort(run_id, payload, actor_id=principal.actor_id, idempotency_key=idempotency_key),
    )
