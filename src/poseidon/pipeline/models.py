"""Strategy-pipeline tables.

These live on their own declarative base so the main Alembic stream and every
``Base.metadata.create_all`` in the test-suite never see them; the separate
``pipeline`` Alembic stream (``alembic -n pipeline``) owns their DDL. ORM
listeners here mirror the database triggers in ``p001`` (two layers).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from poseidon.pipeline.state import STAGE_TERMINAL, RunStatus, StageStatus, Verdict, utcnow

ARTIFACT_JSON_MAX_BYTES = 256 * 1024

RUN_SOURCES = ("telegram", "scout", "api")
TRIAL_STATES = ("complete", "pruned", "fail")


def _in(column: str, values) -> str:
    return f"{column} IN ({', '.join(repr(str(value)) for value in values)})"


def _sha(column: str, *, nullable: bool = False) -> str:
    check = f"char_length({column}) = 64"
    return f"({column} IS NULL OR {check})" if nullable else check


class PipelineBase(DeclarativeBase):
    pass


class PipelineRun(PipelineBase):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        UniqueConstraint("run_no", name="uq_pipeline_runs_run_no"),
        UniqueConstraint("created_by", "create_idempotency_key", name="uq_pipeline_runs_create_idempotency"),
        CheckConstraint(_in("status", list(RunStatus)), name="ck_pipeline_runs_status"),
        CheckConstraint(_in("source", RUN_SOURCES), name="ck_pipeline_runs_source"),
        CheckConstraint("market = 'tw_futures' AND symbol = 'TX'", name="ck_pipeline_runs_instrument"),
        CheckConstraint(_sha("config_sha256"), name="ck_pipeline_runs_config_sha256"),
        CheckConstraint(
            "(create_idempotency_key IS NULL AND create_request_sha256 IS NULL) OR "
            "(create_idempotency_key IS NOT NULL AND create_request_sha256 IS NOT NULL "
            "AND char_length(create_request_sha256) = 64)",
            name="ck_pipeline_runs_idempotency_pair",
        ),
        CheckConstraint("btrim(idea_text) <> ''", name="ck_pipeline_runs_idea_text"),
        CheckConstraint("btrim(created_by) <> ''", name="ck_pipeline_runs_created_by"),
        CheckConstraint("revision >= 1", name="ck_pipeline_runs_revision"),
        CheckConstraint(
            "(status IN ('completed', 'failed', 'unsupported', 'error', 'aborted')) = (finished_at IS NOT NULL)",
            name="ck_pipeline_runs_finished",
        ),
        Index("ix_pipeline_runs_status_updated", "status", "updated_at"),
        Index("ix_pipeline_runs_family", "family_id"),
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    run_no: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False)
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    idea_text: Mapped[str] = mapped_column(Text, nullable=False)
    market: Mapped[str] = mapped_column(String(16), nullable=False, default="tw_futures", server_default="tw_futures")
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, default="TX", server_default="TX")
    input_spec_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    create_idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    create_request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued", server_default="queued")
    current_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    abort_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


RUN_FROZEN_COLUMNS = (
    "id",
    "run_no",
    "family_id",
    "source",
    "idea_text",
    "market",
    "symbol",
    "input_spec_json",
    "config_json",
    "config_sha256",
    "created_by",
    "create_idempotency_key",
    "create_request_sha256",
    "created_at",
)


class StageRun(PipelineBase):
    __tablename__ = "stage_runs"
    __table_args__ = (
        UniqueConstraint("run_id", "stage", "attempt", name="uq_stage_runs_attempt"),
        CheckConstraint(_in("status", list(StageStatus)), name="ck_stage_runs_status"),
        CheckConstraint(f"verdict IS NULL OR {_in('verdict', list(Verdict))}", name="ck_stage_runs_verdict"),
        CheckConstraint("(status = 'completed') = (verdict IS NOT NULL)", name="ck_stage_runs_verdict_completed"),
        CheckConstraint("trigger IN ('advance', 'retry', 'rerun')", name="ck_stage_runs_trigger"),
        CheckConstraint("attempt >= 1 AND seq >= 0", name="ck_stage_runs_ordinals"),
        CheckConstraint(_sha("input_sha256", nullable=True), name="ck_stage_runs_input_sha256"),
        CheckConstraint(
            "(artifact_json IS NULL AND artifact_sha256 IS NULL) OR "
            "(artifact_json IS NOT NULL AND artifact_sha256 IS NOT NULL AND char_length(artifact_sha256) = 64)",
            name="ck_stage_runs_artifact_pair",
        ),
        CheckConstraint(
            f"artifact_json IS NULL OR octet_length(artifact_json::text) <= {ARTIFACT_JSON_MAX_BYTES}",
            name="ck_stage_runs_artifact_size",
        ),
        CheckConstraint(
            "status <> 'running' OR (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND input_sha256 IS NOT NULL AND code_version IS NOT NULL)",
            name="ck_stage_runs_running_lease",
        ),
        CheckConstraint("status <> 'waiting' OR wait_until IS NOT NULL", name="ck_stage_runs_waiting"),
        CheckConstraint(
            "(status IN ('completed', 'aborted')) = (finished_at IS NOT NULL)",
            name="ck_stage_runs_finished",
        ),
        Index(
            "uq_stage_runs_open_attempt",
            "run_id",
            "stage",
            unique=True,
            postgresql_where=text("status IN ('pending', 'running', 'waiting')"),
        ),
        Index(
            "uq_stage_runs_one_running",
            "run_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
        Index("ix_stage_runs_status_lease", "status", "lease_expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    trigger: Mapped[str] = mapped_column(String(8), nullable=False)
    code_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    input_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    lease_owner: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    wait_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )


STAGE_RUN_IDENTITY_COLUMNS = ("id", "run_id", "stage", "seq", "attempt", "trigger", "created_at")


class PipelineTrial(PipelineBase):
    """DSR trial ledger. Every trial counts, whatever its attempt ended as."""

    __tablename__ = "pipeline_trials"
    __table_args__ = (
        UniqueConstraint("stage_run_id", "trial_no", name="uq_pipeline_trials_number"),
        CheckConstraint(_in("state", TRIAL_STATES), name="ck_pipeline_trials_state"),
        CheckConstraint(
            f"{_sha('params_sha256')} AND {_sha('window_sha256')}",
            name="ck_pipeline_trials_hashes",
        ),
        CheckConstraint("trial_no >= 0 AND (n_obs IS NULL OR n_obs >= 0)", name="ck_pipeline_trials_ordinals"),
        Index("ix_pipeline_trials_family", "family_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    stage_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("stage_runs.id"), nullable=False)
    trial_no: Mapped[int] = mapped_column(Integer, nullable=False)
    params_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    params_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_obs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    skew: Mapped[float | None] = mapped_column(Float, nullable=True)
    kurtosis: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )


class PipelineEvent(PipelineBase):
    """Append-only audit trail and idempotency ledger (mirrors ``DecisionEvent``)."""

    __tablename__ = "pipeline_events"
    __table_args__ = (
        UniqueConstraint("run_id", "idempotency_key", name="uq_pipeline_events_idempotency_key"),
        CheckConstraint(
            "(idempotency_key IS NULL AND request_sha256 IS NULL) OR "
            "(idempotency_key IS NOT NULL AND request_sha256 IS NOT NULL)",
            name="ck_pipeline_events_idempotency_pair",
        ),
        CheckConstraint(_sha("request_sha256", nullable=True), name="ck_pipeline_events_request_sha256"),
        Index("ix_pipeline_events_run_created", "run_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=False)
    stage_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stage_runs.id"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    expected_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )


# --- ORM-layer immutability (the p001 triggers are the second layer) ---


def _changed(target, columns) -> bool:
    state = inspect(target)
    return any(state.attrs[column].history.has_changes() for column in columns)


@event.listens_for(PipelineRun, "before_update")
def _prevent_frozen_run_update(mapper, connection, target):
    if _changed(target, RUN_FROZEN_COLUMNS):
        raise ValueError("pipeline run identity and config are frozen")


@event.listens_for(PipelineRun, "before_delete")
def _prevent_run_delete(mapper, connection, target):
    raise ValueError("pipeline runs cannot be deleted")


@event.listens_for(StageRun, "before_update")
def _prevent_terminal_stage_update(mapper, connection, target):
    history = inspect(target).attrs["status"].history
    old_status = history.deleted[0] if history.deleted else target.status
    if old_status in STAGE_TERMINAL:
        raise ValueError(f"stage attempt is {old_status} and immutable")
    if _changed(target, STAGE_RUN_IDENTITY_COLUMNS):
        raise ValueError("stage attempt identity is immutable")


@event.listens_for(StageRun, "before_delete")
def _prevent_stage_delete(mapper, connection, target):
    raise ValueError("stage attempts cannot be deleted")


def _reject_append_only_update(mapper, connection, target):
    if inspect(target).modified:
        raise ValueError(f"{target.__tablename__} is append-only")


def _reject_append_only_delete(mapper, connection, target):
    raise ValueError(f"{target.__tablename__} is append-only")


for _append_only_model in (PipelineTrial, PipelineEvent):
    event.listen(_append_only_model, "before_update", _reject_append_only_update)
    event.listen(_append_only_model, "before_delete", _reject_append_only_delete)
