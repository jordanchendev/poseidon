"""TrainingRun ORM model — tracks Qlib training job lifecycle (Phase 41).

A TrainingRun is created by ``POST /api/v1/models/train`` and progresses
through the status enum:

    pending -> running -> succeeded | failed | cancelled

Only **succeeded** runs promote to a ``ModelVersion`` row (D-06).  The
``model_version_id`` FK is NULL until promotion completes.

Schema: D-04 (training_runs columns)
Status: D-05 (pending / running / succeeded / failed / cancelled)
Promotion: D-06 (only succeeded runs link to model_versions)
Error: D-07 (failed runs preserve error text)
"""

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class TrainingRun(Base):
    """Historical training run — run_id / status / handler / model / metrics."""

    __tablename__ = "training_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    handler_class = Column(String(64), nullable=False)
    handler_params = Column(JSONB, nullable=False, default=dict)
    model_class = Column(String(64), nullable=False)
    model_params = Column(JSONB, nullable=False, default=dict)
    market = Column(String(32), nullable=False)
    symbols = Column(JSONB, nullable=False, default=list)
    interval = Column(String(8), nullable=False)
    segments = Column(JSONB, nullable=False, default=dict)
    lookback = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="pending")
    metrics = Column(JSONB, nullable=True)
    model_version_id = Column(
        UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=True
    )
    mlflow_run_id = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    requested_by = Column(String(16), nullable=False, default="api")
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="ck_training_runs_status",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TrainingRun run_id={self.run_id} status={self.status} "
            f"handler_class={self.handler_class} model_class={self.model_class}>"
        )
