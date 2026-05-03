"""RLExecutionRun ORM model — tracks Qlib RL execution job lifecycle (Phase 90).

A RLExecutionRun is created by ``POST /research/rl-execution/run`` (Wave
4b, Plan 90-05b) and progresses through the same status enum as
TrainingRun (Phase 41 D-05 — D-25 mandates verbatim lifecycle reuse):

    pending -> running -> succeeded | failed | cancelled

Only **succeeded** runs populate ``summary`` (per-algo metrics dict),
``verdict`` (deploy / refine / kill, set by the Wave 5 verdict gate),
and ``result_dir`` (path to ``local_dev/rl-execution/runs/<run_id>/``
where fills.parquet + summary.json + comparison.parquet live).

Schema: D-25 (rl_execution_runs columns; alembic 038)
Status: D-25 -> Phase 41 D-05 (pending / running / succeeded / failed / cancelled)
Result persistence: D-24 (per-run parquet under local_dev/rl-execution/runs/<run_id>/)
Verdict: D-17/D-18/D-19 (frozen YAML gate + 3-way deploy/refine/kill)
"""

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class RLExecutionRun(Base):
    """Historical RL execution run — run_id / status / algos / summary / verdict."""

    __tablename__ = "rl_execution_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    algos = Column(JSONB, nullable=False, default=list)
    dates = Column(JSONB, nullable=True)
    mode = Column(String(16), nullable=False, default="full")
    status = Column(String(16), nullable=False, default="pending")
    summary = Column(JSONB, nullable=True)
    verdict = Column(String(16), nullable=True)
    result_dir = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    requested_by = Column(String(16), nullable=False, default="api")
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','cancelled')",
            name="ck_rl_execution_runs_status",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RLExecutionRun run_id={self.run_id} status={self.status} algos={self.algos} verdict={self.verdict}>"
