"""BackfillJob ORM model — replaces v7.0 BackfillProgress (Phase 38 D-10).

Phase 39 expands the row to carry request-level metadata: ``symbols``,
``intervals``, ``requested_by``, ``started_at``, ``finished_at``. The legacy
``symbol``/``interval`` columns become nullable for multi-tuple jobs but are
still populated by the dispatcher path for single-tuple automatic backfills.
"""

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class BackfillJob(Base):
    """Historical backfill job — job_id / status / cursor / progress / error.

    Phase 38 introduces the substrate; Phase 39 expands it to carry
    request-level API metadata (`symbols`, `intervals`, `requested_by`,
    `started_at`, `finished_at`). See:
    - .planning/phases/38-data-foundation/38-CONTEXT.md D-10..D-12
    - .planning/phases/39-backfill-api-coverage/39-CONTEXT.md D-01..D-09
    """

    __tablename__ = "backfill_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(16), nullable=False, default="pending")
    market = Column(String(32), nullable=False)
    # symbol/interval are nullable for multi-tuple request-level API jobs
    # (Phase 39 migration 021). Legacy dispatcher single-tuple jobs still
    # populate them.
    symbol = Column(String(32), nullable=True)
    interval = Column(String(8), nullable=True)
    symbols = Column(JSONB, nullable=False, default=list)
    intervals = Column(JSONB, nullable=False, default=list)
    requested_by = Column(String(16), nullable=False, default="api")
    cursor = Column(JSONB, nullable=True)
    progress = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
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
            name="ck_backfill_jobs_status",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BackfillJob job_id={self.job_id} status={self.status} "
            f"market={self.market} symbols={self.symbols} "
            f"intervals={self.intervals}>"
        )
