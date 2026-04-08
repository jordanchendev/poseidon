"""BackfillJob ORM model — replaces v7.0 BackfillProgress (Phase 38 D-10)."""

import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class BackfillJob(Base):
    """Historical backfill job — job_id / status / cursor / progress / error.

    Phase 38 introduces the substrate; Phase 39 adds the REST dispatcher.
    See .planning/phases/38-data-foundation/38-CONTEXT.md D-10..D-12.
    """

    __tablename__ = "backfill_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(16), nullable=False, default="pending")
    market = Column(String(32), nullable=False)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    cursor = Column(JSONB, nullable=True)
    progress = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
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
            name="ck_backfill_jobs_status",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BackfillJob job_id={self.job_id} status={self.status} "
            f"{self.market}/{self.symbol}/{self.interval}>"
        )
