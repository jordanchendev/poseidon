"""Terminal result for one member of an evaluation universe."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class EvaluationSnapshot(Base):
    __tablename__ = "evaluation_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "evaluation_run_id",
            "symbol",
            "instrument",
            name="uq_evaluation_snapshots_run_symbol_instrument",
        ),
        Index("ix_evaluation_snapshots_run_status", "evaluation_run_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    recommendation_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    technical_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    research_revision_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    reason_codes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
