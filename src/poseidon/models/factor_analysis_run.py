"""Factor analysis run ORM model (Phase 47)."""

import uuid

from sqlalchemy import CheckConstraint, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class FactorAnalysisRun(Base):
    """Persisted factor analysis run state and results."""

    __tablename__ = "factor_analysis_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    run_type: Mapped[str] = mapped_column(String(16), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    results_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "run_type IN ('ic', 'shapley', 'centrality')",
            name="ck_factor_analysis_runs_run_type",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed')",
            name="ck_factor_analysis_runs_status",
        ),
        Index("ix_factor_analysis_runs_market", "market"),
        Index("ix_factor_analysis_runs_created_at", "created_at"),
    )
