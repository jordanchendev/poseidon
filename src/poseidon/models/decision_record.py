"""Frozen account-level portfolio decision."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class DecisionRecord(Base):
    __tablename__ = "decision_records"
    __table_args__ = (
        UniqueConstraint("creation_sha256", name="uq_decision_records_creation_sha256"),
        UniqueConstraint("execution_key", name="uq_decision_records_execution_key"),
        Index(
            "ix_decision_records_account_status_valid_created",
            "account_scope",
            "status",
            "valid_until",
            "created_at",
        ),
        Index("ix_decision_records_claim_selection", "status", "valid_until", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=False
    )
    strategy_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategy_versions.id"), nullable=False
    )
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    decision_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    creation_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    original_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    final_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    portfolio_snapshot_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    risk_snapshot_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    execution_key: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
