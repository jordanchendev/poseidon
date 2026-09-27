"""Immutable comparison of independent broker and internal account state."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class AccountReconciliation(Base):
    __tablename__ = "account_reconciliations"
    __table_args__ = (
        UniqueConstraint(
            "account_scope",
            "account_generation",
            "as_of",
            "broker_state_watermark",
            "internal_state_watermark",
            "broker_snapshot_sha256",
            "policy_sha256",
            name="uq_account_reconciliations_replay",
        ),
        Index(
            "ix_account_reconciliations_account_as_of",
            "account_scope",
            "account_generation",
            "as_of",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    account_generation: Mapped[str] = mapped_column(String(256), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    broker_state_watermark: Mapped[str] = mapped_column(String(128), nullable=False)
    internal_state_watermark: Mapped[str] = mapped_column(String(128), nullable=False)
    broker_snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    broker_snapshot_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    internal_snapshot_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    difference_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    policy_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
