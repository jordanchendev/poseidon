"""Append-only event and idempotency ledger for portfolio decisions."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class DecisionEvent(Base):
    __tablename__ = "decision_events"
    __table_args__ = (
        UniqueConstraint(
            "decision_id",
            "event_type",
            "expected_revision",
            name="uq_decision_events_type_revision",
        ),
        UniqueConstraint(
            "decision_id",
            "idempotency_key",
            name="uq_decision_events_idempotency_key",
        ),
        CheckConstraint(
            "(idempotency_key IS NULL AND request_sha256 IS NULL) OR "
            "(idempotency_key IS NOT NULL AND request_sha256 IS NOT NULL)",
            name="ck_decision_events_idempotency_pair",
        ),
        Index("ix_decision_events_decision_created", "decision_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decision_records.id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(24), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    expected_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
