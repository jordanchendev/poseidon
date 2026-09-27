"""Replay-safe allocation of a closing fill to an opening lot."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class FillAllocation(Base):
    __tablename__ = "fill_allocations"
    __table_args__ = (
        UniqueConstraint(
            "closing_fill_id",
            "position_lot_id",
            name="uq_fill_allocations_closing_fill_lot",
        ),
        CheckConstraint(
            "quantity > 0 AND quantity <= 1e308",
            name="ck_fill_allocations_quantity_positive",
        ),
        Index("ix_fill_allocations_lot_created", "position_lot_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    closing_fill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("order_fills.id"), nullable=False)
    position_lot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("position_lots.id"), nullable=False
    )
    closing_decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decision_records.id"), nullable=False
    )
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    realized_cost_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
