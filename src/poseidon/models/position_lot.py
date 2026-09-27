"""Fill-derived FIFO position lot."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class PositionLot(Base):
    __tablename__ = "position_lots"
    __table_args__ = (
        UniqueConstraint("opening_fill_id", name="uq_position_lots_opening_fill"),
        CheckConstraint(
            "original_quantity > 0 AND original_quantity <= 1e308",
            name="ck_position_lots_original_quantity_positive",
        ),
        CheckConstraint(
            "open_quantity >= 0 AND open_quantity <= original_quantity AND open_quantity <= 1e308",
            name="ck_position_lots_open_quantity_range",
        ),
        CheckConstraint(
            "reserved_close_quantity >= 0 AND reserved_close_quantity <= open_quantity "
            "AND reserved_close_quantity <= 1e308",
            name="ck_position_lots_reserved_close_quantity_range",
        ),
        Index(
            "ix_position_lots_fifo",
            "account_scope",
            "account_generation",
            "market",
            "symbol",
            "instrument",
            "side",
            "opened_at",
            "id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    account_generation: Mapped[str] = mapped_column(String(256), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    opening_fill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("order_fills.id"), nullable=False)
    opening_decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decision_records.id"), nullable=False
    )
    original_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    open_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    reserved_close_quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0, server_default="0")
    cost_basis_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
