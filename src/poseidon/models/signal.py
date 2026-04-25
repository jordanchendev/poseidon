"""SQLAlchemy ORM model for persisted trading signals."""

import uuid

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    quantity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_time: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_until: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    interval: Mapped[str] = mapped_column(String(8), nullable=False, server_default="1d")
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")

    # Limit order fields (Phase 49)
    order_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    order_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
