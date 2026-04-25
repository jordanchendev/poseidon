"""SQLAlchemy ORM model for order fills."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class OrderFillRecord(Base):
    __tablename__ = "order_fills"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    order_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    fill_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    broker_fill_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", nullable=False)
