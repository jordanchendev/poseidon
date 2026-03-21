"""SQLAlchemy ORM model for virtual portfolio positions."""

import uuid

from sqlalchemy import Boolean, DateTime, Float, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class VirtualPositionRecord(Base):
    __tablename__ = "virtual_positions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()"
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity_pct: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.0")
    entry_signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    entry_time: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False)
    closed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    close_signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    close_time: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
