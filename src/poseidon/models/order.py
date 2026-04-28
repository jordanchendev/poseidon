"""SQLAlchemy ORM model for trading orders."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class OrderRecord(Base):
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # buy / sell
    order_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="'market'")
    target_weight: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)  # limit price
    side: Mapped[str] = mapped_column(String(16), nullable=False, server_default="'long'")
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="'pending'")
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    broker_mode: Mapped[str] = mapped_column(String(16), nullable=False)  # paper / live
    # TRUTH-03 (D-13): structured 4-key dict {check_name, rule, shortfall, details}
    # built via poseidon.risk.reject_reason.build_reject_reason. NULL when not rejected.
    reject_reason: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", nullable=False)
