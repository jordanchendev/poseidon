"""SQLAlchemy ORM model for trade log records (realized PnL tracking)."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class TradeLogRecord(Base):
    __tablename__ = "trade_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()"
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    exit_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    holding_days: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()", nullable=False
    )

    __table_args__ = (
        Index("ix_trade_logs_strategy_symbol", "strategy_name", "symbol"),
    )
