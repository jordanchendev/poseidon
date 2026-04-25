"""SQLAlchemy ORM models for backtest results.

Three tables: backtests (metadata + metrics), backtest_trades (per-trade),
backtest_equity (per-bar equity curve). Separate tables enable cross-backtest
comparison queries.
"""

import uuid

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from poseidon.models.base import Base


class BacktestRecord(Base):
    """Main backtest run record."""

    __tablename__ = "backtests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    strategy_type: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False, server_default="1d")
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    walk_forward: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="running")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    trades: Mapped[list["BacktestTradeRecord"]] = relationship(back_populates="backtest", cascade="all, delete-orphan")
    equity_points: Mapped[list["BacktestEquityRecord"]] = relationship(
        back_populates="backtest", cascade="all, delete-orphan"
    )


class BacktestTradeRecord(Base):
    """Individual trade within a backtest run."""

    __tablename__ = "backtest_trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    backtest_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("backtests.id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    entry_time: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_price: Mapped[float] = mapped_column(Numeric, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    quantity: Mapped[float] = mapped_column(Numeric, nullable=False)
    pnl: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    fees: Mapped[float] = mapped_column(Numeric, nullable=False, server_default="0")
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")

    # Relationship
    backtest: Mapped["BacktestRecord"] = relationship(back_populates="trades")


class BacktestEquityRecord(Base):
    """Per-bar equity curve point within a backtest run."""

    __tablename__ = "backtest_equity"

    backtest_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("backtests.id"), primary_key=True)
    time: Mapped[str] = mapped_column(DateTime(timezone=True), primary_key=True)
    equity: Mapped[float] = mapped_column(Numeric, nullable=False)
    drawdown: Mapped[float] = mapped_column(Numeric, nullable=False, server_default="0")

    # Relationship
    backtest: Mapped["BacktestRecord"] = relationship(back_populates="equity_points")
