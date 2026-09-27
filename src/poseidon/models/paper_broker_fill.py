"""Independent durable paper-broker fill."""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class PaperBrokerFill(Base):
    __tablename__ = "paper_broker_fills"
    __table_args__ = (
        UniqueConstraint(
            "paper_broker_order_id",
            "broker_fill_id",
            name="uq_paper_broker_fills_order_fill",
        ),
        ForeignKeyConstraint(
            [
                "paper_broker_order_id",
                "account_scope",
                "account_generation",
                "market",
                "symbol",
                "instrument",
                "side",
            ],
            [
                "paper_broker_orders.id",
                "paper_broker_orders.account_scope",
                "paper_broker_orders.account_generation",
                "paper_broker_orders.market",
                "paper_broker_orders.symbol",
                "paper_broker_orders.instrument",
                "paper_broker_orders.side",
            ],
            name="fk_paper_broker_fills_order_identity",
        ),
        CheckConstraint(
            "fill_quantity > 0 AND fill_quantity <= 1e308",
            name="ck_paper_broker_fills_quantity_positive",
        ),
        CheckConstraint(
            "fill_price >= 0 AND fill_price <= 1e308",
            name="ck_paper_broker_fills_price_nonnegative",
        ),
        Index(
            "ix_paper_broker_fills_account_time",
            "account_scope",
            "account_generation",
            "fill_time",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    paper_broker_order_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    account_generation: Mapped[str] = mapped_column(String(256), nullable=False)
    broker_fill_id: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_quantity: Mapped[float] = mapped_column(Float, nullable=False)
    fill_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
