"""Independent durable paper-broker order acceptance."""

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


class PaperBrokerOrder(Base):
    __tablename__ = "paper_broker_orders"
    __table_args__ = (
        UniqueConstraint(
            "account_scope",
            "account_generation",
            "client_order_ref",
            name="uq_paper_broker_orders_client_ref",
        ),
        UniqueConstraint(
            "account_scope",
            "account_generation",
            "broker_order_id",
            name="uq_paper_broker_orders_broker_order",
        ),
        UniqueConstraint(
            "id",
            "account_scope",
            "account_generation",
            "market",
            "symbol",
            "instrument",
            "side",
            name="uq_paper_broker_orders_identity",
        ),
        ForeignKeyConstraint(
            ["account_scope", "account_generation"],
            ["paper_broker_accounts.account_scope", "paper_broker_accounts.account_generation"],
            name="fk_paper_broker_orders_account",
        ),
        CheckConstraint(
            "quantity > 0 AND quantity <= 1e308",
            name="ck_paper_broker_orders_quantity_positive",
        ),
        CheckConstraint(
            "price IS NULL OR (price >= 0 AND price <= 1e308)",
            name="ck_paper_broker_orders_price_nonnegative_finite",
        ),
        Index(
            "ix_paper_broker_orders_recovery",
            "account_scope",
            "account_generation",
            "status",
            "updated_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    account_generation: Mapped[str] = mapped_column(String(256), nullable=False)
    client_order_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    broker_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
