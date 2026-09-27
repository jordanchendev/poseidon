"""SQLAlchemy ORM model for trading orders."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class OrderRecord(Base):
    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint(
            "account_scope",
            "account_generation",
            "client_order_ref",
            name="uq_orders_account_generation_client_ref",
        ),
        CheckConstraint(
            "reserved_quantity IS NULL OR (reserved_quantity >= 0 AND reserved_quantity <= 1e308)",
            name="ck_orders_reserved_quantity_nonnegative_finite",
        ),
        Index("ix_orders_decision_id", "decision_id"),
        Index("ix_orders_execution_key", "execution_key"),
        Index(
            "ix_orders_recovery_state",
            "account_scope",
            "account_generation",
            "status",
            "reservation_status",
            "reconciliation_status",
        ),
    )

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
    # TRUTH-03: structured 4-key dict {check_name, rule, shortfall, details}
    # built via poseidon.risk.reject_reason.build_reject_reason. NULL when not rejected.
    reject_reason: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # FK to signals(id) so we can audit whether this Order was driven by
    # an upstream PASSED signal. Indexed for the mini-audit join
    # (signals.status='passed' -> orders.signal_id). NULL for portfolio
    # rebalance / protective close paths.
    signal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("signals.id"),
        nullable=True,
        index=True,
    )
    # Audit whitelist: origin tag distinguishes signal-driven flows from
    # protective close paths. Combined with signal_id NULL it lets the
    # mini-audit reject false positives:
    # NULL + origin=signal           -> wiring breach (must be 0)
    # NULL + origin=stop_loss/liquidation/manual -> legitimate protective exit
    order_origin: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="signal",
        index=True,
    )
    decision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("decision_records.id"), nullable=True
    )
    account_scope: Mapped[str | None] = mapped_column(String(256), nullable=True)
    account_generation: Mapped[str | None] = mapped_column(String(256), nullable=True)
    execution_key: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    client_order_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    instrument: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intent_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    intent_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reserved_cash_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    reserved_quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    reservation_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reconciliation_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    submit_attempted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    protective_context_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", nullable=False)
