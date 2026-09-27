"""Append-only paper-broker cash movement."""

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


class PaperCashMovement(Base):
    __tablename__ = "paper_cash_movements"
    __table_args__ = (
        UniqueConstraint(
            "account_scope",
            "account_generation",
            "state_version",
            name="uq_paper_cash_movements_account_version",
        ),
        ForeignKeyConstraint(
            ["account_scope", "account_generation"],
            ["paper_broker_accounts.account_scope", "paper_broker_accounts.account_generation"],
            name="fk_paper_cash_movements_account",
        ),
        CheckConstraint(
            "amount != 0 AND amount >= -1e308 AND amount <= 1e308",
            name="ck_paper_cash_movements_amount_nonzero",
        ),
        CheckConstraint("state_version > 0", name="ck_paper_cash_movements_state_version_positive"),
        Index(
            "ix_paper_cash_movements_account_time",
            "account_scope",
            "account_generation",
            "occurred_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    account_generation: Mapped[str] = mapped_column(String(256), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    movement_type: Mapped[str] = mapped_column(String(32), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
