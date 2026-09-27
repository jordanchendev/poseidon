"""Independent paper-broker account generation and opening cash."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class PaperBrokerAccount(Base):
    __tablename__ = "paper_broker_accounts"
    __table_args__ = (
        UniqueConstraint(
            "account_scope",
            "account_generation",
            name="uq_paper_broker_accounts_scope_generation",
        ),
        CheckConstraint(
            "opening_cash >= 0 AND opening_cash <= 1e308",
            name="ck_paper_broker_accounts_opening_cash_nonnegative",
        ),
        CheckConstraint("state_version >= 0", name="ck_paper_broker_accounts_state_version_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    account_generation: Mapped[str] = mapped_column(String(256), nullable=False)
    opening_cash: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
