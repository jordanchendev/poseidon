"""SQLAlchemy ORM model for daily NAV snapshots."""

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Float, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class NavSnapshotRecord(Base):
    __tablename__ = "nav_snapshots"
    __table_args__ = (UniqueConstraint("snapshot_date", "market", name="uq_nav_snapshot_date_market"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_nav: Mapped[float] = mapped_column(Float, nullable=False)
    holdings_value: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    holdings_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # TRUTH-04 (D-05): cash flow audit trail. Positive=deposit, negative=withdrawal.
    # NUMERIC(18, 2) preserves cent-level precision. Default 0 for normal EOD writes.
    cash_flow: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, server_default="0", default=Decimal("0"))
    market: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", nullable=False)
