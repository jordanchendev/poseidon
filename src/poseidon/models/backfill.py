from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint, func

from poseidon.models.base import Base


class BackfillProgress(Base):
    __tablename__ = "backfill_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    market = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    last_fetched_date = Column(DateTime(timezone=True), nullable=True)
    target_start_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "market", "interval", name="uq_backfill_symbol_market_interval"),
    )
