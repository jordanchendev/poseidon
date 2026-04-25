"""ORM model for perpetual contract funding rate records."""

from sqlalchemy import Column, DateTime, Float, Index, PrimaryKeyConstraint, String

from poseidon.models.base import Base


class FundingRateRecord(Base):
    """Funding rate settlement record for perpetual contracts.

    Each row represents a single funding rate settlement (typically every 8 hours).
    Timestamps are rounded to 8h boundaries (00:00, 08:00, 16:00 UTC) to prevent
    duplicate entries from slight timestamp drift across API calls.
    """

    __tablename__ = "funding_rates"

    time = Column(DateTime(timezone=True), nullable=False)  # settlement time (8h boundary)
    symbol = Column(String(32), nullable=False)  # CCXT perp format e.g. "BTC/USDT:USDT"
    funding_rate = Column(Float, nullable=False)  # e.g., 0.0001 (0.01%)
    mark_price = Column(Float, nullable=True)
    index_price = Column(Float, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("time", "symbol", name="pk_funding_rates"),
        Index("idx_funding_rates_symbol_time", "symbol", time.desc()),
    )

    def __repr__(self) -> str:
        return f"<FundingRateRecord(time={self.time}, symbol={self.symbol}, rate={self.funding_rate})>"
