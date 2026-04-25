"""ORM model for open interest records on perpetual contracts."""

from sqlalchemy import Column, DateTime, Index, Numeric, PrimaryKeyConstraint, String

from poseidon.models.base import Base


class OpenInterest(Base):
    """Open interest snapshot for perpetual contracts.

    Each row represents an OI data point at a given interval (e.g., 5m, 1h, 4h).
    Used by OIChange and OIBuildup features for liquidity sweep detection.
    """

    __tablename__ = "open_interest"

    time = Column(DateTime(timezone=True), nullable=False)  # OI snapshot timestamp
    symbol = Column(String(32), nullable=False)  # CCXT perp format e.g. "BTC/USDT:USDT"
    market = Column(String(32), nullable=False)  # "crypto_perp"
    interval = Column(String(8), nullable=False)  # "5m", "1h", "4h"
    open_interest = Column(Numeric, nullable=False)  # OI in contracts
    open_interest_value = Column(Numeric, nullable=True)  # OI in USDT (optional)

    __table_args__ = (
        PrimaryKeyConstraint("time", "symbol", "interval", name="pk_open_interest"),
        Index("idx_open_interest_symbol_time", "symbol", time.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<OpenInterest(time={self.time}, symbol={self.symbol}, interval={self.interval}, oi={self.open_interest})>"
        )
