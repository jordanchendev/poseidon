from sqlalchemy import Column, DateTime, Index, Numeric, PrimaryKeyConstraint, String

from poseidon.models.base import Base


class OHLCV(Base):
    __tablename__ = "ohlcv"

    time = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    market = Column(String(32), nullable=False)
    instrument = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    open = Column(Numeric, nullable=False)
    high = Column(Numeric, nullable=False)
    low = Column(Numeric, nullable=False)
    close = Column(Numeric, nullable=False)
    volume = Column(Numeric, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("time", "symbol", "market", "interval", name="pk_ohlcv"),
        Index("idx_ohlcv_symbol_market_interval_time", "symbol", "market", "interval", time.desc()),
    )
