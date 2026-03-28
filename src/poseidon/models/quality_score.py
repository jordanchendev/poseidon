"""Data quality score per symbol+interval, stored in TimescaleDB."""

from sqlalchemy import Column, DateTime, Numeric, PrimaryKeyConstraint, String

from poseidon.models.base import Base


class QualityScore(Base):
    """Quality score for OHLCV data stored in TimescaleDB hypertable."""

    __tablename__ = "quality_scores"

    time = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    score = Column(Numeric, nullable=False)
    completeness = Column(Numeric, nullable=False)
    consistency = Column(Numeric, nullable=False)
    anomaly_free = Column(Numeric, nullable=False)
    timeliness = Column(Numeric, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("time", "symbol", "interval", name="pk_quality_scores"),
    )
