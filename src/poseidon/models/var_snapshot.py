"""VaRSnapshot ORM model for TimescaleDB hypertable.

Stores point-in-time VaR computation results for historical tracking
and compliance reporting.
"""

from sqlalchemy import Column, DateTime, Integer, Numeric, PrimaryKeyConstraint, String
from sqlalchemy.dialects.postgresql import JSONB

from poseidon.models.base import Base


class VaRSnapshot(Base):
    """VaR computation snapshot stored in TimescaleDB hypertable."""

    __tablename__ = "var_snapshots"

    time = Column(DateTime(timezone=True), nullable=False)
    method = Column(String(32), nullable=False)
    var_95 = Column(Numeric, nullable=False)
    var_99 = Column(Numeric, nullable=False)
    cvar_95 = Column(Numeric, nullable=False)
    cvar_99 = Column(Numeric, nullable=False)
    portfolio_value = Column(Numeric, nullable=False)
    holding_period = Column(Integer, nullable=False, server_default="1")
    details = Column(JSONB, nullable=True)

    __table_args__ = (PrimaryKeyConstraint("time", "method", name="pk_var_snapshots"),)
