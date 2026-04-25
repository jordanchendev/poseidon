import uuid

from sqlalchemy import Column, Date, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class Fundamentals(Base):
    __tablename__ = "fundamentals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(32), nullable=False)
    market = Column(String(32), nullable=False)
    date = Column(Date, nullable=False)
    data = Column(JSONB, nullable=False)

    __table_args__ = (UniqueConstraint("symbol", "market", "date", name="uq_fundamentals_symbol_market_date"),)
