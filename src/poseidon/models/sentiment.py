import uuid

from sqlalchemy import Column, DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import UUID

from poseidon.models.base import Base


class Sentiment(Base):
    __tablename__ = "sentiment"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(32), nullable=False)
    market = Column(String(32), nullable=False)
    source_type = Column(String(32), nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
