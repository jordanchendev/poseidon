"""UniverseSnapshotRecord ORM model.

Stores point-in-time universe resolution snapshots for auditing,
reproducibility, and survivorship-bias-free backtesting.
"""

import uuid

from sqlalchemy import Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from poseidon.models.base import Base


class UniverseSnapshotRecord(Base):
    """A snapshot of the resolved universe at a specific point in time."""

    __tablename__ = "universe_snapshots"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default="gen_random_uuid()",
    )
    market = Column(String(32), nullable=False, index=True)
    snapshot_time = Column(DateTime(timezone=True), nullable=False, index=True)
    symbols = Column(JSONB, nullable=False)  # list of {id, name, ccxt_symbol?}
    source_type = Column(String(64), nullable=False)
    filter_config = Column(JSONB, nullable=True)  # applied filter names + params
    created_at = Column(
        DateTime(timezone=True),
        server_default="now()",
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_universe_snapshots_market_time",
            "market",
            snapshot_time.desc(),
        ),
    )
