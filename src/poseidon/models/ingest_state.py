"""IngestState ORM model — cursor bookkeeping for self-healing ingest (Phase 38)."""

from sqlalchemy import Boolean, Column, DateTime, String, Text, func

from poseidon.models.base import Base


class IngestState(Base):
    """Per-(symbol, market, interval) cursor state for the self-healing ingest loop.

    See .planning/phases/38-data-foundation/38-CONTEXT.md D-01..D-03.
    """

    __tablename__ = "ingest_state"

    symbol = Column(String(32), primary_key=True)
    market = Column(String(32), primary_key=True)
    interval = Column(String(8), primary_key=True)
    last_successful_ts = Column(DateTime(timezone=True), nullable=True)
    last_attempt_ts = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    first_backfill_done = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IngestState symbol={self.symbol} market={self.market} "
            f"interval={self.interval} last_successful_ts={self.last_successful_ts}>"
        )
