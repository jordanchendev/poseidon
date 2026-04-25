"""DataGap ORM model — Phase 40 D-04 audit substrate.

One row per detected gap window. The daily ``data_gap_audit`` beat task
(plan 40-02) INSERTs with ``ON CONFLICT (market, symbol, interval,
gap_start) DO NOTHING`` against the unique index installed by migration
023. A later audit pass that confirms the window is now fully populated
updates ``healed_at = now()`` (D-07). Rows are never deleted — history is
the whole point.

See .planning/phases/40-data-health-observability/40-CONTEXT.md D-04..D-09.
"""

import uuid

from sqlalchemy import Column, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID

from poseidon.models.base import Base


class DataGap(Base):
    """Per-gap-window audit row.

    Columns follow 40-CONTEXT.md D-04 exactly:

    * ``gap_id``       — UUID primary key
    * ``market`` / ``symbol`` / ``interval`` — tuple key (joins against
      ``data_coverage_mv`` and ``ingest_state``)
    * ``gap_start`` / ``gap_end`` — time window of the missing bars
    * ``missing_bars`` — integer count, drives the dashboard heatmap color
    * ``detected_at`` — when the audit first recorded this row
    * ``healed_at`` — non-null after a later audit confirms the window is
      now fully populated (D-07); NULL = still open
    """

    __tablename__ = "data_gaps"

    gap_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    market = Column(String(32), nullable=False)
    symbol = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    gap_start = Column(DateTime(timezone=True), nullable=False)
    gap_end = Column(DateTime(timezone=True), nullable=False)
    missing_bars = Column(Integer, nullable=False)
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    healed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DataGap market={self.market} symbol={self.symbol} "
            f"interval={self.interval} gap_start={self.gap_start} "
            f"missing_bars={self.missing_bars} "
            f"healed={self.healed_at is not None}>"
        )
