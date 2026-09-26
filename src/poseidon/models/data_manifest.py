"""Immutable point-in-time evidence manifest."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class DataManifest(Base):
    __tablename__ = "data_manifests"
    __table_args__ = (
        UniqueConstraint("content_sha256", name="uq_data_manifests_content_sha256"),
        Index("ix_data_manifests_market_as_of", "market", "as_of"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    capability_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    sources_json: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    payload_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
