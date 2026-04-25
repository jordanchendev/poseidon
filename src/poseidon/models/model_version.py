"""SQLAlchemy ORM model for ML model version tracking."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="training")
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    feature_list: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    train_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    train_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
