"""SQLAlchemy ORM model for experiment tracking.

Stores Optuna trial results and experiment metadata for the automated
parameter search pipeline. Each record captures the full config, metrics,
and optional linkage to an Optuna study/trial.
"""

import uuid

from sqlalchemy import DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class ExperimentRecord(Base):
    """Experiment run record for parameter search and optimization tracking.

    Fields follow D-04 spec: id, study_name, config_json, metrics_json,
    composite_score, wfe_score, status, market, interval, created_at, updated_at.
    D-05: optuna_study_name and optuna_trial_number provide optional linkage
    without foreign keys (Optuna manages its own tables in the optuna schema).
    """

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    study_name: Mapped[str] = mapped_column(String(128), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metrics_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    wfe_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="running"
    )
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    optuna_study_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    optuna_trial_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    holdout_boundary: Mapped[str | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_experiments_market_interval", "market", "interval"),
        Index("ix_experiments_created_at", "created_at"),
    )
