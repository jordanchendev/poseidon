"""SQLAlchemy ORM model for experiment tracking.

Stores Optuna trial results and experiment metadata for the automated
parameter search pipeline. Each record captures the full config, metrics,
and optional linkage to an Optuna study/trial.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class ExperimentRecord(Base):
    """Experiment run record for parameter search and optimization tracking.

    Fields: id, study_name, config_json, metrics_json, composite_score,
    wfe_score, status, market, interval, created_at, updated_at.
    optuna_study_name and optuna_trial_number provide optional linkage
    without foreign keys (Optuna manages its own tables in the optuna schema).
    """

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    study_name: Mapped[str] = mapped_column(String(128), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metrics_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    wfe_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="running")
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    optuna_study_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    optuna_trial_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    holdout_boundary: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    campaign_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_campaigns.id"), nullable=True
    )
    original_trial_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    trial_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    strategy_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategy_versions.id"), nullable=True
    )
    ablation_arm: Mapped[str | None] = mapped_column(String(32), nullable=True)
    paired_sample_key_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    terminal_state: Mapped[str | None] = mapped_column(String(40), nullable=True)
    terminal_reason_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_experiments_market_interval", "market", "interval"),
        Index("ix_experiments_created_at", "created_at"),
        Index(
            "uq_experiments_campaign_paired_cell",
            "campaign_id",
            "original_trial_id",
            "strategy_version_id",
            "ablation_arm",
            "paired_sample_key_sha256",
            unique=True,
            postgresql_where=text("campaign_id IS NOT NULL"),
        ),
        CheckConstraint(
            "trial_role IS NULL OR trial_role IN ('search', 'paired_evaluation')",
            name="ck_experiments_trial_role",
        ),
        CheckConstraint(
            "ablation_arm IS NULL OR ablation_arm IN ('fundamental_only', 'technical_only', 'combined')",
            name="ck_experiments_ablation_arm",
        ),
        CheckConstraint(
            "terminal_state IS NULL OR terminal_state IN "
            "('succeeded', 'optimizer_failed', 'constraint_rejected', 'insufficient_data', "
            "'statistically_inconclusive', 'capability_unavailable')",
            name="ck_experiments_terminal_state",
        ),
        CheckConstraint(
            "(paired_sample_key_sha256 IS NULL OR char_length(paired_sample_key_sha256) = 64) "
            "AND (input_sha256 IS NULL OR char_length(input_sha256) = 64) "
            "AND (result_sha256 IS NULL OR char_length(result_sha256) = 64)",
            name="ck_experiments_phase99_hashes",
        ).ddl_if(dialect="postgresql"),
        CheckConstraint(
            "campaign_id IS NULL OR (campaign_id IS NOT NULL AND original_trial_id IS NOT NULL AND trial_role IS NOT NULL "
            "AND strategy_version_id IS NOT NULL AND ablation_arm IS NOT NULL "
            "AND paired_sample_key_sha256 IS NOT NULL AND input_sha256 IS NOT NULL "
            "AND result_sha256 IS NOT NULL AND started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND terminal_state IS NOT NULL AND terminal_state IN "
            "('succeeded', 'optimizer_failed', 'constraint_rejected', 'insufficient_data', "
            "'statistically_inconclusive', 'capability_unavailable'))",
            name="ck_experiments_campaign_link_complete",
        ),
    )


@event.listens_for(ExperimentRecord, "before_update")
def _prevent_linked_experiment_update(mapper, connection, target):
    history = inspect(target).attrs.campaign_id.history
    was_linked = any(value is not None for value in history.deleted)
    if target.campaign_id is not None or was_linked:
        raise ValueError("campaign-linked experiments are append-only")


@event.listens_for(ExperimentRecord, "before_delete")
def _prevent_linked_experiment_delete(mapper, connection, target):
    if target.campaign_id is not None:
        raise ValueError("campaign-linked experiments are append-only")
