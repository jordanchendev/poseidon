"""Immutable strategy release authority event."""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
    event,
    func,
    inspect,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


class StrategyReleaseEvent(Base):
    __tablename__ = "strategy_release_events"
    __table_args__ = (
        UniqueConstraint("strategy_id", "id", name="uq_strategy_release_events_strategy_id_id"),
        UniqueConstraint("strategy_id", "idempotency_key", name="uq_strategy_release_events_idempotency"),
        UniqueConstraint("strategy_id", "resulting_revision", name="uq_strategy_release_events_revision"),
        ForeignKeyConstraint(
            ["strategy_id", "candidate_version_id"],
            ["strategy_versions.strategy_id", "strategy_versions.id"],
            name="fk_strategy_release_events_candidate_version",
        ),
        ForeignKeyConstraint(
            ["strategy_id", "from_version_id"],
            ["strategy_versions.strategy_id", "strategy_versions.id"],
            name="fk_strategy_release_events_from_version",
        ),
        ForeignKeyConstraint(
            ["strategy_id", "to_version_id"],
            ["strategy_versions.strategy_id", "strategy_versions.id"],
            name="fk_strategy_release_events_to_version",
        ),
        ForeignKeyConstraint(
            ["strategy_id", "source_release_event_id"],
            ["strategy_release_events.strategy_id", "strategy_release_events.id"],
            name="fk_strategy_release_events_source_release",
        ),
        ForeignKeyConstraint(
            ["campaign_id", "review_id"],
            ["campaign_reviews.campaign_id", "campaign_reviews.id"],
            name="fk_strategy_release_events_review",
        ),
        ForeignKeyConstraint(
            ["campaign_id", "holdout_use_id"],
            ["holdout_uses.campaign_id", "holdout_uses.id"],
            name="fk_strategy_release_events_holdout_use",
        ),
        CheckConstraint(
            "event_type IN ('release', 'pause', 'rollback', 'reject', 'bootstrap')",
            name="ck_strategy_release_events_type",
        ),
        CheckConstraint(
            "resulting_revision = expected_revision + 1 AND expected_revision >= 0",
            name="ck_strategy_release_events_revision",
        ),
        CheckConstraint(
            "char_length(request_sha256) = 64",
            name="ck_strategy_release_events_request_sha256",
        ).ddl_if(dialect="postgresql"),
        CheckConstraint(
            "btrim(actor_id) <> '' AND btrim(account_scope) <> '' "
            "AND btrim(idempotency_key) <> '' AND btrim(reason) <> ''",
            name="ck_strategy_release_events_nonempty_text",
        ).ddl_if(dialect="postgresql"),
        CheckConstraint(
            "(event_type = 'release' AND candidate_version_id IS NOT NULL "
            "AND to_version_id = candidate_version_id AND source_release_event_id IS NULL "
            "AND campaign_id IS NOT NULL AND review_id IS NOT NULL "
            "AND holdout_use_id IS NOT NULL AND reconciliation_id IS NOT NULL) OR "
            "(event_type = 'pause' AND candidate_version_id IS NULL "
            "AND to_version_id IS NULL AND source_release_event_id IS NULL) OR "
            "(event_type = 'rollback' AND candidate_version_id IS NOT NULL "
            "AND to_version_id = candidate_version_id AND source_release_event_id IS NOT NULL "
            "AND reconciliation_id IS NOT NULL) OR "
            "(event_type = 'reject' AND candidate_version_id IS NOT NULL "
            "AND to_version_id IS NOT DISTINCT FROM from_version_id "
            "AND source_release_event_id IS NULL) OR "
            "(event_type = 'bootstrap' AND candidate_version_id IS NOT NULL "
            "AND to_version_id = candidate_version_id AND from_version_id IS NULL "
            "AND expected_revision = 0 AND source_release_event_id IS NULL "
            "AND campaign_id IS NULL AND review_id IS NULL "
            "AND holdout_use_id IS NULL AND reconciliation_id IS NULL)",
            name="ck_strategy_release_events_shape",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    candidate_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    from_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    to_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    source_release_event_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    campaign_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_campaigns.id"), nullable=True
    )
    review_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    holdout_use_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    reconciliation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("account_reconciliations.id"), nullable=True
    )
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    account_scope: Mapped[str] = mapped_column(String(256), nullable=False)
    expected_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    resulting_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    gate_json: Mapped[list | dict] = mapped_column(JSONB, nullable=False)
    response_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


@event.listens_for(StrategyReleaseEvent, "before_update")
def _reject_release_event_update(mapper, connection, target):
    if inspect(target).modified:
        raise ValueError("strategy_release_events is append-only")


@event.listens_for(StrategyReleaseEvent, "before_delete")
def _reject_release_event_delete(mapper, connection, target):
    raise ValueError("strategy_release_events is append-only")
