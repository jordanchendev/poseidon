"""Frozen experiment campaigns and append-only derived evidence."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint, event, func, inspect
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.decision_loop.manifest import content_sha256
from poseidon.models.base import Base


class ExperimentCampaign(Base):
    __tablename__ = "experiment_campaigns"
    __table_args__ = (
        UniqueConstraint(
            "contract_sha256",
            name="uq_experiment_campaigns_contract_sha256",
        ),
        CheckConstraint(
            "char_length(incumbent_content_sha256) = 64 "
            "AND char_length(candidate_content_sha256) = 64 "
            "AND char_length(contract_sha256) = 64",
            name="ck_experiment_campaigns_hashes",
        ),
        CheckConstraint("btrim(hypothesis) <> ''", name="ck_experiment_campaigns_hypothesis"),
        CheckConstraint("btrim(created_by) <> ''", name="ck_experiment_campaigns_created_by"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    incumbent_strategy_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategy_versions.id"), nullable=False
    )
    candidate_strategy_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategy_versions.id"), nullable=False
    )
    incumbent_content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    declared_difference_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    hypothesis: Mapped[str] = mapped_column(String(1024), nullable=False)
    contract_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    contract_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def verify_contract(self) -> None:
        if not isinstance(self.declared_difference_json, dict) or not self.declared_difference_json:
            raise ValueError("declared_difference_json must be a non-empty object")
        if not isinstance(self.contract_json, dict) or not self.contract_json:
            raise ValueError("contract_json must be a non-empty object")
        for field in ("hypothesis", "created_by"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty text")
        for field in ("incumbent_content_sha256", "candidate_content_sha256", "contract_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field} must be a 64-character digest")
        if self.contract_sha256 != content_sha256(self.contract_json):
            raise ValueError("experiment campaign contract hash mismatch")


class CampaignEvent(Base):
    __tablename__ = "campaign_events"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "idempotency_sha256",
            name="uq_campaign_events_idempotency",
        ),
        CheckConstraint("btrim(event_type) <> ''", name="ck_campaign_events_type"),
        CheckConstraint(
            "char_length(idempotency_sha256) = 64",
            name="ck_campaign_events_idempotency_sha256",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_campaigns.id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    idempotency_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class HoldoutUse(Base):
    __tablename__ = "holdout_uses"
    __table_args__ = (
        UniqueConstraint(
            "holdout_identity_sha256",
            name="uq_holdout_uses_identity",
        ),
        CheckConstraint(
            "char_length(holdout_identity_sha256) = 64 "
            "AND char_length(campaign_contract_sha256) = 64",
            name="ck_holdout_uses_hashes",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    holdout_identity_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_campaigns.id"), nullable=False
    )
    campaign_contract_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    audit_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    consumed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CampaignReview(Base):
    __tablename__ = "campaign_reviews"
    __table_args__ = (
        UniqueConstraint(
            "campaign_id",
            "input_sha256",
            name="uq_campaign_reviews_replay",
        ),
        CheckConstraint(
            "status IN ('passed', 'failed', 'inconclusive', 'unavailable')",
            name="ck_campaign_reviews_status",
        ),
        CheckConstraint(
            "char_length(input_sha256) = 64 AND char_length(result_sha256) = 64",
            name="ck_campaign_reviews_hashes",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_campaigns.id"), nullable=False
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    result_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    result_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


@event.listens_for(ExperimentCampaign, "before_insert")
def _verify_experiment_campaign_insert(mapper, connection, target):
    target.verify_contract()


def _reject_campaign_fact_update(mapper, connection, target):
    if inspect(target).modified:
        raise ValueError(f"{target.__tablename__} is append-only")


def _reject_campaign_fact_delete(mapper, connection, target):
    raise ValueError(f"{target.__tablename__} is append-only")


for _append_only_model in (ExperimentCampaign, CampaignEvent, HoldoutUse, CampaignReview):
    event.listen(_append_only_model, "before_update", _reject_campaign_fact_update)
    event.listen(_append_only_model, "before_delete", _reject_campaign_fact_delete)
