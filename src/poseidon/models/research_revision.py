"""Immutable research request identity and completed narrative."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base

# SEED-002 C3: why a completed revision was corrected (migration 048).
CORRECTION_REASONS = ("source_error", "excerpt_error", "numeric_error", "label_policy", "stale_evidence", "other")
_REASON_LIST = ", ".join(f"'{reason}'" for reason in CORRECTION_REASONS)
CORRECTION_REASON_CHECK = f"correction_reason IS NULL OR correction_reason IN ({_REASON_LIST})"
CORRECTION_LINK_CHECK = (
    "(correction_reason IS NULL OR supersedes_revision_id IS NOT NULL) "
    "AND (supersedes_revision_id IS NULL OR status <> 'completed' OR correction_reason IS NOT NULL)"
)
SUPERSEDES_WHERE = "supersedes_revision_id IS NOT NULL AND status = 'completed'"


class ResearchRevision(Base):
    __tablename__ = "research_revisions"
    __table_args__ = (
        UniqueConstraint("request_sha256", name="uq_research_revisions_request_sha256"),
        Index("ix_research_revisions_scope_created", "scope_key", "created_at"),
        # One completed correction per superseded revision: the correction chain is linear (048).
        Index(
            "uq_research_revisions_supersedes_completed",
            "supersedes_revision_id",
            unique=True,
            postgresql_where=text(SUPERSEDES_WHERE),
            sqlite_where=text(SUPERSEDES_WHERE),
        ),
        CheckConstraint(CORRECTION_REASON_CHECK, name="ck_research_revisions_correction_reason"),
        CheckConstraint(CORRECTION_LINK_CHECK, name="ck_research_revisions_correction_link"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    scope_key: Mapped[str] = mapped_column(String(256), nullable=False)
    manifest_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("data_manifests.id"), nullable=False)
    previous_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_revisions.id"), nullable=True
    )
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    runtime_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="prepared", server_default="prepared")
    research_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    # SEED-002 C3 / C2 (migration 048).
    supersedes_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_revisions.id", name="fk_research_revisions_supersedes_revision_id"),
        nullable=True,
    )
    correction_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Host-owned C2 claim precheck results; never part of research_json or content_sha256.
    precheck_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    @property
    def claim_precheck(self):
        """C2 precheck results of the completion (persisted in precheck_json since 048)."""
        return self.precheck_json
