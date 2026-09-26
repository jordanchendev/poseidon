"""Immutable executable version of a strategy identity."""

import hashlib
import json
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, event, func, inspect
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from poseidon.models.base import Base


def strategy_version_digest(config_json: dict, policy_json: dict, artifact_json: dict) -> str:
    payload = {"artifact_json": artifact_json, "config_json": config_json, "policy_json": policy_json}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StrategyVersion(Base):
    __tablename__ = "strategy_versions"
    __table_args__ = (
        UniqueConstraint("strategy_id", "version_no", name="uq_strategy_versions_number"),
        UniqueConstraint("strategy_id", "content_sha256", name="uq_strategy_versions_content"),
        Index("ix_strategy_versions_strategy_status", "strategy_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid()
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", server_default="draft")
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    policy_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    artifact_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def verify_content(self) -> None:
        if not all(isinstance(value, dict) for value in (self.config_json, self.policy_json, self.artifact_json)):
            raise ValueError("strategy version content must be JSON objects")
        if self.content_sha256 != strategy_version_digest(self.config_json, self.policy_json, self.artifact_json):
            raise ValueError("strategy version content hash mismatch")


@event.listens_for(StrategyVersion, "before_insert")
def _verify_strategy_version_insert(mapper, connection, target):
    state = inspect(target)
    for field in ("policy_json", "artifact_json"):
        if getattr(target, field) is None and not state.attrs[field].history.added:
            setattr(target, field, {})
    target.verify_content()


@event.listens_for(StrategyVersion, "before_update")
def _prevent_strategy_version_identity_update(mapper, connection, target):
    target.verify_content()
    state = inspect(target)
    immutable = (
        "strategy_id",
        "version_no",
        "config_json",
        "policy_json",
        "artifact_json",
        "content_sha256",
    )
    if any(state.attrs[field].history.has_changes() for field in immutable):
        raise ValueError("strategy version content is immutable")
