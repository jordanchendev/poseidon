"""ModelManager — CRUD operations and lifecycle management for model versions."""

import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from poseidon.ml import artifacts
from poseidon.ml.lifecycle import InvalidTransitionError, validate_transition
from poseidon.models.model_version import ModelVersion

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model version creation, lifecycle transitions, and artifact coordination."""

    def __init__(self, session: Session):
        self.session = session

    def create_version(
        self,
        name: str,
        params: dict,
        feature_list: list[str],
        train_start: datetime | None = None,
        train_end: datetime | None = None,
    ) -> ModelVersion:
        """Create a new model version in 'training' state.

        Auto-increments the version number for the given model name.
        """
        # Get next version number
        max_version = self.session.execute(
            select(func.max(ModelVersion.version)).where(ModelVersion.name == name)
        ).scalar()
        next_version = (max_version or 0) + 1

        # Prepare artifact directory
        artifact_path = str(artifacts.ensure_version_dir(name, next_version))

        mv = ModelVersion(
            name=name,
            version=next_version,
            status="training",
            params=params,
            feature_list=feature_list,
            train_start=train_start,
            train_end=train_end,
            artifact_path=artifact_path,
        )
        self.session.add(mv)
        self.session.commit()
        self.session.refresh(mv)
        logger.info("Created model version: %s v%d (id=%s)", name, next_version, mv.id)
        return mv

    def transition(self, version_id: UUID, target_status: str, metrics: dict | None = None, error_message: str | None = None) -> ModelVersion:
        """Transition a model version to a new lifecycle state.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
            ValueError: If the version is not found.
        """
        mv = self.session.get(ModelVersion, version_id)
        if mv is None:
            raise ValueError(f"Model version not found: {version_id}")

        validate_transition(mv.status, target_status)

        # If activating, retire any currently active version of the same model
        if target_status == "active":
            self._retire_active(mv.name, exclude_id=version_id)
            artifacts.set_active_symlink(mv.name, mv.version)

        mv.status = target_status
        if metrics is not None:
            mv.metrics = metrics
        if error_message is not None:
            mv.error_message = error_message

        self.session.commit()
        self.session.refresh(mv)
        logger.info("Transitioned %s v%d: -> %s", mv.name, mv.version, target_status)
        return mv

    def get_version(self, version_id: UUID) -> ModelVersion | None:
        """Get a model version by ID."""
        return self.session.get(ModelVersion, version_id)

    def get_active(self, name: str) -> ModelVersion | None:
        """Get the active version of a model by name."""
        return self.session.execute(
            select(ModelVersion).where(
                ModelVersion.name == name, ModelVersion.status == "active"
            )
        ).scalar_one_or_none()

    def list_versions(self, name: str) -> list[ModelVersion]:
        """List all versions of a model, ordered by version number."""
        return list(
            self.session.execute(
                select(ModelVersion)
                .where(ModelVersion.name == name)
                .order_by(ModelVersion.version)
            ).scalars()
        )

    def _retire_active(self, name: str, exclude_id: UUID) -> None:
        """Retire any currently active version (except exclude_id)."""
        active = self.session.execute(
            select(ModelVersion).where(
                ModelVersion.name == name,
                ModelVersion.status == "active",
                ModelVersion.id != exclude_id,
            )
        ).scalars().all()
        for mv in active:
            mv.status = "retired"
            logger.info("Auto-retired %s v%d", mv.name, mv.version)
