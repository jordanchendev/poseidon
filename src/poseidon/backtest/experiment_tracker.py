"""ExperimentTracker -- DB persistence for experiment/optimization results.

Follows BacktestRepository pattern (D-06): session-based repository with
CRUD operations. Per D-07: rejected trials are recorded with status="rejected",
not discarded.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from poseidon.models.experiment import ExperimentRecord


class ExperimentTracker:
    """Repository for persisting and querying experiment records.

    Tracks Optuna trials, parameter search results, and walk-forward
    experiment outcomes. Each experiment is stored with full config,
    metrics, and optional Optuna study/trial linkage.
    """

    def __init__(self, db_session) -> None:  # noqa: ANN001
        self._db = db_session

    def save(
        self,
        *,
        study_name: str,
        config_json: dict,
        market: str,
        interval: str,
        metrics_json: dict | None = None,
        composite_score: float | None = None,
        wfe_score: float | None = None,
        status: str = "running",
        optuna_study_name: str | None = None,
        optuna_trial_number: int | None = None,
        holdout_boundary: datetime | None = None,
    ) -> uuid.UUID:
        """Persist a new experiment record.

        Args:
            study_name: Human-readable experiment name.
            config_json: Full experiment configuration as dict.
            market: Market identifier (e.g., "crypto_spot").
            interval: Time interval (e.g., "1d", "1h").
            metrics_json: Optional metrics dict.
            composite_score: Optional composite performance score.
            wfe_score: Optional Walk-Forward Efficiency score.
            status: Experiment status (default "running").
            optuna_study_name: Optional Optuna study name for linkage.
            optuna_trial_number: Optional Optuna trial number for linkage.
            holdout_boundary: Optional holdout boundary datetime.

        Returns:
            UUID of the created ExperimentRecord.
        """
        record_id = uuid.uuid4()
        record = ExperimentRecord(
            id=record_id,
            study_name=study_name,
            config_json=config_json,
            market=market,
            interval=interval,
            metrics_json=metrics_json,
            composite_score=composite_score,
            wfe_score=wfe_score,
            status=status,
            optuna_study_name=optuna_study_name,
            optuna_trial_number=optuna_trial_number,
            holdout_boundary=holdout_boundary,
        )
        self._db.add(record)
        self._db.flush()
        return record_id

    def get_by_id(self, experiment_id: uuid.UUID) -> ExperimentRecord | None:
        """Retrieve an experiment record by its ID.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            ExperimentRecord or None if not found.
        """
        return (
            self._db.query(ExperimentRecord)
            .filter(ExperimentRecord.id == experiment_id)
            .first()
        )

    def list_by_date_range(
        self, start: datetime, end: datetime, limit: int = 100
    ) -> list[ExperimentRecord]:
        """List experiments within a date range.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            limit: Maximum number of records.

        Returns:
            List of ExperimentRecord objects, newest first.
        """
        return (
            self._db.query(ExperimentRecord)
            .filter(
                ExperimentRecord.created_at >= start,
                ExperimentRecord.created_at <= end,
            )
            .order_by(ExperimentRecord.created_at.desc())
            .limit(limit)
            .all()
        )

    def list_by_market(
        self, market: str, interval: str, limit: int = 100
    ) -> list[ExperimentRecord]:
        """List experiments filtered by market and interval.

        Args:
            market: Market identifier.
            interval: Time interval.
            limit: Maximum number of records.

        Returns:
            List of ExperimentRecord objects, sorted by composite_score desc.
        """
        return (
            self._db.query(ExperimentRecord)
            .filter(
                ExperimentRecord.market == market,
                ExperimentRecord.interval == interval,
            )
            .order_by(ExperimentRecord.composite_score.desc().nulls_last())
            .limit(limit)
            .all()
        )

    def query_passed_by_study(
        self, study_name: str, limit: int = 10
    ) -> list[ExperimentRecord]:
        """Query passed experiments for a study, ranked by composite_score.

        Used by report generation (D-15) to find best configs per market.

        Args:
            study_name: Study name to filter by.
            limit: Maximum number of records to return.

        Returns:
            List of ExperimentRecord with status='passed', ordered by composite_score desc.
        """
        return (
            self._db.query(ExperimentRecord)
            .filter(
                ExperimentRecord.study_name == study_name,
                ExperimentRecord.status == "passed",
            )
            .order_by(ExperimentRecord.composite_score.desc().nulls_last())
            .limit(limit)
            .all()
        )

    def mark_rejected(self, experiment_id: uuid.UUID) -> None:
        """Mark an experiment as rejected.

        Per D-07: rejected trials are recorded, not discarded.

        Args:
            experiment_id: UUID of the experiment to reject.
        """
        record = (
            self._db.query(ExperimentRecord)
            .filter(ExperimentRecord.id == experiment_id)
            .first()
        )
        if record is not None:
            record.status = "rejected"
            self._db.flush()

    def mark_passed(self, experiment_id: uuid.UUID) -> None:
        """Mark an experiment as passed.

        Args:
            experiment_id: UUID of the experiment to pass.
        """
        record = (
            self._db.query(ExperimentRecord)
            .filter(ExperimentRecord.id == experiment_id)
            .first()
        )
        if record is not None:
            record.status = "passed"
            self._db.flush()
