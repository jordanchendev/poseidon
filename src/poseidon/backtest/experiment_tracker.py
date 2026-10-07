"""ExperimentTracker -- DB persistence for experiment/optimization results.

Follows BacktestRepository pattern: session-based repository with
CRUD operations. Rejected trials are recorded with status="rejected",
not discarded.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.exc import IntegrityError

from poseidon.decision_loop.manifest import canonical_json
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import ExperimentCampaign
from poseidon.models.strategy_version import StrategyVersion
from poseidon.research.campaign import _digest, _json_object, _lock, _required_text, _uuid


class CampaignTrialValidationError(ValueError):
    """A linked terminal-trial payload violates its frozen campaign."""


class CampaignTrialIdentityConflict(RuntimeError):
    """A paired-cell identity already contains different immutable truth."""


_TERMINAL_STATES = {
    "succeeded",
    "optimizer_failed",
    "constraint_rejected",
    "insufficient_data",
    "statistically_inconclusive",
    "capability_unavailable",
}


def _timestamp(value, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CampaignTrialValidationError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _trial_equal(row: ExperimentRecord, values: dict) -> bool:
    def normalized_time(value):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    for name in (
        "study_name",
        "config_json",
        "market",
        "interval",
        "metrics_json",
        "composite_score",
        "wfe_score",
        "status",
        "campaign_id",
        "original_trial_id",
        "trial_role",
        "strategy_version_id",
        "ablation_arm",
        "paired_sample_key_sha256",
        "input_sha256",
        "result_sha256",
        "started_at",
        "completed_at",
        "terminal_state",
        "terminal_reason_json",
    ):
        stored = getattr(row, name)
        expected = values[name]
        if name in {"started_at", "completed_at"}:
            stored = normalized_time(stored)
        elif name in {"composite_score", "wfe_score"} and stored is not None and expected is not None:
            stored, expected = Decimal(str(stored)), Decimal(str(expected))
        if stored != expected:
            return False
    return True


class ExperimentTracker:
    """Repository for persisting and querying experiment records.

    Tracks Optuna trials, parameter search results, and walk-forward
    experiment outcomes. Each experiment is stored with full config,
    metrics, and optional Optuna study/trial linkage.
    """

    def __init__(self, db_session) -> None:
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

    def append_campaign_terminal_trial(
        self,
        *,
        campaign_id,
        campaign_contract_sha256: str,
        original_trial_id: str,
        trial_role: str,
        strategy_version_id,
        ablation_arm: str,
        paired_sample_key_sha256: str,
        input_sha256: str,
        result_sha256: str,
        started_at: datetime,
        completed_at: datetime,
        terminal_state: str,
        terminal_reason_json: dict,
        study_name: str,
        config_json: dict,
        market: str,
        interval: str,
        metrics_json: dict | None = None,
        composite_score: float | None = None,
        wfe_score: float | None = None,
    ) -> ExperimentRecord:
        """Append or exactly replay one declared terminal campaign fact."""

        campaign_id = _uuid(campaign_id, "campaign_id")
        strategy_version_id = _uuid(strategy_version_id, "strategy_version_id")
        campaign_digest = _digest(campaign_contract_sha256, "campaign_contract_sha256")
        original_trial_id = _required_text(original_trial_id, "original_trial_id")
        study_name = _required_text(study_name, "study_name")
        market = _required_text(market, "market")
        interval = _required_text(interval, "interval")
        if trial_role not in {"search", "paired_evaluation"}:
            raise CampaignTrialValidationError("trial_role is invalid")
        if ablation_arm not in {"fundamental_only", "technical_only", "combined"}:
            raise CampaignTrialValidationError("ablation_arm is invalid")
        if terminal_state not in _TERMINAL_STATES:
            raise CampaignTrialValidationError("terminal_state is invalid")
        paired_digest = _digest(paired_sample_key_sha256, "paired_sample_key_sha256")
        input_digest = _digest(input_sha256, "input_sha256")
        result_digest = _digest(result_sha256, "result_sha256")
        started = _timestamp(started_at, "started_at")
        completed = _timestamp(completed_at, "completed_at")
        if completed < started:
            raise CampaignTrialValidationError("completed_at must not precede started_at")
        reason = _json_object(terminal_reason_json, "terminal_reason_json", nonempty=False)
        config = _json_object(config_json, "config_json", nonempty=False)
        if metrics_json is not None and not isinstance(metrics_json, dict):
            raise CampaignTrialValidationError("metrics_json must be an object or null")
        metrics = None if metrics_json is None else json.loads(canonical_json(metrics_json))

        campaign = self._db.get(ExperimentCampaign, campaign_id)
        if campaign is None or campaign.contract_sha256 != campaign_digest:
            raise CampaignTrialValidationError("campaign and exact contract hash are required")
        version = self._db.get(StrategyVersion, strategy_version_id)
        if version is None:
            raise CampaignTrialValidationError("strategy version does not exist")
        declared = campaign.contract_json.get("declared_trials", [])
        if not any(
            item.get("original_trial_id") == original_trial_id
            and item.get("trial_role") == trial_role
            and item.get("strategy_version_id") == str(strategy_version_id)
            and item.get("ablation_arm") == ablation_arm
            for item in declared
            if isinstance(item, dict)
        ):
            raise CampaignTrialValidationError("trial is not declared by the frozen campaign")

        identity = canonical_json(
            [
                str(campaign_id),
                original_trial_id,
                str(strategy_version_id),
                ablation_arm,
                paired_digest,
            ]
        )
        _lock(self._db, "phase99:paired-cell:", identity)
        query = self._db.query(ExperimentRecord).filter_by(
            campaign_id=campaign_id,
            original_trial_id=original_trial_id,
            strategy_version_id=strategy_version_id,
            ablation_arm=ablation_arm,
            paired_sample_key_sha256=paired_digest,
        )
        values = {
            "study_name": study_name,
            "config_json": config,
            "market": market,
            "interval": interval,
            "metrics_json": metrics,
            "composite_score": composite_score,
            "wfe_score": wfe_score,
            "status": "complete",
            "campaign_id": campaign_id,
            "original_trial_id": original_trial_id,
            "trial_role": trial_role,
            "strategy_version_id": strategy_version_id,
            "ablation_arm": ablation_arm,
            "paired_sample_key_sha256": paired_digest,
            "input_sha256": input_digest,
            "result_sha256": result_digest,
            "started_at": started,
            "completed_at": completed,
            "terminal_state": terminal_state,
            "terminal_reason_json": reason,
        }
        existing = query.one_or_none()
        if existing is not None:
            if not _trial_equal(existing, values):
                raise CampaignTrialIdentityConflict("paired-cell identity identifies different immutable content")
            return existing

        try:
            with self._db.begin_nested():
                record = ExperimentRecord(id=uuid.uuid4(), **values)
                self._db.add(record)
                self._db.flush()
            return record
        except IntegrityError as error:
            existing = query.one_or_none()
            if existing is None:
                raise
            if not _trial_equal(existing, values):
                raise CampaignTrialIdentityConflict(
                    "paired-cell identity identifies different immutable content"
                ) from error
            return existing

    def get_by_id(self, experiment_id: uuid.UUID) -> ExperimentRecord | None:
        """Retrieve an experiment record by its ID.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            ExperimentRecord or None if not found.
        """
        return self._db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()

    def list_by_date_range(self, start: datetime, end: datetime, limit: int = 100) -> list[ExperimentRecord]:
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

    def list_by_market(self, market: str, interval: str, limit: int = 100) -> list[ExperimentRecord]:
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

    def query_passed_by_study(self, study_name: str, limit: int = 10) -> list[ExperimentRecord]:
        """Query passed experiments for a study, ranked by composite_score.

        Used by report generation to find best configs per market.

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

        Rejected trials are recorded, not discarded.

        Args:
            experiment_id: UUID of the experiment to reject.
        """
        record = self._db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
        if record is not None:
            if record.campaign_id is not None:
                raise ValueError("campaign-linked experiments are append-only")
            record.status = "rejected"
            self._db.flush()

    def mark_passed(self, experiment_id: uuid.UUID) -> None:
        """Mark an experiment as passed.

        Args:
            experiment_id: UUID of the experiment to pass.
        """
        record = self._db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
        if record is not None:
            if record.campaign_id is not None:
                raise ValueError("campaign-linked experiments are append-only")
            record.status = "passed"
            self._db.flush()
