"""Trial ledger writers for ``pipeline_trials`` (append-only, DSR's N).

A stage appends every trial the moment it finishes, each in its own short
transaction, so trials of an attempt that later errors, is aborted or loses
its lease still count (owner decision: DSR counts them, conservatively).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from poseidon.decision_loop.manifest import content_sha256

TRIAL_COMPLETE = "complete"
TRIAL_PRUNED = "pruned"  # ran, but produced no Sharpe (no trades / zero-variance daily PnL)
TRIAL_FAIL = "fail"  # the backtest raised or the parameters did not validate


@dataclass(frozen=True)
class TrialRecord:
    trial_no: int
    params: dict
    params_sha256: str
    state: str
    sharpe: float | None
    n_obs: int | None
    skew: float | None
    kurtosis: float | None
    window_sha256: str


def make_record(trial_no: int, params: dict, state: str, stats: dict | None, window_sha256: str) -> TrialRecord:
    stats = stats or {}
    return TrialRecord(
        trial_no=trial_no,
        params=dict(params),
        params_sha256=content_sha256(params),
        state=state,
        sharpe=stats.get("sharpe_daily"),
        n_obs=stats.get("n_obs"),
        skew=stats.get("skew"),
        kurtosis=stats.get("kurtosis"),
        window_sha256=window_sha256,
    )


class MemoryTrialLedger:
    """In-process ledger for direct stage calls and unit tests."""

    def __init__(self):
        self.records: list[TrialRecord] = []

    def append(self, record: TrialRecord) -> None:
        if any(existing.trial_no == record.trial_no for existing in self.records):
            raise ValueError(f"trial {record.trial_no} already recorded")
        self.records.append(record)


class DbTrialLedger:
    """Writes each trial to ``pipeline_trials`` in its own committed transaction."""

    def __init__(self, session_factory: Callable[[], Session], *, run_id, family_id, stage_run_id):
        self.session_factory = session_factory
        self.run_id = uuid.UUID(str(run_id))
        self.family_id = uuid.UUID(str(family_id))
        self.stage_run_id = uuid.UUID(str(stage_run_id))

    def append(self, record: TrialRecord) -> None:
        from poseidon.pipeline.models import PipelineTrial

        with self.session_factory() as session:
            session.add(
                PipelineTrial(
                    family_id=self.family_id,
                    run_id=self.run_id,
                    stage_run_id=self.stage_run_id,
                    trial_no=record.trial_no,
                    params_json=record.params,
                    params_sha256=record.params_sha256,
                    state=record.state,
                    sharpe=record.sharpe,
                    n_obs=record.n_obs,
                    skew=record.skew,
                    kurtosis=record.kurtosis,
                    window_sha256=record.window_sha256,
                )
            )
            session.commit()
