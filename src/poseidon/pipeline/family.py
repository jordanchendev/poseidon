"""What a stage may know about the rest of its run's family.

``dsr`` needs every ``pipeline_trials`` row of the family (N and the trial
Sharpe variance, owner decision: errored and aborted attempts included);
``holdout`` records each look at the locked holdout as a ``holdout_evaluated``
event and reports how many other runs of the family already looked. The
dispatcher hands stages a :class:`DbFamily`; direct stage calls and unit tests
use :class:`MemoryFamily`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

HOLDOUT_EVALUATED = "holdout_evaluated"
HOLDOUT_ACTOR = "system:pipeline-stage:holdout"


class MemoryFamily:
    """In-process family view: trials from a ``MemoryTrialLedger``, holdout looks in a list."""

    def __init__(self, ledger=None, *, run_id=None, holdout_runs: list | None = None):
        self.ledger = ledger
        self.run_id = run_id
        self.holdout_runs: list = list(holdout_runs or [])
        self.events: list[dict] = []

    def trials(self) -> list:
        return list(getattr(self.ledger, "records", []))

    def holdout_evaluations(self) -> dict:
        others = sorted({str(run) for run in self.holdout_runs if run != self.run_id})
        return {
            "other_runs": len(others),
            "other_run_ids": others,
            "this_run_previous_evaluations": sum(1 for run in self.holdout_runs if run == self.run_id),
        }

    def record_holdout_evaluated(self, payload: dict) -> None:
        self.holdout_runs.append(self.run_id)
        self.events.append({"event_type": HOLDOUT_EVALUATED, "payload": payload})


class DbFamily:
    """Reads the family's ledger / events and appends holdout events, each in its own transaction."""

    def __init__(self, session_factory: Callable[[], Session], *, run_id, family_id, stage_run_id):
        self.session_factory = session_factory
        self.run_id = uuid.UUID(str(run_id))
        self.family_id = uuid.UUID(str(family_id))
        self.stage_run_id = uuid.UUID(str(stage_run_id))

    def trials(self) -> list:
        from poseidon.pipeline.models import PipelineTrial

        with self.session_factory() as session:
            rows = session.scalars(
                select(PipelineTrial)
                .where(PipelineTrial.family_id == self.family_id)
                .order_by(PipelineTrial.created_at, PipelineTrial.trial_no)
            ).all()
            for row in rows:
                session.expunge(row)
            return list(rows)

    def holdout_evaluations(self) -> dict:
        from poseidon.pipeline.models import PipelineEvent, PipelineRun

        with self.session_factory() as session:
            rows = session.execute(
                select(PipelineEvent.run_id, func.count())
                .join(PipelineRun, PipelineRun.id == PipelineEvent.run_id)
                .where(PipelineRun.family_id == self.family_id, PipelineEvent.event_type == HOLDOUT_EVALUATED)
                .group_by(PipelineEvent.run_id)
            ).all()
        others = sorted(str(run) for run, _count in rows if run != self.run_id)
        return {
            "other_runs": len(others),
            "other_run_ids": others[:50],
            "this_run_previous_evaluations": int(sum(count for run, count in rows if run == self.run_id)),
        }

    def record_holdout_evaluated(self, payload: dict) -> None:
        from poseidon.pipeline.models import PipelineEvent, PipelineRun

        with self.session_factory() as session:
            revision = session.scalar(select(PipelineRun.revision).where(PipelineRun.id == self.run_id))
            session.add(
                PipelineEvent(
                    run_id=self.run_id,
                    stage_run_id=self.stage_run_id,
                    event_type=HOLDOUT_EVALUATED,
                    actor_id=HOLDOUT_ACTOR,
                    expected_revision=int(revision),
                    payload_json=payload,
                )
            )
            session.commit()
