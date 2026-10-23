"""The single idempotent pipeline driver and its recovery sweeper.

``advance(run_id)``:

1. Txn A -- lock the run ``FOR UPDATE SKIP LOCKED``, pick the next stage, claim
   one attempt with a lease, record an event, commit.
2. Run the stage outside any lock; a heartbeat thread extends the lease every
   ``heartbeat_seconds`` and tells the stage to stop on abort / lost lease.
3. Txn B -- lock the run again, write verdict + artifact only if the lease
   owner is unchanged and no abort was requested (the fence), commit, then
   enqueue the next ``advance``.

Every caller (Celery task, sweeper, API, tests) goes through this one path, so
a redelivered or duplicated message is a no-op rather than a second attempt.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from poseidon.core.config import settings
from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline import artifacts
from poseidon.pipeline.family import DbFamily
from poseidon.pipeline.models import PipelineEvent, PipelineRun, StageRun
from poseidon.pipeline.plan import completion_reason
from poseidon.pipeline.stages import (
    STAGE_REGISTRY,
    StageContext,
    StageResult,
    StageStopped,
    TransientStageError,
    UpstreamResult,
)
from poseidon.pipeline.state import (
    RUN_TERMINAL,
    VERDICT_RUN_STATUS,
    AttemptTrigger,
    RunStatus,
    StageStatus,
    Verdict,
    transition_run,
    transition_stage,
    utcnow,
)
from poseidon.pipeline.trials import DbTrialLedger

logger = logging.getLogger(__name__)

DISPATCHER_ACTOR = "system:pipeline-dispatcher"
SWEEPER_ACTOR = "system:pipeline-sweeper"
HEARTBEAT_SECONDS = 60
QUEUED_IDLE_SECONDS = 300
ERROR_MESSAGE_LIMIT = 2000


def default_enqueue(run_id: uuid.UUID) -> None:
    from poseidon.pipeline.tasks import enqueue_advance

    enqueue_advance(run_id)


def latest_attempts(attempts) -> dict[str, StageRun]:
    latest: dict[str, StageRun] = {}
    for row in attempts:
        if row.stage not in latest or row.attempt > latest[row.stage].attempt:
            latest[row.stage] = row
    return latest


def select_next_stage(stage_plan, attempts) -> tuple[str | None, StageRun | None]:
    """First stage whose latest attempt is not a pass; upstream passes are reused."""

    latest = latest_attempts(attempts)
    for stage in stage_plan:
        row = latest.get(stage)
        if row is not None and row.status == StageStatus.COMPLETED and row.verdict == Verdict.PASS:
            continue
        return stage, row
    return None, None


def retry_allowed(attempts, current: StageRun, max_auto_attempts: int) -> bool:
    """At most ``max_auto_attempts`` consecutive automatic attempts per manual start."""

    by_number = {row.attempt: row for row in attempts if row.stage == current.stage}
    chain, number = 1, current.attempt
    while by_number.get(number) is not None and by_number[number].trigger == AttemptTrigger.RETRY:
        chain += 1
        number -= 1
    return chain < max_auto_attempts


def add_event(session: Session, run: PipelineRun, event_type: str, actor_id: str, payload: dict, **extra) -> None:
    session.add(
        PipelineEvent(
            run_id=run.id,
            event_type=event_type,
            actor_id=actor_id,
            expected_revision=run.revision,
            payload_json=payload,
            **extra,
        )
    )


@dataclass
class Claim:
    run_id: uuid.UUID
    stage_run_id: uuid.UUID
    stage: str
    attempt: int
    lease_owner: uuid.UUID
    lease_seconds: int
    context: StageContext


class PipelineDispatcher:
    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        *,
        registry=None,
        enqueue: Callable[[uuid.UUID], None] | None = None,
        artifacts_root=None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        clock: Callable[[], datetime] = utcnow,
        code_version: str | None = None,
    ):
        if session_factory is None:
            from poseidon.core.database import SessionLocal

            session_factory = SessionLocal
        self.session_factory = session_factory
        self.registry = registry if registry is not None else STAGE_REGISTRY
        self.enqueue = enqueue or default_enqueue
        self.artifacts_root = artifacts.artifacts_root(artifacts_root)
        self.heartbeat_seconds = heartbeat_seconds
        self.clock = clock
        self.code_version = (code_version or settings.pipeline_code_version or "unversioned")[:64]

    # ------------------------------------------------------------------ advance

    def advance(self, run_id) -> dict:
        run_id = uuid.UUID(str(run_id))
        claim, outcome = self._claim(run_id)
        if claim is None:
            if outcome.pop("enqueue", False):
                self.enqueue(run_id)
            return outcome
        result, failure = self._execute(claim)
        outcome = self._commit(claim, result, failure)
        if outcome.pop("enqueue", False):
            self.enqueue(run_id)
        return outcome

    def _lock_run(self, session: Session, run_id, *, skip_locked: bool) -> PipelineRun | None:
        return (
            session.query(PipelineRun)
            .filter(PipelineRun.id == run_id)
            .with_for_update(skip_locked=skip_locked)
            .populate_existing()
            .one_or_none()
        )

    def _claim(self, run_id) -> tuple[Claim | None, dict]:
        """Txn A."""

        with self.session_factory() as session:
            run = self._lock_run(session, run_id, skip_locked=True)
            if run is None:
                exists = session.scalar(select(PipelineRun.id).where(PipelineRun.id == run_id))
                return None, {"run_id": str(run_id), "skipped": "locked" if exists else "not_found"}
            base = {"run_id": str(run_id)}
            if run.status in RUN_TERMINAL:
                return None, {**base, "skipped": "terminal", "status": run.status}
            if run.abort_requested_at is not None:
                return None, {**base, "skipped": "abort_requested"}
            now = self.clock()
            attempts = session.query(StageRun).filter(StageRun.run_id == run.id).with_for_update().all()
            if any(row.status == StageStatus.RUNNING for row in attempts):
                return None, {**base, "skipped": "running"}
            plan = run.config_json["stage_plan"]
            stage_name, latest = select_next_stage(plan, attempts)
            if stage_name is None:
                transition_run(run, RunStatus.COMPLETED, now=now, reason=completion_reason(plan))
                run.current_stage = None
                add_event(session, run, "run_completed", DISPATCHER_ACTOR, {"stage_plan": plan})
                session.commit()
                return None, {**base, "status": RunStatus.COMPLETED.value}
            stage = self.registry.get(stage_name)
            if stage is None:
                return None, self._fail_run_locked(session, run, now, f"stage {stage_name} is not registered")

            if latest is not None and latest.status == StageStatus.WAITING:
                if latest.wait_until > now:
                    return None, {**base, "skipped": "waiting", "wait_until": latest.wait_until.isoformat()}
                row = latest
            elif latest is not None and latest.status == StageStatus.PENDING:
                row = latest
            elif latest is None:
                row = StageRun(
                    id=uuid.uuid4(),
                    run_id=run.id,
                    stage=stage_name,
                    seq=plan.index(stage_name),
                    attempt=1,
                    trigger=AttemptTrigger.ADVANCE.value,
                    status=StageStatus.PENDING.value,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                # Latest attempt completed without a pass but the run is not
                # terminal: only a rerun may continue from here.
                return None, self._fail_run_locked(
                    session, run, now, f"stage {stage_name} attempt {latest.attempt} ended {latest.verdict}"
                )

            upstream = self._upstream(attempts, stage)
            context = StageContext(
                run_id=run.id,
                family_id=run.family_id,
                idea_text=run.idea_text,
                input_spec_json=run.input_spec_json,
                config=run.config_json,
                config_sha256=run.config_sha256,
                stage=stage_name,
                attempt=row.attempt,
                upstream=upstream,
                stage_run_id=row.id,
                artifacts_root=self.artifacts_root,
                trials=DbTrialLedger(self.session_factory, run_id=run.id, family_id=run.family_id, stage_run_id=row.id),
                family=DbFamily(self.session_factory, run_id=run.id, family_id=run.family_id, stage_run_id=row.id),
            )
            input_sha256 = content_sha256(stage.input_payload(context))
            if row.input_sha256 is not None and row.input_sha256 != input_sha256:
                return None, self._fail_run_locked(session, run, now, f"stage {stage_name} input changed")
            context.input_sha256 = input_sha256

            lease_owner = uuid.uuid4()
            row.input_sha256 = input_sha256
            row.code_version = self.code_version
            row.lease_owner = lease_owner
            row.lease_expires_at = now + timedelta(seconds=stage.lease_seconds)
            row.heartbeat_at = now
            row.started_at = row.started_at or now
            row.wait_until = None
            transition_stage(row, StageStatus.RUNNING, now=now)
            if run.status != RunStatus.RUNNING:
                transition_run(run, RunStatus.RUNNING, now=now)
            run.current_stage = stage_name
            session.flush()
            add_event(
                session,
                run,
                "stage_claimed",
                DISPATCHER_ACTOR,
                {
                    "stage": stage_name,
                    "attempt": row.attempt,
                    "input_sha256": input_sha256,
                    "code_version": self.code_version,
                    "lease_expires_at": row.lease_expires_at.isoformat(),
                },
                stage_run_id=row.id,
            )
            session.commit()
            return (
                Claim(run.id, row.id, stage_name, row.attempt, lease_owner, stage.lease_seconds, context),
                {},
            )

    def _upstream(self, attempts, stage) -> dict[str, UpstreamResult]:
        latest = latest_attempts(attempts)
        upstream = {}
        for name in stage.requires:
            row = latest[name]  # validate_stage_plan guarantees a passing attempt exists
            artifact = row.artifact_json or {}
            upstream[name] = UpstreamResult(
                name,
                row.attempt,
                row.input_sha256,
                row.artifact_sha256,
                artifact.get("report"),
                artifact_uri=row.artifact_uri,
                files=tuple(artifact.get("files") or ()),
            )
        return upstream

    def _fail_run_locked(self, session, run, now, reason) -> dict:
        transition_run(run, RunStatus.ERROR, now=now, reason=reason)
        add_event(session, run, "run_error", DISPATCHER_ACTOR, {"reason": reason})
        session.commit()
        return {"run_id": str(run.id), "status": RunStatus.ERROR.value, "reason": reason}

    # ------------------------------------------------------------------ execute

    def _execute(self, claim: Claim) -> tuple[StageResult | None, dict | None]:
        stop = threading.Event()
        done = threading.Event()

        def heartbeat():
            while not done.wait(self.heartbeat_seconds):
                try:
                    if not self._heartbeat(claim):
                        stop.set()
                except Exception:  # a failed heartbeat must not kill the stage thread
                    logger.exception("pipeline heartbeat failed for %s", claim.stage_run_id)

        claim.context.should_stop = stop.is_set
        beat = threading.Thread(target=heartbeat, name=f"pipeline-heartbeat-{claim.stage_run_id}", daemon=True)
        beat.start()
        try:
            stage = self.registry[claim.stage]
            result = stage.run(claim.context)
            if not isinstance(result, StageResult):
                raise TypeError(f"stage {claim.stage} returned {type(result).__name__}, not StageResult")
            return result, None
        except TransientStageError as error:
            return None, self._failure("transient", error)
        except StageStopped as error:
            return None, self._failure("stopped", error)
        except Exception as error:
            return None, self._failure("exception", error)
        finally:
            done.set()
            beat.join()

    @staticmethod
    def _failure(kind: str, error: BaseException) -> dict:
        return {
            "kind": kind,
            "type": type(error).__name__,
            "message": str(error)[:ERROR_MESSAGE_LIMIT],
            "traceback": "".join(traceback.format_exception(error))[-ERROR_MESSAGE_LIMIT:],
        }

    def _heartbeat(self, claim: Claim) -> bool:
        """Extend the lease; ``False`` means stop (abort requested or lease lost)."""

        now = self.clock()
        with self.session_factory() as session:
            extended = session.execute(
                update(StageRun)
                .where(
                    StageRun.id == claim.stage_run_id,
                    StageRun.lease_owner == claim.lease_owner,
                    StageRun.status == StageStatus.RUNNING.value,
                )
                .values(
                    lease_expires_at=now + timedelta(seconds=claim.lease_seconds),
                    heartbeat_at=now,
                    updated_at=now,
                )
            ).rowcount
            abort_requested = session.scalar(
                select(PipelineRun.abort_requested_at).where(PipelineRun.id == claim.run_id)
            )
            session.commit()
        return bool(extended) and abort_requested is None

    # ------------------------------------------------------------------ commit

    def _commit(self, claim: Claim, result: StageResult | None, failure: dict | None) -> dict:
        """Txn B (fenced)."""

        manifest: list[dict] = []
        artifact_uri = None
        artifact_json = artifact_sha256 = None
        if result is not None and failure is None:
            try:
                if result.files:
                    directory = artifacts.attempt_dir(self.artifacts_root, claim.run_id, claim.stage, claim.attempt)
                    manifest = artifacts.write_files(directory, result.files)
                    artifact_uri = directory.as_uri()
                artifact_json, artifact_sha256 = artifacts.build_artifact(result.report, manifest)
            except (OSError, ValueError) as error:
                result, failure = None, self._failure("artifact", error)

        base = {"run_id": str(claim.run_id), "stage": claim.stage, "attempt": claim.attempt}
        with self.session_factory() as session:
            run = self._lock_run(session, claim.run_id, skip_locked=False)
            row = (
                session.query(StageRun)
                .filter(StageRun.id == claim.stage_run_id)
                .with_for_update()
                .populate_existing()
                .one()
            )
            now = self.clock()
            if row.status != StageStatus.RUNNING or row.lease_owner != claim.lease_owner:
                add_event(
                    session,
                    run,
                    "stage_result_fenced",
                    DISPATCHER_ACTOR,
                    {**base, "why": "lease_lost", "attempt_status": row.status},
                    stage_run_id=row.id,
                )
                session.commit()
                return {**base, "fenced": "lease_lost"}
            if run.abort_requested_at is not None:
                transition_stage(row, StageStatus.ABORTED, now=now, reason="abort requested")
                transition_run(run, RunStatus.ABORTED, now=now)
                run.current_stage = claim.stage
                add_event(
                    session,
                    run,
                    "stage_aborted",
                    DISPATCHER_ACTOR,
                    {**base, "why": "abort_requested", "result_discarded": True},
                    stage_run_id=row.id,
                )
                session.commit()
                return {**base, "fenced": "aborted", "status": RunStatus.ABORTED.value}

            attempts = session.query(StageRun).filter(StageRun.run_id == run.id).all()
            enqueue = False
            if failure is not None:
                row.error_json = failure
                reason = f"{failure['kind']}: {failure['type']}: {failure['message']}"[:ERROR_MESSAGE_LIMIT]
                transition_stage(row, StageStatus.COMPLETED, now=now, verdict=Verdict.ERROR, reason=reason)
                stage = self.registry[claim.stage]
                if failure["kind"] == "transient" and retry_allowed(attempts, row, stage.max_auto_attempts):
                    session.flush()
                    self._add_retry(session, run, row, now)
                    transition_run(run, RunStatus.QUEUED, now=now)
                    enqueue = True
                else:
                    transition_run(run, RunStatus.ERROR, now=now, reason=f"{claim.stage}: {reason}")
                verdict = Verdict.ERROR
            elif result.wait_until is not None:
                row.wait_until = result.wait_until
                transition_stage(row, StageStatus.WAITING, now=now, reason=result.reason)
                transition_run(run, RunStatus.WAITING, now=now)
                verdict = None
            else:
                verdict = Verdict(result.verdict)
                row.artifact_json = artifact_json
                row.artifact_sha256 = artifact_sha256
                row.artifact_uri = artifact_uri
                transition_stage(row, StageStatus.COMPLETED, now=now, verdict=verdict, reason=result.reason)
                if verdict == Verdict.PASS:
                    session.flush()
                    next_stage, _ = select_next_stage(run.config_json["stage_plan"], attempts)
                    if next_stage is None:
                        transition_run(
                            run, RunStatus.COMPLETED, now=now, reason=completion_reason(run.config_json["stage_plan"])
                        )
                        run.current_stage = None
                    else:
                        transition_run(run, RunStatus.QUEUED, now=now)
                        run.current_stage = next_stage
                        enqueue = True
                else:
                    detail = result.reason or verdict.value
                    transition_run(run, VERDICT_RUN_STATUS[verdict], now=now, reason=f"{claim.stage}: {detail}")
            add_event(
                session,
                run,
                "stage_completed" if verdict is not None else "stage_waiting",
                DISPATCHER_ACTOR,
                {
                    **base,
                    "verdict": verdict.value if verdict is not None else None,
                    "reason": row.reason,
                    "input_sha256": row.input_sha256,
                    "artifact_sha256": row.artifact_sha256,
                    "run_status": run.status,
                },
                stage_run_id=row.id,
            )
            session.commit()
            return {**base, "verdict": verdict.value if verdict else None, "status": run.status, "enqueue": enqueue}

    def _add_retry(self, session, run, previous: StageRun, now) -> StageRun:
        retry = StageRun(
            id=uuid.uuid4(),
            run_id=run.id,
            stage=previous.stage,
            seq=previous.seq,
            attempt=previous.attempt + 1,
            trigger=AttemptTrigger.RETRY.value,
            status=StageStatus.PENDING.value,
            created_at=now,
            updated_at=now,
        )
        session.add(retry)
        return retry

    # ------------------------------------------------------------------ sweep

    def sweep(self, *, now: datetime | None = None, queued_idle_seconds: float = QUEUED_IDLE_SECONDS) -> dict:
        now = now or self.clock()
        to_enqueue: set[uuid.UUID] = set()
        lost = 0
        with self.session_factory() as session:
            expired = session.execute(
                select(StageRun.id, StageRun.run_id).where(
                    StageRun.status == StageStatus.RUNNING.value, StageRun.lease_expires_at < now
                )
            ).all()
        for stage_run_id, run_id in expired:
            outcome = self._recover_lost(stage_run_id, run_id, now)
            if outcome is not None:
                lost += 1
                if outcome:
                    to_enqueue.add(run_id)
        with self.session_factory() as session:
            idle = session.scalars(
                select(PipelineRun.id).where(
                    PipelineRun.status == RunStatus.QUEUED.value,
                    PipelineRun.updated_at < now - timedelta(seconds=queued_idle_seconds),
                )
            ).all()
            due = session.scalars(
                select(StageRun.run_id).where(StageRun.status == StageStatus.WAITING.value, StageRun.wait_until <= now)
            ).all()
        to_enqueue.update(idle)
        to_enqueue.update(due)
        for run_id in sorted(to_enqueue, key=str):
            self.enqueue(run_id)
        return {"worker_lost": lost, "requeued_idle": len(idle), "waiting_due": len(due), "enqueued": len(to_enqueue)}

    def _recover_lost(self, stage_run_id, run_id, now) -> bool | None:
        """Close an expired lease. Returns None if skipped, else whether to enqueue."""

        with self.session_factory() as session:
            run = self._lock_run(session, run_id, skip_locked=True)
            if run is None:
                return None  # an advance holds the run; it will see the lease state itself
            row = (
                session.query(StageRun).filter(StageRun.id == stage_run_id).with_for_update().populate_existing().one()
            )
            if row.status != StageStatus.RUNNING or row.lease_expires_at >= now:
                session.rollback()
                return None
            error = {
                "kind": "worker_lost",
                "lease_owner": str(row.lease_owner),
                "lease_expires_at": row.lease_expires_at.isoformat(),
                "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
            }
            base = {"stage": row.stage, "attempt": row.attempt}
            if run.abort_requested_at is not None:
                row.error_json = error
                transition_stage(row, StageStatus.ABORTED, now=now, reason="worker_lost after abort request")
                transition_run(run, RunStatus.ABORTED, now=now)
                add_event(session, run, "stage_aborted", SWEEPER_ACTOR, {**base, **error}, stage_run_id=row.id)
                session.commit()
                return False
            attempts = session.query(StageRun).filter(StageRun.run_id == run.id).all()
            row.error_json = error
            transition_stage(row, StageStatus.COMPLETED, now=now, verdict=Verdict.ERROR, reason="worker_lost")
            stage = self.registry.get(row.stage)
            session.flush()
            enqueue = False
            if stage is not None and retry_allowed(attempts, row, stage.max_auto_attempts):
                retry = self._add_retry(session, run, row, now)
                transition_run(run, RunStatus.QUEUED, now=now)
                payload = {**base, **error, "retry_attempt": retry.attempt}
                enqueue = True
            else:
                transition_run(run, RunStatus.ERROR, now=now, reason=f"{row.stage}: worker_lost")
                payload = {**base, **error, "retry_attempt": None}
            add_event(session, run, "stage_worker_lost", SWEEPER_ACTOR, payload, stage_run_id=row.id)
            session.commit()
            return enqueue
