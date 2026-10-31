"""Status vocabulary and the only legal transitions for runs and stage attempts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    ABORTED = "aborted"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ABORTED = "aborted"


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class AttemptTrigger(StrEnum):
    ADVANCE = "advance"  # first attempt, created by the dispatcher
    RETRY = "retry"  # automatic attempt after a transient error or worker loss
    RERUN = "rerun"  # manual attempt requested through the API


RUN_TERMINAL = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.UNSUPPORTED, RunStatus.ERROR, RunStatus.ABORTED}
)
# Owner decision 2026-10-22: a gate ``fail`` closes the run for good. Only runs
# that stopped for a reason unrelated to the idea itself may be rerun.
RUN_RERUNNABLE = frozenset({RunStatus.ERROR, RunStatus.ABORTED, RunStatus.UNSUPPORTED})
RUN_ABORTABLE = frozenset({RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.WAITING})
STAGE_TERMINAL = frozenset({StageStatus.COMPLETED, StageStatus.ABORTED})
STAGE_OPEN = frozenset({StageStatus.PENDING, StageStatus.RUNNING, StageStatus.WAITING})

RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.ABORTED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.QUEUED,
            RunStatus.WAITING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.UNSUPPORTED,
            RunStatus.ERROR,
            RunStatus.ABORTED,
        }
    ),
    RunStatus.WAITING: frozenset({RunStatus.RUNNING, RunStatus.ERROR, RunStatus.ABORTED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.UNSUPPORTED: frozenset({RunStatus.QUEUED}),
    RunStatus.ERROR: frozenset({RunStatus.QUEUED}),
    RunStatus.ABORTED: frozenset({RunStatus.QUEUED}),
}

STAGE_TRANSITIONS: dict[StageStatus, frozenset[StageStatus]] = {
    StageStatus.PENDING: frozenset({StageStatus.RUNNING, StageStatus.ABORTED}),
    StageStatus.RUNNING: frozenset({StageStatus.COMPLETED, StageStatus.WAITING, StageStatus.ABORTED}),
    StageStatus.WAITING: frozenset({StageStatus.RUNNING, StageStatus.ABORTED}),
    StageStatus.COMPLETED: frozenset(),
    StageStatus.ABORTED: frozenset(),
}

# Terminal run status reached when a stage attempt completes with this verdict.
VERDICT_RUN_STATUS = {
    Verdict.FAIL: RunStatus.FAILED,
    Verdict.UNSUPPORTED: RunStatus.UNSUPPORTED,
    Verdict.ERROR: RunStatus.ERROR,
}


class IllegalTransition(ValueError):
    """A status change outside the declared state machine."""


def utcnow() -> datetime:
    return datetime.now(UTC)


def check_run_transition(old: str, new: str) -> RunStatus:
    old_status, new_status = RunStatus(old), RunStatus(new)
    if new_status not in RUN_TRANSITIONS[old_status]:
        raise IllegalTransition(f"pipeline run cannot move from {old_status} to {new_status}")
    return new_status


def check_stage_transition(old: str, new: str) -> StageStatus:
    old_status, new_status = StageStatus(old), StageStatus(new)
    if new_status not in STAGE_TRANSITIONS[old_status]:
        raise IllegalTransition(f"stage attempt cannot move from {old_status} to {new_status}")
    return new_status


def transition_run(run, new: str, *, now: datetime, reason: str | None = None) -> None:
    """Move a run, bump its optimistic-lock revision, and keep finished_at coherent."""

    status = check_run_transition(run.status, new)
    run.status = status.value
    run.revision += 1
    run.updated_at = now
    run.finished_at = now if status in RUN_TERMINAL else None
    if reason is not None:
        run.reason = reason


def touch_run(run, *, now: datetime) -> None:
    """Record a run mutation that keeps the status (e.g. abort requested while running)."""

    run.revision += 1
    run.updated_at = now


def transition_stage(
    attempt,
    new: str,
    *,
    now: datetime,
    verdict: str | None = None,
    reason: str | None = None,
) -> None:
    status = check_stage_transition(attempt.status, new)
    if (status == StageStatus.COMPLETED) != (verdict is not None):
        raise IllegalTransition("only a completed stage attempt carries a verdict")
    attempt.status = status.value
    attempt.verdict = Verdict(verdict).value if verdict is not None else None
    attempt.updated_at = now
    if status in STAGE_TERMINAL:
        attempt.finished_at = now
    if reason is not None:
        attempt.reason = reason
