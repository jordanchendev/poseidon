"""Step 7 against PostgreSQL: shadow_paper and awaiting_release under the real dispatcher + sweeper."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from poseidon.pipeline.plan import GATE_STAGES, V1_STAGE_PLAN
from poseidon.pipeline.shadow_paper import AWAITING_RELEASE_REASON, ShadowPaperStage
from tests.pipeline.conftest import Harness, SyntheticTxSource, synthetic_registry
from tests.pipeline.gate_support import passing, stub_train_gates
from tests.pipeline.paper_support import PAPER_START, STAGE_START, PaperSource

pytestmark = pytest.mark.usefixtures("pipeline_engine")


class WideTrainSource(SyntheticTxSource):
    """The conftest synthetic window with every price move ``scale`` x wider (bigger train drawdowns)."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def fetch(self, symbol, market, interval, start, end):
        frame = super().fetch(symbol, market, interval, start, end).copy()
        for column in ("open", "high", "low", "close"):
            frame[column] = 23000.0 + self.scale * (frame[column] - 23000.0)
        return frame


def _harness(pipeline_sessions, tmp_path, clock, source, *, train_scale: float = 1.0):
    """Real spec/backtest/optimize on synthetic bars, train gates + holdout stubbed to pass, real shadow_paper."""

    registry = stub_train_gates(holdout=passing("holdout", ("optimize", *GATE_STAGES[:5])))
    if train_scale != 1.0:
        registry.update(synthetic_registry(WideTrainSource(train_scale), **{k: registry[k] for k in GATE_STAGES}))
    registry["shadow_paper"] = ShadowPaperStage(source)
    clock.now = STAGE_START  # Thu 2026-10-22 10:00 Asia/Taipei
    return Harness(pipeline_sessions, tmp_path / "artifacts", clock, registry=registry)


def _latest(harness, run_id, stage):
    rows = [row for row in harness.attempts(run_id) if row.stage == stage]
    return max(rows, key=lambda row: row.attempt) if rows else None


def _abort(harness, run_id, reason="owner abort"):
    from poseidon.pipeline.service import PipelineService

    with harness.sessions() as session:
        run = harness.run(run_id)
        PipelineService(session, registry=harness.dispatcher.registry, clock=harness.clock).abort(
            run_id,
            {"expected_revision": run.revision, "reason": reason},
            actor_id="human:researcher",
            idempotency_key=uuid.uuid4().hex,
        )
        session.commit()


def _wake(harness, source, row):
    """Jump the clock to the attempt's wait_until, let Thalassa catch up, sweep, drain."""

    harness.clock.now = row.wait_until
    source.at(harness.clock.now)
    swept = harness.dispatcher.sweep()
    harness.drain()
    return swept


def _walk(harness, source, run_id, limit=40) -> int:
    wakes = 0
    while (row := _latest(harness, run_id, "shadow_paper")).status == "waiting":
        _wake(harness, source, row)
        wakes += 1
        assert wakes <= limit, "shadow paper never finished"
    return wakes


def test_end_to_end_holdout_pass_shadow_paper_awaiting_release(pipeline_sessions, tmp_path, clock):
    # The protection limits come from 15 synthetic train days; at the paper's volatility those
    # limits trip within days (see the next test), so the train bars here move 5x wider.
    source = PaperSource(vol=3.0)
    harness = _harness(pipeline_sessions, tmp_path, clock, source, train_scale=5.0)
    run_id = harness.create(stage_plan=None)
    harness.drain()
    run = harness.run(run_id)
    assert run.config_json["stage_plan"] == list(V1_STAGE_PLAN) and run.config_json["version"] == 4
    assert _latest(harness, run_id, "holdout").verdict == "pass"
    assert (run.status, run.current_stage) == ("waiting", "shadow_paper")

    shas, wakes = set(), 0
    while (row := _latest(harness, run_id, "shadow_paper")).status == "waiting":
        assert row.attempt == 1 and row.wait_until is not None
        run = harness.run(run_id)
        assert run.status == "waiting" and run.reason == row.reason and run.reason.startswith("shadow_paper ⏳")
        shas.add(row.artifact_sha256)
        swept = _wake(harness, source, row)
        assert swept["waiting_due"] == 1
        wakes += 1
        assert wakes <= 40, "shadow paper never finished"

    paper = _latest(harness, run_id, "shadow_paper")
    report = paper.artifact_json["report"]
    assert (paper.status, paper.verdict, paper.attempt) == ("completed", "pass", 1), (
        paper.reason,
        report["protection"],
        report["trading_days"],
    )
    assert report["trading_days"] == 20 and report["paper"]["n_trades"] >= 10
    assert report["paper_start_utc"] == PAPER_START.isoformat()
    assert report["consistency"]["ratio"] == 1.0 and report["revisions"]["count"] == 0
    assert report["cutoff_utc"] == "2026-11-19T07:00:00+00:00"  # Thu 11-19 15:00 Taipei, judged Fri 11-20 09:30
    assert wakes == 29 and len(shas) == wakes  # Fri 10-23 .. Fri 11-20, a fresh artifact every wake-up
    # the final (completed) files sit in the attempt directory; every waiting wake-up wrote its own wake-*
    directory = Path(url2pathname(urlparse(paper.artifact_uri).path))
    assert directory.name == "1" and (directory / "paper_ledger.json").is_file()
    assert len(list(directory.glob("wake-*"))) == wakes  # the first claim + 28 wake-ups that kept waiting
    events = harness.events(run_id)
    assert events.count("stage_waiting") == wakes + 1  # + awaiting_release

    held = _latest(harness, run_id, "awaiting_release")
    run = harness.run(run_id)
    assert (run.status, run.current_stage, run.reason) == ("waiting", "awaiting_release", AWAITING_RELEASE_REASON)
    assert (held.status, held.wait_until, held.verdict) == ("waiting", None, None)
    assert held.artifact_json["report"]["shadow_paper"]["artifact_sha256"] == paper.artifact_sha256

    # never auto-advances: neither the sweeper nor a stray advance message moves it
    clock.advance(days=90)
    swept = harness.dispatcher.sweep()
    assert (swept["waiting_due"], swept["enqueued"]) == (0, 0)
    assert harness.dispatcher.advance(run_id)["skipped"] == "held"
    assert harness.run(run_id).status == "waiting" and len(harness.attempts(run_id)) == len(V1_STAGE_PLAN)

    _abort(harness, run_id)
    assert harness.run(run_id).status == "aborted"
    assert _latest(harness, run_id, "awaiting_release").status == "aborted"


def test_abort_while_shadow_paper_waits_then_rerun_resumes_the_paper(pipeline_sessions, tmp_path, clock):
    source = PaperSource()
    harness = _harness(pipeline_sessions, tmp_path, clock, source)
    run_id = harness.create(stage_plan=None)
    harness.drain()
    for _ in range(5):  # Fri 10-23 .. Tue 10-27: Friday and Monday booked
        _wake(harness, source, _latest(harness, run_id, "shadow_paper"))
    waiting = _latest(harness, run_id, "shadow_paper")
    booked = waiting.artifact_json["report"]
    assert waiting.status == "waiting" and booked["trading_days"] == 2

    _abort(harness, run_id)
    aborted = _latest(harness, run_id, "shadow_paper")
    assert harness.run(run_id).status == "aborted"
    assert aborted.status == "aborted" and aborted.artifact_sha256 == waiting.artifact_sha256  # progress kept
    clock.now = aborted.wait_until + timedelta(hours=1)
    assert harness.dispatcher.sweep()["waiting_due"] == 0
    assert harness.dispatcher.advance(run_id)["skipped"] == "terminal"

    from poseidon.pipeline.service import PipelineService

    source.at(clock.now)
    with harness.sessions() as session:
        run = harness.run(run_id)
        PipelineService(session, registry=harness.dispatcher.registry, clock=clock).rerun(
            run_id,
            {"expected_revision": run.revision, "reason": "resume paper"},
            actor_id="human:researcher",
            idempotency_key=uuid.uuid4().hex,
        )
        session.commit()
    harness.queue.append(run_id)
    harness.drain()
    resumed = _latest(harness, run_id, "shadow_paper")
    report = resumed.artifact_json["report"]
    assert (resumed.attempt, resumed.trigger, resumed.status) == (2, "rerun", "waiting")
    assert report["paper_start_utc"] == booked["paper_start_utc"]  # anchored on the first attempt
    assert report["resume"] == {"from_attempt": 1}
    assert report["trading_days"] == 3  # Tuesday caught up on top of the two days attempt 1 booked


def test_a_waiting_attempt_needs_wait_until_except_awaiting_release(pipeline_sessions, tmp_path, clock):
    source = PaperSource()
    harness = _harness(pipeline_sessions, tmp_path, clock, source)
    run_id = harness.create(stage_plan=None)
    harness.drain()
    waiting = _latest(harness, run_id, "shadow_paper")
    with pipeline_sessions() as session, pytest.raises(IntegrityError, match="ck_stage_runs_waiting"):
        session.execute(text("UPDATE stage_runs SET wait_until = NULL WHERE id = :id"), {"id": waiting.id})
        session.commit()


def test_old_runs_keep_ending_at_holdout(pipeline_sessions, tmp_path, clock):
    source = PaperSource()
    harness = _harness(pipeline_sessions, tmp_path, clock, source)
    plan = list(V1_STAGE_PLAN[: V1_STAGE_PLAN.index("holdout") + 1])
    run_id = harness.create(stage_plan=plan)
    harness.drain()
    run = harness.run(run_id)
    assert (run.status, run.reason) == ("completed", "passed all validation gates")
    assert _latest(harness, run_id, "shadow_paper") is None and source.calls == []


def test_a_protection_trip_fails_the_run_for_good(pipeline_sessions, tmp_path, clock):
    from poseidon.pipeline.service import PipelineConflict, PipelineService

    source = PaperSource()  # same volatility as the 15 train days: the train max drawdown is small
    harness = _harness(pipeline_sessions, tmp_path, clock, source)
    run_id = harness.create(stage_plan=None)
    harness.drain()
    _walk(harness, source, run_id)
    paper = _latest(harness, run_id, "shadow_paper")
    run = harness.run(run_id)
    assert (paper.verdict, paper.artifact_json["report"]["reason_code"]) == ("fail", "paper_protection_trip")
    assert run.status == "failed" and run.reason.startswith("shadow_paper: protection trip: ")
    assert _latest(harness, run_id, "awaiting_release") is None
    with harness.sessions() as session, pytest.raises(PipelineConflict):
        PipelineService(session, registry=harness.dispatcher.registry, clock=clock).rerun(
            run_id,
            {"expected_revision": run.revision, "reason": "again"},
            actor_id="human:researcher",
            idempotency_key=uuid.uuid4().hex,
        )
