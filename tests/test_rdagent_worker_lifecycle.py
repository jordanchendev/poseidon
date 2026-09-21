"""Isolated Postgres lifecycle tests for the RD-Agent Celery worker.

The worker dependencies are replaced before invocation, so these tests never
make an HTTP, Docker, qrun, or LLM call. Run inside qlib-research with the
isolated PostgreSQL URL supplied by the Phase 91 executor.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def session_factory(monkeypatch):
    """Bind the worker and test rows to the executor's isolated database."""
    database_url = os.environ.get("POSEIDON_REAL_DATABASE_URL")
    if not database_url:
        pytest.skip("POSEIDON_REAL_DATABASE_URL is required for lifecycle tests")

    factory = sessionmaker(bind=create_engine(database_url))
    from poseidon.workers import qlib_rdagent_tasks as worker

    @contextmanager
    def isolated_db_session():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(worker, "SessionLocal", factory)
    monkeypatch.setattr(worker, "db_session", isolated_db_session)
    return factory


@pytest.fixture
def rd_run(session_factory):
    from poseidon.models.rd_agent_run import RDAgentRun

    session = session_factory()
    run = RDAgentRun(challenge="mean reversion", time_budget_hours=1, cost_cap_usd=5)
    session.add(run)
    session.commit()
    session.refresh(run)
    try:
        yield run
    finally:
        session.query(RDAgentRun).filter_by(run_id=run.run_id).delete()
        session.commit()
        session.close()


def test_pending_cancel_is_not_claimed(rd_run, session_factory):
    from poseidon.models.rd_agent_run import RDAgentRun

    session = session_factory()
    try:
        row = session.query(RDAgentRun).filter_by(run_id=rd_run.run_id).one()
        row.cancel_requested = True
        row.cancel_reason = "user cancelled"
        session.commit()
    finally:
        session.close()

    from poseidon.workers.qlib_rdagent_tasks import qlib_rdagent_run

    result = qlib_rdagent_run.run(str(rd_run.run_id))
    assert result["status"] == "pending"


def test_invalid_run_id_returns_failed_without_worker_crash(session_factory):
    from poseidon.workers.qlib_rdagent_tasks import qlib_rdagent_run

    result = qlib_rdagent_run.run("not-a-uuid")
    assert result["status"] == "failed"


@pytest.fixture
def worker_doubles(monkeypatch, tmp_path):
    """Keep lifecycle tests in real PostgreSQL while excluding paid/runtime work."""
    import qlib
    from rdagent.app.qlib_rd_loop import quant

    from poseidon.rdagent import audit, budget_guard, dataset_builder, sandbox, scenario
    from poseidon.workers import qlib_rdagent_tasks as worker

    sandbox_dir = tmp_path / "sandbox"
    (sandbox_dir / "workspace").mkdir(parents=True)
    (sandbox_dir / "logs").mkdir()
    provider = tmp_path / "provider"
    monkeypatch.setattr(qlib, "init", lambda **_kwargs: None)
    monkeypatch.setattr(dataset_builder, "ensure_qlib_bin_dump", lambda **_kwargs: provider)
    monkeypatch.setattr(sandbox, "resolve_sandbox", lambda _run_id: sandbox_dir)
    monkeypatch.setattr(sandbox, "install_rdagent_env", lambda _sandbox: lambda: None)
    monkeypatch.setattr(scenario.PoseidonQuantScenario, "set_challenge", lambda *_args: None)
    monkeypatch.setattr(worker, "_install_chat_env", lambda: lambda: None)
    monkeypatch.setattr(worker, "_install_run_settings", lambda _sandbox: lambda: None)
    monkeypatch.setattr(worker, "_install_factor_data", lambda _sandbox, _provider: lambda: None)
    monkeypatch.setattr(worker, "_install_poseidon_templates", lambda _sandbox, _provider: lambda: None)
    monkeypatch.setattr(worker, "_install_conda_bypass", lambda: lambda: None)
    monkeypatch.setattr(worker, "_install_generated_code_env_guard", lambda: lambda: None)
    monkeypatch.setattr(worker, "_install_stop_guards", lambda *_args: lambda: None)
    harvested = []

    def harvest(loop, sandbox_dir, run):
        harvested.append((loop, sandbox_dir, run.run_id))
        verdict = f"verdict for {run.status}\n"
        (sandbox_dir / "verdict.md").write_text(verdict)
        return {"summary": {"artifact_count": 7, "cost_acc_usd": run.token_cost_acc_usd}, "verdict": verdict}

    monkeypatch.setattr(audit, "harvest_artifacts", harvest)

    guards = []

    class Guard:
        def __init__(self, **_kwargs):
            self.started = False
            guards.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def current_cost(self):
            return 0.0

    monkeypatch.setattr(budget_guard, "BudgetGuard", Guard)

    def set_loop(loop_class):
        monkeypatch.setattr(quant, "QuantRDLoop", loop_class)

    return {"sandbox": sandbox_dir, "harvested": harvested, "guards": guards, "set_loop": set_loop}


def test_success_persists_terminal_metadata_and_audit(rd_run, session_factory, worker_doubles):
    class SuccessfulLoop:
        LoopTerminationError = RuntimeError

        def __init__(self, _settings):
            pass

        async def run(self, **_kwargs):
            return None

    worker_doubles["set_loop"](SuccessfulLoop)
    from poseidon.models.rd_agent_run import RDAgentRun
    from poseidon.workers.qlib_rdagent_tasks import qlib_rdagent_run

    result = qlib_rdagent_run.run(str(rd_run.run_id))
    session = session_factory()
    try:
        row = session.query(RDAgentRun).filter_by(run_id=rd_run.run_id).one()
        assert result["status"] == "succeeded"
        assert row.status == "succeeded"
        assert row.finished_at is not None
        assert row.token_cost_acc_usd == 0.0
        assert row.summary["artifact_count"] == 7
        assert row.verdict == "verdict for succeeded\n"
    finally:
        session.close()
    assert len(worker_doubles["harvested"]) == 1


def test_constructor_failure_persists_failed_audit(rd_run, session_factory, worker_doubles):
    class BrokenConstructor:
        def __init__(self, _settings):
            raise RuntimeError("constructor failed")

    worker_doubles["set_loop"](BrokenConstructor)
    from poseidon.models.rd_agent_run import RDAgentRun
    from poseidon.workers.qlib_rdagent_tasks import qlib_rdagent_run

    result = qlib_rdagent_run.run(str(rd_run.run_id))
    session = session_factory()
    try:
        row = session.query(RDAgentRun).filter_by(run_id=rd_run.run_id).one()
        assert result["status"] == "failed"
        assert row.status == "failed"
        assert row.finished_at is not None
        assert "RuntimeError" in row.error
        assert row.summary["artifact_count"] == 7
        assert row.verdict == "verdict for failed\n"
    finally:
        session.close()
    assert len(worker_doubles["harvested"]) == 1


def test_provider_stop_persists_cancelled_audit(rd_run, session_factory, worker_doubles):
    from poseidon.workers.qlib_rdagent_tasks import ProviderCallStopped

    class CancelledLoop:
        LoopTerminationError = RuntimeError

        def __init__(self, _settings):
            pass

        async def run(self, **_kwargs):
            raise ProviderCallStopped("cost reserve exceeds cap")

    worker_doubles["set_loop"](CancelledLoop)
    from poseidon.models.rd_agent_run import RDAgentRun
    from poseidon.workers.qlib_rdagent_tasks import qlib_rdagent_run

    result = qlib_rdagent_run.run(str(rd_run.run_id))
    session = session_factory()
    try:
        row = session.query(RDAgentRun).filter_by(run_id=rd_run.run_id).one()
        assert result["status"] == "cancelled"
        assert row.status == "cancelled"
        assert row.cancel_requested is True
        assert row.cancel_reason == "cost reserve exceeds cap"
        assert row.summary["artifact_count"] == 7
        assert row.verdict == "verdict for cancelled\n"
    finally:
        session.close()
    assert len(worker_doubles["harvested"]) == 1
    assert worker_doubles["guards"][0].started is False
