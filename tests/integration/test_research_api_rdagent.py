"""RD-Agent REST contract with a thread-safe SQLite fixture."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from poseidon.api import rdagent
from poseidon.api.rdagent import router
from poseidon.core.database import get_db
from poseidon.models.base import Base
from poseidon.models.rd_agent_run import RDAgentRun


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(bind=_engine)
_app = FastAPI()
_app.include_router(router, prefix="/research/rd-agent")


def _db():
    db = Session()
    try:
        yield db
    finally:
        db.close()


_app.dependency_overrides[get_db] = _db


@pytest.fixture(autouse=True)
def _database(tmp_path, monkeypatch):
    Base.metadata.create_all(_engine)
    monkeypatch.setattr(rdagent, "_ROOT", tmp_path)
    yield
    Base.metadata.drop_all(_engine)


@pytest.fixture(autouse=True)
def _celery(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rdagent.celery_app, "send_task", lambda name, args=None, queue=None, **kw: calls.append((name, args, queue))
    )
    return calls


@pytest.fixture
def client():
    return TestClient(_app)


def _insert(status="pending"):
    db = Session()
    try:
        run = RDAgentRun(challenge="test", status=status)
        db.add(run)
        db.commit()
        return str(run.run_id)
    finally:
        db.close()


def test_post_get_list_and_cancel(client, _celery):
    response = client.post("/research/rd-agent/run", json={"challenge": "find TX signals"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert _celery == [("poseidon.workers.qlib_tasks.qlib_rdagent_run", [run_id], "poseidon_qlib")]
    assert client.get(f"/research/rd-agent/runs/{run_id}").status_code == 200
    assert client.get("/research/rd-agent/runs?limit=1").json()["total"] == 1
    assert client.post(f"/research/rd-agent/runs/{run_id}/cancel").json()["status"] == "cancelled"
    running = _insert("running")
    assert client.post(f"/research/rd-agent/runs/{running}/cancel").json()["status"] == "running"
    assert client.post(f"/research/rd-agent/runs/{_insert('succeeded')}/cancel").status_code == 409


def test_input_dispatch_and_artifact_guards(client, monkeypatch, tmp_path):
    assert client.post("/research/rd-agent/run", json={"challenge": "bad `prompt`"}).status_code == 422
    assert client.get("/research/rd-agent/runs/not-a-uuid").status_code == 422
    monkeypatch.setattr(rdagent.celery_app, "send_task", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret")))
    assert client.post("/research/rd-agent/run", json={"challenge": "valid"}).status_code == 503
    db = Session()
    failed = db.query(RDAgentRun).one()
    db.close()
    assert failed.status == "failed" and failed.finished_at is not None and "secret" not in failed.error
    run_id = _insert()
    root = tmp_path / "local_dev" / "rd-agent" / "runs" / run_id
    root.mkdir(parents=True)
    (root / "verdict.md").write_text("ok")
    (root / "escape").symlink_to(Path("/tmp"), target_is_directory=True)
    assert client.get(f"/research/rd-agent/runs/{run_id}/artifacts").json()["artifacts"] == [
        {"path": "verdict.md", "size": 2}
    ]
