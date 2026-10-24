"""``/api/v1/pipelines`` on real PostgreSQL: roles, idempotency, error mapping."""

from __future__ import annotations

import hashlib
import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from poseidon.api import pipelines
from poseidon.api.errors import http_exception_handler, validation_exception_handler
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.pipeline.models import PipelineRun
from tests.pipeline.conftest import STEP1_PLAN, Harness, tx_spec

pytestmark = pytest.mark.postgresql

KEYS = {
    "researcher": ("researcher-secret", {"actor_id": "strategy-bridge", "roles": ["researcher"], "account_scopes": []}),
    "viewer": ("viewer-secret", {"actor_id": "human:viewer", "roles": ["viewer"], "account_scopes": []}),
    "release": ("release-secret", {"actor_id": "human:release", "roles": ["release_manager"], "account_scopes": []}),
    "pm": ("pm-secret", {"actor_id": "human:pm", "roles": ["portfolio_manager"], "account_scopes": []}),
}


@pytest.fixture
def api(pipeline_sessions, tmp_path, clock, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy-secret")
    monkeypatch.setattr(
        settings,
        "api_principals_json",
        json.dumps({hashlib.sha256(key.encode()).hexdigest(): record for key, record in KEYS.values()}),
    )
    harness = Harness(pipeline_sessions, tmp_path, clock)
    app = FastAPI()
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(pipelines.router, prefix="/api/v1/pipelines")

    def db():
        session = pipeline_sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[pipelines.get_pipeline_enqueuer] = lambda: harness.queue.append
    client = TestClient(app)
    client.harness = harness
    return client


def _headers(who="researcher", key=None):
    secret = "legacy-secret" if who == "legacy" else KEYS[who][0]
    headers = {"X-API-Key": secret}
    if key is not None:
        headers["Idempotency-Key"] = key
    return headers


def _create(api, key="create-1", **body):
    payload = {"idea_text": "TX ORB", "source": "telegram", "spec_json": tx_spec(), "stage_plan": STEP1_PLAN}
    payload.update(body)
    return api.post("/api/v1/pipelines", json=payload, headers=_headers(key=key))


def test_create_is_idempotent_and_conflicts_on_a_different_body(api, pipeline_sessions):
    first = _create(api)
    assert first.status_code == 201, first.text
    body = first.json()
    assert (body["status"], body["created_by"], body["run_no"], body["revision"]) == (
        "queued",
        "strategy-bridge",
        1,
        1,
    )
    assert body["family_id"] == body["id"] and body["current_stage"] == "spec_generate"
    assert api.harness.queue == [uuid.UUID(body["id"])]

    replay = _create(api)
    assert replay.status_code == 201 and replay.json() == body
    assert api.harness.queue == [uuid.UUID(body["id"])]  # a replay never enqueues twice

    conflict = _create(api, idea_text="something else")
    assert conflict.status_code == 409 and "different request" in conflict.json()["detail"]
    with pipeline_sessions() as session:
        assert session.query(PipelineRun).count() == 1

    assert (
        api.post("/api/v1/pipelines", json={"idea_text": "x", "source": "api"}, headers=_headers()).status_code == 422
    )


def test_roles(api):
    run_id = _create(api).json()["id"]
    paths = [
        "/api/v1/pipelines",
        f"/api/v1/pipelines/{run_id}",
        f"/api/v1/pipelines/{run_id}/stages",
        f"/api/v1/pipelines/{run_id}/events",
    ]
    for who in ("viewer", "researcher", "release"):
        assert all(api.get(path, headers=_headers(who)).status_code == 200 for path in paths)
    for who in ("legacy", "pm"):
        assert all(api.get(path, headers=_headers(who)).status_code == 403 for path in paths)
    for who in ("legacy", "viewer", "release", "pm"):
        response = api.post(
            "/api/v1/pipelines",
            json={"idea_text": "x", "source": "api"},
            headers=_headers(who, key=f"k-{who}"),
        )
        assert response.status_code == 403, who
        rerun = api.post(
            f"/api/v1/pipelines/{run_id}/abort",
            json={"expected_revision": 1, "reason": "x"},
            headers=_headers(who, key=f"a-{who}"),
        )
        assert rerun.status_code == 403, who
    assert api.get("/api/v1/pipelines", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        api.post("/api/v1/pipelines", json={"idea_text": "x", "source": "api"}, headers=_headers()).status_code == 422
    )


def test_error_mapping(api):
    missing = uuid.uuid4()
    assert api.get(f"/api/v1/pipelines/{missing}", headers=_headers()).status_code == 404
    assert api.get(f"/api/v1/pipelines/families/{missing}/trials", headers=_headers()).status_code == 404
    assert (
        api.post(
            f"/api/v1/pipelines/{missing}/rerun",
            json={"expected_revision": 1, "reason": "x"},
            headers=_headers(key="r"),
        ).status_code
        == 404
    )
    assert _create(api, key="bad-plan", stage_plan=["spec_generate", "backtest"]).status_code == 422
    assert _create(api, key="bad-family", family_id=str(missing)).status_code == 422
    assert _create(api, key="extra", thresholds={"wfe_min": 0}).status_code == 422  # no client overrides
    run_id = _create(api).json()["id"]
    assert api.get(f"/api/v1/pipelines/{run_id}/stages/backtest/attempts/1", headers=_headers()).status_code == 404
    stale = api.post(
        f"/api/v1/pipelines/{run_id}/abort",
        json={"expected_revision": 7, "reason": "x"},
        headers=_headers(key="stale"),
    )
    assert stale.status_code == 409 and "revision" in stale.json()["detail"]


def test_full_flow_reads_and_rerun_rules(api):
    created = _create(api).json()
    api.harness.drain()
    run_id = created["id"]
    run = api.get(f"/api/v1/pipelines/{run_id}", headers=_headers("viewer")).json()
    assert run["status"] == "completed" and run["revision"] > created["revision"]
    stages = api.get(f"/api/v1/pipelines/{run_id}/stages", headers=_headers("viewer")).json()["stages"]
    assert [(entry["stage"], entry["latest"]["verdict"]) for entry in stages] == [
        ("spec_generate", "pass"),
        ("spec_validate", "pass"),
        ("backtest", "pass"),
    ]
    attempt = api.get(f"/api/v1/pipelines/{run_id}/stages/backtest/attempts/1", headers=_headers("viewer")).json()
    assert attempt["artifact_json"]["report"]["metrics"]["n_trades"] > 0 and attempt["artifact_sha256"]
    events = api.get(f"/api/v1/pipelines/{run_id}/events", headers=_headers("viewer")).json()["items"]
    assert events[0]["event_type"] == "run_created" and events[-1]["event_type"] == "stage_completed"
    listing = api.get("/api/v1/pipelines?status=completed", headers=_headers("viewer")).json()["items"]
    assert [item["id"] for item in listing] == [run_id]
    trials = api.get(f"/api/v1/pipelines/families/{run_id}/trials", headers=_headers("viewer")).json()
    assert trials == {"family_id": run_id, "trial_count": 0, "items": []}
    done = api.post(
        f"/api/v1/pipelines/{run_id}/rerun",
        json={"expected_revision": run["revision"], "reason": "again"},
        headers=_headers(key="rerun-completed"),
    )
    assert done.status_code == 409

    failed = _create(api, key="create-bad", spec_json=tx_spec(market="tw_stock")).json()
    api.harness.drain()
    failed_run = api.get(f"/api/v1/pipelines/{failed['id']}", headers=_headers()).json()
    assert failed_run["status"] == "failed"
    refused = api.post(
        f"/api/v1/pipelines/{failed['id']}/rerun",
        json={"expected_revision": failed_run["revision"], "reason": "try again"},
        headers=_headers(key="rerun-failed"),
    )
    assert refused.status_code == 409 and "final" in refused.json()["detail"]

    unsupported = _create(api, key="create-null", spec_json=None).json()
    api.harness.drain()
    revision = api.get(f"/api/v1/pipelines/{unsupported['id']}", headers=_headers()).json()["revision"]
    body = {"expected_revision": revision, "reason": "after DSL v2", "from_stage": "spec_generate"}
    accepted = api.post(f"/api/v1/pipelines/{unsupported['id']}/rerun", json=body, headers=_headers(key="rerun-null"))
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["status"] == "queued" and accepted.json()["revision"] == revision + 1
    replay = api.post(f"/api/v1/pipelines/{unsupported['id']}/rerun", json=body, headers=_headers(key="rerun-null"))
    assert replay.status_code == 202 and replay.json() == accepted.json()
    other = api.post(
        f"/api/v1/pipelines/{unsupported['id']}/rerun",
        json={**body, "reason": "different"},
        headers=_headers(key="rerun-null"),
    )
    assert other.status_code == 409

    aborted = api.post(
        f"/api/v1/pipelines/{unsupported['id']}/abort",
        json={"expected_revision": revision + 1, "reason": "stop"},
        headers=_headers(key="abort-null"),
    )
    assert aborted.status_code == 202 and aborted.json()["status"] == "aborted"
    again = api.post(
        f"/api/v1/pipelines/{unsupported['id']}/abort",
        json={"expected_revision": revision + 2, "reason": "stop"},
        headers=_headers(key="abort-null-2"),
    )
    assert again.status_code == 409
    for method in ("put", "delete"):
        assert getattr(api, method)(f"/api/v1/pipelines/{run_id}", headers=_headers()).status_code == 405


def test_capabilities_serve_the_checked_in_brief_and_schema_to_readers(api):
    from poseidon.pipeline.v3.brief import BRIEF_FILE, GENERATED_DIR, SCHEMA_FILE
    from poseidon.pipeline.v3.registry import REGISTRY_VERSION

    response = api.get("/api/v1/pipelines/capabilities", headers=_headers("viewer"))
    assert response.status_code == 200, response.text
    body = response.json()
    brief = (GENERATED_DIR / BRIEF_FILE).read_text(encoding="utf-8")
    assert body["spec_version"] == 3 and body["registry_version"] == REGISTRY_VERSION
    assert body["brief"] == brief and body["brief_sha256"] == hashlib.sha256(brief.encode()).hexdigest()
    assert body["schema"] == json.loads((GENERATED_DIR / SCHEMA_FILE).read_text(encoding="utf-8"))
    assert api.get("/api/v1/pipelines/capabilities", headers=_headers("researcher")).status_code == 200
    assert api.get("/api/v1/pipelines/capabilities", headers=_headers("pm")).status_code == 403
    assert api.get("/api/v1/pipelines/capabilities", headers=_headers("legacy")).status_code == 403
