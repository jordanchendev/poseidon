"""HTTP contract for principal-scoped manifest freeze and research revisions."""

import copy
import hashlib
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from poseidon.api import research_loop
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.decision_loop.research import FAILURE_CODES
from poseidon.main import app as poseidon_app
from poseidon.models.base import Base
from poseidon.models.data_manifest import DataManifest
from tests.test_decision_loop_foundation import manifest_request
from tests.test_research_submission import narrative

ACCOUNT = "paper:pilot"
OTHER_ACCOUNT = "paper:other"
RESEARCHER_KEY = "research-loop-researcher"
OTHER_RESEARCHER_KEY = "research-loop-other-researcher"
WORKER_KEY = "research-loop-worker"
VIEWER_KEY = "research-loop-viewer"
MANAGER_KEY = "research-loop-manager"
SCOPE_KEY = "tw_stock:spot:2330"
PREPARE_FIELDS = {"policy_version": "pilot-v1", "provider": "human", "model": "manual", "runtime_digest": "host-v1"}


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


def _fingerprint(key):
    return hashlib.sha256(key.encode()).hexdigest()


def _principal(actor_id, roles, scopes):
    return {"actor_id": actor_id, "roles": roles, "account_scopes": scopes}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "research-loop-legacy")
    monkeypatch.setattr(
        settings,
        "api_principals_json",
        json.dumps(
            {
                _fingerprint(RESEARCHER_KEY): _principal("human:researcher", ["researcher"], [ACCOUNT]),
                _fingerprint(OTHER_RESEARCHER_KEY): _principal("human:other", ["researcher"], [OTHER_ACCOUNT]),
                _fingerprint(WORKER_KEY): _principal("service:decision-worker", ["decision-worker"], [ACCOUNT]),
                _fingerprint(VIEWER_KEY): _principal("human:viewer", ["viewer"], [ACCOUNT]),
                _fingerprint(MANAGER_KEY): _principal("human:manager", ["portfolio_manager"], [ACCOUNT]),
            }
        ),
    )
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(
        engine, tables=[Base.metadata.tables[name] for name in ("data_manifests", "research_revisions")]
    )
    app = FastAPI()
    app.include_router(research_loop.router, prefix="/api/v1/research")
    with Session(engine) as session:
        app.dependency_overrides[get_db] = lambda: session
        yield TestClient(app), session
    engine.dispose()


def _headers(key):
    return {"X-API-Key": key}


def _freeze(client, key=RESEARCHER_KEY, request=None):
    return client.post(
        "/api/v1/research/manifests",
        json={"request": request or manifest_request()},
        headers=_headers(key),
    )


def _prepare(client, manifest_id, key=RESEARCHER_KEY, scope_key=SCOPE_KEY, **changes):
    body = {"scope_key": scope_key, "manifest_id": str(manifest_id), **PREPARE_FIELDS, **changes}
    return client.post("/api/v1/research/revisions", json=body, headers=_headers(key))


def _prepared(client):
    manifest = _freeze(client).json()
    revision = _prepare(client, manifest["id"]).json()
    return manifest, revision


def _complete(client, revision, manifest, key=RESEARCHER_KEY, account_scope=ACCOUNT, research=None):
    body = {
        "research": research or _narrative_for(manifest),
        "account_scope": account_scope,
        "request_sha256": revision["request_sha256"],
    }
    return client.post(f"/api/v1/research/revisions/{revision['id']}/complete", json=body, headers=_headers(key))


def _narrative_for(_manifest):
    # narrative() binds the frozen manifest cutoff; manifest_request() freezes this exact as_of.
    return narrative(SimpleNamespace(payload_json={"as_of": manifest_request()["as_of"]}))


def test_openapi_exposes_exactly_the_research_loop_operations():
    poseidon_app.openapi_schema = None
    try:
        schema = poseidon_app.openapi()
    finally:
        poseidon_app.openapi_schema = None
    paths = {
        path: set(operations) for path, operations in schema["paths"].items() if path.startswith("/api/v1/research")
    }
    assert paths == {
        "/api/v1/research/manifests": {"post"},
        "/api/v1/research/manifests/{manifest_id}": {"get"},
        "/api/v1/research/revisions": {"get", "post"},
        "/api/v1/research/revisions/{revision_id}": {"get"},
        "/api/v1/research/revisions/{revision_id}/complete": {"post"},
        "/api/v1/research/revisions/{revision_id}/provider-failed": {"post"},
    }
    serialized = json.dumps({path: schema["paths"][path] for path in paths})
    for forbidden in ("actor", "active_version_id", "approve", "order", "api_key", "credential"):
        assert forbidden not in serialized


def test_manifest_freeze_replays_to_the_same_row_for_researcher_and_worker(client):
    client, _session = client
    first = _freeze(client)
    assert first.status_code == 200, first.text
    replay = _freeze(client, key=WORKER_KEY, request=copy.deepcopy(manifest_request()))
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    assert first.json()["account_scope"] == ACCOUNT
    assert set(first.json()) == {"id", "content_sha256", "as_of", "market", "interval", "account_scope"}


@pytest.mark.parametrize(
    ("key", "mutate", "status"),
    [
        (VIEWER_KEY, None, 403),
        (MANAGER_KEY, None, 403),
        (OTHER_RESEARCHER_KEY, None, 403),
        (RESEARCHER_KEY, lambda request: request.pop("account_scope"), 403),
        (RESEARCHER_KEY, lambda request: request["evidence"].pop(), 422),
    ],
)
def test_manifest_freeze_requires_role_scope_and_valid_evidence(client, key, mutate, status):
    client, session = client
    request = manifest_request()
    if mutate is not None:
        mutate(request)
    response = _freeze(client, key=key, request=request)
    assert response.status_code == status, response.text
    assert session.query(DataManifest).count() == 0


def test_request_bodies_reject_caller_supplied_actor(client):
    client, _session = client
    response = client.post(
        "/api/v1/research/manifests",
        json={"request": manifest_request(), "actor": "human:spoofed"},
        headers=_headers(RESEARCHER_KEY),
    )
    assert response.status_code == 422
    manifest = _freeze(client).json()
    response = _prepare(client, manifest["id"], actor="human:spoofed")
    assert response.status_code == 422


def test_prepare_binds_every_request_dimension_and_replays(client):
    client, _session = client
    manifest = _freeze(client).json()
    first = _prepare(client, manifest["id"])
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "prepared"
    assert first.json()["previous_revision_id"] is None
    assert _prepare(client, manifest["id"], key=WORKER_KEY).json()["id"] == first.json()["id"]
    digests = {first.json()["request_sha256"]}
    digests.add(_prepare(client, manifest["id"], scope_key="tw_stock:sector:semiconductor").json()["request_sha256"])
    for field in PREPARE_FIELDS:
        digests.add(_prepare(client, manifest["id"], **{field: "changed"}).json()["request_sha256"])
    assert len(digests) == 6
    stale = _prepare(client, manifest["id"], expected_previous_revision_id=str(uuid.uuid4()))
    assert stale.status_code == 422


def test_prepare_requires_role_and_manifest_scope(client):
    client, _session = client
    manifest = _freeze(client).json()
    assert _prepare(client, manifest["id"], key=VIEWER_KEY).status_code == 403
    assert _prepare(client, manifest["id"], key=OTHER_RESEARCHER_KEY).status_code == 403
    assert _prepare(client, uuid.uuid4()).status_code == 404


def test_only_an_in_scope_researcher_completes_a_narrative(client):
    client, _session = client
    manifest, revision = _prepared(client)
    worker = _complete(client, revision, manifest, key=WORKER_KEY)
    assert worker.status_code == 403
    outside = _complete(client, revision, manifest, key=OTHER_RESEARCHER_KEY, account_scope=OTHER_ACCOUNT)
    assert outside.status_code == 403
    mismatched = _complete(client, revision, manifest, key=RESEARCHER_KEY, account_scope=OTHER_ACCOUNT)
    assert mismatched.status_code == 403

    completed = _complete(client, revision, manifest)
    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "completed"
    assert completed.json()["content_sha256"]
    replay = _complete(client, revision, manifest)
    assert replay.status_code == 200
    assert replay.json()["content_sha256"] == completed.json()["content_sha256"]
    changed = _complete(client, revision, manifest, research=dict(_narrative_for(manifest), thesis="changed"))
    assert changed.status_code == 422
    missing = client.post(
        f"/api/v1/research/revisions/{uuid.uuid4()}/complete",
        json={"research": _narrative_for(manifest), "account_scope": ACCOUNT, "request_sha256": "a" * 64},
        headers=_headers(RESEARCHER_KEY),
    )
    assert missing.status_code == 404


def test_provider_failed_is_worker_only_and_reason_bounded(client):
    client, _session = client
    _manifest, revision = _prepared(client)
    path = f"/api/v1/research/revisions/{revision['id']}/provider-failed"
    reason = sorted(FAILURE_CODES)[0]
    assert client.post(path, json={"reason_code": reason}, headers=_headers(RESEARCHER_KEY)).status_code == 403
    unknown = client.post(path, json={"reason_code": "made_up"}, headers=_headers(WORKER_KEY))
    assert unknown.status_code == 422
    failed = client.post(path, json={"reason_code": reason}, headers=_headers(WORKER_KEY))
    assert failed.status_code == 200, failed.text
    assert failed.json()["status"] == "needs_review"
    assert failed.json()["research_json"] == {"review_reason_codes": [reason]}


def test_reads_are_scope_filtered_and_never_leak_narratives(client):
    client, _session = client
    manifest, revision = _prepared(client)
    assert _complete(client, revision, manifest).status_code == 200

    own = client.get(f"/api/v1/research/revisions/{revision['id']}", headers=_headers(VIEWER_KEY))
    assert own.status_code == 200
    assert own.json()["research_json"]["thesis"]
    own_manifest = client.get(f"/api/v1/research/manifests/{manifest['id']}", headers=_headers(VIEWER_KEY))
    assert own_manifest.status_code == 200
    assert own_manifest.json()["payload_json"]["account_scope"] == ACCOUNT

    other_revision = client.get(f"/api/v1/research/revisions/{revision['id']}", headers=_headers(OTHER_RESEARCHER_KEY))
    other_manifest = client.get(f"/api/v1/research/manifests/{manifest['id']}", headers=_headers(OTHER_RESEARCHER_KEY))
    assert other_revision.status_code == 403
    assert other_manifest.status_code == 403
    assert "thesis" not in other_revision.text

    listed = client.get(
        "/api/v1/research/revisions",
        params={"scope_key": SCOPE_KEY, "status": "completed"},
        headers=_headers(VIEWER_KEY),
    )
    assert [item["id"] for item in listed.json()] == [revision["id"]]
    hidden = client.get(
        "/api/v1/research/revisions", params={"scope_key": SCOPE_KEY}, headers=_headers(OTHER_RESEARCHER_KEY)
    )
    assert hidden.status_code == 200
    assert hidden.json() == []
    prepared_only = client.get(
        "/api/v1/research/revisions",
        params={"scope_key": SCOPE_KEY, "status": "prepared"},
        headers=_headers(VIEWER_KEY),
    )
    assert prepared_only.json() == []

    assert client.get(f"/api/v1/research/revisions/{uuid.uuid4()}", headers=_headers(VIEWER_KEY)).status_code == 404
    assert client.get(f"/api/v1/research/manifests/{uuid.uuid4()}", headers=_headers(VIEWER_KEY)).status_code == 404
