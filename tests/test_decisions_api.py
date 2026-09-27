"""HTTP contract for scoped Phase 97 portfolio decisions."""

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from poseidon.api import decisions as decisions_api
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionNotFoundError
from poseidon.decision_loop.manifest import ValidationError
from poseidon.main import app as poseidon_app

ACCOUNT = "paper:pilot"
OTHER_ACCOUNT = "paper:other"
WORKER_KEY = "decision-worker-secret"
MANAGER_KEY = "portfolio-manager-secret"
VIEWER_KEY = "viewer-secret"
CROSS_ACCOUNT_KEY = "cross-account-secret"
LEGACY_KEY = "legacy-secret"


class TrackingSession:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.refreshes = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, _record):
        self.refreshes += 1


def _record(decision_id=None):
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=decision_id or uuid.uuid4(),
        evaluation_run_id=uuid.uuid4(),
        strategy_version_id=uuid.uuid4(),
        account_scope=ACCOUNT,
        decision_as_of=now,
        valid_until=now + timedelta(hours=1),
        status="pending_approval",
        revision=1,
        creation_sha256="a" * 64,
        policy_sha256="b" * 64,
        original_json={"final_action": "hold"},
        final_json={"final_action": "hold"},
        portfolio_snapshot_json={"cash": 1000},
        risk_snapshot_json={"hard_failures": [], "allowed_actions": ["hold"]},
        created_at=now,
        updated_at=now,
    )


class StubDecisionService:
    def __init__(self):
        self.decision = _record()
        self.calls = []
        self.failures = {}
        self.transition_events = 0
        self.replays = {}

    def _fail(self, operation):
        error = self.failures.get(operation)
        if error is not None:
            raise error

    def create_decision(self, evaluation_run_id, **kwargs):
        self.calls.append(("create", evaluation_run_id, kwargs))
        principal = kwargs["principal"]
        principal.require_role("decision-worker")
        principal.require_account_scope(kwargs["account_scope"])
        self._fail("create")
        return self.decision

    def get(self, decision_id, *, principal):
        self.calls.append(("get", decision_id, principal))
        principal.require_account_scope(self.decision.account_scope)
        self._fail("get")
        return self.decision

    def list_pending(self, *, principal):
        self.calls.append(("list", principal))
        principal.require_account_scope(self.decision.account_scope)
        self._fail("list")
        return [self.decision]

    def _transition(self, operation, decision_id, body, *, principal, idempotency_key):
        self.calls.append((operation, decision_id, body, principal, idempotency_key))
        principal.require_role("portfolio_manager")
        principal.require_account_scope(self.decision.account_scope)
        self._fail(operation)
        replay_key = (operation, decision_id, idempotency_key)
        response = {
            "id": str(decision_id),
            "status": "approved" if operation == "approve" else "rejected",
            "revision": 2,
            "final_json": dict(self.decision.final_json),
        }
        if replay_key not in self.replays:
            self.replays[replay_key] = response
            self.transition_events += 1
        return self.replays[replay_key]

    def approve(self, decision_id, body, **kwargs):
        return self._transition("approve", decision_id, body, **kwargs)

    def reject(self, decision_id, body, **kwargs):
        return self._transition("reject", decision_id, body, **kwargs)

    def trace(self, decision_id, *, principal):
        self.calls.append(("trace", decision_id, principal))
        principal.require_account_scope(self.decision.account_scope)
        self._fail("trace")
        return {
            "decision": {"id": str(decision_id)},
            "evaluation_run": {"id": str(self.decision.evaluation_run_id)},
            "execution_key": None,
            "orders": [],
            "fills": [],
            "lots": [],
            "allocations": [],
            "reconciliations": [],
            "events": [{"event_type": "created"}],
        }


def _fingerprint(key):
    return hashlib.sha256(key.encode()).hexdigest()


def _principal_config():
    return {
        _fingerprint(WORKER_KEY): {
            "actor_id": "service:decision-worker",
            "roles": ["decision-worker"],
            "account_scopes": [ACCOUNT],
        },
        _fingerprint(MANAGER_KEY): {
            "actor_id": "human:portfolio-manager",
            "roles": ["portfolio_manager"],
            "account_scopes": [ACCOUNT],
        },
        _fingerprint(VIEWER_KEY): {
            "actor_id": "human:viewer",
            "roles": ["viewer"],
            "account_scopes": [ACCOUNT],
        },
        _fingerprint(CROSS_ACCOUNT_KEY): {
            "actor_id": "human:other-manager",
            "roles": ["portfolio_manager"],
            "account_scopes": [OTHER_ACCOUNT],
        },
    }


def _configure_auth(monkeypatch):
    monkeypatch.setattr(settings, "api_key", LEGACY_KEY)
    monkeypatch.setattr(settings, "api_principals_json", json.dumps(_principal_config()))


@pytest.fixture
def api(monkeypatch):
    _configure_auth(monkeypatch)
    session = TrackingSession()
    service = StubDecisionService()
    monkeypatch.setattr(decisions_api, "DecisionService", lambda _db: service)
    app = FastAPI()
    app.include_router(decisions_api.router, prefix="/api/v1/decisions")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app), session, service


@pytest.fixture
def real_api(monkeypatch):
    _configure_auth(monkeypatch)
    session = TrackingSession()
    service = StubDecisionService()
    monkeypatch.setattr(decisions_api, "DecisionService", lambda _db: service)
    poseidon_app.dependency_overrides[get_db] = lambda: session
    try:
        yield TestClient(poseidon_app, raise_server_exceptions=False), session, service
    finally:
        poseidon_app.dependency_overrides.pop(get_db, None)


def _create_body(**overrides):
    body = {
        "evaluation_run_id": str(uuid.uuid4()),
        "account_scope": ACCOUNT,
        "original_json": {"final_action": "hold"},
        "portfolio_snapshot_json": {"cash": 1000},
        "risk_snapshot_json": {"hard_failures": [], "allowed_actions": ["hold"]},
    }
    body.update(overrides)
    return body


def _headers(key, idempotency_key=None):
    headers = {"X-API-Key": key}
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def test_worker_creates_and_scoped_principals_read_list_and_trace(api):
    client, session, service = api
    created = client.post("/api/v1/decisions/", json=_create_body(), headers=_headers(WORKER_KEY))
    assert created.status_code == 200
    assert created.json()["id"] == str(service.decision.id)
    assert session.commits == 1
    assert session.refreshes == 1

    for path in (
        "/api/v1/decisions/?status=pending_approval",
        f"/api/v1/decisions/{service.decision.id}",
        f"/api/v1/decisions/{service.decision.id}/trace",
    ):
        response = client.get(path, headers=_headers(VIEWER_KEY))
        assert response.status_code == 200
    assert session.commits == 1


@pytest.mark.parametrize(
    ("method", "path", "key", "body", "extra_headers"),
    [
        ("POST", "/api/v1/decisions/", MANAGER_KEY, _create_body(), {}),
        (
            "POST",
            "/api/v1/decisions/{id}/approve",
            WORKER_KEY,
            {"expected_revision": 1},
            {"Idempotency-Key": "worker-approve"},
        ),
        (
            "POST",
            "/api/v1/decisions/{id}/reject",
            VIEWER_KEY,
            {"expected_revision": 1, "reason": "no"},
            {"Idempotency-Key": "viewer-reject"},
        ),
        (
            "POST",
            "/api/v1/decisions/{id}/approve",
            CROSS_ACCOUNT_KEY,
            {"expected_revision": 1},
            {"Idempotency-Key": "cross-account"},
        ),
        ("GET", "/api/v1/decisions/{id}", LEGACY_KEY, None, {}),
    ],
)
def test_wrong_role_worker_legacy_and_cross_account_are_forbidden(api, method, path, key, body, extra_headers):
    client, session, service = api
    response = client.request(
        method,
        path.format(id=service.decision.id),
        json=body,
        headers={**_headers(key), **extra_headers},
    )
    assert response.status_code == 403
    if method == "POST":
        assert session.rollbacks == 1


@pytest.mark.parametrize("field", ["actor", "actor_id", "credential", "unknown"])
def test_extra_create_fields_are_rejected_before_service(api, field):
    client, _session, service = api
    response = client.post(
        "/api/v1/decisions/",
        json=_create_body(**{field: "untrusted"}),
        headers=_headers(WORKER_KEY),
    )
    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("path", ["approve", "reject"])
@pytest.mark.parametrize("field", ["actor", "actor_id", "service", "fingerprint", "key", "unknown"])
def test_extra_transition_fields_are_rejected_before_service(api, path, field):
    client, _session, service = api
    body = {"expected_revision": 1, field: "untrusted"}
    if path == "reject":
        body["reason"] = "not suitable"
    response = client.post(
        f"/api/v1/decisions/{service.decision.id}/{path}",
        json=body,
        headers=_headers(MANAGER_KEY, f"strict-{path}-{field}"),
    )
    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    ("operation", "error", "status"),
    [
        ("create", DecisionNotFoundError("missing"), 404),
        ("approve", DecisionNotFoundError("missing"), 404),
        ("approve", DecisionConflictError("stale"), 409),
        ("reject", DecisionConflictError("state"), 409),
        ("create", ValidationError("bad frozen input"), 422),
        ("approve", ValidationError("bad override"), 422),
    ],
)
def test_write_errors_are_mapped_and_rolled_back(api, operation, error, status):
    client, session, service = api
    service.failures[operation] = error
    if operation == "create":
        response = client.post("/api/v1/decisions/", json=_create_body(), headers=_headers(WORKER_KEY))
    else:
        body = {"expected_revision": 1}
        if operation == "reject":
            body["reason"] = "not suitable"
        response = client.post(
            f"/api/v1/decisions/{service.decision.id}/{operation}",
            json=body,
            headers=_headers(MANAGER_KEY, f"error-{operation}"),
        )
    assert response.status_code == status
    assert session.commits == 0
    assert session.rollbacks == 1


def test_missing_read_is_404_without_commit(api):
    client, session, service = api
    service.failures["get"] = DecisionNotFoundError("missing")
    response = client.get(f"/api/v1/decisions/{uuid.uuid4()}", headers=_headers(VIEWER_KEY))
    assert response.status_code == 404
    assert session.commits == 0


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("approve", {"expected_revision": 0}),
        ("approve", {"expected_revision": True}),
        ("approve", {"expected_revision": 1, "final_action": "buy"}),
        ("reject", {"expected_revision": 1, "reason": ""}),
        ("reject", {"expected_revision": 1, "reason": "   "}),
    ],
)
def test_invalid_transition_body_is_422_before_service(api, path, body):
    client, _session, service = api
    response = client.post(
        f"/api/v1/decisions/{service.decision.id}/{path}",
        json=body,
        headers=_headers(MANAGER_KEY, f"invalid-{path}"),
    )
    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("path", ["approve", "reject"])
def test_transition_requires_idempotency_header(api, path):
    client, _session, service = api
    body = {"expected_revision": 1}
    if path == "reject":
        body["reason"] = "not suitable"
    response = client.post(
        f"/api/v1/decisions/{service.decision.id}/{path}",
        json=body,
        headers=_headers(MANAGER_KEY),
    )
    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("path", ["approve", "reject"])
def test_exact_transition_replay_returns_original_response_with_one_event(api, path):
    client, session, service = api
    body = {"expected_revision": 1}
    if path == "reject":
        body["reason"] = "not suitable"

    def request():
        return client.post(
            f"/api/v1/decisions/{service.decision.id}/{path}",
            json=body,
            headers=_headers(MANAGER_KEY, f"same-{path}"),
        )

    first = request()
    second = request()
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert service.transition_events == 1
    assert session.commits == 2


def test_unknown_write_failure_rolls_back_and_is_not_reclassified(api):
    client, session, service = api
    service.failures["create"] = RuntimeError("unexpected")
    with pytest.raises(RuntimeError, match="unexpected"):
        client.post("/api/v1/decisions/", json=_create_body(), headers=_headers(WORKER_KEY))
    assert session.commits == 0
    assert session.rollbacks == 1


def test_authorization_http_exception_is_preserved(api):
    client, session, service = api
    service.failures["create"] = HTTPException(status_code=403, detail="Role not authorized")
    response = client.post("/api/v1/decisions/", json=_create_body(), headers=_headers(WORKER_KEY))
    assert response.status_code == 403
    assert response.json()["detail"] == "Role not authorized"
    assert session.rollbacks == 1


def test_real_app_mapped_principal_reaches_decisions_without_legacy_gate(real_api):
    client, session, service = real_api
    response = client.post("/api/v1/decisions/", json=_create_body(), headers=_headers(WORKER_KEY))
    assert response.status_code == 200
    assert response.json()["id"] == str(service.decision.id)
    assert session.commits == 1


def test_real_app_legacy_key_has_no_decision_authority_but_keeps_legacy_access(real_api):
    client, _session, service = real_api
    decision_response = client.get(
        f"/api/v1/decisions/{service.decision.id}",
        headers=_headers(LEGACY_KEY),
    )
    assert decision_response.status_code == 403

    legacy_response = client.get("/api/strategies", headers=_headers(LEGACY_KEY))
    assert legacy_response.status_code not in (401, 403)
    mapped_response = client.get("/api/strategies", headers=_headers(WORKER_KEY))
    assert mapped_response.status_code == 401


def test_real_app_exposes_only_the_decision_contract(real_api):
    client, _session, service = real_api
    schema = client.get("/openapi.json").json()
    decision_paths = {
        path: set(operations) for path, operations in schema["paths"].items() if path.startswith("/api/v1/decisions")
    }
    assert decision_paths == {
        "/api/v1/decisions/": {"get", "post"},
        "/api/v1/decisions/{decision_id}": {"get"},
        "/api/v1/decisions/{decision_id}/approve": {"post"},
        "/api/v1/decisions/{decision_id}/reject": {"post"},
        "/api/v1/decisions/{decision_id}/trace": {"get"},
    }

    for suffix in (
        "submit",
        "claim",
        "execute",
        "retry",
        "release",
        "orders",
        "manual-order",
        "manual-emergency",
    ):
        response = client.post(
            f"/api/v1/decisions/{service.decision.id}/{suffix}",
            headers=_headers(MANAGER_KEY),
        )
        assert response.status_code in (404, 405)


def test_real_app_trace_exposes_read_only_execution_audit_only(real_api):
    client, _session, service = real_api
    response = client.get(
        f"/api/v1/decisions/{service.decision.id}/trace",
        headers=_headers(VIEWER_KEY),
    )
    assert response.status_code == 200

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert {
        "execution_key",
        "orders",
        "fills",
        "lots",
        "allocations",
        "reconciliations",
    }.issubset(keys(response.json()))
    assert keys(response.json()).isdisjoint(
        {"credential", "credentials", "broker_snapshot_json", "internal_snapshot_json", "dsh"}
    )
