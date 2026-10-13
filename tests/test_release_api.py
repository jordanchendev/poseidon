"""Typed release API contracts for Phase 100."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api import capabilities as capabilities_api
from poseidon.api import releases as releases_api
from poseidon.api.auth import get_auth_principal
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError
from poseidon.decision_loop.manifest import ValidationError


ACCOUNT = "paper:pilot"
MANAGER_KEY = "release-manager-secret"
WORKER_KEY = "worker-secret"
VIEWER_KEY = "viewer-secret"
LEGACY_KEY = "legacy-secret"


def _fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _configure_auth(monkeypatch):
    principals = {
        _fingerprint(MANAGER_KEY): {
            "actor_id": "human:release-manager",
            "roles": ["portfolio_manager", "release_manager"],
            "account_scopes": [ACCOUNT],
        },
        _fingerprint(WORKER_KEY): {
            "actor_id": "service:decision-worker",
            "roles": ["decision-worker"],
            "account_scopes": [ACCOUNT],
        },
        _fingerprint(VIEWER_KEY): {
            "actor_id": "human:viewer",
            "roles": ["viewer"],
            "account_scopes": [ACCOUNT],
        },
    }
    monkeypatch.setattr(settings, "api_key", LEGACY_KEY)
    monkeypatch.setattr(settings, "api_principals_json", json.dumps(principals))


class TrackingSession:
    def __init__(self, strategy):
        self.strategy = strategy
        self.commits = 0
        self.rollbacks = 0

    def get(self, model, identity):
        if model.__name__ == "StrategyRecord" and identity == self.strategy.id:
            return self.strategy
        if model.__name__ == "StrategyVersion" and identity == self.strategy.active_version_id:
            return SimpleNamespace(
                id=identity,
                strategy_id=self.strategy.id,
                version_no=2,
                content_sha256="a" * 64,
                policy_json={"account_scope": ACCOUNT},
            )
        return None

    def scalars(self, _statement):
        return SimpleNamespace(all=lambda: [])

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class StubReleaseService:
    def __init__(self, strategy):
        self.strategy = strategy
        self.calls = []
        self.failure = None

    def evaluate_gate(self, *args, principal, **kwargs):
        self.calls.append(("gate", args, principal, kwargs))
        principal.require_account_scope(ACCOUNT)
        predicate = SimpleNamespace(
            as_json=lambda: {
                "code": "candidate_identity",
                "state": "passed",
                "reason": "candidate matches",
                "evidence": {"candidate_version_id": str(args[1])},
            }
        )
        return SimpleNamespace(allowed=True, predicates=(predicate,), as_json=lambda: {
            "allowed": True,
            "predicates": [predicate.as_json()],
        })

    def _mutate(self, operation, strategy_id, *args, principal, **kwargs):
        self.calls.append((operation, strategy_id, args, principal, kwargs))
        principal.require_role("release_manager")
        principal.require_account_scope(ACCOUNT)
        if self.failure is not None:
            raise self.failure
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": operation,
            "strategy_id": str(strategy_id),
            "from_version_id": None,
            "to_version_id": str(args[0]) if operation in {"release", "rollback"} else None,
            "release_revision": 1,
        }

    release = lambda self, strategy_id, candidate_version_id, *args, **kwargs: self._mutate(
        "release", strategy_id, candidate_version_id, *args, **kwargs
    )
    pause = lambda self, strategy_id, **kwargs: self._mutate("pause", strategy_id, **kwargs)
    rollback = lambda self, strategy_id, target_version_id, *args, **kwargs: self._mutate(
        "rollback", strategy_id, target_version_id, *args, **kwargs
    )
    reject = lambda self, strategy_id, candidate_version_id, **kwargs: self._mutate(
        "reject", strategy_id, candidate_version_id, **kwargs
    )


@pytest.fixture
def api(monkeypatch):
    _configure_auth(monkeypatch)
    strategy = SimpleNamespace(
        id=uuid.uuid4(), active_version_id=uuid.uuid4(), release_revision=0
    )
    session = TrackingSession(strategy)
    service = StubReleaseService(strategy)
    monkeypatch.setattr(releases_api, "ReleaseService", lambda _db: service)
    app = FastAPI()
    app.include_router(capabilities_api.router, prefix="/api/v1")
    app.include_router(releases_api.router, prefix="/api/v1/strategies")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app), session, service, strategy


def _headers(key, idempotency_key=None):
    result = {"X-API-Key": key}
    if idempotency_key:
        result["Idempotency-Key"] = idempotency_key
    return result


def _base_body(strategy, **overrides):
    body = {
        "expected_revision": strategy.release_revision,
        "expected_active_version_id": str(strategy.active_version_id),
        "reason": "operator reviewed exact evidence",
    }
    body.update(overrides)
    return body


def test_principal_contract_is_server_derived_and_capability_typed(api):
    client, _session, _service, _strategy = api
    response = client.get("/api/v1/principal", headers=_headers(MANAGER_KEY))
    assert response.status_code == 200
    assert response.json() == {
        "actor_id": "human:release-manager",
        "kind": "human",
        "roles": ["portfolio_manager", "release_manager"],
        "account_scopes": [ACCOUNT],
        "can_approve": True,
        "can_release": True,
    }
    assert set(response.json()).isdisjoint({"api_key", "credential", "fingerprint"})


def test_release_gate_and_four_mutations_use_typed_identity_contract(api):
    client, session, service, strategy = api
    candidate, campaign, review, reconciliation = (uuid.uuid4() for _ in range(4))
    gate = client.get(
        f"/api/v1/strategies/{strategy.id}/release-gate",
        params={
            "candidate_version_id": str(candidate),
            "campaign_id": str(campaign),
            "review_id": str(review),
            "reconciliation_id": str(reconciliation),
        },
        headers=_headers(VIEWER_KEY),
    )
    assert gate.status_code == 200
    assert gate.json()["predicates"][0]["state"] == "passed"

    cases = {
        "release": _base_body(
            strategy,
            candidate_version_id=str(candidate),
            campaign_id=str(campaign),
            review_id=str(review),
            reconciliation_id=str(reconciliation),
        ),
        "pause": _base_body(strategy),
        "rollback": _base_body(
            strategy,
            target_version_id=str(candidate),
            reconciliation_id=str(reconciliation),
        ),
        "reject": _base_body(strategy, candidate_version_id=str(candidate)),
    }
    for operation, body in cases.items():
        response = client.post(
            f"/api/v1/strategies/{strategy.id}/{operation}",
            json=body,
            headers=_headers(MANAGER_KEY, f"key-{operation}"),
        )
        assert response.status_code == 200
        assert response.json()["event_type"] == operation
    assert session.commits == 4
    assert [call[0] for call in service.calls] == ["gate", "release", "pause", "rollback", "reject"]


@pytest.mark.parametrize("key", [WORKER_KEY, VIEWER_KEY, LEGACY_KEY])
def test_non_release_principals_receive_403_and_transaction_rolls_back(api, key):
    client, session, service, strategy = api
    response = client.post(
        f"/api/v1/strategies/{strategy.id}/pause",
        json=_base_body(strategy),
        headers=_headers(key, "denied-key"),
    )
    assert response.status_code == 403
    assert session.commits == 0 and session.rollbacks == 1
    assert not service.calls or service.calls[-1][0] == "pause"


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (DecisionConflictError("strategy release revision changed"), 409),
        (DecisionConflictError("release gate did not pass: review_identity"), 422),
        (ValidationError("invalid release input"), 422),
    ],
)
def test_release_errors_are_stable_and_rollback(api, failure, status):
    client, session, service, strategy = api
    service.failure = failure
    response = client.post(
        f"/api/v1/strategies/{strategy.id}/pause",
        json=_base_body(strategy),
        headers=_headers(MANAGER_KEY, "error-key"),
    )
    assert response.status_code == status
    assert session.commits == 0 and session.rollbacks == 1


def test_release_api_openapi_has_generated_client_models_and_errors(api):
    client, _session, _service, _strategy = api
    schema = client.get("/openapi.json").json()
    schemas = schema["components"]["schemas"]
    for model in (
        "PrincipalResponse",
        "ReleaseGateResponse",
        "ReleaseEventResponse",
        "ReleaseRequest",
        "PauseRequest",
        "RollbackRequest",
        "RejectRequest",
        "ReleaseMutationResponse",
        "ReleaseAPIError",
    ):
        assert model in schemas
    mutation = schema["paths"]["/api/v1/strategies/{strategy_id}/release"]["post"]
    assert {"403", "404", "409", "422"}.issubset(mutation["responses"])
    request_properties = schemas["ReleaseRequest"]["properties"]
    assert set(request_properties).isdisjoint(
        {"actor", "actor_id", "roles", "account_scope", "credential", "api_key"}
    )
    assert all("order" not in path for path in schema["paths"])
