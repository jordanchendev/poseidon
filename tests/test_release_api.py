"""Typed release API contracts for Phase 100."""

import hashlib
import importlib
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from poseidon.api import capabilities as capabilities_api
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError
from poseidon.decision_loop.manifest import ValidationError
from poseidon.decision_loop.releases import (
    ReleaseGateRejected,
    ReleaseGateResult,
    ReleasePredicate,
)
from poseidon.decision_loop.releases import (
    ReleaseService as DomainReleaseService,
)
from poseidon.main import app as poseidon_app
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion

try:
    releases_api = importlib.import_module("poseidon.api.releases")
except ModuleNotFoundError:
    releases_api = SimpleNamespace(router=APIRouter(), ReleaseService=None)


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
        self.versions = {
            strategy.active_version_id: SimpleNamespace(
                id=strategy.active_version_id,
                strategy_id=strategy.id,
                version_no=2,
                content_sha256="a" * 64,
                policy_json={"account_scope": ACCOUNT},
            )
        }
        self.events = []
        self.commits = 0
        self.rollbacks = 0

    def get(self, model, identity):
        if model.__name__ == "StrategyRecord" and identity == self.strategy.id:
            return self.strategy
        if model.__name__ == "StrategyVersion":
            return self.versions.get(identity)
        return None

    def scalars(self, statement):
        entity = statement.column_descriptions[0].get("entity")
        if entity is StrategyReleaseEvent:
            values = list(self.events)
        elif entity is StrategyVersion:
            values = list(self.versions.values())
        else:
            values = []
        return SimpleNamespace(all=lambda: values)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class StubReleaseService:
    def __init__(self, strategy):
        self.strategy = strategy
        self.calls = []
        self.failure = None
        self.gate_allowed = True
        self.replays = {}

    def evaluate_gate(self, *args, principal, **kwargs):
        self.calls.append(("gate", args, principal, kwargs))
        principal.require_account_scope(ACCOUNT)
        state = "passed" if self.gate_allowed else "blocked"
        predicate = SimpleNamespace(
            as_json=lambda: {
                "code": "candidate_identity",
                "state": state,
                "reason": "candidate matches" if self.gate_allowed else "candidate is unsafe",
                "evidence": {"candidate_version_id": str(args[1])},
            }
        )
        return SimpleNamespace(
            allowed=self.gate_allowed,
            predicates=(predicate,),
            as_json=lambda: {
                "allowed": self.gate_allowed,
                "predicates": [predicate.as_json()],
            },
        )

    def _mutate(self, operation, strategy_id, *args, principal, **kwargs):
        self.calls.append((operation, strategy_id, args, principal, kwargs))
        principal.require_role("release_manager")
        principal.require_account_scope(ACCOUNT)
        if self.failure is not None:
            raise self.failure
        request = (operation, str(strategy_id), args, tuple(sorted(kwargs.items(), key=lambda item: item[0])))
        replay_key = (operation, strategy_id, kwargs["idempotency_key"])
        if replay_key in self.replays:
            if self.replays[replay_key][0] != request:
                raise DecisionConflictError("idempotency key was used for a different request")
            return self.replays[replay_key][1]
        response = {
            "event_id": str(uuid.uuid4()),
            "event_type": operation,
            "strategy_id": str(strategy_id),
            "from_version_id": None,
            "to_version_id": str(args[0]) if operation in {"release", "rollback"} else None,
            "release_revision": 1,
        }
        self.replays[replay_key] = (request, response)
        return response

    def _current_or_paused_authority_version(self, _strategy):
        return next(iter(self.strategy_session.versions.values()))

    def _policy_for_version(self, version, _strategy, principal):
        principal.require_account_scope(version.policy_json["account_scope"])
        return SimpleNamespace(
            account_scope=version.policy_json["account_scope"],
            reconciliation=SimpleNamespace(account_generation="pilot-generation"),
        )

    def _rollback_gate(self, _strategy, target, _reconciliation, _policy, principal, _now):
        principal.require_account_scope(ACCOUNT)
        eligible = target.content_sha256 == "a" * 64
        predicate = SimpleNamespace(
            as_json=lambda: {
                "code": "rollback_target",
                "state": "passed" if eligible else "blocked",
                "reason": "prior released immutable target matches" if eligible else "target is unsafe",
                "evidence": {"strategy_version_id": str(target.id)},
            }
        )
        source = next(event for event in self.strategy_session.events if event.to_version_id == target.id)
        return SimpleNamespace(allowed=eligible, predicates=(predicate,)), source, _policy

    def read_release_state(self, strategy):
        self.calls.append(("state", strategy.id))
        version = self.strategy_session.versions[strategy.active_version_id]
        identity = {
            "strategy_version_id": version.id,
            "version_no": version.version_no,
            "status": "draft",
            "content_sha256": version.content_sha256,
            "policy_sha256": "c" * 64,
            "account_scope": ACCOUNT,
            "created_at": datetime(2026, 10, 13, tzinfo=UTC),
        }
        return {
            "strategy_id": strategy.id,
            "release_revision": strategy.release_revision,
            "active_version_id": strategy.active_version_id,
            "pointer_state": "active",
            "active_version": identity,
            "authority_version": identity,
            "latest_event_id": None,
            "latest_event_type": None,
            "authority_reconciliation": None,
            "candidates": [],
        }

    def release(self, strategy_id, candidate_version_id, *args, **kwargs):
        return self._mutate("release", strategy_id, candidate_version_id, *args, **kwargs)

    def pause(self, strategy_id, **kwargs):
        return self._mutate("pause", strategy_id, **kwargs)

    def rollback(self, strategy_id, target_version_id, *args, **kwargs):
        return self._mutate("rollback", strategy_id, target_version_id, *args, **kwargs)

    def reject(self, strategy_id, candidate_version_id, **kwargs):
        return self._mutate("reject", strategy_id, candidate_version_id, **kwargs)


@pytest.fixture
def api(monkeypatch):
    _configure_auth(monkeypatch)
    strategy = SimpleNamespace(id=uuid.uuid4(), active_version_id=uuid.uuid4(), release_revision=0)
    session = TrackingSession(strategy)
    service = StubReleaseService(strategy)
    service.strategy_session = session
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


def test_legacy_principal_kind_is_explicit_not_human(api):
    client, _session, _service, _strategy = api
    response = client.get("/api/v1/principal", headers=_headers(LEGACY_KEY))
    assert response.status_code == 200
    assert response.json()["kind"] == "legacy"
    assert response.json()["can_approve"] is False
    assert response.json()["can_release"] is False


def test_release_gate_and_four_mutations_use_typed_identity_contract(api):
    client, session, service, strategy = api
    candidate, campaign, review, reconciliation = (uuid.uuid4() for _ in range(4))
    session.versions[candidate] = SimpleNamespace(
        id=candidate,
        strategy_id=strategy.id,
        version_no=3,
        content_sha256="b" * 64,
        policy_json={"account_scope": ACCOUNT},
    )
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


def test_pause_exact_replay_is_resolved_before_active_pointer_lookup(api):
    client, session, service, strategy = api
    body = _base_body(strategy)
    first = client.post(
        f"/api/v1/strategies/{strategy.id}/pause",
        json=body,
        headers=_headers(MANAGER_KEY, "pause-replay"),
    )
    strategy.active_version_id = None
    second = client.post(
        f"/api/v1/strategies/{strategy.id}/pause",
        json=body,
        headers=_headers(MANAGER_KEY, "pause-replay"),
    )

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(service.replays) == 1
    assert session.commits == 2


def test_gate_failure_returns_all_typed_predicates(api):
    client, session, service, strategy = api
    service.gate_allowed = False
    candidate, campaign, review, reconciliation = (uuid.uuid4() for _ in range(4))
    service.failure = ReleaseGateRejected(
        "release",
        ReleaseGateResult(
            (
                ReleasePredicate(
                    "candidate_identity",
                    "blocked",
                    "candidate is unsafe",
                    {"candidate_version_id": str(candidate)},
                ),
            )
        ),
    )
    session.versions[candidate] = SimpleNamespace(
        id=candidate,
        strategy_id=strategy.id,
        version_no=3,
        content_sha256="b" * 64,
        policy_json={"account_scope": ACCOUNT},
    )
    response = client.post(
        f"/api/v1/strategies/{strategy.id}/release",
        json=_base_body(
            strategy,
            candidate_version_id=str(candidate),
            campaign_id=str(campaign),
            review_id=str(review),
            reconciliation_id=str(reconciliation),
        ),
        headers=_headers(MANAGER_KEY, "blocked-release"),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "release_gate_failed",
        "message": "Release gate did not pass",
        "gate": {
            "allowed": False,
            "predicates": [
                {
                    "code": "candidate_identity",
                    "state": "blocked",
                    "reason": "candidate is unsafe",
                    "evidence": {"candidate_version_id": str(candidate)},
                }
            ],
        },
    }
    assert session.rollbacks == 1


def test_release_reads_authorize_empty_results_and_filter_unsafe_rollback_targets(api):
    client, session, _service, strategy = api
    current, released, unsafe = strategy.active_version_id, uuid.uuid4(), uuid.uuid4()
    session.versions[released] = SimpleNamespace(
        id=released,
        strategy_id=strategy.id,
        version_no=1,
        content_sha256="a" * 64,
        policy_json={"account_scope": ACCOUNT},
    )
    session.versions[unsafe] = SimpleNamespace(
        id=unsafe,
        strategy_id=strategy.id,
        version_no=1,
        content_sha256="0" * 64,
        policy_json={"account_scope": ACCOUNT},
    )
    for revision, version in enumerate((released, unsafe), start=1):
        session.events.append(
            SimpleNamespace(
                id=uuid.uuid4(),
                strategy_id=strategy.id,
                event_type="release",
                candidate_version_id=version,
                from_version_id=None,
                to_version_id=version,
                source_release_event_id=None,
                campaign_id=uuid.uuid4(),
                review_id=uuid.uuid4(),
                holdout_use_id=uuid.uuid4(),
                reconciliation_id=uuid.uuid4(),
                actor_id="human:release-manager",
                account_scope=ACCOUNT,
                expected_revision=revision - 1,
                resulting_revision=revision,
                reason="released",
                gate_json={"allowed": True, "predicates": []},
                created_at=datetime.now(UTC),
            )
        )

    history = client.get(
        f"/api/v1/strategies/{strategy.id}/release-events",
        headers=_headers(VIEWER_KEY),
    )
    targets = client.get(
        f"/api/v1/strategies/{strategy.id}/rollback-targets",
        params={"reconciliation_id": str(uuid.uuid4())},
        headers=_headers(MANAGER_KEY),
    )
    assert history.status_code == 200
    assert [event["resulting_revision"] for event in history.json()] == [1, 2]
    assert targets.status_code == 200
    assert [target["strategy_version_id"] for target in targets.json()] == [str(released)]
    assert all(target["strategy_version_id"] != str(current) for target in targets.json())
    assert targets.json()[0]["eligible"] is True
    assert targets.json()[0]["predicates"][0]["code"] == "rollback_target"

    session.events.clear()
    denied = client.get(
        f"/api/v1/strategies/{strategy.id}/release-events",
        headers=_headers(LEGACY_KEY),
    )
    assert denied.status_code == 403


def test_missing_release_read_is_404(api):
    client, _session, _service, _strategy = api
    response = client.get(
        f"/api/v1/strategies/{uuid.uuid4()}/release-events",
        headers=_headers(VIEWER_KEY),
    )
    assert response.status_code == 404


@pytest.mark.postgresql
def test_real_release_service_pause_replay_and_conflicts_through_api(
    monkeypatch,
    phase99_session_factory,
):
    from tests.test_strategy_release import _release, _seed_gate_graph

    _configure_auth(monkeypatch)
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session)
        session.commit()
        _release(DomainReleaseService(session), graph)
        session.commit()

        app = FastAPI()
        app.include_router(releases_api.router, prefix="/api/v1/strategies")
        app.dependency_overrides[get_db] = lambda: session
        client = TestClient(app)
        body = {
            "expected_revision": 1,
            "expected_active_version_id": str(graph.candidate_id),
            "reason": "operator pause",
        }

        first = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/pause",
            json=body,
            headers=_headers(MANAGER_KEY, "real-pause"),
        )
        replay = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/pause",
            json=body,
            headers=_headers(MANAGER_KEY, "real-pause"),
        )
        reused = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/pause",
            json=body | {"reason": "changed"},
            headers=_headers(MANAGER_KEY, "real-pause"),
        )
        stale = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/pause",
            json=body,
            headers=_headers(MANAGER_KEY, "real-pause-stale"),
        )

        assert first.status_code == replay.status_code == 200
        assert first.json() == replay.json()
        assert reused.status_code == stale.status_code == 409
        assert (
            session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id, event_type="pause").count()
            == 1
        )


@pytest.mark.postgresql
def test_real_failed_review_gate_and_never_released_targets_are_serializable(
    monkeypatch,
    phase99_session_factory,
):
    from tests.test_strategy_release import _seed_gate_graph

    _configure_auth(monkeypatch)
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session, review_status="failed")
        session.commit()
        app = FastAPI()
        app.include_router(releases_api.router, prefix="/api/v1/strategies")
        app.dependency_overrides[get_db] = lambda: session
        client = TestClient(app, raise_server_exceptions=False)

        gate = client.get(
            f"/api/v1/strategies/{graph.strategy_id}/release-gate",
            params={
                "candidate_version_id": str(graph.candidate_id),
                "campaign_id": str(graph.campaign_id),
                "review_id": str(graph.review_id),
                "reconciliation_id": str(graph.reconciliation_id),
            },
            headers=_headers(MANAGER_KEY),
        )
        targets = client.get(
            f"/api/v1/strategies/{graph.strategy_id}/rollback-targets",
            params={"reconciliation_id": str(graph.reconciliation_id)},
            headers=_headers(MANAGER_KEY),
        )
        mutation = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/release",
            json={
                "expected_revision": 0,
                "expected_active_version_id": None,
                "reason": "reviewed",
                "candidate_version_id": str(graph.candidate_id),
                "campaign_id": str(graph.campaign_id),
                "review_id": str(graph.review_id),
                "reconciliation_id": str(graph.reconciliation_id),
            },
            headers=_headers(MANAGER_KEY, "failed-review"),
        )

        assert gate.status_code == 200
        review_gate = next(item for item in gate.json()["predicates"] if item["code"] == "review_identity")
        assert review_gate["evidence"]["status"] == "failed"
        assert targets.status_code == 200
        assert targets.json() == []
        assert mutation.status_code == 422
        failed_review = next(
            item for item in mutation.json()["detail"]["gate"]["predicates"] if item["code"] == "review_identity"
        )
        assert failed_review["evidence"]["status"] == "failed"
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=graph.strategy_id).count() == 0


def test_formal_app_422_contract_covers_request_and_gate_validation(api):
    _client, session, _service, strategy = api
    poseidon_app.dependency_overrides[get_db] = lambda: session
    poseidon_app.openapi_schema = None
    try:
        client = TestClient(poseidon_app)
        schema = client.get("/openapi.json").json()
        response_schema = schema["paths"]["/api/v1/strategies/{strategy_id}/release"]["post"]["responses"]["422"][
            "content"
        ]["application/json"]["schema"]
        refs = {branch["$ref"] for branch in response_schema["anyOf"]}
        assert refs == {
            "#/components/schemas/ReleaseUnprocessableResponse",
            "#/components/schemas/RequestValidationResponse",
        }

        invalid = client.post(
            f"/api/v1/strategies/{strategy.id}/release",
            json={},
            headers=_headers(MANAGER_KEY, "invalid-body"),
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"] == "Validation error"
        assert invalid.json()["error_code"] == "VALIDATION_ERROR"
        assert invalid.json()["status_code"] == 422
        assert {"loc", "msg", "type"} <= set(invalid.json()["errors"][0])
    finally:
        poseidon_app.dependency_overrides.pop(get_db, None)
        poseidon_app.openapi_schema = None


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
        (
            ReleaseGateRejected(
                "release",
                ReleaseGateResult(
                    (
                        ReleasePredicate(
                            "review_identity",
                            "blocked",
                            "review is unsafe",
                            {"review_id": str(uuid.uuid4())},
                        ),
                    )
                ),
            ),
            422,
        ),
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
        "ReleasePredicateEvidence",
        "ReleaseGateFailureDetail",
    ):
        assert model in schemas
    mutation = schema["paths"]["/api/v1/strategies/{strategy_id}/release"]["post"]
    assert {"403", "404", "409", "422"}.issubset(mutation["responses"])
    request_properties = schemas["ReleaseRequest"]["properties"]
    assert set(request_properties).isdisjoint({"actor", "actor_id", "roles", "account_scope", "credential", "api_key"})
    assert all("order" not in path for path in schema["paths"])
    assert schemas["PrincipalResponse"]["properties"]["kind"]["enum"] == [
        "human",
        "worker",
        "legacy",
    ]
    gate_json = schemas["ReleaseEventResponse"]["properties"]["gate_json"]
    assert gate_json == {"$ref": "#/components/schemas/ReleaseGateResponse"}
    evidence = schemas["ReleasePredicateResponse"]["properties"]["evidence"]
    assert evidence == {"$ref": "#/components/schemas/ReleasePredicateEvidence"}


def test_release_state_read_matches_sibling_status_semantics(api):
    client, session, service, strategy = api
    state = client.get(f"/api/v1/strategies/{strategy.id}/release-state", headers=_headers(VIEWER_KEY))

    assert state.status_code == 200
    body = state.json()
    assert body["strategy_id"] == str(strategy.id)
    assert body["release_revision"] == strategy.release_revision
    assert body["active_version_id"] == str(strategy.active_version_id)
    assert body["active_version"]["content_sha256"] == "a" * 64
    assert body["latest_event_id"] is None and "authority_reconciliation" in body
    assert body["candidates"] == []

    unknown = uuid.uuid4()
    for path in ("release-state", "release-events"):
        missing = client.get(f"/api/v1/strategies/{unknown}/{path}", headers=_headers(VIEWER_KEY))
        assert missing.status_code == 404
    unauthenticated = client.get(f"/api/v1/strategies/{strategy.id}/release-state")
    sibling_unauthenticated = client.get(f"/api/v1/strategies/{strategy.id}/release-events")
    assert unauthenticated.status_code == sibling_unauthenticated.status_code
    assert unauthenticated.status_code in {401, 403}
    legacy = client.get(f"/api/v1/strategies/{strategy.id}/release-state", headers=_headers(LEGACY_KEY))
    sibling_legacy = client.get(f"/api/v1/strategies/{strategy.id}/release-events", headers=_headers(LEGACY_KEY))
    assert legacy.status_code == sibling_legacy.status_code == 403
    assert session.commits == 0 and session.rollbacks == 0
    assert [call[0] for call in service.calls] == ["state"]


def test_release_state_openapi_is_typed_read_only_contract(api):
    client, _session, _service, _strategy = api
    schema = client.get("/openapi.json").json()
    path = schema["paths"]["/api/v1/strategies/{strategy_id}/release-state"]
    assert set(path) == {"get"}
    operation = path["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ReleaseStateResponse"
    }
    assert {"403", "404"}.issubset(operation["responses"])
    assert [parameter["name"] for parameter in operation.get("parameters", [])] == ["strategy_id"]
    schemas = schema["components"]["schemas"]
    for model in (
        "ReleaseStateResponse",
        "ReleaseStateCandidate",
        "ReleaseStateVersionIdentity",
        "ReleaseStateReconciliation",
        "ReleaseStateReview",
        "ReleaseStateHoldout",
        "ReleaseStateHoldoutDenial",
        "ReleaseStateTrial",
    ):
        assert model in schemas
    state = schemas["ReleaseStateResponse"]
    assert {"release_revision", "active_version_id", "candidates", "pointer_state"} <= set(state["required"])
    assert state["properties"]["pointer_state"]["enum"] == ["unreleased", "active", "paused"]
    assert schemas["ReleaseStateHoldout"]["properties"]["state"]["enum"] == [
        "consumed_by_this_campaign",
        "consumed_by_another_campaign",
        "reuse_denied",
        "ownership_mismatch",
        "unavailable",
    ]
    candidate = schemas["ReleaseStateCandidate"]["properties"]
    assert set(candidate).isdisjoint({"allowed", "eligible", "predicates", "sharpe"})


def _release_api_client(session):
    app = FastAPI()
    app.include_router(releases_api.router, prefix="/api/v1/strategies")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def _authority_counts(session, strategy_id, campaign_ids):
    from poseidon.models.account_reconciliation import AccountReconciliation
    from poseidon.models.experiment_campaign import CampaignEvent, CampaignReview, HoldoutUse
    from poseidon.models.strategy import StrategyRecord

    session.expire_all()
    strategy = session.get(StrategyRecord, strategy_id)
    return {
        "pointer": (strategy.active_version_id, strategy.release_revision),
        "release_events": session.query(StrategyReleaseEvent).filter_by(strategy_id=strategy_id).count(),
        "campaign_events": session.query(CampaignEvent).filter(CampaignEvent.campaign_id.in_(campaign_ids)).count(),
        "reviews": session.query(CampaignReview).filter(CampaignReview.campaign_id.in_(campaign_ids)).count(),
        "holdouts": session.query(HoldoutUse).count(),
        "reconciliations": session.query(AccountReconciliation).count(),
        "versions": session.query(StrategyVersion).filter_by(strategy_id=strategy_id).count(),
    }


@pytest.mark.postgresql
def test_real_release_state_for_unreleased_strategy_without_candidates(monkeypatch, phase99_session_factory):
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_version import strategy_version_digest
    from tests.test_decision_service import synthetic_policy, synthetic_reconciliation_policy

    _configure_auth(monkeypatch)
    with phase99_session_factory() as session:
        marker = uuid.uuid4().hex
        strategy = StrategyRecord(
            name=f"phase100-state-empty-{marker}",
            strategy_type="technical",
            config={},
            symbol="2330",
            market="tw_stock",
            interval="1d",
            active=False,
        )
        session.add(strategy)
        session.flush()
        policy = synthetic_policy(reconciliation=synthetic_reconciliation_policy(account_generation=f"gen-{marker}"))
        config = {"marker": marker}
        session.add(
            StrategyVersion(
                strategy_id=strategy.id,
                version_no=1,
                status="draft",
                config_json=config,
                policy_json=policy,
                artifact_json={},
                content_sha256=strategy_version_digest(config, policy, {}),
            )
        )
        session.commit()
        before = _authority_counts(session, strategy.id, [])

        response = _release_api_client(session).get(
            f"/api/v1/strategies/{strategy.id}/release-state",
            headers=_headers(VIEWER_KEY),
        )

        assert response.status_code == 200
        assert response.json() == {
            "strategy_id": str(strategy.id),
            "release_revision": 0,
            "active_version_id": None,
            "pointer_state": "unreleased",
            "active_version": None,
            "authority_version": None,
            "latest_event_id": None,
            "latest_event_type": None,
            "authority_reconciliation": None,
            "candidates": [],
        }
        assert _authority_counts(session, strategy.id, []) == before


def _fresh_reconciliation(session, graph):
    from poseidon.models.account_reconciliation import AccountReconciliation

    seeded = session.get(AccountReconciliation, graph.reconciliation_id)
    fresh = AccountReconciliation(
        account_scope=seeded.account_scope,
        account_generation=seeded.account_generation,
        as_of=datetime.now(UTC),
        broker_state_watermark=f"{seeded.broker_state_watermark}:fresh",
        internal_state_watermark=f"{seeded.internal_state_watermark}:fresh",
        broker_snapshot_sha256=seeded.broker_snapshot_sha256,
        broker_snapshot_json={},
        internal_snapshot_json={},
        difference_json={},
        policy_sha256=seeded.policy_sha256,
        status="matched",
    )
    session.add(fresh)
    session.flush()
    return fresh


@pytest.mark.postgresql
def test_real_release_state_identities_feed_gate_release_and_pause(monkeypatch, phase99_session_factory):
    from poseidon.models.experiment_campaign import CampaignReview, ExperimentCampaign
    from tests.test_decision_service import synthetic_reconciliation_policy
    from tests.test_strategy_release import _seed_gate_graph

    _configure_auth(monkeypatch)
    with phase99_session_factory() as session:
        generation = f"gen-{uuid.uuid4().hex}"
        graph = _seed_gate_graph(
            session,
            policy_changes={"reconciliation": synthetic_reconciliation_policy(account_generation=generation)},
        )
        fresh = _fresh_reconciliation(session, graph)
        session.commit()
        campaign = session.get(ExperimentCampaign, graph.campaign_id)
        review = session.get(CampaignReview, graph.review_id)
        candidate_version = session.get(StrategyVersion, graph.candidate_id)
        client = _release_api_client(session)
        before = _authority_counts(session, graph.strategy_id, [graph.campaign_id])

        state = client.get(f"/api/v1/strategies/{graph.strategy_id}/release-state", headers=_headers(VIEWER_KEY))

        assert state.status_code == 200
        body = state.json()
        assert _authority_counts(session, graph.strategy_id, [graph.campaign_id]) == before
        assert body["release_revision"] == 0
        assert body["active_version_id"] is None
        assert body["pointer_state"] == "unreleased"
        assert body["authority_reconciliation"] is None
        assert len(body["candidates"]) == 1
        candidate = body["candidates"][0]
        assert candidate["candidate"]["strategy_version_id"] == str(graph.candidate_id)
        assert candidate["candidate"]["content_sha256"] == candidate_version.content_sha256
        assert candidate["candidate"]["account_scope"] == ACCOUNT
        assert candidate["campaign_id"] == str(graph.campaign_id)
        assert candidate["hypothesis"] == campaign.hypothesis
        assert candidate["contract_sha256"] == campaign.contract_sha256
        assert candidate["incumbent_version_id"] == str(campaign.incumbent_strategy_version_id)
        assert candidate["incumbent_content_sha256"] == campaign.incumbent_content_sha256
        assert candidate["declared_trial_count"] == 6
        assert candidate["terminal_trial_count"] == 6
        assert {(trial["version_role"], trial["ablation_arm"]) for trial in candidate["trials"]} == {
            (role, arm)
            for role in ("incumbent", "candidate")
            for arm in ("fundamental_only", "technical_only", "combined")
        }
        assert candidate["holdout_reuse_denials"] == []
        assert [row["review_id"] for row in candidate["reviews"]] == [str(graph.review_id)]
        assert candidate["reviews"][0]["status"] == "passed"
        assert candidate["reviews"][0]["input_sha256"] == review.input_sha256
        assert candidate["reviews"][0]["result_sha256"] == review.result_sha256
        assert candidate["reviews"][0]["holdout"] == {
            "state": "consumed_by_this_campaign",
            "holdout_use_id": str(review.holdout_use_id),
            "holdout_identity_sha256": graph.holdout_identity,
            "owner_campaign_id": str(graph.campaign_id),
        }
        assert candidate["account_scope"] == ACCOUNT
        assert candidate["account_generation"] == generation
        assert candidate["reconciliation"]["reconciliation_id"] == str(fresh.id)
        assert candidate["reconciliation"]["status"] == "matched"

        gate_query = {
            "candidate_version_id": candidate["candidate"]["strategy_version_id"],
            "campaign_id": candidate["campaign_id"],
            "review_id": candidate["reviews"][0]["review_id"],
            "reconciliation_id": candidate["reconciliation"]["reconciliation_id"],
        }
        gate = client.get(
            f"/api/v1/strategies/{graph.strategy_id}/release-gate",
            params=gate_query,
            headers=_headers(MANAGER_KEY),
        )
        assert gate.status_code == 200
        assert gate.json()["allowed"] is True, gate.json()

        released = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/release",
            json={
                "expected_revision": body["release_revision"],
                "expected_active_version_id": body["active_version_id"],
                "reason": "release from read-only state identities",
                **gate_query,
            },
            headers=_headers(MANAGER_KEY, f"state-release-{generation}"),
        )
        assert released.status_code == 200, released.json()

        after_release = client.get(
            f"/api/v1/strategies/{graph.strategy_id}/release-state",
            headers=_headers(VIEWER_KEY),
        ).json()
        assert after_release["release_revision"] == released.json()["release_revision"] == 1
        assert after_release["active_version_id"] == str(graph.candidate_id)
        assert after_release["pointer_state"] == "active"
        assert after_release["active_version"]["content_sha256"] == candidate_version.content_sha256
        assert after_release["latest_event_id"] == released.json()["event_id"]
        assert after_release["latest_event_type"] == "release"
        assert after_release["authority_reconciliation"]["reconciliation_id"] == str(fresh.id)

        paused = client.post(
            f"/api/v1/strategies/{graph.strategy_id}/pause",
            json={
                "expected_revision": after_release["release_revision"],
                "expected_active_version_id": after_release["active_version_id"],
                "reason": "pause from read-only state identities",
            },
            headers=_headers(MANAGER_KEY, f"state-pause-{generation}"),
        )
        assert paused.status_code == 200, paused.json()
        after_pause = client.get(
            f"/api/v1/strategies/{graph.strategy_id}/release-state",
            headers=_headers(VIEWER_KEY),
        ).json()
        assert after_pause["pointer_state"] == "paused"
        assert after_pause["active_version_id"] is None
        assert after_pause["active_version"] is None
        assert after_pause["authority_version"]["strategy_version_id"] == str(graph.candidate_id)
        assert after_pause["release_revision"] == 2
        assert after_pause["latest_event_type"] == "pause"


@pytest.mark.postgresql
def test_real_release_state_labels_foreign_and_denied_holdouts_like_gate(monkeypatch, phase99_session_factory):
    from poseidon.decision_loop.manifest import content_sha256
    from poseidon.models.experiment import ExperimentRecord
    from poseidon.models.experiment_campaign import CampaignReview, ExperimentCampaign
    from poseidon.research.campaign import CampaignService, build_review_input_identity
    from tests.test_strategy_release import _seed_gate_graph

    _configure_auth(monkeypatch)
    with phase99_session_factory() as session:
        owner = _seed_gate_graph(session)
        foreign = _seed_gate_graph(session)
        denied = _seed_gate_graph(session)
        foreign_campaign = session.get(ExperimentCampaign, foreign.campaign_id)
        foreign_review = session.get(CampaignReview, foreign.review_id)
        trials = session.query(ExperimentRecord).filter_by(campaign_id=foreign.campaign_id).all()
        borrowed_identity = build_review_input_identity(
            foreign_campaign,
            trials,
            holdout_identity_sha256=owner.holdout_identity,
            panel_sha256="f" * 64,
        )
        borrowed = CampaignReview(
            campaign_id=foreign.campaign_id,
            input_sha256=content_sha256(borrowed_identity),
            result_sha256=foreign_review.result_sha256,
            status="passed",
            result_json=foreign_review.result_json,
            holdout_use_id=foreign_review.holdout_use_id,
            input_identity_json=borrowed_identity,
        )
        session.add(borrowed)
        CampaignService(session).append_event(
            denied.campaign_id,
            event_type="holdout_reuse_denied",
            payload_json={
                "holdout_identity_sha256": denied.holdout_identity,
                "owner_campaign_id": str(owner.campaign_id),
                "owner_campaign_contract_sha256": "0" * 64,
            },
        )
        session.commit()
        client = _release_api_client(session)

        def holdout_rows(graph):
            response = client.get(f"/api/v1/strategies/{graph.strategy_id}/release-state", headers=_headers(VIEWER_KEY))
            assert response.status_code == 200
            (candidate,) = response.json()["candidates"]
            return candidate, {row["review_id"]: row["holdout"] for row in candidate["reviews"]}

        def gate_holdout(graph, review_id):
            response = client.get(
                f"/api/v1/strategies/{graph.strategy_id}/release-gate",
                params={
                    "candidate_version_id": str(graph.candidate_id),
                    "campaign_id": str(graph.campaign_id),
                    "review_id": str(review_id),
                    "reconciliation_id": str(graph.reconciliation_id),
                },
                headers=_headers(MANAGER_KEY),
            )
            assert response.status_code == 200
            return next(item for item in response.json()["predicates"] if item["code"] == "holdout_ownership")

        before = _authority_counts(session, foreign.strategy_id, [foreign.campaign_id, denied.campaign_id])
        _candidate, foreign_rows = holdout_rows(foreign)
        assert foreign_rows[str(foreign.review_id)]["state"] == "consumed_by_this_campaign"
        assert foreign_rows[str(borrowed.id)] == {
            "state": "consumed_by_another_campaign",
            "holdout_use_id": str(foreign_review.holdout_use_id),
            "holdout_identity_sha256": owner.holdout_identity,
            "owner_campaign_id": str(owner.campaign_id),
        }
        assert gate_holdout(foreign, foreign.review_id)["state"] == "passed"
        foreign_gate = gate_holdout(foreign, borrowed.id)
        assert foreign_gate["state"] == "blocked"
        assert foreign_gate["evidence"]["owner_campaign_id"] == str(owner.campaign_id)

        denied_candidate, denied_rows = holdout_rows(denied)
        assert denied_rows[str(denied.review_id)]["state"] == "reuse_denied"
        assert [row["holdout_identity_sha256"] for row in denied_candidate["holdout_reuse_denials"]] == [
            denied.holdout_identity
        ]
        assert denied_candidate["holdout_reuse_denials"][0]["owner_campaign_id"] == str(owner.campaign_id)
        denied_gate = gate_holdout(denied, denied.review_id)
        assert denied_gate["state"] == "blocked"
        assert "denied" in denied_gate["reason"]
        assert _authority_counts(session, foreign.strategy_id, [foreign.campaign_id, denied.campaign_id]) == before
