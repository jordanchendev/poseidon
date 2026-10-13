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
    released, unsafe = strategy.active_version_id, uuid.uuid4()
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
