"""Documented first-activation bootstrap release event (Phase 101 D-23.1, D-19, D-27).

Bootstrap is the only campaign-free way to set a strategy's first active version.
It must run the real candidate/policy identity predicates, require a human release
manager in scope, a fresh paper account, and a resolvable tw_stock stop-loss, and it
must leave the 044 pointer trigger in force.
"""

import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from poseidon.api import releases as releases_api
from poseidon.api.auth import AuthPrincipal
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionPolicy
from poseidon.decision_loop.releases import ReleaseGateRejected, ReleaseService
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest
from tests.test_decision_service import synthetic_policy, synthetic_reconciliation_policy
from tests.test_phase101_protective_exit_config import sma_cross_config
from tests.test_position_lot_allocation import project
from tests.test_position_lot_allocation import seed as seed

NOW = datetime(2026, 10, 15, 2, 0, tzinfo=UTC)
BOOTSTRAP_CODES = [
    "candidate_identity",
    "policy_identity",
    "bootstrap_eligibility",
    "protective_exit",
    "account_fresh",
    "release_authority",
]
STOP_LOSS = {"stop_loss_pct": 0.08}

MANAGER_KEY = "phase101-release-manager-secret"
OTHER_SCOPE_MANAGER_KEY = "phase101-other-scope-manager-secret"
WORKER_KEY = "phase101-worker-secret"
VIEWER_KEY = "phase101-viewer-secret"
LEGACY_KEY = "phase101-legacy-secret"


def _scope():
    return f"paper:phase101:bootstrap:{uuid.uuid4().hex}"


def bootstrap_manager(scope):
    return AuthPrincipal("human:phase101-release-manager", frozenset({"release_manager"}), frozenset({scope}))


def bootstrap_policy(scope, generation="generation-1", **changes):
    policy = synthetic_policy(
        account_scope=scope,
        market="tw_stock",
        mode="human",
        reconciliation=synthetic_reconciliation_policy(account_generation=generation),
    )
    policy.update(changes)
    return policy


def _seed_bootstrap(
    session,
    *,
    scope=None,
    generation="generation-1",
    config=None,
    policy=None,
    paper_account=True,
):
    """Seed a never-released tw_stock strategy plus one candidate and a fresh paper account."""
    scope = _scope() if scope is None else scope
    marker = uuid.uuid4().hex
    strategy = StrategyRecord(
        name=f"phase101-bootstrap-{marker}",
        strategy_type="technical",
        config={},
        symbol="2330",
        market="tw_stock",
        interval="1d",
        active=False,
    )
    session.add(strategy)
    session.flush()
    config = sma_cross_config(protective_exit=STOP_LOSS) if config is None else copy.deepcopy(config)
    policy = bootstrap_policy(scope, generation) if policy is None else copy.deepcopy(policy)
    artifact = {"uri": f"fixture://phase101/bootstrap/{marker}"}
    version = StrategyVersion(
        strategy_id=strategy.id,
        version_no=1,
        status="draft",
        config_json=config,
        policy_json=policy,
        artifact_json=artifact,
        content_sha256=strategy_version_digest(config, policy, artifact),
    )
    session.add(version)
    if paper_account:
        reconciliation = policy["reconciliation"]
        existing = session.scalar(
            select(PaperBrokerAccount).where(
                PaperBrokerAccount.account_scope == scope,
                PaperBrokerAccount.account_generation == generation,
            )
        )
        if existing is None:
            session.add(
                PaperBrokerAccount(
                    account_scope=scope,
                    account_generation=generation,
                    opening_cash=reconciliation["opening_cash"],
                    currency=reconciliation["currency"],
                )
            )
    session.flush()
    return SimpleNamespace(strategy_id=strategy.id, candidate_id=version.id, scope=scope, generation=generation)


def _bootstrap(session, graph, *, key="bootstrap-key", reason="phase101 pilot bootstrap", principal=None, **changes):
    values = {
        "expected_revision": 0,
        "expected_active_version_id": None,
        "principal": principal or bootstrap_manager(graph.scope),
        "idempotency_key": key,
        "reason": reason,
    }
    values.update(changes)
    return ReleaseService(session).bootstrap(graph.strategy_id, graph.candidate_id, **values)


def _events(session, strategy_id):
    return session.scalars(
        select(StrategyReleaseEvent)
        .where(StrategyReleaseEvent.strategy_id == strategy_id)
        .order_by(StrategyReleaseEvent.resulting_revision)
    ).all()


def _pointer(session, strategy_id):
    return session.execute(
        text("SELECT active_version_id, release_revision FROM strategies WHERE id = :id"),
        {"id": strategy_id},
    ).one()


def _by_code(gate, code):
    return next(predicate for predicate in gate.predicates if predicate.code == code)


def _assert_gate_blocked(session, graph, code, **changes):
    with pytest.raises(ReleaseGateRejected) as error:
        _bootstrap(session, graph, **changes)
    session.rollback()
    assert error.value.operation == "bootstrap"
    assert [predicate.code for predicate in error.value.result.predicates] == BOOTSTRAP_CODES
    assert _by_code(error.value.result, code).state != "passed"
    assert _pointer(session, graph.strategy_id) == (None, 0)
    assert _events(session, graph.strategy_id) == []
    return error.value.result


# --------------------------------------------------------------------------- service


@pytest.mark.postgresql
def test_bootstrap_activates_first_version_with_campaign_free_event(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        session.commit()

        response = _bootstrap(session, graph)
        session.commit()

        assert _pointer(session, graph.strategy_id) == (graph.candidate_id, 1)
        events = _events(session, graph.strategy_id)
        assert len(events) == 1
        event = events[0]
        assert event.event_type == "bootstrap"
        assert (event.candidate_version_id, event.to_version_id, event.from_version_id) == (
            graph.candidate_id,
            graph.candidate_id,
            None,
        )
        assert (event.expected_revision, event.resulting_revision) == (0, 1)
        assert event.campaign_id is None and event.review_id is None
        assert event.holdout_use_id is None and event.reconciliation_id is None
        assert event.source_release_event_id is None
        assert event.actor_id == "human:phase101-release-manager"
        assert event.account_scope == graph.scope
        assert event.gate_json["bootstrap"] is True
        assert event.gate_json["allowed"] is True
        assert [predicate["code"] for predicate in event.gate_json["predicates"]] == BOOTSTRAP_CODES
        assert all(predicate["state"] == "passed" for predicate in event.gate_json["predicates"])
        assert response == event.response_json
        assert response["event_type"] == "bootstrap"
        assert response["release_revision"] == 1
        assert response["from_version_id"] is None
        assert response["to_version_id"] == str(graph.candidate_id)


@pytest.mark.postgresql
def test_bootstrap_replay_returns_identical_response_and_reused_key_conflicts(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        session.commit()
        first = _bootstrap(session, graph, key="replay-key")
        session.commit()

        replay = _bootstrap(session, graph, key="replay-key")
        session.commit()
        assert replay == first
        assert len(_events(session, graph.strategy_id)) == 1

        with pytest.raises(DecisionConflictError, match="idempotency key was used for a different request"):
            _bootstrap(session, graph, key="replay-key", reason="different reason")
        session.rollback()
        assert len(_events(session, graph.strategy_id)) == 1


@pytest.mark.postgresql
def test_second_bootstrap_and_non_initial_expectations_conflict(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        session.commit()
        _bootstrap(session, graph, key="first")
        session.commit()

        with pytest.raises(DecisionConflictError, match="release revision changed"):
            _bootstrap(session, graph, key="second")
        session.rollback()
        with pytest.raises(DecisionConflictError, match="bootstrap requires expected_revision 0"):
            _bootstrap(
                session,
                graph,
                key="third",
                expected_revision=1,
                expected_active_version_id=graph.candidate_id,
            )
        session.rollback()
        assert len(_events(session, graph.strategy_id)) == 1
        assert _pointer(session, graph.strategy_id) == (graph.candidate_id, 1)

    with phase99_session_factory() as session:
        fresh = _seed_bootstrap(session)
        session.commit()
        with pytest.raises(DecisionConflictError, match="bootstrap requires expected_revision 0"):
            _bootstrap(session, fresh, expected_active_version_id=fresh.candidate_id)
        session.rollback()
        assert _events(session, fresh.strategy_id) == []
        assert _pointer(session, fresh.strategy_id) == (None, 0)


@pytest.mark.postgresql
def test_missing_protective_exit_blocks_bootstrap(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, config=sma_cross_config())
        session.commit()
        gate = _assert_gate_blocked(session, graph, "protective_exit")
        assert _by_code(gate, "protective_exit").state == "blocked"
        assert _by_code(gate, "candidate_identity").state == "passed"


@pytest.mark.postgresql
def test_non_live_candidate_config_blocks_candidate_identity(phase99_session_factory):
    config = sma_cross_config(protective_exit=STOP_LOSS)
    config["payload"]["rules"][0]["condition"] = {"type": "feature_above", "column": "rsi_14", "threshold": 70.0}
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, config=config)
        session.commit()
        gate = _assert_gate_blocked(session, graph, "candidate_identity")
        assert _by_code(gate, "candidate_identity").state == "blocked"


@pytest.mark.postgresql
@pytest.mark.parametrize(
    "policy_changes",
    [
        {"mode": "assisted"},
        {"release_gate": {"minimum_mature_samples": 10, "requires_human_release": False}},
    ],
)
def test_non_human_release_policy_conflicts_before_any_predicate(phase99_session_factory, policy_changes):
    scope = _scope()
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, scope=scope, policy=bootstrap_policy(scope, **policy_changes))
        session.commit()
        with pytest.raises(DecisionConflictError) as error:
            _bootstrap(session, graph)
        session.rollback()
        assert not isinstance(error.value, ReleaseGateRejected)
        assert "human-release-only" in str(error.value) or "policy" in str(error.value)
        assert _events(session, graph.strategy_id) == []
        assert _pointer(session, graph.strategy_id) == (None, 0)


@pytest.mark.postgresql
def test_missing_paper_account_blocks_account_fresh(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, paper_account=False)
        session.commit()
        gate = _assert_gate_blocked(session, graph, "account_fresh")
        assert _by_code(gate, "account_fresh").evidence["paper_accounts"] == 0


@pytest.mark.postgresql
def test_paper_broker_order_blocks_account_fresh(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        session.add(
            PaperBrokerOrder(
                account_scope=graph.scope,
                account_generation=graph.generation,
                client_order_ref=f"phase101-{uuid.uuid4().hex}",
                broker_order_id=f"PAPER-{uuid.uuid4().hex}",
                market="tw_stock",
                symbol="2330",
                instrument="spot",
                action="buy",
                side="long",
                order_type="market",
                quantity=1.0,
                price=100.0,
                status="submitted",
                state_version=1,
                accepted_at=NOW,
            )
        )
        session.commit()
        gate = _assert_gate_blocked(session, graph, "account_fresh")
        assert _by_code(gate, "account_fresh").evidence["paper_broker_orders"] == 1


@pytest.mark.postgresql
def test_decision_record_in_scope_blocks_account_fresh(phase99_session_factory, seed):
    seed.fill()
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, scope=seed.account, generation="generation-1")
        session.commit()
        gate = _assert_gate_blocked(session, graph, "account_fresh")
        evidence = _by_code(gate, "account_fresh").evidence
        assert evidence["decision_records"] >= 1
        assert evidence["paper_broker_orders"] == 0


@pytest.mark.postgresql
def test_open_position_lot_blocks_account_fresh(phase99_session_factory, seed):
    project(seed, seed.fill())
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, scope=seed.account, generation="generation-1")
        session.commit()
        gate = _assert_gate_blocked(session, graph, "account_fresh")
        assert _by_code(gate, "account_fresh").evidence["open_position_lots"] >= 1


@pytest.mark.postgresql
def test_direct_pointer_update_after_bootstrap_still_raises_23514(phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        other = StrategyVersion(
            strategy_id=graph.strategy_id,
            version_no=2,
            status="draft",
            config_json=sma_cross_config(protective_exit=STOP_LOSS),
            policy_json=bootstrap_policy(graph.scope),
            artifact_json={"uri": "fixture://phase101/bootstrap/other"},
            content_sha256=strategy_version_digest(
                sma_cross_config(protective_exit=STOP_LOSS),
                bootstrap_policy(graph.scope),
                {"uri": "fixture://phase101/bootstrap/other"},
            ),
        )
        session.add(other)
        session.commit()
        _bootstrap(session, graph)
        session.commit()

        savepoint = session.begin_nested()
        try:
            with pytest.raises(DBAPIError) as error:
                session.execute(
                    text("UPDATE strategies SET active_version_id = :version WHERE id = :strategy"),
                    {"version": other.id, "strategy": graph.strategy_id},
                )
            sqlstate = getattr(error.value.orig, "sqlstate", None) or getattr(error.value.orig, "pgcode", None)
            assert sqlstate == "23514"
        finally:
            savepoint.rollback()
        session.rollback()
        assert _pointer(session, graph.strategy_id) == (graph.candidate_id, 1)


# ------------------------------------------------------------------ predicate units


class _CountingSession:
    """Stub session that answers the account_fresh queries by table name."""

    def __init__(self, accounts, counts):
        self.accounts = accounts
        self.counts = counts

    def scalars(self, _statement):
        return SimpleNamespace(all=lambda: list(self.accounts))

    def scalar(self, statement):
        sql = str(statement)
        for table, value in self.counts.items():
            if f"FROM {table}" in sql:
                return value
        raise AssertionError(f"unexpected account_fresh query: {sql}")


def _fresh_counts(**changes):
    counts = {
        "paper_broker_orders": 0,
        "paper_broker_fills": 0,
        "position_lots": 0,
        "decision_records": 0,
    }
    counts.update(changes)
    return counts


def _matching_account(policy, **changes):
    values = {"opening_cash": policy.reconciliation.opening_cash, "currency": policy.reconciliation.currency}
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("accounts_factory", "counts", "expected"),
    [
        (lambda policy: [_matching_account(policy)], _fresh_counts(), "passed"),
        (lambda _policy: [], _fresh_counts(), "blocked"),
        (lambda policy: [_matching_account(policy), _matching_account(policy)], _fresh_counts(), "blocked"),
        (lambda policy: [_matching_account(policy, opening_cash=1.0)], _fresh_counts(), "blocked"),
        (lambda policy: [_matching_account(policy, currency="USD")], _fresh_counts(), "blocked"),
        (lambda policy: [_matching_account(policy)], _fresh_counts(paper_broker_orders=1), "blocked"),
        (lambda policy: [_matching_account(policy)], _fresh_counts(paper_broker_fills=1), "blocked"),
        (lambda policy: [_matching_account(policy)], _fresh_counts(position_lots=1), "blocked"),
        (lambda policy: [_matching_account(policy)], _fresh_counts(decision_records=1), "blocked"),
    ],
)
def test_account_fresh_predicate_matrix(accounts_factory, counts, expected):
    policy = DecisionPolicy.model_validate(bootstrap_policy("paper:phase101:unit"))
    service = ReleaseService(_CountingSession(accounts_factory(policy), counts))

    predicate = service._account_fresh_predicate(policy)

    assert predicate.code == "account_fresh"
    assert predicate.state == expected


def test_account_fresh_requires_frozen_reconciliation_policy():
    policy = DecisionPolicy.model_validate(synthetic_policy(account_scope="paper:phase101:unit"))
    assert policy.reconciliation is None
    predicate = ReleaseService(_CountingSession([], _fresh_counts()))._account_fresh_predicate(policy)
    assert predicate.state == "blocked"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (sma_cross_config(protective_exit=STOP_LOSS), "passed"),
        (sma_cross_config(), "blocked"),
        ({"strategy_type": "rule"}, "blocked"),
        (sma_cross_config(market="crypto_spot"), "passed"),
    ],
)
def test_protective_exit_predicate(config, expected):
    candidate = SimpleNamespace(id=uuid.uuid4(), config_json=config)
    assert ReleaseService._protective_exit_predicate(candidate).state == expected


# --------------------------------------------------------------------------- API


def _fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _configure_bootstrap_auth(monkeypatch, scope):
    principals = {
        _fingerprint(MANAGER_KEY): {
            "actor_id": "human:phase101-release-manager",
            "roles": ["portfolio_manager", "release_manager"],
            "account_scopes": [scope],
        },
        _fingerprint(OTHER_SCOPE_MANAGER_KEY): {
            "actor_id": "human:phase101-other-scope-manager",
            "roles": ["release_manager"],
            "account_scopes": ["paper:phase101:other-scope"],
        },
        _fingerprint(WORKER_KEY): {
            "actor_id": "service:decision-worker",
            "roles": ["decision-worker"],
            "account_scopes": [scope],
        },
        _fingerprint(VIEWER_KEY): {
            "actor_id": "human:viewer",
            "roles": ["viewer"],
            "account_scopes": [scope],
        },
    }
    monkeypatch.setattr(settings, "api_key", LEGACY_KEY)
    monkeypatch.setattr(settings, "api_principals_json", json.dumps(principals))


def _headers(key, idempotency_key):
    return {"X-API-Key": key, "Idempotency-Key": idempotency_key}


def _client(session):
    app = FastAPI()
    app.include_router(releases_api.router, prefix="/api/v1/strategies")
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app, raise_server_exceptions=False)


def _body(graph, **changes):
    body = {
        "expected_revision": 0,
        "expected_active_version_id": None,
        "candidate_version_id": str(graph.candidate_id),
        "reason": "phase101 pilot bootstrap",
    }
    body.update(changes)
    return body


def _event_count(session, strategy_id):
    return session.scalar(
        select(func.count()).select_from(StrategyReleaseEvent).where(StrategyReleaseEvent.strategy_id == strategy_id)
    )


@pytest.mark.postgresql
def test_bootstrap_route_activates_replays_and_conflicts(monkeypatch, phase99_session_factory):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        session.commit()
        _configure_bootstrap_auth(monkeypatch, graph.scope)
        client = _client(session)
        path = f"/api/v1/strategies/{graph.strategy_id}/bootstrap"

        first = client.post(path, json=_body(graph), headers=_headers(MANAGER_KEY, "api-bootstrap"))
        replay = client.post(path, json=_body(graph), headers=_headers(MANAGER_KEY, "api-bootstrap"))
        reused = client.post(path, json=_body(graph, reason="changed"), headers=_headers(MANAGER_KEY, "api-bootstrap"))
        second = client.post(path, json=_body(graph), headers=_headers(MANAGER_KEY, "api-bootstrap-2"))
        non_null = client.post(
            path,
            json=_body(graph, expected_revision=1, expected_active_version_id=str(graph.candidate_id)),
            headers=_headers(MANAGER_KEY, "api-bootstrap-3"),
        )

        assert first.status_code == 200, first.text
        assert first.json()["event_type"] == "bootstrap"
        assert first.json()["release_revision"] == 1
        assert replay.status_code == 200
        assert replay.json() == first.json()
        assert reused.status_code == 409
        assert second.status_code == 409
        assert non_null.status_code == 409
        assert _event_count(session, graph.strategy_id) == 1
        assert _pointer(session, graph.strategy_id) == (graph.candidate_id, 1)

        events = client.get(
            f"/api/v1/strategies/{graph.strategy_id}/release-events", headers={"X-API-Key": MANAGER_KEY}
        )
        assert events.status_code == 200, events.text
        assert [event["event_type"] for event in events.json()] == ["bootstrap"]
        assert events.json()[0]["gate_json"]["bootstrap"] is True
        assert events.json()[0].get("campaign_id") is None  # route excludes None fields
        assert [item["code"] for item in events.json()[0]["gate_json"]["predicates"]] == BOOTSTRAP_CODES


@pytest.mark.postgresql
@pytest.mark.parametrize("key", [WORKER_KEY, VIEWER_KEY, LEGACY_KEY, OTHER_SCOPE_MANAGER_KEY])
def test_bootstrap_route_denies_non_release_principals_with_403(monkeypatch, phase99_session_factory, key):
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session)
        session.commit()
        _configure_bootstrap_auth(monkeypatch, graph.scope)
        response = _client(session).post(
            f"/api/v1/strategies/{graph.strategy_id}/bootstrap",
            json=_body(graph),
            headers=_headers(key, "denied-bootstrap"),
        )
        assert response.status_code == 403
        assert _event_count(session, graph.strategy_id) == 0
        assert _pointer(session, graph.strategy_id) == (None, 0)


@pytest.mark.postgresql
def test_bootstrap_route_gate_failures_are_422_with_predicates(monkeypatch, phase99_session_factory):
    with phase99_session_factory() as session:
        no_stop = _seed_bootstrap(session, config=sma_cross_config())
        session.commit()
        _configure_bootstrap_auth(monkeypatch, no_stop.scope)
        response = _client(session).post(
            f"/api/v1/strategies/{no_stop.strategy_id}/bootstrap",
            json=_body(no_stop),
            headers=_headers(MANAGER_KEY, "no-stop"),
        )
        assert response.status_code == 422, response.text
        predicates = {item["code"]: item["state"] for item in response.json()["detail"]["gate"]["predicates"]}
        assert predicates["protective_exit"] == "blocked"
        assert _event_count(session, no_stop.strategy_id) == 0

    with phase99_session_factory() as session:
        stale_account = _seed_bootstrap(session, paper_account=False)
        session.commit()
        _configure_bootstrap_auth(monkeypatch, stale_account.scope)
        response = _client(session).post(
            f"/api/v1/strategies/{stale_account.strategy_id}/bootstrap",
            json=_body(stale_account),
            headers=_headers(MANAGER_KEY, "no-account"),
        )
        assert response.status_code == 422, response.text
        predicates = {item["code"]: item["state"] for item in response.json()["detail"]["gate"]["predicates"]}
        assert predicates["account_fresh"] == "blocked"
        assert _event_count(session, stale_account.strategy_id) == 0


@pytest.mark.postgresql
def test_bootstrap_route_rejects_non_human_policy_with_409(monkeypatch, phase99_session_factory):
    scope = _scope()
    with phase99_session_factory() as session:
        graph = _seed_bootstrap(session, scope=scope, policy=bootstrap_policy(scope, mode="assisted"))
        session.commit()
        _configure_bootstrap_auth(monkeypatch, graph.scope)
        response = _client(session).post(
            f"/api/v1/strategies/{graph.strategy_id}/bootstrap",
            json=_body(graph),
            headers=_headers(MANAGER_KEY, "assisted"),
        )
        assert response.status_code == 409
        assert _event_count(session, graph.strategy_id) == 0


def test_bootstrap_route_is_typed_in_openapi():
    app = FastAPI()
    app.include_router(releases_api.router, prefix="/api/v1/strategies")
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/strategies/{strategy_id}/bootstrap"]["post"]
    assert {"403", "404", "409", "422"}.issubset(operation["responses"])
    schemas = schema["components"]["schemas"]
    assert set(schemas["BootstrapRequest"]["properties"]) == {
        "expected_revision",
        "expected_active_version_id",
        "reason",
        "candidate_version_id",
    }
    assert "bootstrap" in schemas["ReleaseMutationResponse"]["properties"]["event_type"]["enum"]
    assert "bootstrap" in schemas["ReleaseEventResponse"]["properties"]["event_type"]["enum"]
    codes = schemas["ReleasePredicateResponse"]["properties"]["code"]["enum"]
    assert {"bootstrap_eligibility", "protective_exit", "account_fresh"} <= set(codes)
