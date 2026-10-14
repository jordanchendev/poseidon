"""HTTP contract for release-manager campaign freeze and paired-review enqueue."""

import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api import campaigns as campaigns_api
from poseidon.core.config import settings
from poseidon.core.database import get_db
from poseidon.main import app as poseidon_app
from poseidon.models.experiment_campaign import ExperimentCampaign
from poseidon.models.strategy_version import StrategyVersion
from poseidon.research.campaign import CampaignContractValidationError, FrozenCampaignIdentityConflict

ACCOUNT = "paper:phase99"
OTHER_ACCOUNT = "paper:other"
RELEASE_KEY = "campaign-release-manager"
OTHER_RELEASE_KEY = "campaign-other-release-manager"
MANAGER_KEY = "campaign-portfolio-manager"
VIEWER_KEY = "campaign-viewer"
WORKER_KEY = "campaign-worker"
RESEARCHER_KEY = "campaign-researcher"


def _fingerprint(key):
    return hashlib.sha256(key.encode()).hexdigest()


def _configure_auth(monkeypatch):
    def principal(actor_id, roles, scopes):
        return {"actor_id": actor_id, "roles": roles, "account_scopes": scopes}

    monkeypatch.setattr(settings, "api_key", "campaign-legacy")
    monkeypatch.setattr(
        settings,
        "api_principals_json",
        json.dumps(
            {
                _fingerprint(RELEASE_KEY): principal("human:release-manager", ["release_manager"], [ACCOUNT]),
                _fingerprint(OTHER_RELEASE_KEY): principal("human:other-release", ["release_manager"], [OTHER_ACCOUNT]),
                _fingerprint(MANAGER_KEY): principal("human:portfolio-manager", ["portfolio_manager"], [ACCOUNT]),
                _fingerprint(VIEWER_KEY): principal("human:viewer", ["viewer"], [ACCOUNT]),
                _fingerprint(WORKER_KEY): principal("service:decision-worker", ["decision-worker"], [ACCOUNT]),
                _fingerprint(RESEARCHER_KEY): principal("human:researcher", ["researcher"], [ACCOUNT]),
            }
        ),
    )


class FakeSession:
    def __init__(self):
        self.objects = {}
        self.commits = 0
        self.rollbacks = 0

    def add_object(self, model, key, value):
        self.objects[(model, key)] = value

    def get(self, model, key):
        return self.objects.get((model, key))

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, _record):
        pass


class StubCampaignService:
    def __init__(self):
        self.contracts = []
        self.by_digest = {}
        self.failure = None

    def create_frozen_campaign(self, contract):
        self.contracts.append(copy.deepcopy(contract))
        if self.failure is not None:
            raise self.failure
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if digest not in self.by_digest:
            self.by_digest[digest] = SimpleNamespace(
                id=uuid.uuid4(),
                contract_sha256=digest,
                incumbent_strategy_version_id=uuid.UUID(contract["incumbent_strategy_version_id"]),
                candidate_strategy_version_id=uuid.UUID(contract["candidate_strategy_version_id"]),
                hypothesis=contract["hypothesis"],
                created_by=contract["created_by"],
                created_at=datetime.now(UTC),
            )
        return self.by_digest[digest]


class StubCelery:
    def __init__(self):
        self.sent = []

    def send_task(self, name, args=None, queue=None, **_kwargs):
        self.sent.append((name, args, queue))
        return SimpleNamespace(id=f"task-{len(self.sent)}")


def _holdout_identity():
    return {
        "manifest_set": [{"manifest_id": str(uuid.uuid4()), "content_sha256": "c" * 64}],
        "evidence_slice": {"fold": "holdout", "slice": "2025-H2"},
        "window": {"start": "2025-07-01", "end": "2025-12-31"},
        "universe": ["0050", "2317", "2330"],
        "label": {"version": "forward-return-v1", "horizons": ["1_session"]},
        "benchmark": {"symbol": "0050"},
        "purge_gap": {"gap_eligible_sessions": 20, "overlap_method": "purged_embargo"},
    }


@pytest.fixture
def api(monkeypatch):
    _configure_auth(monkeypatch)
    session = FakeSession()
    incumbent, candidate, foreign = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for version_id, scope in ((incumbent, ACCOUNT), (candidate, ACCOUNT), (foreign, OTHER_ACCOUNT)):
        session.add_object(
            StrategyVersion, version_id, SimpleNamespace(id=version_id, policy_json={"account_scope": scope})
        )
    campaign = SimpleNamespace(
        id=uuid.uuid4(), incumbent_strategy_version_id=incumbent, candidate_strategy_version_id=candidate
    )
    session.add_object(ExperimentCampaign, campaign.id, campaign)
    service = StubCampaignService()
    celery = StubCelery()
    monkeypatch.setattr(campaigns_api, "CampaignService", lambda _db: service)
    monkeypatch.setattr(campaigns_api, "celery_app", celery)
    monkeypatch.setattr(
        "poseidon.research.campaign.run_paired_review_in_transactions",
        lambda *args, **kwargs: pytest.fail("review must not run inline in the API process"),
    )
    app = FastAPI()
    app.include_router(campaigns_api.router, prefix="/api/v1/campaigns")
    app.dependency_overrides[get_db] = lambda: session
    return SimpleNamespace(
        client=TestClient(app),
        session=session,
        service=service,
        celery=celery,
        incumbent=incumbent,
        candidate=candidate,
        foreign=foreign,
        campaign=campaign,
    )


def _contract(api, **changes):
    body = {
        "incumbent_strategy_version_id": str(api.incumbent),
        "candidate_strategy_version_id": str(api.candidate),
        "incumbent_content_sha256": "a" * 64,
        "candidate_content_sha256": "b" * 64,
        "declared_difference_json": {"feature": "added"},
        "hypothesis": "Candidate improves paired net return.",
        "contract_json": {"manifest_set": [{"manifest_id": "m", "content_sha256": "c" * 64}]},
    }
    body.update(changes)
    return body


def _headers(key):
    return {"X-API-Key": key}


def test_openapi_adds_only_campaign_writers_and_keeps_release_state_as_the_reader():
    poseidon_app.openapi_schema = None
    try:
        schema = poseidon_app.openapi()
    finally:
        poseidon_app.openapi_schema = None
    paths = {
        path: set(operations) for path, operations in schema["paths"].items() if path.startswith("/api/v1/campaigns")
    }
    assert paths == {"/api/v1/campaigns/": {"post"}, "/api/v1/campaigns/{campaign_id}/review": {"post"}}
    assert set(schema["paths"]["/api/v1/strategies/{strategy_id}/release-state"]) == {"get"}
    create_body = schema["components"]["schemas"]["CampaignCreateRequest"]["properties"]
    assert "created_by" not in create_body
    assert "active_version_id" not in json.dumps({path: schema["paths"][path] for path in paths})


def test_release_manager_freezes_campaign_with_principal_actor_and_replays(api):
    first = api.client.post("/api/v1/campaigns/", json=_contract(api), headers=_headers(RELEASE_KEY))
    assert first.status_code == 200, first.text
    assert first.json()["created_by"] == "human:release-manager"
    assert api.service.contracts[0]["created_by"] == "human:release-manager"
    replay = api.client.post("/api/v1/campaigns/", json=_contract(api), headers=_headers(RELEASE_KEY))
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    assert api.session.commits == 2


def test_campaign_body_cannot_supply_its_own_actor(api):
    response = api.client.post(
        "/api/v1/campaigns/",
        json=_contract(api, created_by="human:spoofed"),
        headers=_headers(RELEASE_KEY),
    )
    assert response.status_code == 422
    assert api.service.contracts == []


@pytest.mark.parametrize("key", [MANAGER_KEY, VIEWER_KEY, WORKER_KEY, RESEARCHER_KEY, OTHER_RELEASE_KEY])
def test_campaign_creation_requires_in_scope_release_manager(api, key):
    response = api.client.post("/api/v1/campaigns/", json=_contract(api), headers=_headers(key))
    assert response.status_code == 403
    assert api.service.contracts == []
    assert api.session.commits == 0


def test_campaign_creation_scope_checks_both_versions(api):
    foreign = api.client.post(
        "/api/v1/campaigns/",
        json=_contract(api, candidate_strategy_version_id=str(api.foreign)),
        headers=_headers(RELEASE_KEY),
    )
    assert foreign.status_code == 403
    missing = api.client.post(
        "/api/v1/campaigns/",
        json=_contract(api, incumbent_strategy_version_id=str(uuid.uuid4())),
        headers=_headers(RELEASE_KEY),
    )
    assert missing.status_code == 404
    assert api.service.contracts == []


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (CampaignContractValidationError("contract_json missing required fields"), 422),
        (FrozenCampaignIdentityConflict("campaign digest identifies different immutable content"), 409),
    ],
)
def test_campaign_service_failures_are_mapped_and_rolled_back(api, failure, status):
    api.service.failure = failure
    response = api.client.post("/api/v1/campaigns/", json=_contract(api), headers=_headers(RELEASE_KEY))
    assert response.status_code == status
    assert api.session.commits == 0
    assert api.session.rollbacks == 1


def test_review_is_enqueued_with_202_and_never_executed_inline(api):
    holdout = _holdout_identity()
    response = api.client.post(
        f"/api/v1/campaigns/{api.campaign.id}/review",
        json={"holdout_identity": holdout},
        headers=_headers(RELEASE_KEY),
    )
    assert response.status_code == 202, response.text
    assert response.json()["task_id"] == "task-1"
    assert len(response.json()["holdout_identity_sha256"]) == 64
    assert api.celery.sent == [(campaigns_api.RUN_REVIEW_TASK, [str(api.campaign.id), holdout], "poseidon_cpu")]


@pytest.mark.parametrize("key", [MANAGER_KEY, VIEWER_KEY, WORKER_KEY, OTHER_RELEASE_KEY])
def test_review_enqueue_requires_in_scope_release_manager(api, key):
    response = api.client.post(
        f"/api/v1/campaigns/{api.campaign.id}/review",
        json={"holdout_identity": _holdout_identity()},
        headers=_headers(key),
    )
    assert response.status_code == 403
    assert api.celery.sent == []


def test_review_enqueue_rejects_unknown_campaign_and_invalid_holdout(api):
    missing = api.client.post(
        f"/api/v1/campaigns/{uuid.uuid4()}/review",
        json={"holdout_identity": _holdout_identity()},
        headers=_headers(RELEASE_KEY),
    )
    assert missing.status_code == 404
    holdout = _holdout_identity()
    holdout.pop("window")
    invalid = api.client.post(
        f"/api/v1/campaigns/{api.campaign.id}/review",
        json={"holdout_identity": holdout},
        headers=_headers(RELEASE_KEY),
    )
    assert invalid.status_code == 422
    assert api.celery.sent == []


@pytest.mark.postgresql
def test_real_campaign_service_replays_identical_contract_through_the_route(monkeypatch, phase99_session_factory):
    from tests.test_campaign_contract import _complete_campaign_contract, _seed_strategy_versions

    _configure_auth(monkeypatch)
    monkeypatch.setattr(campaigns_api, "celery_app", StubCelery())
    with phase99_session_factory() as session:
        incumbent, candidate, _alternate = _seed_strategy_versions(session)
        session.commit()
        contract = _complete_campaign_contract(incumbent, candidate, marker=uuid.uuid4().hex)
        contract.pop("created_by")
        body = json.loads(json.dumps(contract, default=str))
        app = FastAPI()
        app.include_router(campaigns_api.router, prefix="/api/v1/campaigns")
        app.dependency_overrides[get_db] = lambda: session
        client = TestClient(app)
        first = client.post("/api/v1/campaigns/", json=body, headers=_headers(RELEASE_KEY))
        assert first.status_code == 200, first.text
        replay = client.post("/api/v1/campaigns/", json=body, headers=_headers(RELEASE_KEY))
        assert replay.status_code == 200
        assert replay.json()["id"] == first.json()["id"]
        stored = session.get(ExperimentCampaign, uuid.UUID(first.json()["id"]))
        assert stored.created_by == "human:release-manager"
