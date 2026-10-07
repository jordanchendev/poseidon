"""PostgreSQL race proofs for frozen campaign creation."""

from __future__ import annotations

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import ExperimentCampaign
from poseidon.models.strategy_version import StrategyVersion
from tests.test_campaign_contract import _campaign_api, _complete_campaign_contract, _seed_strategy_versions

pytestmark = pytest.mark.postgresql


def _committed_versions(session_factory) -> tuple[uuid.UUID, uuid.UUID]:
    with session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        ids = (incumbent.id, candidate.id)
        session.commit()
        return ids


def _sentinel(marker: str) -> ExperimentRecord:
    return ExperimentRecord(
        id=uuid.uuid4(),
        study_name=f"phase99-campaign-sentinel-{marker}",
        config_json={"sentinel": marker},
        market="tw_stock",
        interval="1d",
        status="complete",
    )


def test_exact_campaign_creation_race_returns_one_frozen_campaign(
    phase99_session_factory,
    phase99_barrier,
):
    api = _campaign_api()
    incumbent_id, candidate_id = _committed_versions(phase99_session_factory)
    marker = uuid.uuid4().hex

    def create(worker: str):
        with phase99_session_factory() as session:
            incumbent = session.get(StrategyVersion, incumbent_id)
            candidate = session.get(StrategyVersion, candidate_id)
            contract = _complete_campaign_contract(incumbent, candidate, marker=marker)
            phase99_barrier.wait(timeout=10)
            campaign = api.CampaignService(session).create_frozen_campaign(contract)
            assert session.in_transaction(), "campaign service committed the caller-owned transaction"
            sentinel = _sentinel(f"{marker}-{worker}")
            session.add(sentinel)
            session.flush()
            result = (campaign.id, campaign.contract_sha256, sentinel.id)
            session.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(create, "a"), pool.submit(create, "b"))]

    assert len({(campaign_id, digest) for campaign_id, digest, _ in results}) == 1
    campaign_id, digest, _ = results[0]
    sentinel_ids = [sentinel_id for _, _, sentinel_id in results]
    with phase99_session_factory() as session:
        assert session.query(ExperimentCampaign).filter_by(contract_sha256=digest).count() == 1
        assert session.get(ExperimentCampaign, campaign_id).contract_sha256 == digest
        assert session.query(ExperimentRecord).filter(ExperimentRecord.id.in_(sentinel_ids)).count() == 2


def test_campaign_same_hash_different_contract_raises_frozen_campaign_identity_conflict(
    phase99_session_factory,
    monkeypatch,
):
    api = _campaign_api()
    with phase99_session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        original_contract = _complete_campaign_contract(incumbent, candidate)
        original = api.CampaignService(session).create_frozen_campaign(original_contract)
        original_id = original.id
        original_digest = original.contract_sha256
        session.commit()

    with phase99_session_factory() as session:
        before = session.execute(
            text("SELECT to_jsonb(c)::text FROM experiment_campaigns AS c WHERE id = :id"),
            {"id": original_id},
        ).scalar_one()
        changed = copy.deepcopy(original_contract)
        changed["hypothesis"] = "Conflicting immutable content forced onto the same identity."
        monkeypatch.setattr(api, "experiment_campaign_contract_sha256", lambda **_fields: original_digest)

        with pytest.raises(api.FrozenCampaignIdentityConflict):
            api.CampaignService(session).create_frozen_campaign(changed)
        assert session.in_transaction(), "identity conflict aborted the caller-owned transaction"
        sentinel = _sentinel(f"conflict-{uuid.uuid4().hex}")
        session.add(sentinel)
        session.flush()
        sentinel_id = sentinel.id
        session.commit()

    with phase99_session_factory() as session:
        after = session.execute(
            text("SELECT to_jsonb(c)::text FROM experiment_campaigns AS c WHERE id = :id"),
            {"id": original_id},
        ).scalar_one()
        assert after == before
        assert session.get(ExperimentRecord, sentinel_id) is not None
