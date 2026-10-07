"""PostgreSQL race proofs for insert-once paired campaign cells."""

from __future__ import annotations

import importlib
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import ExperimentCampaign
from poseidon.models.strategy_version import StrategyVersion
from tests.test_campaign_contract import (
    _campaign_api,
    _complete_campaign_contract,
    _seed_strategy_versions,
    _terminal_trial_kwargs,
)

pytestmark = pytest.mark.postgresql


def _tracker_conflict_type():
    module = importlib.import_module("poseidon.backtest.experiment_tracker")
    assert hasattr(module, "CampaignTrialIdentityConflict"), "typed paired-cell conflict is not implemented"
    return module.CampaignTrialIdentityConflict


def _committed_campaign(session_factory) -> tuple[uuid.UUID, uuid.UUID]:
    api = _campaign_api()
    with session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        campaign = api.CampaignService(session).create_frozen_campaign(
            _complete_campaign_contract(incumbent, candidate)
        )
        result = (campaign.id, candidate.id)
        session.commit()
        return result


def _sentinel(marker: str) -> ExperimentRecord:
    return ExperimentRecord(
        id=uuid.uuid4(),
        study_name=f"phase99-paired-sentinel-{marker}",
        config_json={"sentinel": marker},
        market="tw_stock",
        interval="1d",
        status="complete",
    )


def test_duplicate_paired_cell_race_inserts_one_terminal_fact(
    phase99_session_factory,
    phase99_barrier,
):
    campaign_id, candidate_id = _committed_campaign(phase99_session_factory)
    marker = uuid.uuid4().hex

    def append(worker: str):
        with phase99_session_factory() as session:
            campaign = session.get(ExperimentCampaign, campaign_id)
            candidate = session.get(StrategyVersion, candidate_id)
            kwargs = _terminal_trial_kwargs(campaign, candidate)
            phase99_barrier.wait(timeout=10)
            row = ExperimentTracker(session).append_campaign_terminal_trial(**kwargs)
            assert session.in_transaction(), "trial append committed the caller-owned transaction"
            sentinel = _sentinel(f"{marker}-{worker}")
            session.add(sentinel)
            session.flush()
            result = (row.id, row.result_sha256, sentinel.id)
            session.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(append, "a"), pool.submit(append, "b"))]

    assert len({(row_id, result_sha256) for row_id, result_sha256, _ in results}) == 1
    row_id, result_sha256, _ = results[0]
    sentinel_ids = [sentinel_id for _, _, sentinel_id in results]
    with phase99_session_factory() as session:
        row = session.get(ExperimentRecord, row_id)
        assert row.result_sha256 == result_sha256
        assert (
            session.query(ExperimentRecord)
            .filter_by(
                campaign_id=campaign_id,
                original_trial_id="candidate-combined",
                strategy_version_id=candidate_id,
                ablation_arm="combined",
                paired_sample_key_sha256="3" * 64,
            )
            .count()
            == 1
        )
        assert session.query(ExperimentRecord).filter(ExperimentRecord.id.in_(sentinel_ids)).count() == 2


def test_paired_cell_same_identity_different_result_raises_campaign_trial_identity_conflict(
    phase99_session_factory,
):
    conflict_type = _tracker_conflict_type()
    campaign_id, candidate_id = _committed_campaign(phase99_session_factory)
    with phase99_session_factory() as session:
        campaign = session.get(ExperimentCampaign, campaign_id)
        candidate = session.get(StrategyVersion, candidate_id)
        original = ExperimentTracker(session).append_campaign_terminal_trial(
            **_terminal_trial_kwargs(campaign, candidate)
        )
        original_id = original.id
        session.commit()

    with phase99_session_factory() as session:
        campaign = session.get(ExperimentCampaign, campaign_id)
        candidate = session.get(StrategyVersion, candidate_id)
        before = session.execute(
            text("SELECT to_jsonb(e)::text FROM experiments AS e WHERE id = :id"),
            {"id": original_id},
        ).scalar_one()
        conflicting = _terminal_trial_kwargs(campaign, candidate)
        conflicting.update(
            input_sha256="6" * 64,
            result_sha256="7" * 64,
            terminal_state="statistically_inconclusive",
            terminal_reason_json={"reason": "changed-result"},
            metrics_json=None,
        )

        with pytest.raises(conflict_type):
            ExperimentTracker(session).append_campaign_terminal_trial(**conflicting)
        assert session.in_transaction(), "paired-cell conflict aborted the caller-owned transaction"
        sentinel = _sentinel(f"conflict-{uuid.uuid4().hex}")
        session.add(sentinel)
        session.flush()
        sentinel_id = sentinel.id
        session.commit()

    with phase99_session_factory() as session:
        after = session.execute(
            text("SELECT to_jsonb(e)::text FROM experiments AS e WHERE id = :id"),
            {"id": original_id},
        ).scalar_one()
        assert after == before
        assert session.get(ExperimentRecord, sentinel_id) is not None
