"""Global consume-before-read holdout proofs on PostgreSQL."""

from __future__ import annotations

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.models.experiment_campaign import CampaignEvent, ExperimentCampaign, HoldoutUse
from poseidon.models.strategy_version import StrategyVersion
from tests.test_campaign_contract import (
    _campaign_api,
    _complete_campaign_contract,
    _seed_strategy_versions,
    _terminal_trial_kwargs,
)


pytestmark = pytest.mark.postgresql


def _holdout_api():
    module = _campaign_api()
    required = (
        "HoldoutReadPermit",
        "HoldoutReuseDenied",
        "build_holdout_identity",
        "consume_holdout_in_committed_transaction",
    )
    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, f"holdout API is incomplete: {missing}"
    return module


def _holdout_contract(marker: str | None = None) -> dict:
    marker = marker or uuid.uuid4().hex
    return {
        "manifest_set": [{"manifest_id": f"manifest-{marker}", "content_sha256": marker * 2}],
        "evidence_slice": {"fold": "holdout", "slice": "2025-H2", "sample_ids_sha256": marker[::-1] * 2},
        "window": {"start": "2025-07-01", "end": "2025-12-31"},
        "universe": ["0050", "2317", "2330"],
        "label": {"version": "forward-return-v1", "horizons": ["1_session", "5_sessions", "20_sessions"]},
        "benchmark": {"symbol": "0050", "return": "total_return"},
        "purge_gap": {"gap_eligible_sessions": 20, "overlap_method": "purged_embargo"},
        # These fields are deliberately outside the evidence identity.
        "campaign_id": str(uuid.uuid4()),
        "incumbent_strategy_version_id": str(uuid.uuid4()),
        "candidate_strategy_version_id": str(uuid.uuid4()),
        "candidate_content_sha256": "a" * 64,
        "hypothesis": "candidate-specific text",
    }


def _committed_campaigns(session_factory) -> tuple[tuple[uuid.UUID, str, uuid.UUID], tuple[uuid.UUID, str, uuid.UUID]]:
    api = _holdout_api()
    with session_factory() as session:
        incumbent, candidate, alternate = _seed_strategy_versions(session)
        first = api.CampaignService(session).create_frozen_campaign(
            _complete_campaign_contract(incumbent, candidate, marker=uuid.uuid4().hex)
        )
        second_contract = _complete_campaign_contract(incumbent, alternate, marker=uuid.uuid4().hex)
        second_contract["hypothesis"] = "A different candidate must not evade the same holdout."
        second = api.CampaignService(session).create_frozen_campaign(second_contract)
        result = (
            (first.id, first.contract_sha256, candidate.id),
            (second.id, second.contract_sha256, alternate.id),
        )
        session.commit()
        return result


def test_holdout_identity_excludes_candidate_but_changes_with_evidence():
    api = _holdout_api()
    original = _holdout_contract()
    candidate_changed = copy.deepcopy(original)
    candidate_changed.update(
        campaign_id=str(uuid.uuid4()),
        incumbent_strategy_version_id=str(uuid.uuid4()),
        candidate_strategy_version_id=str(uuid.uuid4()),
        candidate_content_sha256="b" * 64,
        hypothesis="entirely different candidate",
    )
    evidence_changed = copy.deepcopy(original)
    evidence_changed["window"]["end"] = "2026-01-30"

    assert api.build_holdout_identity(candidate_changed) == api.build_holdout_identity(original)
    assert api.build_holdout_identity(evidence_changed) != api.build_holdout_identity(original)


def test_same_campaign_concurrent_consumption_returns_one_use_row(
    phase99_session_factory,
    phase99_barrier,
):
    api = _holdout_api()
    owner, _competitor = _committed_campaigns(phase99_session_factory)
    campaign_id, contract_sha256, _version_id = owner
    identity = api.build_holdout_identity(_holdout_contract())

    def consume():
        phase99_barrier.wait(timeout=10)
        permit = api.consume_holdout_in_committed_transaction(
            phase99_session_factory,
            campaign_id,
            contract_sha256,
            identity,
        )
        return permit.holdout_use_id, permit.holdout_identity_sha256

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(consume), pool.submit(consume))]

    assert len(set(results)) == 1
    with phase99_session_factory() as session:
        assert session.query(HoldoutUse).filter_by(holdout_identity_sha256=identity).count() == 1


def test_competing_campaign_race_has_one_owner_committed_denial_and_zero_loser_reads(
    phase99_session_factory,
    phase99_barrier,
):
    api = _holdout_api()
    campaigns = _committed_campaigns(phase99_session_factory)
    identity_payload = _holdout_contract()
    identity = api.build_holdout_identity(identity_payload)
    calls = {campaign_id: 0 for campaign_id, _digest, _version_id in campaigns}
    calls_lock = Lock()

    def loader(*, permit, frozen_slice):
        with calls_lock:
            calls[permit.campaign_id] += 1
        with phase99_session_factory() as observer:
            visible = observer.get(HoldoutUse, permit.holdout_use_id)
            assert visible is not None, "loader could not see the committed use row"
            assert visible.holdout_identity_sha256 == identity
        return frozen_slice

    def compete(campaign):
        campaign_id, contract_sha256, _version_id = campaign
        phase99_barrier.wait(timeout=10)
        try:
            permit = api.consume_holdout_in_committed_transaction(
                phase99_session_factory,
                campaign_id,
                contract_sha256,
                identity,
            )
        except api.HoldoutReuseDenied:
            return "denied", campaign_id, None
        materialized = permit.materialize(loader, frozen_slice=identity_payload)
        return "owner", campaign_id, materialized

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in (pool.submit(compete, campaigns[0]), pool.submit(compete, campaigns[1]))]

    assert sorted(status for status, _campaign_id, _result in results) == ["denied", "owner"]
    owner_id = next(campaign_id for status, campaign_id, _result in results if status == "owner")
    loser_id = next(campaign_id for status, campaign_id, _result in results if status == "denied")
    assert calls == {owner_id: 1, loser_id: 0}
    with phase99_session_factory() as session:
        use = session.query(HoldoutUse).filter_by(holdout_identity_sha256=identity).one()
        assert use.campaign_id == owner_id
        denial = session.query(CampaignEvent).filter_by(
            campaign_id=loser_id,
            event_type="holdout_reuse_denied",
        ).one()
        assert denial.payload_json["owner_campaign_id"] == str(owner_id)


def test_crash_after_consumption_commit_keeps_evidence_consumed_and_allows_exact_replay(
    phase99_session_factory,
):
    api = _holdout_api()
    owner, competitor = _committed_campaigns(phase99_session_factory)
    identity = api.build_holdout_identity(_holdout_contract())

    with pytest.raises(RuntimeError, match="crash before materialization"):
        permit = api.consume_holdout_in_committed_transaction(
            phase99_session_factory,
            owner[0],
            owner[1],
            identity,
        )
        assert permit.holdout_identity_sha256 == identity
        raise RuntimeError("crash before materialization")

    with pytest.raises(api.HoldoutReuseDenied):
        api.consume_holdout_in_committed_transaction(
            phase99_session_factory,
            competitor[0],
            competitor[1],
            identity,
        )
    replay = api.consume_holdout_in_committed_transaction(
        phase99_session_factory,
        owner[0],
        owner[1],
        identity,
    )
    with phase99_session_factory() as session:
        use = session.query(HoldoutUse).filter_by(holdout_identity_sha256=identity).one()
        assert (replay.holdout_use_id, use.campaign_id) == (use.id, owner[0])


@pytest.mark.parametrize(
    "terminal_state",
    ("optimizer_failed", "statistically_inconclusive", "capability_unavailable"),
)
def test_terminal_failure_after_exposure_never_releases_holdout(
    phase99_session_factory,
    terminal_state,
):
    api = _holdout_api()
    owner, _competitor = _committed_campaigns(phase99_session_factory)
    campaign_id, contract_sha256, candidate_id = owner
    identity = api.build_holdout_identity(_holdout_contract())
    permit = api.consume_holdout_in_committed_transaction(
        phase99_session_factory,
        campaign_id,
        contract_sha256,
        identity,
    )

    with phase99_session_factory() as session:
        campaign = session.get(ExperimentCampaign, campaign_id)
        candidate = session.get(StrategyVersion, candidate_id)
        kwargs = _terminal_trial_kwargs(campaign, candidate)
        arm_by_state = {
            "optimizer_failed": "fundamental_only",
            "statistically_inconclusive": "technical_only",
            "capability_unavailable": "combined",
        }
        arm = arm_by_state[terminal_state]
        kwargs.update(
            original_trial_id=f"candidate-{arm}",
            ablation_arm=arm,
            paired_sample_key_sha256={
                "optimizer_failed": "b" * 64,
                "statistically_inconclusive": "c" * 64,
                "capability_unavailable": "d" * 64,
            }[terminal_state],
            terminal_state=terminal_state,
            terminal_reason_json={"state": terminal_state},
            metrics_json=None,
        )
        ExperimentTracker(session).append_campaign_terminal_trial(**kwargs)
        session.commit()

    with phase99_session_factory() as session:
        uses = session.query(HoldoutUse).filter_by(holdout_identity_sha256=identity).all()
        assert [(row.id, row.campaign_id) for row in uses] == [(permit.holdout_use_id, campaign_id)]


def test_holdout_permit_cannot_be_caller_constructed(phase99_session_factory):
    api = _holdout_api()
    owner, _competitor = _committed_campaigns(phase99_session_factory)
    identity = api.build_holdout_identity(_holdout_contract())
    permit = api.consume_holdout_in_committed_transaction(
        phase99_session_factory,
        owner[0],
        owner[1],
        identity,
    )
    with pytest.raises(TypeError):
        api.HoldoutReadPermit()
    assert permit.materialize(lambda *, permit: permit.holdout_use_id) == permit.holdout_use_id
