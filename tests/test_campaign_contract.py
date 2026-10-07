"""Frozen campaign and terminal-trial contracts for Phase 99."""

from __future__ import annotations

import copy
import importlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import CampaignEvent, ExperimentCampaign
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest

pytestmark = pytest.mark.postgresql

TERMINAL_STATES = (
    "succeeded",
    "optimizer_failed",
    "constraint_rejected",
    "insufficient_data",
    "statistically_inconclusive",
    "capability_unavailable",
)
ARMS = ("fundamental_only", "technical_only", "combined")
REQUIRED_CAMPAIGN_PATHS = (
    ("incumbent_strategy_version_id",),
    ("candidate_strategy_version_id",),
    ("incumbent_content_sha256",),
    ("candidate_content_sha256",),
    ("declared_difference_json",),
    ("hypothesis",),
    ("created_by",),
    ("contract_json", "manifest_set"),
    ("contract_json", "universe"),
    ("contract_json", "windows", "train"),
    ("contract_json", "windows", "validation"),
    ("contract_json", "windows", "holdout"),
    ("contract_json", "label", "version"),
    ("contract_json", "label", "horizons"),
    ("contract_json", "benchmark"),
    ("contract_json", "cost_fx_contract", "cost_scenarios"),
    ("contract_json", "cost_fx_contract", "fx"),
    ("contract_json", "purge_gap", "gap_eligible_sessions"),
    ("contract_json", "purge_gap", "overlap_method"),
    ("contract_json", "regimes"),
    ("contract_json", "uncertainty_estimator", "name"),
    ("contract_json", "uncertainty_estimator", "kernel"),
    ("contract_json", "uncertainty_estimator", "maxlags_by_horizon"),
    ("contract_json", "uncertainty_estimator", "small_sample_correction"),
    ("contract_json", "uncertainty_estimator", "alpha"),
    ("contract_json", "ablation_arms"),
    ("contract_json", "gates", "minimum_symbols_per_date"),
    ("contract_json", "gates", "minimum_dates_per_symbol"),
    ("contract_json", "gates", "minimum_effective_paired_dates"),
    ("contract_json", "gates", "minimum_coverage"),
    ("contract_json", "runtime_artifact_identity", "runtime"),
    ("contract_json", "runtime_artifact_identity", "artifact_sha256"),
    ("contract_json", "seed"),
    ("contract_json", "turnover", "formula"),
    ("contract_json", "turnover", "cash_included"),
    ("contract_json", "capacity", "adv_lookback_sessions"),
    ("contract_json", "capacity", "participation_cap"),
    ("contract_json", "capacity", "price_volume_adjustment"),
    ("contract_json", "capacity", "aggregation"),
    ("contract_json", "declared_trials"),
)


def _campaign_api():
    """Load the wished-for API while keeping pre-implementation RED collectible."""

    try:
        module = importlib.import_module("poseidon.research.campaign")
    except ModuleNotFoundError:
        pytest.fail("poseidon.research.campaign is not implemented")
    required = (
        "CampaignContractValidationError",
        "CampaignService",
        "FrozenCampaignIdentityConflict",
    )
    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, f"campaign API is incomplete: {missing}"
    return module


def _seed_strategy_versions(session, count: int = 3, marker: str | None = None) -> list[StrategyVersion]:
    marker = marker or uuid.uuid4().hex
    versions = []
    for index in range(count):
        strategy = StrategyRecord(
            name=f"phase99-campaign-{marker}-{index}",
            strategy_type="technical",
            config={},
            symbol="2330",
            market="tw_stock",
            interval="1d",
            active=False,
        )
        session.add(strategy)
        session.flush()
        config = {"marker": marker, "variant": index}
        policy = {"account_scope": "paper:phase99", "variant": index}
        artifact = {"uri": f"fixture://phase99/{marker}/{index}"}
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
        session.flush()
        versions.append(version)
    return versions


def _complete_campaign_contract(
    incumbent: StrategyVersion,
    candidate: StrategyVersion,
    *,
    marker: str | None = None,
) -> dict:
    marker = marker or uuid.uuid4().hex
    declared_trials = [
        {
            "original_trial_id": f"{role}-{arm}",
            "version_role": role,
            "strategy_version_id": str(version.id),
            "trial_role": "paired_evaluation",
            "ablation_arm": arm,
        }
        for role, version in (("incumbent", incumbent), ("candidate", candidate))
        for arm in ARMS
    ]
    return {
        "incumbent_strategy_version_id": incumbent.id,
        "candidate_strategy_version_id": candidate.id,
        "incumbent_content_sha256": incumbent.content_sha256,
        "candidate_content_sha256": candidate.content_sha256,
        "declared_difference_json": {
            "parameter": "lookback_sessions",
            "incumbent": 20,
            "candidate": 40,
        },
        "hypothesis": "A longer lookback improves out-of-sample stability.",
        "created_by": f"human:phase99:{marker}",
        "contract_json": {
            "manifest_set": [
                {"manifest_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"manifest:{marker}")), "content_sha256": "1" * 64}
            ],
            "universe": [
                {"symbol": "2330", "market": "tw_stock", "instrument": "spot"},
                {"symbol": "2317", "market": "tw_stock", "instrument": "spot"},
                {"symbol": "0050", "market": "tw_stock", "instrument": "spot"},
            ],
            "windows": {
                "train": {"start": "2024-01-02", "end": "2024-12-31"},
                "validation": {"start": "2025-01-02", "end": "2025-06-30"},
                "holdout": {"start": "2025-07-01", "end": "2025-12-31"},
            },
            "label": {
                "version": "forward-return-v1",
                "horizons": ["1_session", "5_sessions", "20_sessions"],
            },
            "benchmark": {"symbol": "0050", "return": "total_return"},
            "cost_fx_contract": {
                "reporting_currency": "TWD",
                "fx": {"version": "tw-cb-close-v1", "source": "fixture://fx"},
                "cost_scenarios": [
                    {"name": "base", "commission_bps": "2.5", "tax_bps": "30.0", "slippage_bps": "3.0"},
                    {"name": "stressed", "commission_bps": "4.0", "tax_bps": "30.0", "slippage_bps": "12.0"},
                ],
            },
            "purge_gap": {"gap_eligible_sessions": 20, "overlap_method": "purged_embargo"},
            "regimes": ["risk_on", "risk_off", "high_volatility"],
            "uncertainty_estimator": {
                "name": "ols_hac_intercept",
                "kernel": "bartlett",
                "maxlags_by_horizon": {"1_session": 0, "5_sessions": 4, "20_sessions": 19},
                "small_sample_correction": True,
                "alpha": "0.05",
            },
            "ablation_arms": list(ARMS),
            "gates": {
                "minimum_symbols_per_date": 3,
                "minimum_dates_per_symbol": 30,
                "minimum_effective_paired_dates": 30,
                "minimum_coverage": "0.80",
            },
            "runtime_artifact_identity": {
                "runtime": "poseidon-qlib-py312",
                "artifact_sha256": "2" * 64,
            },
            "seed": 20261007,
            "turnover": {
                "formula": "0.5*sum(abs(w_t-w_t_minus_1))",
                "cash_included": True,
            },
            "capacity": {
                "adv_lookback_sessions": 20,
                "participation_cap": "0.05",
                "price_volume_adjustment": "split_adjusted",
                "aggregation": "min_symbol_capacity",
            },
            "declared_trials": declared_trials,
        },
    }


def _terminal_trial_kwargs(campaign: ExperimentCampaign, version: StrategyVersion, *, arm: str = "combined") -> dict:
    now = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
    return {
        "campaign_id": campaign.id,
        "campaign_contract_sha256": campaign.contract_sha256,
        "original_trial_id": f"candidate-{arm}",
        "trial_role": "paired_evaluation",
        "strategy_version_id": version.id,
        "ablation_arm": arm,
        "paired_sample_key_sha256": "3" * 64,
        "input_sha256": "4" * 64,
        "result_sha256": "5" * 64,
        "started_at": now,
        "completed_at": now + timedelta(minutes=1),
        "terminal_state": "succeeded",
        "terminal_reason_json": {},
        "study_name": "phase99-paired-review",
        "config_json": {"frozen": True},
        "market": "tw_stock",
        "interval": "1d",
        "metrics_json": {"paired_effect": "0.02"},
        "composite_score": None,
        "wfe_score": None,
    }


def _remove_path(value: dict, path: tuple[str, ...]) -> None:
    parent = value
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]


def _assert_sqlstate(session, statement: str, expected: str = "23514", **params) -> None:
    savepoint = session.begin_nested()
    try:
        with pytest.raises(DBAPIError) as error:
            session.execute(text(statement), params)
        sqlstate = getattr(error.value.orig, "sqlstate", None) or getattr(error.value.orig, "pgcode", None)
        assert sqlstate == expected
    finally:
        savepoint.rollback()


def test_missing_any_frozen_field_leaves_no_campaign_row(phase99_session_factory):
    api = _campaign_api()
    with phase99_session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        complete = _complete_campaign_contract(incumbent, candidate)
        initial_count = session.query(ExperimentCampaign).count()

        for path in REQUIRED_CAMPAIGN_PATHS:
            incomplete = copy.deepcopy(complete)
            _remove_path(incomplete, path)
            with pytest.raises(api.CampaignContractValidationError, match=r"required|missing|complete|must|requires"):
                api.CampaignService(session).create_frozen_campaign(incomplete)
            assert session.query(ExperimentCampaign).count() == initial_count, path
        session.rollback()


def test_exact_replay_returns_same_campaign_and_contract_changes_change_identity(phase99_session_factory):
    api = _campaign_api()
    with phase99_session_factory() as session:
        incumbent, candidate, alternate = _seed_strategy_versions(session)
        complete = _complete_campaign_contract(incumbent, candidate)
        service = api.CampaignService(session)
        original = service.create_frozen_campaign(complete)
        replay = service.create_frozen_campaign(copy.deepcopy(complete))
        assert (replay.id, replay.contract_sha256) == (original.id, original.contract_sha256)
        assert session.query(ExperimentCampaign).filter_by(contract_sha256=original.contract_sha256).count() == 1

        mutations = []
        changed_candidate = copy.deepcopy(complete)
        changed_candidate["candidate_strategy_version_id"] = alternate.id
        changed_candidate["candidate_content_sha256"] = alternate.content_sha256
        for trial in changed_candidate["contract_json"]["declared_trials"]:
            if trial["version_role"] == "candidate":
                trial["strategy_version_id"] = str(alternate.id)
        mutations.append(changed_candidate)
        for path, replacement in (
            (("contract_json", "uncertainty_estimator", "alpha"), "0.01"),
            (("contract_json", "gates", "minimum_coverage"), "0.90"),
            (("contract_json", "cost_fx_contract", "cost_scenarios", 0, "slippage_bps"), "7.0"),
            (("contract_json", "windows", "holdout", "end"), "2026-01-30"),
            (("contract_json", "seed"), 20261008),
        ):
            changed = copy.deepcopy(complete)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            mutations.append(changed)

        changed_rows = [service.create_frozen_campaign(contract) for contract in mutations]
        hashes = {row.contract_sha256 for row in changed_rows}
        assert original.contract_sha256 not in hashes
        assert len(hashes) == len(mutations)
        session.rollback()


def test_campaign_declares_exact_six_paired_cells_and_appends_distinct_terminal_states(phase99_session_factory):
    api = _campaign_api()
    with phase99_session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        contract = _complete_campaign_contract(incumbent, candidate)
        declared = contract["contract_json"]["declared_trials"]
        assert {(cell["version_role"], cell["ablation_arm"]) for cell in declared} == {
            (role, arm) for role in ("incumbent", "candidate") for arm in ARMS
        }
        campaign = api.CampaignService(session).create_frozen_campaign(contract)
        tracker = ExperimentTracker(session)

        rows = []
        for index, terminal_state in enumerate(TERMINAL_STATES):
            arm = ARMS[index % len(ARMS)]
            role = "incumbent" if index < len(ARMS) else "candidate"
            version = incumbent if role == "incumbent" else candidate
            kwargs = _terminal_trial_kwargs(campaign, version, arm=arm)
            kwargs.update(
                original_trial_id=f"{role}-{arm}",
                paired_sample_key_sha256=f"{10 + index:064x}",
                input_sha256=f"{20 + index:064x}",
                result_sha256=f"{30 + index:064x}",
                terminal_state=terminal_state,
                terminal_reason_json={"state": terminal_state},
                metrics_json={"paired_effect": "0.02"} if terminal_state == "succeeded" else None,
            )
            rows.append(tracker.append_campaign_terminal_trial(**kwargs))

        assert [row.terminal_state for row in rows] == list(TERMINAL_STATES)
        assert all(row.composite_score is None and row.wfe_score is None for row in rows)
        assert all(row.metrics_json is None for row in rows[1:])
        assert session.query(ExperimentRecord).filter(ExperimentRecord.campaign_id == campaign.id).count() == 6
        session.rollback()


def test_campaign_event_append_is_insert_once(phase99_session_factory):
    api = _campaign_api()
    with phase99_session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        campaign = api.CampaignService(session).create_frozen_campaign(
            _complete_campaign_contract(incumbent, candidate)
        )
        service = api.CampaignService(session)
        payload = {"contract_sha256": campaign.contract_sha256}
        first = service.append_event(campaign.id, event_type="campaign_frozen", payload_json=payload)
        replay = service.append_event(campaign.id, event_type="campaign_frozen", payload_json=copy.deepcopy(payload))
        assert replay.id == first.id
        assert session.query(CampaignEvent).filter_by(campaign_id=campaign.id).count() == 1
        session.rollback()


def test_linked_trial_rejects_tracker_and_direct_sql_mutation_but_legacy_stays_unbound(phase99_session_factory):
    api = _campaign_api()
    with phase99_session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session)
        campaign = api.CampaignService(session).create_frozen_campaign(
            _complete_campaign_contract(incumbent, candidate)
        )
        tracker = ExperimentTracker(session)
        linked = tracker.append_campaign_terminal_trial(**_terminal_trial_kwargs(campaign, candidate))
        session.flush()
        before = session.execute(
            text("SELECT to_jsonb(e)::text FROM experiments AS e WHERE id = :id"), {"id": linked.id}
        ).scalar_one()

        with pytest.raises(ValueError, match="campaign-linked"):
            tracker.mark_rejected(linked.id)
        with pytest.raises(ValueError, match="campaign-linked"):
            tracker.mark_passed(linked.id)
        _assert_sqlstate(session, "UPDATE experiments SET status = 'rejected' WHERE id = :id", id=linked.id)
        _assert_sqlstate(session, "DELETE FROM experiments WHERE id = :id", id=linked.id)
        after = session.execute(
            text("SELECT to_jsonb(e)::text FROM experiments AS e WHERE id = :id"), {"id": linked.id}
        ).scalar_one()
        assert after == before

        legacy_id = tracker.save(
            study_name="phase99-legacy",
            config_json={},
            market="tw_stock",
            interval="1d",
        )
        disposable_id = tracker.save(
            study_name="phase99-legacy-delete",
            config_json={},
            market="tw_stock",
            interval="1d",
        )
        assert tracker.get_by_id(legacy_id).campaign_id is None
        tracker.mark_passed(legacy_id)
        assert tracker.get_by_id(legacy_id).status == "passed"
        _assert_sqlstate(
            session,
            "UPDATE experiments SET campaign_id = :campaign_id WHERE id = :id",
            id=legacy_id,
            campaign_id=campaign.id,
        )
        session.execute(text("DELETE FROM experiments WHERE id = :id"), {"id": disposable_id})
        assert session.query(ExperimentRecord).filter_by(id=disposable_id).count() == 0
        session.rollback()
