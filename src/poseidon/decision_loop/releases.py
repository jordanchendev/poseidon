"""Fail-closed strategy release evidence and authority transitions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionPolicy
from poseidon.decision_loop.manifest import canonical_json, content_sha256, timestamp
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import (
    CampaignEvent,
    CampaignReview,
    ExperimentCampaign,
    HoldoutUse,
)
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion
from poseidon.research.campaign import (
    CampaignContractValidationError,
    _validate_complete_contract,
    build_review_input_identity,
)


PredicateState = Literal["passed", "blocked", "unavailable", "inconclusive"]
_ARMS = ("fundamental_only", "technical_only", "combined")
_ROLES = ("incumbent", "candidate")


@dataclass(frozen=True)
class ReleasePredicate:
    code: str
    state: PredicateState
    reason: str
    evidence: dict

    def as_json(self) -> dict:
        return json.loads(
            canonical_json(
                {
                    "code": self.code,
                    "state": self.state,
                    "reason": self.reason,
                    "evidence": self.evidence,
                }
            )
        )


@dataclass(frozen=True)
class ReleaseGateResult:
    predicates: tuple[ReleasePredicate, ...]

    @property
    def allowed(self) -> bool:
        return bool(self.predicates) and all(predicate.state == "passed" for predicate in self.predicates)

    def as_json(self) -> dict:
        return {
            "allowed": self.allowed,
            "predicates": [predicate.as_json() for predicate in self.predicates],
        }


def _stored_time(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _predicate(code: str, state: PredicateState, reason: str, **evidence) -> ReleasePredicate:
    return ReleasePredicate(code, state, reason, json.loads(canonical_json(evidence)))


class ReleaseService:
    """Read exact release evidence without owning the caller's transaction."""

    def __init__(self, session) -> None:
        self.session = session

    def _candidate_predicate(self, strategy, candidate, candidate_id):
        evidence = {"candidate_version_id": str(candidate_id)}
        if strategy is None:
            return _predicate("candidate_identity", "unavailable", "strategy does not exist", **evidence)
        evidence["strategy_id"] = str(strategy.id)
        if candidate is None:
            return _predicate("candidate_identity", "unavailable", "candidate version does not exist", **evidence)
        evidence["content_sha256"] = candidate.content_sha256
        try:
            candidate.verify_content()
        except (TypeError, ValueError):
            return _predicate("candidate_identity", "blocked", "candidate immutable content hash changed", **evidence)
        if candidate.strategy_id != strategy.id:
            return _predicate("candidate_identity", "blocked", "candidate belongs to another strategy", **evidence)
        return _predicate("candidate_identity", "passed", "candidate content and ownership match", **evidence)

    def _policy_predicate(self, candidate):
        evidence = {"candidate_version_id": None if candidate is None else str(candidate.id)}
        if candidate is None:
            return _predicate("policy_identity", "unavailable", "candidate policy is unavailable", **evidence), None
        evidence["policy_sha256"] = content_sha256(candidate.policy_json)
        try:
            policy = DecisionPolicy.model_validate(candidate.policy_json)
        except (PydanticValidationError, TypeError, ValueError):
            return _predicate("policy_identity", "unavailable", "frozen strategy policy is incomplete", **evidence), None
        evidence.update(
            account_scope=policy.account_scope,
            mode=policy.mode,
            requires_human_release=policy.release_gate.requires_human_release,
        )
        if policy.mode != "human" or policy.release_gate.requires_human_release is not True:
            return _predicate(
                "policy_identity", "blocked", "pilot policy must remain human-release-only", **evidence
            ), policy
        if policy.reconciliation is None:
            return _predicate(
                "policy_identity", "unavailable", "frozen reconciliation policy is missing", **evidence
            ), policy
        return _predicate("policy_identity", "passed", "frozen release policy matches", **evidence), policy

    def _campaign_predicate(self, strategy, candidate, campaign):
        evidence = {"campaign_id": None if campaign is None else str(campaign.id)}
        if campaign is None:
            return _predicate("campaign_identity", "unavailable", "campaign does not exist", **evidence)
        evidence["contract_sha256"] = campaign.contract_sha256
        try:
            campaign.verify_contract()
            _validate_complete_contract(
                {
                    "incumbent_strategy_version_id": campaign.incumbent_strategy_version_id,
                    "candidate_strategy_version_id": campaign.candidate_strategy_version_id,
                    "incumbent_content_sha256": campaign.incumbent_content_sha256,
                    "candidate_content_sha256": campaign.candidate_content_sha256,
                    "declared_difference_json": campaign.declared_difference_json,
                    "hypothesis": campaign.hypothesis,
                    "contract_json": campaign.contract_json,
                    "created_by": campaign.created_by,
                }
            )
        except (CampaignContractValidationError, TypeError, ValueError):
            return _predicate("campaign_identity", "unavailable", "frozen campaign contract is invalid", **evidence)
        if candidate is None:
            return _predicate("campaign_identity", "unavailable", "candidate is unavailable", **evidence)
        if (
            campaign.candidate_strategy_version_id != candidate.id
            or campaign.candidate_content_sha256 != candidate.content_sha256
        ):
            return _predicate("campaign_identity", "blocked", "campaign candidate identity changed", **evidence)
        incumbent = self.session.get(StrategyVersion, campaign.incumbent_strategy_version_id)
        if (
            incumbent is None
            or strategy is None
            or incumbent.strategy_id != strategy.id
            or incumbent.content_sha256 != campaign.incumbent_content_sha256
        ):
            return _predicate("campaign_identity", "blocked", "campaign incumbent identity changed", **evidence)
        try:
            incumbent.verify_content()
        except (TypeError, ValueError):
            return _predicate("campaign_identity", "blocked", "campaign incumbent content changed", **evidence)
        return _predicate("campaign_identity", "passed", "campaign and version identities match", **evidence)

    @staticmethod
    def _incumbent_predicate(strategy, campaign):
        evidence = {
            "strategy_id": None if strategy is None else str(strategy.id),
            "active_version_id": (
                None if strategy is None or strategy.active_version_id is None else str(strategy.active_version_id)
            ),
            "campaign_incumbent_version_id": (
                None if campaign is None else str(campaign.incumbent_strategy_version_id)
            ),
        }
        if strategy is None or campaign is None:
            return _predicate("incumbent_pointer", "unavailable", "incumbent evidence is unavailable", **evidence)
        if strategy.active_version_id is None:
            return _predicate("incumbent_pointer", "passed", "strategy has no released incumbent", **evidence)
        if strategy.active_version_id != campaign.incumbent_strategy_version_id:
            return _predicate("incumbent_pointer", "blocked", "active pointer drifted from campaign incumbent", **evidence)
        return _predicate("incumbent_pointer", "passed", "active pointer matches campaign incumbent", **evidence)

    def _cells_predicate(self, campaign):
        evidence = {"campaign_id": None if campaign is None else str(campaign.id), "trial_ids": []}
        if campaign is None:
            return _predicate("six_cell_evidence", "unavailable", "campaign trials are unavailable", **evidence), []
        trials = self.session.scalars(
            select(ExperimentRecord)
            .where(ExperimentRecord.campaign_id == campaign.id)
            .order_by(ExperimentRecord.id)
        ).all()
        evidence["trial_ids"] = [str(trial.id) for trial in trials]
        cells = {
            (
                "incumbent"
                if trial.strategy_version_id == campaign.incumbent_strategy_version_id
                else "candidate",
                trial.ablation_arm,
            )
            for trial in trials
            if trial.strategy_version_id
            in {campaign.incumbent_strategy_version_id, campaign.candidate_strategy_version_id}
        }
        expected = {(role, arm) for role in _ROLES for arm in _ARMS}
        if len(trials) != 6 or cells != expected:
            return _predicate("six_cell_evidence", "unavailable", "exact six terminal cells are missing", **evidence), trials
        terminal_states = {trial.terminal_state for trial in trials}
        evidence["terminal_states"] = sorted(str(state) for state in terminal_states)
        if "statistically_inconclusive" in terminal_states:
            return _predicate(
                "six_cell_evidence", "inconclusive", "paired trial evidence is inconclusive", **evidence
            ), trials
        if terminal_states != {"succeeded"}:
            return _predicate("six_cell_evidence", "unavailable", "paired trial evidence is not complete", **evidence), trials
        memberships = []
        for trial in trials:
            membership = (trial.metrics_json or {}).get("sample_membership")
            if not isinstance(membership, list) or not membership:
                return _predicate(
                    "six_cell_evidence", "unavailable", "paired sample membership is missing", **evidence
                ), trials
            memberships.append(content_sha256(sorted(membership, key=canonical_json)))
        paired_keys = {trial.paired_sample_key_sha256 for trial in trials}
        if len(set(memberships)) != 1 or len(paired_keys) != 1:
            return _predicate(
                "six_cell_evidence", "blocked", "paired trial membership identity changed", **evidence
            ), trials
        evidence["paired_sample_sha256"] = memberships[0]
        evidence["paired_sample_key_sha256"] = next(iter(paired_keys))
        return _predicate("six_cell_evidence", "passed", "exact six terminal cells match", **evidence), trials

    def _review_predicate(self, campaign, trials, review, policy):
        evidence = {"review_id": None if review is None else str(review.id)}
        if campaign is None or review is None:
            return _predicate("review_identity", "unavailable", "campaign review does not exist", **evidence)
        evidence.update(status=review.status, input_sha256=review.input_sha256, result_sha256=review.result_sha256)
        if review.campaign_id != campaign.id:
            return _predicate("review_identity", "blocked", "review belongs to another campaign", **evidence)
        if review.holdout_use_id is None or not isinstance(review.input_identity_json, dict):
            return _predicate("review_identity", "unavailable", "legacy review lacks exact release binding", **evidence)
        if review.input_sha256 != content_sha256(review.input_identity_json):
            return _predicate("review_identity", "blocked", "review input digest changed", **evidence)
        if review.result_sha256 != content_sha256(review.result_json):
            return _predicate("review_identity", "blocked", "review result digest changed", **evidence)
        try:
            expected_identity = build_review_input_identity(
                campaign,
                trials,
                holdout_identity_sha256=review.input_identity_json["holdout_identity_sha256"],
                panel_sha256=review.input_identity_json["panel_sha256"],
            )
        except (CampaignContractValidationError, KeyError, TypeError, ValueError):
            return _predicate("review_identity", "unavailable", "review input components are incomplete", **evidence)
        if expected_identity != review.input_identity_json:
            return _predicate("review_identity", "blocked", "review input components changed", **evidence)
        if review.status == "inconclusive":
            return _predicate("review_identity", "inconclusive", "campaign review is inconclusive", **evidence)
        if review.status == "unavailable":
            return _predicate("review_identity", "unavailable", "campaign review is unavailable", **evidence)
        if review.status != "passed" or review.result_json.get("status") != "passed":
            return _predicate("review_identity", "blocked", "campaign review did not pass", **evidence)
        coverage = review.result_json.get("coverage_matrix")
        hac = review.result_json.get("hac")
        complete_coverage = (
            isinstance(coverage, dict)
            and set(coverage) == {f"{role}:{arm}" for role in _ROLES for arm in _ARMS}
            and all(
                isinstance(value, dict)
                and value.get("present") is True
                and value.get("terminal_state") == "succeeded"
                for value in coverage.values()
            )
        )
        mature = (
            policy is not None
            and isinstance(hac, dict)
            and hac.get("status") == "available"
            and type(hac.get("sample_count")) is int
            and hac["sample_count"] >= policy.release_gate.minimum_mature_samples
        )
        complete_outputs = all(
            review.result_json.get(field) is not None
            for field in ("cross_sectional_ic", "time_series_ic", "capacity")
        ) and review.result_json.get("failed_gates") == []
        if not complete_coverage or not mature or not complete_outputs:
            return _predicate("review_identity", "inconclusive", "passed review lacks mature complete evidence", **evidence)
        return _predicate("review_identity", "passed", "passed review identity and evidence match", **evidence)

    def _holdout_predicate(self, campaign, review):
        identity = (
            review.input_identity_json.get("holdout_identity_sha256")
            if review is not None and isinstance(review.input_identity_json, dict)
            else None
        )
        evidence = {
            "holdout_use_id": None if review is None or review.holdout_use_id is None else str(review.holdout_use_id),
            "holdout_identity_sha256": identity,
        }
        owner = (
            None
            if identity is None
            else self.session.scalar(select(HoldoutUse).where(HoldoutUse.holdout_identity_sha256 == identity))
        )
        if owner is not None and campaign is not None and owner.campaign_id != campaign.id:
            evidence["owner_campaign_id"] = str(owner.campaign_id)
            return _predicate("holdout_ownership", "blocked", "holdout is owned by another campaign", **evidence)
        if campaign is None or review is None or review.holdout_use_id is None or identity is None:
            return _predicate("holdout_ownership", "unavailable", "review holdout binding is missing", **evidence)
        use = self.session.get(HoldoutUse, review.holdout_use_id)
        if (
            use is None
            or use.campaign_id != campaign.id
            or use.campaign_contract_sha256 != campaign.contract_sha256
            or use.holdout_identity_sha256 != identity
        ):
            return _predicate("holdout_ownership", "blocked", "committed holdout ownership does not match", **evidence)
        events = self.session.scalars(select(CampaignEvent).where(CampaignEvent.event_type.in_((
            "holdout_consumed",
            "holdout_replayed",
            "holdout_reuse_denied",
        )))).all()
        own_use = any(
            event.campaign_id == campaign.id
            and event.event_type in {"holdout_consumed", "holdout_replayed"}
            and event.payload_json.get("holdout_identity_sha256") == identity
            and event.payload_json.get("holdout_use_id") == str(use.id)
            for event in events
        )
        denied = any(
            event.event_type == "holdout_reuse_denied"
            and event.payload_json.get("holdout_identity_sha256") == identity
            for event in events
        )
        if denied:
            return _predicate("holdout_ownership", "blocked", "holdout reuse was denied", **evidence)
        if not own_use:
            return _predicate("holdout_ownership", "unavailable", "committed holdout consume event is missing", **evidence)
        return _predicate("holdout_ownership", "passed", "same-campaign committed holdout use matches", **evidence)

    def _reconciliation_predicate(self, reconciliation, policy, policy_sha256, now):
        evidence = {"reconciliation_id": None if reconciliation is None else str(reconciliation.id)}
        if policy is None or policy.reconciliation is None:
            return _predicate("reconciliation", "unavailable", "reconciliation policy is unavailable", **evidence)
        evidence.update(
            account_scope=policy.account_scope,
            account_generation=policy.reconciliation.account_generation,
            policy_sha256=None if reconciliation is None else reconciliation.policy_sha256,
        )
        if reconciliation is None:
            return _predicate("reconciliation", "unavailable", "account reconciliation does not exist", **evidence)
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        age = (current_time - _stored_time(reconciliation.as_of)).total_seconds()
        if (
            reconciliation.account_scope != policy.account_scope
            or reconciliation.account_generation != policy.reconciliation.account_generation
            or reconciliation.policy_sha256 != policy_sha256
        ):
            return _predicate("reconciliation", "blocked", "reconciliation policy identity changed", **evidence)
        if reconciliation.status != "matched":
            return _predicate("reconciliation", "blocked", "reconciliation is unresolved", **evidence)
        if age < 0 or age > policy.reconciliation.max_reconciliation_age_seconds:
            evidence["age_seconds"] = age
            return _predicate("reconciliation", "blocked", "reconciliation is stale", **evidence)
        if not reconciliation.broker_state_watermark or not reconciliation.internal_state_watermark:
            return _predicate("reconciliation", "unavailable", "reconciliation watermarks are missing", **evidence)
        newer = self.session.scalars(
            select(AccountReconciliation).where(
                AccountReconciliation.account_scope == reconciliation.account_scope,
                AccountReconciliation.account_generation == reconciliation.account_generation,
                AccountReconciliation.as_of > reconciliation.as_of,
            )
        ).all()
        if newer:
            evidence["newer_reconciliation_ids"] = [str(row.id) for row in newer]
            return _predicate("reconciliation", "blocked", "a newer reconciliation state exists", **evidence)
        evidence.update(
            broker_state_watermark=reconciliation.broker_state_watermark,
            internal_state_watermark=reconciliation.internal_state_watermark,
        )
        return _predicate("reconciliation", "passed", "current exact-policy reconciliation matches", **evidence)

    @staticmethod
    def _authority_predicate(principal: AuthPrincipal, policy):
        evidence = {
            "actor_id": None if principal is None else principal.actor_id,
            "account_scope": None if policy is None else policy.account_scope,
        }
        if principal is None or policy is None:
            return _predicate("release_authority", "unavailable", "release principal or policy is unavailable", **evidence)
        try:
            principal.require_role("release_manager")
            principal.require_account_scope(policy.account_scope)
        except HTTPException:
            return _predicate("release_authority", "blocked", "principal lacks human release authority", **evidence)
        return _predicate("release_authority", "passed", "human release manager covers account scope", **evidence)

    def evaluate_gate(
        self,
        strategy_id,
        candidate_version_id,
        campaign_id,
        review_id,
        reconciliation_id,
        *,
        principal: AuthPrincipal,
        now=None,
    ) -> ReleaseGateResult:
        """Re-hash and retain every release predicate; never mutate or commit."""

        strategy = self.session.get(StrategyRecord, strategy_id)
        candidate = self.session.get(StrategyVersion, candidate_version_id)
        campaign = self.session.get(ExperimentCampaign, campaign_id)
        review = self.session.get(CampaignReview, review_id)
        reconciliation = self.session.get(AccountReconciliation, reconciliation_id)

        candidate_predicate = self._candidate_predicate(strategy, candidate, candidate_version_id)
        policy_predicate, policy = self._policy_predicate(candidate)
        campaign_predicate = self._campaign_predicate(strategy, candidate, campaign)
        incumbent_predicate = self._incumbent_predicate(strategy, campaign)
        cells_predicate, trials = self._cells_predicate(campaign)
        review_predicate = self._review_predicate(campaign, trials, review, policy)
        holdout_predicate = self._holdout_predicate(campaign, review)
        policy_sha256 = None if candidate is None else content_sha256(candidate.policy_json)
        reconciliation_predicate = self._reconciliation_predicate(
            reconciliation,
            policy,
            policy_sha256,
            now,
        )
        authority_predicate = self._authority_predicate(principal, policy)
        return ReleaseGateResult(
            (
                candidate_predicate,
                policy_predicate,
                campaign_predicate,
                incumbent_predicate,
                cells_predicate,
                review_predicate,
                holdout_predicate,
                reconciliation_predicate,
                authority_predicate,
            )
        )
