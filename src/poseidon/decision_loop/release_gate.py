"""Release gate: typed predicates over candidate, policy, campaign, review, holdout, reconciliation, and authority."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from poseidon.core.authz import AuthorizationError, AuthPrincipal
from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.timeutil import as_utc, timestamp
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionPolicy
from poseidon.decision_loop.strategy_config import (
    ExecutableStrategyConfig,
    LiveFeaturePlanError,
)
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import (
    CampaignEvent,
    CampaignReview,
    ExperimentCampaign,
    HoldoutUse,
)
from poseidon.models.order import OrderRecord
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion
from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS
from poseidon.research.campaign import (
    CampaignContractValidationError,
    _validate_complete_contract,
    build_review_input_identity,
)

PredicateState = Literal["passed", "blocked", "unavailable", "inconclusive"]


_ARMS = ("fundamental_only", "technical_only", "combined")


_ROLES = ("incumbent", "candidate")


_OUTSTANDING_ORDER_STATES = (
    "pending",
    "pending_submit",
    "reconciliation_required",
    "submitted",
    "partially_filled",
)


_PASSED_REVIEW_OBJECTS = (
    "cross_sectional_ic",
    "time_series_ic",
    "horizon_decay",
    "slices",
    "hac",
    "gross",
    "net",
    "drawdown",
    "turnover",
    "capacity",
    "cost_sensitivity",
)


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


class ReleaseGateRejected(DecisionConflictError):
    """Exact gate result that rejected a release mutation."""

    def __init__(self, operation: str, result: ReleaseGateResult) -> None:
        self.operation = operation
        self.result = result
        failed = ", ".join(predicate.code for predicate in result.predicates if predicate.state != "passed")
        super().__init__(f"{operation} gate did not pass: {failed}")


def _predicate(code: str, state: PredicateState, reason: str, **evidence) -> ReleasePredicate:
    return ReleasePredicate(code, state, reason, json.loads(canonical_json(evidence)))


HoldoutResolutionKind = Literal[
    "foreign_owner",
    "binding_missing",
    "ownership_mismatch",
    "reuse_denied",
    "consume_event_missing",
    "own_campaign",
]


_HOLDOUT_OUTCOMES: dict[str, tuple[PredicateState, str]] = {
    "foreign_owner": ("blocked", "holdout is owned by another campaign"),
    "binding_missing": ("unavailable", "review holdout binding is missing"),
    "ownership_mismatch": ("blocked", "committed holdout ownership does not match"),
    "reuse_denied": ("blocked", "holdout reuse was denied"),
    "consume_event_missing": ("unavailable", "committed holdout consume event is missing"),
    "own_campaign": ("passed", "same-campaign committed holdout use matches"),
}


_HOLDOUT_STATE_LABELS: dict[str, str] = {
    "foreign_owner": "consumed_by_another_campaign",
    "binding_missing": "unavailable",
    "ownership_mismatch": "ownership_mismatch",
    "reuse_denied": "reuse_denied",
    "consume_event_missing": "unavailable",
    "own_campaign": "consumed_by_this_campaign",
}


@dataclass(frozen=True)
class HoldoutResolution:
    kind: HoldoutResolutionKind
    holdout_use_id: uuid.UUID | None
    holdout_identity_sha256: str | None
    owner_campaign_id: uuid.UUID | None


class ReleaseGateEvaluator:
    """Evaluate release and rollback predicates read-only; the caller owns the transaction."""

    def __init__(self, session) -> None:
        self.session = session

    def candidate_predicate(self, strategy, candidate, candidate_id):
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
        try:
            executable = ExecutableStrategyConfig.model_validate(candidate.config_json)
        except (PydanticValidationError, TypeError, ValueError):
            return _predicate(
                "candidate_identity",
                "blocked",
                "candidate executable config is invalid",
                **evidence,
            )
        try:
            executable.live_feature_specs()
        except LiveFeaturePlanError:
            return _predicate(
                "candidate_identity",
                "blocked",
                "candidate executable config is not live-capable",
                **evidence,
            )
        evidence.update(
            strategy_type=executable.strategy_type,
            symbol=executable.symbol,
            market=executable.market,
            instrument=executable.instrument,
            interval=executable.interval,
        )
        if not executable.routing_matches(strategy):
            return _predicate(
                "candidate_identity",
                "blocked",
                "candidate executable routing identity does not match strategy",
                **evidence,
            )
        return _predicate(
            "candidate_identity",
            "passed",
            "candidate content, ownership, and executable config match",
            **evidence,
        )

    def policy_predicate(self, strategy, candidate):
        evidence = {"candidate_version_id": None if candidate is None else str(candidate.id)}
        if candidate is None:
            return _predicate("policy_identity", "unavailable", "candidate policy is unavailable", **evidence), None
        evidence["policy_sha256"] = content_sha256(candidate.policy_json)
        try:
            policy = DecisionPolicy.model_validate(candidate.policy_json)
        except (PydanticValidationError, TypeError, ValueError):
            return _predicate(
                "policy_identity", "unavailable", "frozen strategy policy is incomplete", **evidence
            ), None
        evidence.update(
            account_scope=policy.account_scope,
            market=policy.market,
            mode=policy.mode,
            requires_human_release=policy.release_gate.requires_human_release,
            strategy_market=None if strategy is None else strategy.market,
        )
        if strategy is None:
            return _predicate(
                "policy_identity", "unavailable", "owning strategy market is unavailable", **evidence
            ), policy
        try:
            executable = ExecutableStrategyConfig.model_validate(candidate.config_json)
        except (PydanticValidationError, TypeError, ValueError):
            return _predicate(
                "policy_identity", "unavailable", "candidate executable config is invalid", **evidence
            ), policy
        if executable.market != policy.market:
            return _predicate(
                "policy_identity",
                "blocked",
                "executable market does not match frozen policy market",
                **evidence,
            ), policy
        if policy.market != strategy.market:
            return _predicate(
                "policy_identity", "blocked", "frozen policy market does not match strategy market", **evidence
            ), policy
        if policy.mode != "human" or policy.release_gate.requires_human_release is not True:
            return _predicate(
                "policy_identity", "blocked", "pilot policy must remain human-release-only", **evidence
            ), policy
        if policy.reconciliation is None:
            return _predicate(
                "policy_identity", "unavailable", "frozen reconciliation policy is missing", **evidence
            ), policy
        if executable.market == "crypto_perp" and executable.symbol not in policy.reconciliation.perp_instrument_rules:
            return _predicate(
                "policy_identity",
                "blocked",
                "executable perp contract lacks an owner-frozen reconciliation rule",
                **evidence,
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
        try:
            executable = ExecutableStrategyConfig.model_validate(candidate.config_json)
        except (PydanticValidationError, TypeError, ValueError):
            return _predicate("campaign_identity", "unavailable", "candidate executable config is invalid", **evidence)
        universe = campaign.contract_json.get("universe")
        if not isinstance(universe, list) or not any(
            isinstance(member, dict)
            and member.get("symbol") == executable.symbol
            and member.get("market") == executable.market
            and member.get("instrument") == executable.instrument
            for member in universe
        ):
            return _predicate(
                "campaign_identity",
                "blocked",
                "candidate executable routing identity is outside campaign universe",
                **evidence,
            )
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
            return _predicate(
                "incumbent_pointer", "blocked", "active pointer drifted from campaign incumbent", **evidence
            )
        return _predicate("incumbent_pointer", "passed", "active pointer matches campaign incumbent", **evidence)

    def _cells_predicate(self, campaign):
        evidence = {"campaign_id": None if campaign is None else str(campaign.id), "trial_ids": []}
        if campaign is None:
            return _predicate("six_cell_evidence", "unavailable", "campaign trials are unavailable", **evidence), []
        trials = self.session.scalars(
            select(ExperimentRecord).where(ExperimentRecord.campaign_id == campaign.id).order_by(ExperimentRecord.id)
        ).all()
        evidence["trial_ids"] = [str(trial.id) for trial in trials]
        declared = campaign.contract_json.get("declared_trials") if isinstance(campaign.contract_json, dict) else None
        if not isinstance(declared, list) or len(declared) != 6 or not all(isinstance(item, dict) for item in declared):
            return _predicate(
                "six_cell_evidence",
                "unavailable",
                "frozen declared trial identity is unavailable",
                **evidence,
            ), trials
        version_roles = {
            campaign.incumbent_strategy_version_id: "incumbent",
            campaign.candidate_strategy_version_id: "candidate",
        }
        identity_fields = (
            "original_trial_id",
            "trial_role",
            "strategy_version_id",
            "version_role",
            "ablation_arm",
        )
        declared_identity = sorted(
            ({field: item.get(field) for field in identity_fields} for item in declared),
            key=canonical_json,
        )
        actual_identity = sorted(
            (
                {
                    "original_trial_id": trial.original_trial_id,
                    "trial_role": trial.trial_role,
                    "strategy_version_id": None
                    if trial.strategy_version_id is None
                    else str(trial.strategy_version_id),
                    "version_role": version_roles.get(trial.strategy_version_id),
                    "ablation_arm": trial.ablation_arm,
                }
                for trial in trials
            ),
            key=canonical_json,
        )
        evidence.update(
            declared_trial_identity_sha256=content_sha256(declared_identity),
            actual_trial_identity_sha256=content_sha256(actual_identity),
        )
        if actual_identity != declared_identity:
            return _predicate(
                "six_cell_evidence",
                "blocked",
                "terminal trials do not match frozen declared identities",
                **evidence,
            ), trials
        cells = {
            (
                "incumbent" if trial.strategy_version_id == campaign.incumbent_strategy_version_id else "candidate",
                trial.ablation_arm,
            )
            for trial in trials
            if trial.strategy_version_id
            in {campaign.incumbent_strategy_version_id, campaign.candidate_strategy_version_id}
        }
        expected = {(role, arm) for role in _ROLES for arm in _ARMS}
        if len(trials) != 6 or cells != expected:
            return _predicate(
                "six_cell_evidence", "unavailable", "exact six terminal cells are missing", **evidence
            ), trials
        terminal_states = {trial.terminal_state for trial in trials}
        evidence["terminal_states"] = sorted(str(state) for state in terminal_states)
        if "statistically_inconclusive" in terminal_states:
            return _predicate(
                "six_cell_evidence", "inconclusive", "paired trial evidence is inconclusive", **evidence
            ), trials
        if terminal_states != {"succeeded"}:
            return _predicate(
                "six_cell_evidence", "unavailable", "paired trial evidence is not complete", **evidence
            ), trials
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
        if review.result_json.get("paired_sample_digest") != review.input_identity_json.get("paired_sample_sha256"):
            return _predicate("review_identity", "blocked", "review result paired sample changed", **evidence)
        coverage = review.result_json.get("coverage_matrix")
        hac = review.result_json.get("hac")
        capacity = review.result_json.get("capacity")
        complete_coverage = (
            isinstance(coverage, dict)
            and set(coverage) == {f"{role}:{arm}" for role in _ROLES for arm in _ARMS}
            and all(
                isinstance(value, dict) and value.get("present") is True and value.get("terminal_state") == "succeeded"
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
        complete_outputs = (
            review.result_json.get("reason_code") is None
            and all(
                isinstance(review.result_json.get(field), dict) and review.result_json[field]
                for field in _PASSED_REVIEW_OBJECTS
            )
            and capacity.get("status") == "available"
            and capacity.get("value") is not None
            and review.result_json.get("failed_gates") == []
        )
        if not complete_coverage or not mature or not complete_outputs:
            return _predicate(
                "review_identity", "inconclusive", "passed review lacks mature complete evidence", **evidence
            )
        return _predicate("review_identity", "passed", "passed review identity and evidence match", **evidence)

    def holdout_resolution(self, campaign, review) -> HoldoutResolution:
        """Resolve stored holdout ownership facts shared by the gate and state reader."""

        identity = (
            review.input_identity_json.get("holdout_identity_sha256")
            if review is not None and isinstance(review.input_identity_json, dict)
            else None
        )
        holdout_use_id = None if review is None else review.holdout_use_id
        owner = (
            None
            if identity is None
            else self.session.scalar(select(HoldoutUse).where(HoldoutUse.holdout_identity_sha256 == identity))
        )
        owner_campaign_id = None if owner is None else owner.campaign_id

        def resolved(kind):
            return HoldoutResolution(kind, holdout_use_id, identity, owner_campaign_id)

        if owner is not None and campaign is not None and owner.campaign_id != campaign.id:
            return resolved("foreign_owner")
        if campaign is None or review is None or review.holdout_use_id is None or identity is None:
            return resolved("binding_missing")
        use = self.session.get(HoldoutUse, review.holdout_use_id)
        if (
            use is None
            or use.campaign_id != campaign.id
            or use.campaign_contract_sha256 != campaign.contract_sha256
            or use.holdout_identity_sha256 != identity
        ):
            return resolved("ownership_mismatch")
        events = self.session.scalars(
            select(CampaignEvent).where(
                CampaignEvent.event_type.in_(
                    (
                        "holdout_consumed",
                        "holdout_replayed",
                        "holdout_reuse_denied",
                    )
                )
            )
        ).all()
        own_use = any(
            event.campaign_id == campaign.id
            and event.event_type in {"holdout_consumed", "holdout_replayed"}
            and event.payload_json.get("holdout_identity_sha256") == identity
            and event.payload_json.get("holdout_use_id") == str(use.id)
            for event in events
        )
        denied = any(
            event.event_type == "holdout_reuse_denied" and event.payload_json.get("holdout_identity_sha256") == identity
            for event in events
        )
        if denied:
            return resolved("reuse_denied")
        if not own_use:
            return resolved("consume_event_missing")
        return resolved("own_campaign")

    def _holdout_predicate(self, campaign, review):
        resolution = self.holdout_resolution(campaign, review)
        evidence = {
            "holdout_use_id": None if resolution.holdout_use_id is None else str(resolution.holdout_use_id),
            "holdout_identity_sha256": resolution.holdout_identity_sha256,
        }
        if resolution.kind == "foreign_owner":
            evidence["owner_campaign_id"] = str(resolution.owner_campaign_id)
        state, reason = _HOLDOUT_OUTCOMES[resolution.kind]
        return _predicate("holdout_ownership", state, reason, **evidence)

    def latest_reconciliation(self, account_scope, account_generation):
        """Return the reconciliation the gate treats as current for one account identity."""

        return self.session.scalar(
            select(AccountReconciliation)
            .where(
                AccountReconciliation.account_scope == account_scope,
                AccountReconciliation.account_generation == account_generation,
            )
            .order_by(
                AccountReconciliation.as_of.desc(),
                AccountReconciliation.created_at.desc(),
                AccountReconciliation.id.desc(),
            )
            .limit(1)
        )

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
        age = (current_time - as_utc(reconciliation.as_of)).total_seconds()
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
        latest = self.latest_reconciliation(reconciliation.account_scope, reconciliation.account_generation)
        if latest is not None and latest.id != reconciliation.id:
            evidence["newer_reconciliation_ids"] = [str(latest.id)]
            return _predicate("reconciliation", "blocked", "a newer reconciliation state exists", **evidence)
        evidence.update(
            broker_state_watermark=reconciliation.broker_state_watermark,
            internal_state_watermark=reconciliation.internal_state_watermark,
        )
        return _predicate("reconciliation", "passed", "current exact-policy reconciliation matches", **evidence)

    @staticmethod
    def authority_predicate(principal: AuthPrincipal, policy):
        evidence = {
            "actor_id": None if principal is None else principal.actor_id,
            "account_scope": None if policy is None else policy.account_scope,
        }
        if principal is None or policy is None:
            return _predicate(
                "release_authority", "unavailable", "release principal or policy is unavailable", **evidence
            )
        try:
            principal.require_role("release_manager")
            principal.require_account_scope(policy.account_scope)
        except AuthorizationError:
            return _predicate("release_authority", "blocked", "principal lacks human release authority", **evidence)
        return _predicate("release_authority", "passed", "human release manager covers account scope", **evidence)

    def evaluate(
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

        candidate_predicate = self.candidate_predicate(strategy, candidate, candidate_version_id)
        policy_predicate, policy = self.policy_predicate(strategy, candidate)
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
        authority_predicate = self.authority_predicate(principal, policy)
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

    @staticmethod
    def account_identity(policy):
        if policy.reconciliation is None:
            return None
        return policy.account_scope, policy.reconciliation.account_generation

    def rollback_gate(self, strategy, target, reconciliation, source_policy, principal, now):
        source = self.session.scalar(
            select(StrategyReleaseEvent)
            .where(
                StrategyReleaseEvent.strategy_id == strategy.id,
                StrategyReleaseEvent.event_type == "release",
                StrategyReleaseEvent.to_version_id == target.id,
            )
            .order_by(StrategyReleaseEvent.resulting_revision.desc())
            .limit(1)
        )
        target_evidence = {
            "strategy_version_id": str(target.id),
            "content_sha256": target.content_sha256,
            "source_release_event_id": None if source is None else str(source.id),
        }
        executable = None
        live_capable = False
        try:
            target.verify_content()
            executable = ExecutableStrategyConfig.model_validate(target.config_json)
            executable.live_feature_specs()
            live_capable = True
            valid_target = (
                target.strategy_id == strategy.id and source is not None and executable.routing_matches(strategy)
            )
        except (PydanticValidationError, TypeError, ValueError):
            valid_target = False
        if executable is None:
            target_reason = "rollback target executable config is invalid"
        elif not live_capable:
            target_reason = "rollback target executable config is not live-capable"
        elif not executable.routing_matches(strategy):
            target_reason = "rollback target executable routing identity does not match strategy"
        elif source is None or target.strategy_id != strategy.id:
            target_reason = "rollback target lacks prior valid release"
        else:
            target_reason = "prior released immutable target matches"
        target_predicate = _predicate(
            "rollback_target",
            "passed" if valid_target else "blocked",
            target_reason,
            **target_evidence,
        )

        policy_predicate, policy = self.policy_predicate(strategy, target)
        if policy is None or policy.reconciliation is None:
            rollback_policy = _predicate(
                "rollback_policy",
                "unavailable",
                "rollback policy is incomplete",
                strategy_version_id=str(target.id),
            )
        else:
            valid_policy = (
                policy_predicate.state == "passed"
                and executable is not None
                and executable.market == policy.market
                and policy.mode == "human"
                and policy.release_gate.requires_human_release is True
                and policy.protective_exit.allowed_without_approval is True
                and policy.market == strategy.market
                and self.account_identity(policy) == self.account_identity(source_policy)
            )
            rollback_policy = _predicate(
                "rollback_policy",
                "passed" if valid_policy else "blocked",
                "source and target account identity plus protective-exit policy match"
                if valid_policy
                else "source and target account identity or protective-exit policy is unsafe",
                strategy_version_id=str(target.id),
                account_scope=policy.account_scope,
                account_generation=policy.reconciliation.account_generation,
                source_account_scope=source_policy.account_scope,
                source_account_generation=(
                    None if source_policy.reconciliation is None else source_policy.reconciliation.account_generation
                ),
                market=policy.market,
                protective_exit=policy.protective_exit.allowed_without_approval,
                policy_state=policy_predicate.state,
            )

        if policy is None or policy.reconciliation is None:
            lots = []
            claimed = []
            orders = []
            holdings_valid = False
        else:
            lots = self.session.scalars(
                select(PositionLot).where(
                    PositionLot.account_scope == policy.account_scope,
                    PositionLot.open_quantity > 0,
                )
            ).all()
            source_decision_ids = {lot.opening_decision_id for lot in lots}
            source_decisions = (
                self.session.scalars(select(DecisionRecord).where(DecisionRecord.id.in_(source_decision_ids))).all()
                if source_decision_ids
                else []
            )
            source_versions = {
                row.id: row
                for row in self.session.scalars(
                    select(StrategyVersion).where(
                        StrategyVersion.id.in_({row.strategy_version_id for row in source_decisions})
                    )
                ).all()
            }
            holdings_valid = (
                len(source_decisions) == len(source_decision_ids)
                and all(
                    lot.account_generation == policy.reconciliation.account_generation and lot.market == policy.market
                    for lot in lots
                )
                and all(
                    source_versions.get(row.strategy_version_id) is not None
                    and source_versions[row.strategy_version_id].strategy_id == strategy.id
                    for row in source_decisions
                )
            )
            strategy_versions = select(StrategyVersion.id).where(StrategyVersion.strategy_id == strategy.id)
            claimed = self.session.scalars(
                select(DecisionRecord).where(
                    DecisionRecord.strategy_version_id.in_(strategy_versions),
                    DecisionRecord.account_scope == policy.account_scope,
                    DecisionRecord.status.in_(("pending_approval", "approved", "execution_claimed")),
                )
            ).all()
            orders = self.session.scalars(
                select(OrderRecord).where(
                    OrderRecord.account_scope == policy.account_scope,
                    OrderRecord.account_generation == policy.reconciliation.account_generation,
                    OrderRecord.status.in_(_OUTSTANDING_ORDER_STATES),
                    ~OrderRecord.order_origin.in_(DURABLE_PROTECTIVE_ORIGINS),
                )
            ).all()
        holdings_predicate = _predicate(
            "rollback_holdings",
            "passed" if holdings_valid else "blocked",
            "current holdings remain in frozen strategy/account/exit scope"
            if holdings_valid
            else "current holdings are outside frozen rollback scope",
            position_lot_ids=[str(lot.id) for lot in lots],
        )
        work_clear = not claimed and not orders
        work_predicate = _predicate(
            "rollback_outstanding_work",
            "passed" if work_clear else "blocked",
            "no ordinary pending, approved, or claimed intent or active order requires takeover"
            if work_clear
            else "ordinary pending, approved, or claimed intent or active order is outstanding",
            claimed_decision_ids=[str(row.id) for row in claimed],
            active_order_ids=[str(row.id) for row in orders],
        )
        reconciliation_predicate = self._reconciliation_predicate(
            reconciliation,
            policy,
            content_sha256(target.policy_json),
            now,
        )
        rollback_reconciliation = _predicate(
            "rollback_reconciliation",
            reconciliation_predicate.state,
            reconciliation_predicate.reason,
            **reconciliation_predicate.evidence,
        )
        authority = self.authority_predicate(principal, policy)
        return (
            ReleaseGateResult(
                (
                    target_predicate,
                    rollback_policy,
                    holdings_predicate,
                    work_predicate,
                    rollback_reconciliation,
                    authority,
                )
            ),
            source,
            policy,
        )
