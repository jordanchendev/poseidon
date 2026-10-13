"""Fail-closed strategy release evidence and authority transitions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select, text

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionPolicy
from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    required_text,
    timestamp,
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
from poseidon.models.strategy import StrategyRecord
from poseidon.models.order import OrderRecord
from poseidon.models.position_lot import PositionLot
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
_OUTSTANDING_ORDER_STATES = ("pending_submit", "reconciliation_required", "submitted", "partially_filled")


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

    @staticmethod
    def _request_digest(
        operation,
        strategy_id,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        body,
    ):
        principal.require_role("release_manager")
        required_text(idempotency_key, "Idempotency-Key")
        if len(idempotency_key) > 256:
            raise ValidationError("Idempotency-Key is too long")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValidationError("expected_revision must be a non-negative integer")
        try:
            expected_pointer = (
                None if expected_active_version_id is None else uuid.UUID(str(expected_active_version_id))
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValidationError("expected_active_version_id must be a UUID or null") from error
        normalized_body = json.loads(canonical_json(body))
        digest = content_sha256(
            {
                "operation": operation,
                "strategy_id": str(strategy_id),
                "expected_revision": expected_revision,
                "expected_active_version_id": None if expected_pointer is None else str(expected_pointer),
                "actor_id": principal.actor_id,
                "body": normalized_body,
            }
        )
        return digest, expected_pointer, normalized_body

    def _idempotency_event(self, strategy_id, idempotency_key):
        return self.session.scalar(
            select(StrategyReleaseEvent).where(
                StrategyReleaseEvent.strategy_id == strategy_id,
                StrategyReleaseEvent.idempotency_key == idempotency_key,
            )
        )

    @staticmethod
    def _replay_event(event, request_sha256):
        if event.request_sha256 != request_sha256:
            raise DecisionConflictError("idempotency key was used for a different request")
        if not isinstance(event.response_json, dict):
            raise DecisionConflictError("stored idempotency response is invalid")
        return json.loads(canonical_json(event.response_json))

    def _lock_strategy(self, strategy_id):
        strategy = self.session.scalar(
            select(StrategyRecord)
            .where(StrategyRecord.id == strategy_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if strategy is None:
            raise DecisionConflictError("strategy does not exist")
        return strategy

    @staticmethod
    def _compare_expected(strategy, expected_revision, expected_pointer):
        if strategy.release_revision != expected_revision:
            raise DecisionConflictError("strategy release revision changed")
        if strategy.active_version_id != expected_pointer:
            raise DecisionConflictError("strategy active pointer changed")

    @staticmethod
    def _policy_for_version(version, strategy, principal):
        if version is None or version.strategy_id != strategy.id:
            raise DecisionConflictError("strategy version does not belong to the strategy")
        try:
            version.verify_content()
            policy = DecisionPolicy.model_validate(version.policy_json)
        except (PydanticValidationError, TypeError, ValueError) as error:
            raise DecisionConflictError("strategy version policy or immutable content changed") from error
        if policy.mode != "human" or policy.release_gate.requires_human_release is not True:
            raise DecisionConflictError("strategy policy is not human-release-only")
        principal.require_account_scope(policy.account_scope)
        return policy

    @staticmethod
    def _response(event_id, operation, strategy, from_version_id, to_version_id):
        return json.loads(
            canonical_json(
                {
                    "event_id": str(event_id),
                    "event_type": operation,
                    "strategy_id": str(strategy.id),
                    "from_version_id": None if from_version_id is None else str(from_version_id),
                    "to_version_id": None if to_version_id is None else str(to_version_id),
                    "release_revision": strategy.release_revision + 1,
                }
            )
        )

    def _append_transition(
        self,
        *,
        operation,
        strategy,
        candidate_version_id,
        to_version_id,
        source_release_event_id,
        campaign_id,
        review_id,
        holdout_use_id,
        reconciliation_id,
        principal,
        account_scope,
        expected_revision,
        idempotency_key,
        request_sha256,
        reason,
        gate_json,
    ):
        event_id = uuid.uuid4()
        response = self._response(
            event_id,
            operation,
            strategy,
            strategy.active_version_id,
            to_version_id,
        )
        event = StrategyReleaseEvent(
            id=event_id,
            strategy_id=strategy.id,
            event_type=operation,
            candidate_version_id=candidate_version_id,
            from_version_id=strategy.active_version_id,
            to_version_id=to_version_id,
            source_release_event_id=source_release_event_id,
            campaign_id=campaign_id,
            review_id=review_id,
            holdout_use_id=holdout_use_id,
            reconciliation_id=reconciliation_id,
            actor_id=principal.actor_id,
            account_scope=account_scope,
            expected_revision=expected_revision,
            resulting_revision=expected_revision + 1,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            reason=reason,
            gate_json=json.loads(canonical_json(gate_json)),
            response_json=response,
        )
        self.session.add(event)
        self.session.flush()
        self.session.execute(
            text("SELECT set_config('poseidon.release_event_id', :event_id, true)"),
            {"event_id": str(event.id)},
        )
        strategy.active_version_id = to_version_id
        strategy.release_revision = expected_revision + 1
        self.session.flush()
        return response

    def _begin_transition(
        self,
        *,
        operation,
        strategy_id,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        body,
    ):
        digest, expected_pointer, normalized_body = self._request_digest(
            operation,
            strategy_id,
            expected_revision,
            expected_active_version_id,
            principal,
            idempotency_key,
            body,
        )
        replay = self._idempotency_event(strategy_id, idempotency_key)
        if replay is not None:
            return None, digest, expected_pointer, normalized_body, self._replay_event(replay, digest)
        strategy = self._lock_strategy(strategy_id)
        replay = self._idempotency_event(strategy_id, idempotency_key)
        if replay is not None:
            return None, digest, expected_pointer, normalized_body, self._replay_event(replay, digest)
        self._compare_expected(strategy, expected_revision, expected_pointer)
        return strategy, digest, expected_pointer, normalized_body, None

    @staticmethod
    def _require_allowed(result, operation):
        if result.allowed:
            return
        failed = ", ".join(predicate.code for predicate in result.predicates if predicate.state != "passed")
        raise DecisionConflictError(f"{operation} gate did not pass: {failed}")

    def release(
        self,
        strategy_id,
        candidate_version_id,
        campaign_id,
        review_id,
        reconciliation_id,
        *,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        reason,
        now=None,
    ):
        reason = required_text(reason, "reason")
        body = {
            "candidate_version_id": str(candidate_version_id),
            "campaign_id": str(campaign_id),
            "review_id": str(review_id),
            "reconciliation_id": str(reconciliation_id),
            "reason": reason,
        }
        strategy, digest, _pointer, _body, replay = self._begin_transition(
            operation="release",
            strategy_id=strategy_id,
            expected_revision=expected_revision,
            expected_active_version_id=expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            body=body,
        )
        if replay is not None:
            return replay
        candidate = self.session.get(StrategyVersion, candidate_version_id)
        policy = self._policy_for_version(candidate, strategy, principal)
        gate = self.evaluate_gate(
            strategy_id,
            candidate_version_id,
            campaign_id,
            review_id,
            reconciliation_id,
            principal=principal,
            now=now,
        )
        self._require_allowed(gate, "release")
        review = self.session.get(CampaignReview, review_id)
        return self._append_transition(
            operation="release",
            strategy=strategy,
            candidate_version_id=candidate_version_id,
            to_version_id=candidate_version_id,
            source_release_event_id=None,
            campaign_id=campaign_id,
            review_id=review_id,
            holdout_use_id=review.holdout_use_id,
            reconciliation_id=reconciliation_id,
            principal=principal,
            account_scope=policy.account_scope,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=digest,
            reason=reason,
            gate_json=gate.as_json(),
        )

    def _simple_transition(
        self,
        operation,
        strategy_id,
        candidate_version_id,
        *,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        reason,
    ):
        reason = required_text(reason, "reason")
        body = {
            "candidate_version_id": None if candidate_version_id is None else str(candidate_version_id),
            "reason": reason,
        }
        strategy, digest, _pointer, _body, replay = self._begin_transition(
            operation=operation,
            strategy_id=strategy_id,
            expected_revision=expected_revision,
            expected_active_version_id=expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            body=body,
        )
        if replay is not None:
            return replay
        authority_version_id = strategy.active_version_id if operation == "pause" else candidate_version_id
        if authority_version_id is None:
            raise DecisionConflictError("pause requires a current active version")
        version = self.session.get(StrategyVersion, authority_version_id)
        policy = self._policy_for_version(version, strategy, principal)
        predicates = (
            _predicate(
                "candidate_identity" if operation == "reject" else "incumbent_pointer",
                "passed",
                "strategy version immutable identity matches",
                strategy_version_id=str(authority_version_id),
                content_sha256=version.content_sha256,
            ),
            self._authority_predicate(principal, policy),
        )
        gate = ReleaseGateResult(predicates)
        self._require_allowed(gate, operation)
        to_version_id = None if operation == "pause" else strategy.active_version_id
        return self._append_transition(
            operation=operation,
            strategy=strategy,
            candidate_version_id=candidate_version_id,
            to_version_id=to_version_id,
            source_release_event_id=None,
            campaign_id=None,
            review_id=None,
            holdout_use_id=None,
            reconciliation_id=None,
            principal=principal,
            account_scope=policy.account_scope,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=digest,
            reason=reason,
            gate_json=gate.as_json(),
        )

    def pause(
        self,
        strategy_id,
        *,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        reason,
    ):
        return self._simple_transition(
            "pause",
            strategy_id,
            None,
            expected_revision=expected_revision,
            expected_active_version_id=expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            reason=reason,
        )

    def reject(
        self,
        strategy_id,
        candidate_version_id,
        *,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        reason,
    ):
        return self._simple_transition(
            "reject",
            strategy_id,
            candidate_version_id,
            expected_revision=expected_revision,
            expected_active_version_id=expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            reason=reason,
        )

    def _rollback_gate(self, strategy, target, reconciliation, principal, now):
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
        try:
            target.verify_content()
            valid_target = target.strategy_id == strategy.id and source is not None
        except (TypeError, ValueError):
            valid_target = False
        target_predicate = _predicate(
            "rollback_target",
            "passed" if valid_target else "blocked",
            "prior released immutable target matches" if valid_target else "rollback target lacks prior valid release",
            **target_evidence,
        )

        policy_predicate, policy = self._policy_predicate(target)
        if policy is None or policy.reconciliation is None:
            rollback_policy = _predicate(
                "rollback_policy",
                "unavailable",
                "rollback policy is incomplete",
                strategy_version_id=str(target.id),
            )
        else:
            valid_policy = (
                policy.mode == "human"
                and policy.release_gate.requires_human_release is True
                and policy.protective_exit.allowed_without_approval is True
                and policy.market == strategy.market
            )
            rollback_policy = _predicate(
                "rollback_policy",
                "passed" if valid_policy else "blocked",
                "frozen account and protective-exit policy match"
                if valid_policy
                else "frozen rollback account or protective-exit policy is unsafe",
                strategy_version_id=str(target.id),
                account_scope=policy.account_scope,
                account_generation=policy.reconciliation.account_generation,
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
                    lot.account_generation == policy.reconciliation.account_generation
                    and lot.market == policy.market
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
                    DecisionRecord.status == "execution_claimed",
                )
            ).all()
            orders = self.session.scalars(
                select(OrderRecord)
                .join(DecisionRecord, DecisionRecord.id == OrderRecord.decision_id)
                .where(
                    DecisionRecord.strategy_version_id.in_(strategy_versions),
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
            "no ordinary claimed intent or active order requires takeover"
            if work_clear
            else "ordinary claimed intent or active order is outstanding",
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
        authority = self._authority_predicate(principal, policy)
        return ReleaseGateResult(
            (
                target_predicate,
                rollback_policy,
                holdings_predicate,
                work_predicate,
                rollback_reconciliation,
                authority,
            )
        ), source, policy

    def rollback(
        self,
        strategy_id,
        target_version_id,
        reconciliation_id,
        *,
        expected_revision,
        expected_active_version_id,
        principal,
        idempotency_key,
        reason,
        now=None,
    ):
        reason = required_text(reason, "reason")
        body = {
            "target_version_id": str(target_version_id),
            "reconciliation_id": str(reconciliation_id),
            "reason": reason,
        }
        strategy, digest, _pointer, _body, replay = self._begin_transition(
            operation="rollback",
            strategy_id=strategy_id,
            expected_revision=expected_revision,
            expected_active_version_id=expected_active_version_id,
            principal=principal,
            idempotency_key=idempotency_key,
            body=body,
        )
        if replay is not None:
            return replay
        target = self.session.get(StrategyVersion, target_version_id)
        if target is None:
            raise DecisionConflictError("rollback target does not exist")
        reconciliation = self.session.get(AccountReconciliation, reconciliation_id)
        gate, source, policy = self._rollback_gate(strategy, target, reconciliation, principal, now)
        self._require_allowed(gate, "rollback")
        principal.require_account_scope(policy.account_scope)
        return self._append_transition(
            operation="rollback",
            strategy=strategy,
            candidate_version_id=target_version_id,
            to_version_id=target_version_id,
            source_release_event_id=source.id,
            campaign_id=None,
            review_id=None,
            holdout_use_id=None,
            reconciliation_id=reconciliation_id,
            principal=principal,
            account_scope=policy.account_scope,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=digest,
            reason=reason,
            gate_json=gate.as_json(),
        )
