"""Fail-closed strategy release evidence and authority transitions."""

from __future__ import annotations

import json
import uuid

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select, text

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.errors import ValidationError
from poseidon.core.locks import lock_account
from poseidon.decision_loop.decisions import DecisionConflictError, DecisionPolicy
from poseidon.decision_loop.manifest import required_text
from poseidon.decision_loop.release_gate import (
    _HOLDOUT_STATE_LABELS,
    ReleaseGateEvaluator,
    ReleaseGateRejected,
    ReleaseGateResult,
    _predicate,
)
from poseidon.decision_loop.strategy_config import (
    ExecutableStrategyConfig,
)
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import (
    CampaignEvent,
    CampaignReview,
    ExperimentCampaign,
)
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
from poseidon.models.position_lot import PositionLot
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_release_event import StrategyReleaseEvent
from poseidon.models.strategy_version import StrategyVersion


def require_active_pointer(session, decision, *, lock: bool, error_type):
    """Require a decision's immutable version to remain the active authority."""
    version = session.get(StrategyVersion, decision.strategy_version_id)
    if version is None:
        raise error_type("decision active pointer is unavailable")
    query = session.query(StrategyRecord).filter(StrategyRecord.id == version.strategy_id)
    if lock:
        query = query.with_for_update()
    strategy = query.populate_existing().one_or_none()
    if strategy is None or strategy.active_version_id != decision.strategy_version_id:
        raise error_type("decision active pointer changed")
    return strategy


class ReleaseService:
    """Read exact release evidence without owning the caller's transaction."""

    def __init__(self, session) -> None:
        self.session = session
        self.gate = ReleaseGateEvaluator(session)

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
    def _replay_event(event, request_sha256, principal):
        if event.request_sha256 != request_sha256:
            raise DecisionConflictError("idempotency key was used for a different request")
        principal.require_account_scope(event.account_scope)
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

    def _lock_reconciliation_authority(self, policy):
        if policy.reconciliation is None:
            return
        lock_account(self.session, policy.account_scope, policy.reconciliation.account_generation)

    def _current_or_paused_authority_version(self, strategy):
        if strategy.active_version_id is not None:
            return self.session.get(StrategyVersion, strategy.active_version_id)
        event = self.session.scalar(
            select(StrategyReleaseEvent)
            .where(
                StrategyReleaseEvent.strategy_id == strategy.id,
                (StrategyReleaseEvent.from_version_id.is_not(None) | StrategyReleaseEvent.to_version_id.is_not(None)),
            )
            .order_by(StrategyReleaseEvent.resulting_revision.desc())
            .limit(1)
        )
        if event is None:
            return None
        version_id = event.to_version_id if event.to_version_id is not None else event.from_version_id
        return self.session.get(StrategyVersion, version_id)

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
            return None, digest, expected_pointer, normalized_body, self._replay_event(replay, digest, principal)
        strategy = self._lock_strategy(strategy_id)
        replay = self._idempotency_event(strategy_id, idempotency_key)
        if replay is not None:
            return None, digest, expected_pointer, normalized_body, self._replay_event(replay, digest, principal)
        self._compare_expected(strategy, expected_revision, expected_pointer)
        return strategy, digest, expected_pointer, normalized_body, None

    @staticmethod
    def _require_allowed(result, operation):
        if result.allowed:
            return
        raise ReleaseGateRejected(operation, result)

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
        self._lock_reconciliation_authority(policy)
        gate = self.gate.evaluate(
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
            self.gate.authority_predicate(principal, policy),
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

    def _bootstrap_eligibility_predicate(self, strategy):
        prior_events = self.session.scalar(
            select(func.count())
            .select_from(StrategyReleaseEvent)
            .where(StrategyReleaseEvent.strategy_id == strategy.id)
        )
        evidence = {
            "strategy_id": str(strategy.id),
            "active_version_id": None if strategy.active_version_id is None else str(strategy.active_version_id),
            "release_revision": strategy.release_revision,
            "prior_release_events": prior_events,
        }
        if strategy.active_version_id is not None or strategy.release_revision != 0 or prior_events != 0:
            return _predicate(
                "bootstrap_eligibility",
                "blocked",
                "bootstrap is only for a strategy with no release history",
                **evidence,
            )
        return _predicate("bootstrap_eligibility", "passed", "strategy has no release history", **evidence)

    @staticmethod
    def _protective_exit_predicate(candidate):
        evidence = {"candidate_version_id": None if candidate is None else str(candidate.id)}
        if candidate is None:
            return _predicate("protective_exit", "unavailable", "candidate version is unavailable", **evidence)
        try:
            executable = ExecutableStrategyConfig.model_validate(candidate.config_json)
        except (PydanticValidationError, TypeError, ValueError):
            return _predicate("protective_exit", "blocked", "candidate executable config is invalid", **evidence)
        evidence["market"] = executable.market
        if executable.market != "tw_stock":
            return _predicate(
                "protective_exit", "passed", "typed protective exit is defined only for tw_stock", **evidence
            )
        if executable.protective_exit is None:
            return _predicate(
                "protective_exit", "blocked", "tw_stock bootstrap requires a typed protective_exit", **evidence
            )
        evidence["stop_loss_pct"] = executable.protective_exit.stop_loss_pct
        return _predicate("protective_exit", "passed", "tw_stock stop-loss is resolvable", **evidence)

    def _account_fresh_predicate(self, policy):
        evidence = {"account_scope": None if policy is None else policy.account_scope}
        if policy is None or policy.reconciliation is None:
            return _predicate("account_fresh", "blocked", "frozen reconciliation policy is missing", **evidence)
        scope = policy.account_scope
        generation = policy.reconciliation.account_generation
        evidence["account_generation"] = generation
        accounts = self.session.scalars(
            select(PaperBrokerAccount).where(
                PaperBrokerAccount.account_scope == scope,
                PaperBrokerAccount.account_generation == generation,
            )
        ).all()
        evidence["paper_accounts"] = len(accounts)
        if len(accounts) != 1:
            return _predicate(
                "account_fresh", "blocked", "exactly one paper account must match the frozen identity", **evidence
            )
        account = accounts[0]
        if account.opening_cash != policy.reconciliation.opening_cash or (
            account.currency != policy.reconciliation.currency
        ):
            return _predicate(
                "account_fresh",
                "blocked",
                "paper account opening cash or currency does not match the frozen policy",
                **evidence,
            )
        evidence.update(
            paper_broker_orders=self.session.scalar(
                select(func.count())
                .select_from(PaperBrokerOrder)
                .where(PaperBrokerOrder.account_scope == scope, PaperBrokerOrder.account_generation == generation)
            ),
            paper_broker_fills=self.session.scalar(
                select(func.count())
                .select_from(PaperBrokerFill)
                .where(PaperBrokerFill.account_scope == scope, PaperBrokerFill.account_generation == generation)
            ),
            open_position_lots=self.session.scalar(
                select(func.count())
                .select_from(PositionLot)
                .where(
                    PositionLot.account_scope == scope,
                    PositionLot.account_generation == generation,
                    PositionLot.open_quantity > 0,
                )
            ),
            decision_records=self.session.scalar(
                select(func.count()).select_from(DecisionRecord).where(DecisionRecord.account_scope == scope)
            ),
        )
        if any(
            evidence[key]
            for key in ("paper_broker_orders", "paper_broker_fills", "open_position_lots", "decision_records")
        ):
            return _predicate("account_fresh", "blocked", "paper account already has trading history", **evidence)
        return _predicate("account_fresh", "passed", "paper account is fresh for the frozen identity", **evidence)

    def bootstrap(
        self,
        strategy_id,
        candidate_version_id,
        *,
        expected_revision,
        expected_active_version_id,
        reason,
        principal,
        idempotency_key,
    ):
        """Activate the first StrategyVersion without campaign evidence (Phase 101 D-23.1).

        Bootstrap is a documented, one-time event: it records no campaign, review,
        holdout, or reconciliation ids, and the pointer still moves only through
        ``_append_transition`` under the 044 authority trigger.
        """
        reason = required_text(reason, "reason")
        body = {"candidate_version_id": str(candidate_version_id), "reason": reason}
        if isinstance(expected_revision, bool) or expected_revision != 0 or expected_active_version_id is not None:
            raise DecisionConflictError("bootstrap requires expected_revision 0 and no active version")
        strategy, digest, _pointer, _body, replay = self._begin_transition(
            operation="bootstrap",
            strategy_id=strategy_id,
            expected_revision=0,
            expected_active_version_id=None,
            principal=principal,
            idempotency_key=idempotency_key,
            body=body,
        )
        if replay is not None:
            return replay
        candidate = self.session.get(StrategyVersion, candidate_version_id)
        policy = self._policy_for_version(candidate, strategy, principal)
        self._lock_reconciliation_authority(policy)
        gate = ReleaseGateResult(
            (
                self.gate.candidate_predicate(strategy, candidate, candidate_version_id),
                self.gate.policy_predicate(strategy, candidate)[0],
                self._bootstrap_eligibility_predicate(strategy),
                self._protective_exit_predicate(candidate),
                self._account_fresh_predicate(policy),
                self.gate.authority_predicate(principal, policy),
            )
        )
        self._require_allowed(gate, "bootstrap")
        gate_json = {**gate.as_json(), "bootstrap": True}
        return self._append_transition(
            operation="bootstrap",
            strategy=strategy,
            candidate_version_id=candidate.id,
            to_version_id=candidate.id,
            source_release_event_id=None,
            campaign_id=None,
            review_id=None,
            holdout_use_id=None,
            reconciliation_id=None,
            principal=principal,
            account_scope=policy.account_scope,
            expected_revision=0,
            idempotency_key=idempotency_key,
            request_sha256=digest,
            reason=reason,
            gate_json=gate_json,
        )

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
        target_policy = self._policy_for_version(target, strategy, principal)
        source_version = self._current_or_paused_authority_version(strategy)
        source_policy = self._policy_for_version(source_version, strategy, principal)
        if self.gate.account_identity(target_policy) == self.gate.account_identity(source_policy):
            self._lock_reconciliation_authority(target_policy)
        reconciliation = self.session.get(AccountReconciliation, reconciliation_id)
        gate, source, policy = self.gate.rollback_gate(
            strategy,
            target,
            reconciliation,
            source_policy,
            principal,
            now,
        )
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

    @staticmethod
    def _version_identity(version):
        if version is None:
            return None
        return {
            "strategy_version_id": version.id,
            "version_no": version.version_no,
            "status": version.status,
            "content_sha256": version.content_sha256,
            "policy_sha256": content_sha256(version.policy_json),
            "account_scope": (
                version.policy_json.get("account_scope") if isinstance(version.policy_json, dict) else None
            ),
            "created_at": version.created_at,
        }

    def _stored_reconciliation(self, version):
        """Resolve the latest stored reconciliation for a version's frozen account identity."""

        if version is None:
            return None, None
        try:
            policy = DecisionPolicy.model_validate(version.policy_json)
        except (PydanticValidationError, TypeError, ValueError):
            return None, None
        identity = self.gate.account_identity(policy)
        if identity is None:
            return None, None
        latest = self.gate.latest_reconciliation(*identity)
        if latest is None:
            return identity, None
        return identity, {
            "reconciliation_id": latest.id,
            "account_scope": latest.account_scope,
            "account_generation": latest.account_generation,
            "status": latest.status,
            "as_of": latest.as_of,
            "policy_sha256": latest.policy_sha256,
        }

    def read_release_state(self, strategy) -> dict:
        """Project stored release identities for one strategy; never evaluate, lock, or write.

        Eligibility stays with ``evaluate_gate`` and ``_rollback_gate``. This reader only
        hands callers the exact stored identities those gates and mutations consume.
        """

        active = (
            None
            if strategy.active_version_id is None
            else self.session.get(StrategyVersion, strategy.active_version_id)
        )
        authority = self._current_or_paused_authority_version(strategy)
        latest_event = self.session.scalar(
            select(StrategyReleaseEvent)
            .where(StrategyReleaseEvent.strategy_id == strategy.id)
            .order_by(StrategyReleaseEvent.resulting_revision.desc())
            .limit(1)
        )
        _authority_account, authority_reconciliation = self._stored_reconciliation(authority)
        if strategy.active_version_id is not None:
            pointer_state = "active"
        elif authority is not None:
            pointer_state = "paused"
        else:
            pointer_state = "unreleased"

        strategy_versions = select(StrategyVersion.id).where(StrategyVersion.strategy_id == strategy.id)
        campaigns = self.session.scalars(
            select(ExperimentCampaign)
            .where(ExperimentCampaign.candidate_strategy_version_id.in_(strategy_versions))
            .order_by(ExperimentCampaign.created_at.desc(), ExperimentCampaign.id.desc())
        ).all()
        candidates = []
        for campaign in campaigns:
            candidate = self.session.get(StrategyVersion, campaign.candidate_strategy_version_id)
            trials = self.session.scalars(
                select(ExperimentRecord)
                .where(ExperimentRecord.campaign_id == campaign.id)
                .order_by(ExperimentRecord.id)
            ).all()
            reviews = self.session.scalars(
                select(CampaignReview)
                .where(CampaignReview.campaign_id == campaign.id)
                .order_by(CampaignReview.created_at.desc(), CampaignReview.id.desc())
            ).all()
            denials = self.session.scalars(
                select(CampaignEvent)
                .where(
                    CampaignEvent.campaign_id == campaign.id,
                    CampaignEvent.event_type == "holdout_reuse_denied",
                )
                .order_by(CampaignEvent.created_at, CampaignEvent.id)
            ).all()
            account, reconciliation = self._stored_reconciliation(candidate)
            declared = (
                campaign.contract_json.get("declared_trials") if isinstance(campaign.contract_json, dict) else None
            )
            version_roles = {
                campaign.incumbent_strategy_version_id: "incumbent",
                campaign.candidate_strategy_version_id: "candidate",
            }
            review_rows = []
            for review in reviews:
                resolution = self.gate.holdout_resolution(campaign, review)
                review_rows.append(
                    {
                        "review_id": review.id,
                        "status": review.status,
                        "input_sha256": review.input_sha256,
                        "result_sha256": review.result_sha256,
                        "created_at": review.created_at,
                        "holdout": {
                            "state": _HOLDOUT_STATE_LABELS[resolution.kind],
                            "holdout_use_id": resolution.holdout_use_id,
                            "holdout_identity_sha256": resolution.holdout_identity_sha256,
                            "owner_campaign_id": resolution.owner_campaign_id,
                        },
                    }
                )
            candidates.append(
                {
                    "candidate": self._version_identity(candidate),
                    "campaign_id": campaign.id,
                    "hypothesis": campaign.hypothesis,
                    "contract_sha256": campaign.contract_sha256,
                    "campaign_created_at": campaign.created_at,
                    "incumbent_version_id": campaign.incumbent_strategy_version_id,
                    "incumbent_content_sha256": campaign.incumbent_content_sha256,
                    "declared_trial_count": len(declared) if isinstance(declared, list) else None,
                    "terminal_trial_count": sum(1 for trial in trials if trial.terminal_state is not None),
                    "trials": [
                        {
                            "experiment_id": trial.id,
                            "original_trial_id": trial.original_trial_id,
                            "trial_role": trial.trial_role,
                            "strategy_version_id": trial.strategy_version_id,
                            "version_role": version_roles.get(trial.strategy_version_id),
                            "ablation_arm": trial.ablation_arm,
                            "terminal_state": trial.terminal_state,
                            "result_sha256": trial.result_sha256,
                        }
                        for trial in trials
                    ],
                    "reviews": review_rows,
                    "holdout_reuse_denials": [
                        {
                            "campaign_event_id": event.id,
                            "holdout_identity_sha256": event.payload_json.get("holdout_identity_sha256"),
                            "owner_campaign_id": event.payload_json.get("owner_campaign_id"),
                            "created_at": event.created_at,
                        }
                        for event in denials
                    ],
                    "account_scope": None if account is None else account[0],
                    "account_generation": None if account is None else account[1],
                    "reconciliation": reconciliation,
                }
            )
        return {
            "strategy_id": strategy.id,
            "release_revision": strategy.release_revision,
            "active_version_id": strategy.active_version_id,
            "pointer_state": pointer_state,
            "active_version": self._version_identity(active),
            "authority_version": self._version_identity(authority),
            "latest_event_id": None if latest_event is None else latest_event.id,
            "latest_event_type": None if latest_event is None else latest_event.event_type,
            "authority_reconciliation": authority_reconciliation,
            "candidates": candidates,
        }
