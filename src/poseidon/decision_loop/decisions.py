"""Frozen portfolio decisions and human-only terminal transitions."""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)
from sqlalchemy import text

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.evaluation import TERMINAL_STATUSES, verify_complete_run
from poseidon.decision_loop.manifest import (
    ValidationError,
    canonical_json,
    content_sha256,
    iso_time,
    required_text,
    timestamp,
    validate_manifest,
)
from poseidon.models.decision_event import DecisionEvent
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.evaluation_run import EvaluationRun
from poseidon.models.research_revision import ResearchRevision
from poseidon.models.strategy_version import StrategyVersion

ACTION_VALUES = frozenset({"watch", "hold", "enter", "add", "reduce", "exit", "reject"})
LIVE_STATUSES = frozenset({"pending_approval", "approved"})
SYSTEM_ACTOR = "system:decision-service"
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class DecisionNotFoundError(LookupError):
    pass


class DecisionConflictError(RuntimeError):
    pass


class HardLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_gross_exposure: PositiveFiniteFloat
    max_position_weight: PositiveFiniteFloat

    @field_validator("max_gross_exposure", "max_position_weight", mode="before")
    @classmethod
    def reject_boolean_limits(cls, value):
        if isinstance(value, bool):
            raise ValueError("hard limits must be numeric")
        return value


class ProtectiveExit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_without_approval: StrictBool


class ReleaseGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_mature_samples: PositiveStrictInt
    requires_human_release: StrictBool


class FreshnessRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_age_seconds: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: Annotated[str, Field(min_length=1)]
    mode: Literal["human", "assisted", "automatic"]
    market: Literal["tw_stock"]
    account_scope: Annotated[str, Field(min_length=1)]
    universe_id: Annotated[str, Field(min_length=1)]
    decision_ttl_seconds: PositiveStrictInt
    required_data: Annotated[dict[str, FreshnessRule], Field(min_length=1)]
    hard_limits: HardLimits
    approval_roles: Annotated[list[str], Field(min_length=1)]
    protective_exit: ProtectiveExit
    release_gate: ReleaseGate

    @field_validator("policy_id", "account_scope", "universe_id")
    @classmethod
    def require_trimmed_text(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("must be non-empty trimmed text")
        return value

    @field_validator("required_data")
    @classmethod
    def require_data_kinds(cls, rules):
        if any(not kind.strip() or kind != kind.strip() for kind in rules):
            raise ValueError("required data kinds must be non-empty trimmed text")
        return rules

    @field_validator("approval_roles")
    @classmethod
    def require_portfolio_manager(cls, roles):
        if len(set(roles)) != len(roles) or any(not role.strip() or role != role.strip() for role in roles):
            raise ValueError("approval roles must be unique non-empty text")
        if "portfolio_manager" not in roles:
            raise ValueError("portfolio_manager approval is required")
        return roles


class ApprovalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: PositiveStrictInt
    final_action: Literal["watch", "hold", "enter", "add", "reduce", "exit", "reject"] | None = None
    override_reason: str | None = None


class RejectionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: PositiveStrictInt
    reason: Annotated[str, Field(min_length=1)]

    @field_validator("reason")
    @classmethod
    def require_reason(cls, value):
        if not value.strip():
            raise ValueError("rejection reason must be non-empty")
        return value


def _json_object(value, field):
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be a JSON object")
    return json.loads(canonical_json(value))


def _validate_selection(snapshots, selected_values, final_action):
    if not isinstance(selected_values, list) or not selected_values:
        raise ValidationError("selected_evaluation_ids must be a non-empty list")
    try:
        selected_ids = [uuid.UUID(value) for value in selected_values]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError("selected_evaluation_ids must contain UUID strings") from error
    if len(set(selected_ids)) != len(selected_ids):
        raise ValidationError("selected_evaluation_ids must be unique")
    if not isinstance(final_action, str) or final_action not in ACTION_VALUES:
        raise ValidationError("final_action is invalid")
    snapshot_by_id = {row.id: row for row in snapshots}
    try:
        selected = [snapshot_by_id[selected_id] for selected_id in selected_ids]
    except KeyError as error:
        raise ValidationError("selected evaluation does not belong to the run") from error
    if any(row.status not in TERMINAL_STATUSES for row in selected):
        raise ValidationError("selected evaluation is not terminal")
    if final_action in {"enter", "add"} and any(row.status != "evaluated" for row in selected):
        raise ValidationError("exposure increase requires evaluated selections")
    return selected


def _stored_time(value, field):
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return timestamp(value, field)


def _decision_response(decision):
    return {
        "id": str(decision.id),
        "status": decision.status,
        "revision": decision.revision,
        "creation_sha256": decision.creation_sha256,
        "policy_sha256": decision.policy_sha256,
        "valid_until": iso_time(_stored_time(decision.valid_until, "valid_until"), "valid_until"),
        "final_json": json.loads(canonical_json(decision.final_json)),
    }


class DecisionService:
    def __init__(self, session):
        self.session = session

    def _lock_scope(self, account_scope, strategy_version_id):
        if self.session.get_bind().dialect.name == "postgresql":
            lock_key = int(
                content_sha256({"account_scope": account_scope, "strategy_version_id": str(strategy_version_id)})[:16],
                16,
            )
            if lock_key >= 2**63:
                lock_key -= 2**64
            self.session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})

    def _append_event(
        self,
        decision,
        event_type,
        actor_id,
        expected_revision,
        *,
        idempotency_key=None,
        request_sha256=None,
        payload_json=None,
    ):
        event = DecisionEvent(
            decision_id=decision.id,
            event_type=event_type,
            actor_id=actor_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            payload_json=payload_json or {},
        )
        self.session.add(event)
        return event

    def _policy_is_current(self, decision):
        version = self.session.get(StrategyVersion, decision.strategy_version_id)
        if version is None:
            return False
        try:
            version.verify_content()
            DecisionPolicy.model_validate(version.policy_json)
        except (TypeError, ValueError, PydanticValidationError):
            return False
        return content_sha256(version.policy_json) == decision.policy_sha256

    def _authorized_decision(self, decision_id, principal, *, role=None):
        if role is not None:
            principal.require_role(role)
        decision = self.session.get(DecisionRecord, decision_id)
        if decision is None:
            raise DecisionNotFoundError("decision does not exist")
        principal.require_account_scope(decision.account_scope)
        return decision

    def _idempotency_event(self, decision_id, idempotency_key):
        return (
            self.session.query(DecisionEvent)
            .filter_by(decision_id=decision_id, idempotency_key=idempotency_key)
            .one_or_none()
        )

    @staticmethod
    def _replay_event(event, request_sha256):
        if event.request_sha256 != request_sha256:
            raise DecisionConflictError("idempotency key was used for a different request")
        response = event.payload_json.get("response")
        if not isinstance(response, dict):
            raise DecisionConflictError("stored idempotency response is invalid")
        return json.loads(canonical_json(response))

    def _transition(
        self,
        operation,
        decision_id,
        raw_body,
        body_type,
        *,
        principal,
        idempotency_key,
        now=None,
    ):
        self._authorized_decision(decision_id, principal, role="portfolio_manager")
        required_text(idempotency_key, "Idempotency-Key")
        if len(idempotency_key) > 256:
            raise ValidationError("Idempotency-Key is too long")
        body = body_type.model_validate(raw_body)
        body_json = body.model_dump(mode="json")
        request_sha256 = content_sha256(
            {
                "operation": operation,
                "decision_id": str(decision_id),
                "expected_revision": body.expected_revision,
                "actor_id": principal.actor_id,
                "body": body_json,
            }
        )
        event = self._idempotency_event(decision_id, idempotency_key)
        if event is not None:
            return self._replay_event(event, request_sha256)

        decision = (
            self.session.query(DecisionRecord)
            .filter(DecisionRecord.id == decision_id)
            .with_for_update()
            .populate_existing()
            .one_or_none()
        )
        if decision is None:
            raise DecisionNotFoundError("decision does not exist")
        event = self._idempotency_event(decision_id, idempotency_key)
        if event is not None:
            return self._replay_event(event, request_sha256)

        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        if decision.status != "pending_approval":
            raise DecisionConflictError("decision is not pending approval")
        if decision.revision != body.expected_revision:
            raise DecisionConflictError("decision revision changed")
        if current_time >= _stored_time(decision.valid_until, "valid_until"):
            raise DecisionConflictError("decision has expired")
        if not self._policy_is_current(decision):
            raise DecisionConflictError("decision policy changed")

        if operation == "approve":
            hard_failures = decision.risk_snapshot_json.get("hard_failures")
            allowed_actions = decision.risk_snapshot_json.get("allowed_actions")
            if not isinstance(hard_failures, list) or hard_failures:
                raise DecisionConflictError("hard risk failures prevent approval")
            final_json = json.loads(canonical_json(decision.original_json))
            final_action = body.final_action or final_json.get("final_action")
            if not isinstance(allowed_actions, list) or final_action not in allowed_actions:
                raise DecisionConflictError("override action is not allowed by frozen risk")
            try:
                _, _, _, snapshots = verify_complete_run(self.session, decision.evaluation_run_id)
                _validate_selection(snapshots, final_json.get("selected_evaluation_ids"), final_action)
            except ValidationError as error:
                raise DecisionConflictError(str(error)) from error
            if final_action != final_json.get("final_action"):
                if not isinstance(body.override_reason, str) or not body.override_reason.strip():
                    raise DecisionConflictError("action override requires a reason")
                final_json["final_action"] = final_action
                final_json["override_reason"] = body.override_reason
            decision.final_json = final_json

        event_type = "approved" if operation == "approve" else "rejected"
        expected_revision = decision.revision
        decision.status = event_type
        decision.revision += 1
        response = _decision_response(decision)
        self._append_event(
            decision,
            event_type,
            principal.actor_id,
            expected_revision,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            payload_json={"request": body_json, "response": response},
        )
        self.session.flush()
        return response

    def approve(self, decision_id, body, *, principal, idempotency_key, now=None):
        return self._transition(
            "approve",
            decision_id,
            body,
            ApprovalBody,
            principal=principal,
            idempotency_key=idempotency_key,
            now=now,
        )

    def reject(self, decision_id, body, *, principal, idempotency_key, now=None):
        return self._transition(
            "reject",
            decision_id,
            body,
            RejectionBody,
            principal=principal,
            idempotency_key=idempotency_key,
            now=now,
        )

    def create_decision(
        self,
        evaluation_run_id,
        *,
        principal: AuthPrincipal,
        account_scope,
        original_json,
        portfolio_snapshot_json,
        risk_snapshot_json,
        now=None,
    ):
        principal.require_role("decision-worker")
        principal.require_account_scope(account_scope)
        if self.session.get(EvaluationRun, evaluation_run_id) is None:
            raise DecisionNotFoundError("evaluation run does not exist")
        run, version, manifest, snapshots = verify_complete_run(self.session, evaluation_run_id)
        policy = DecisionPolicy.model_validate(version.policy_json)
        manifest_payload = manifest.payload_json
        for field, actual, expected in (
            ("market", policy.market, manifest.market),
            ("account_scope", policy.account_scope, account_scope),
            ("manifest.account_scope", manifest_payload.get("account_scope"), account_scope),
            ("universe_id", policy.universe_id, manifest_payload.get("universe_id")),
        ):
            if actual != expected:
                raise ValidationError(f"policy {field} mismatch")
        validate_manifest(
            dict(
                manifest_payload,
                as_of=iso_time(_stored_time(run.decision_as_of, "decision_as_of"), "decision_as_of"),
                required_data={kind: rule.model_dump() for kind, rule in policy.required_data.items()},
            )
        )

        original = _json_object(original_json, "original_json")
        portfolio = _json_object(portfolio_snapshot_json, "portfolio_snapshot_json")
        risk = _json_object(risk_snapshot_json, "risk_snapshot_json")
        final_action = original.get("final_action")
        selected = _validate_selection(snapshots, original.get("selected_evaluation_ids"), final_action)
        original["selected_evaluation_ids"] = [str(row.id) for row in selected]
        hard_failures = risk.get("hard_failures")
        allowed_actions = risk.get("allowed_actions")
        if not isinstance(hard_failures, list):
            raise ValidationError("hard_failures must be a list")
        if (
            not isinstance(allowed_actions, list)
            or not allowed_actions
            or any(not isinstance(action, str) for action in allowed_actions)
            or len(set(allowed_actions)) != len(allowed_actions)
            or any(action not in ACTION_VALUES for action in allowed_actions)
        ):
            raise ValidationError("allowed_actions must be a unique non-empty action list")
        if final_action not in allowed_actions:
            raise ValidationError("final_action is not allowed by frozen risk")

        decision_as_of = _stored_time(run.decision_as_of, "decision_as_of")
        expiry_candidates = [decision_as_of + timedelta(seconds=policy.decision_ttl_seconds)]
        expiry_candidates.extend(
            _stored_time(row.valid_until, "snapshot.valid_until") for row in selected if row.valid_until is not None
        )
        valid_until = min(expiry_candidates)
        if valid_until <= decision_as_of:
            raise ValidationError("decision validity must extend beyond its cutoff")
        if now is not None:
            timestamp(now, "now")

        policy_sha256 = content_sha256(version.policy_json)
        creation_sha256 = content_sha256(
            {
                "evaluation_run_id": str(run.id),
                "evaluation_input_sha256": run.input_sha256,
                "strategy_version_id": str(version.id),
                "strategy_content_sha256": version.content_sha256,
                "account_scope": account_scope,
                "decision_as_of": iso_time(decision_as_of, "decision_as_of"),
                "valid_until": iso_time(valid_until, "valid_until"),
                "policy_sha256": policy_sha256,
                "original_json": original,
                "final_json": original,
                "portfolio_snapshot_json": portfolio,
                "risk_snapshot_json": risk,
            }
        )
        self._lock_scope(account_scope, version.id)
        replay = self.session.query(DecisionRecord).filter_by(creation_sha256=creation_sha256).one_or_none()
        if replay is not None:
            return replay

        decision = DecisionRecord(
            evaluation_run_id=run.id,
            strategy_version_id=version.id,
            account_scope=account_scope,
            decision_as_of=decision_as_of,
            valid_until=valid_until,
            status="pending_approval",
            revision=1,
            creation_sha256=creation_sha256,
            policy_sha256=policy_sha256,
            original_json=original,
            final_json=json.loads(canonical_json(original)),
            portfolio_snapshot_json=portfolio,
            risk_snapshot_json=risk,
        )
        self.session.add(decision)
        self.session.flush()
        self._append_event(
            decision,
            "created",
            principal.actor_id,
            0,
            payload_json={"response": _decision_response(decision)},
        )

        older = (
            self.session.query(DecisionRecord)
            .filter(
                DecisionRecord.id != decision.id,
                DecisionRecord.account_scope == account_scope,
                DecisionRecord.strategy_version_id == version.id,
                DecisionRecord.status.in_(LIVE_STATUSES),
            )
            .with_for_update()
            .populate_existing()
            .order_by(DecisionRecord.valid_until, DecisionRecord.id)
            .all()
        )
        for candidate in older:
            expected_revision = candidate.revision
            candidate.status = "superseded"
            candidate.revision += 1
            self._append_event(
                candidate,
                "superseded",
                SYSTEM_ACTOR,
                expected_revision,
                payload_json={"response": _decision_response(candidate)},
            )
        self.session.flush()
        return decision

    def expire_due(self, as_of):
        cutoff = timestamp(as_of, "as_of")
        due = (
            self.session.query(DecisionRecord)
            .filter(DecisionRecord.status.in_(LIVE_STATUSES), DecisionRecord.valid_until <= cutoff)
            .with_for_update()
            .populate_existing()
            .order_by(DecisionRecord.valid_until, DecisionRecord.id)
            .all()
        )
        for decision in due:
            expected_revision = decision.revision
            decision.status = "expired"
            decision.revision += 1
            self._append_event(
                decision,
                "expired",
                SYSTEM_ACTOR,
                expected_revision,
                payload_json={"response": _decision_response(decision)},
            )
        self.session.flush()
        return due

    def get(self, decision_id, *, principal):
        return self._authorized_decision(decision_id, principal)

    def list_pending(self, *, principal, now=None):
        if not principal.account_scopes:
            raise HTTPException(status_code=403, detail="Account scope not authorized")
        current_time = timestamp(now if now is not None else datetime.now(UTC), "now")
        return (
            self.session.query(DecisionRecord)
            .filter(
                DecisionRecord.status == "pending_approval",
                DecisionRecord.valid_until > current_time,
                DecisionRecord.account_scope.in_(principal.account_scopes),
            )
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
            .all()
        )

    def trace(self, decision_id, *, principal):
        decision = self.get(decision_id, principal=principal)
        run, version, manifest, snapshots = verify_complete_run(self.session, decision.evaluation_run_id)
        selected_ids = {uuid.UUID(value) for value in decision.original_json["selected_evaluation_ids"]}
        evaluations = []
        for snapshot in snapshots:
            revision_ids = list(snapshot.research_revision_ids)
            if revision_ids:
                revisions = [self.session.get(ResearchRevision, uuid.UUID(value)) for value in revision_ids]
                research = {
                    "revision_ids": revision_ids,
                    "revisions": [
                        {"id": str(revision.id), "content_sha256": revision.content_sha256} for revision in revisions
                    ],
                }
            else:
                research = {"research_status": snapshot.recommendation_json["research_status"]}
            evaluations.append(
                {
                    "id": str(snapshot.id),
                    "content_sha256": snapshot.content_sha256,
                    "symbol": snapshot.symbol,
                    "market": snapshot.market,
                    "instrument": snapshot.instrument,
                    "status": snapshot.status,
                    "selected": snapshot.id in selected_ids,
                    "recommendation_json": json.loads(canonical_json(snapshot.recommendation_json)),
                    "technical_json": json.loads(canonical_json(snapshot.technical_json)),
                    "reason_codes": list(snapshot.reason_codes),
                    "valid_until": (
                        iso_time(_stored_time(snapshot.valid_until, "snapshot.valid_until"), "snapshot.valid_until")
                        if snapshot.valid_until is not None
                        else None
                    ),
                    "research": research,
                }
            )
        events = (
            self.session.query(DecisionEvent)
            .filter_by(decision_id=decision.id)
            .order_by(DecisionEvent.created_at, DecisionEvent.id)
            .all()
        )
        return {
            "decision": {
                **_decision_response(decision),
                "evaluation_run_id": str(decision.evaluation_run_id),
                "strategy_version_id": str(decision.strategy_version_id),
                "account_scope": decision.account_scope,
                "decision_as_of": iso_time(_stored_time(decision.decision_as_of, "decision_as_of"), "decision_as_of"),
                "original_json": json.loads(canonical_json(decision.original_json)),
                "portfolio_snapshot_json": json.loads(canonical_json(decision.portfolio_snapshot_json)),
                "risk_snapshot_json": json.loads(canonical_json(decision.risk_snapshot_json)),
            },
            "strategy_version": {
                "id": str(version.id),
                "content_sha256": version.content_sha256,
                "policy_sha256": content_sha256(version.policy_json),
            },
            "evaluation_run": {
                "id": str(run.id),
                "input_sha256": run.input_sha256,
                "coverage_json": json.loads(canonical_json(run.coverage_json)),
                "decision_as_of": iso_time(_stored_time(run.decision_as_of, "decision_as_of"), "decision_as_of"),
            },
            "manifest": {"id": str(manifest.id), "content_sha256": manifest.content_sha256},
            "evaluations": evaluations,
            "events": [
                {
                    "id": str(event.id),
                    "event_type": event.event_type,
                    "actor_id": event.actor_id,
                    "expected_revision": event.expected_revision,
                    "idempotency_key": event.idempotency_key,
                    "request_sha256": event.request_sha256,
                    "payload_json": json.loads(canonical_json(event.payload_json)),
                    "created_at": (
                        iso_time(_stored_time(event.created_at, "event.created_at"), "event.created_at")
                        if event.created_at is not None
                        else None
                    ),
                }
                for event in events
            ],
        }

    def is_execution_eligible(self, decision_id, expected_revision, as_of):
        decision = self.session.get(DecisionRecord, decision_id)
        if decision is None:
            return False
        return (
            decision.status == "approved"
            and decision.revision == expected_revision
            and _stored_time(decision.valid_until, "valid_until") > timestamp(as_of, "as_of")
            and self._policy_is_current(decision)
        )
