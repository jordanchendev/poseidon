"""Fail-closed strategy release evidence and authority transitions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select, text

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
from poseidon.models.order import OrderRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.paper_broker_fill import PaperBrokerFill
from poseidon.models.paper_broker_order import PaperBrokerOrder
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


class _ExecutableConditionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class IndicatorParameters(_ExecutableConditionModel):
    period: Annotated[int, Field(strict=True, gt=0)] | None = None
    fast_period: Annotated[int, Field(strict=True, gt=0)] | None = None
    slow_period: Annotated[int, Field(strict=True, gt=0)] | None = None
    signal_period: Annotated[int, Field(strict=True, gt=0)] | None = None


class CrossParameters(_ExecutableConditionModel):
    fast: Annotated[int, Field(strict=True, gt=0)]
    slow: Annotated[int, Field(strict=True, gt=0)]


class ComparisonParameters(_ExecutableConditionModel):
    period_a: Annotated[int, Field(strict=True, gt=0)]
    period_b: Annotated[int, Field(strict=True, gt=0)]


class VolumeParameters(_ExecutableConditionModel):
    period: Annotated[int, Field(strict=True, gt=0)] = 20
    multiplier: Annotated[int | float, Field(gt=0)] = 2.0


class BollingerParameters(_ExecutableConditionModel):
    period: Annotated[int, Field(strict=True, gt=0)] = 20
    lookback: Annotated[int, Field(strict=True, gt=1)] = 168


class IndicatorThresholdCondition(_ExecutableConditionModel):
    type: Literal["indicator_above", "indicator_below"]
    indicator: Annotated[str, Field(min_length=1)]
    params: IndicatorParameters = Field(default_factory=IndicatorParameters)
    threshold: int | float


class PriceCrossesCondition(_ExecutableConditionModel):
    type: Literal["price_crosses"]
    indicator: Annotated[str, Field(min_length=1)]
    params: IndicatorParameters = Field(default_factory=IndicatorParameters)
    direction: Literal["up", "down"] = "up"


class IndicatorCrossesCondition(_ExecutableConditionModel):
    type: Literal["indicator_crosses"]
    indicator: Annotated[str, Field(min_length=1)] = "sma"
    params: CrossParameters
    direction: Literal["up", "down"] = "up"


class PriceChangeCondition(_ExecutableConditionModel):
    type: Literal["price_change_pct"]
    threshold: int | float
    direction: Literal["up", "down", "absolute"] = "up"


class VolumeSpikeCondition(_ExecutableConditionModel):
    type: Literal["volume_spike"]
    params: VolumeParameters = Field(default_factory=VolumeParameters)


class BollingerCondition(_ExecutableConditionModel):
    type: Literal["bollinger_width_percentile"]
    params: BollingerParameters = Field(default_factory=BollingerParameters)
    threshold: Annotated[int | float, Field(ge=0, le=1)] = 0.85


class IndicatorComparisonCondition(_ExecutableConditionModel):
    type: Literal["indicator_comparison"]
    indicator_a: Annotated[str, Field(min_length=1)]
    indicator_b: Annotated[str, Field(min_length=1)]
    params: ComparisonParameters
    direction: Literal["above", "below"] = "above"


class FeatureThresholdCondition(_ExecutableConditionModel):
    type: Literal["feature_above", "feature_below"]
    column: Annotated[str, Field(min_length=1)]
    threshold: int | float


class ModelPredictionCondition(_ExecutableConditionModel):
    type: Literal["ml_prediction"]
    model_version_id: Annotated[str, Field(min_length=36, max_length=36)]
    threshold: Annotated[int | float, Field(ge=0)] = 0.5
    direction: Literal["long", "short"] = "long"

    @field_validator("model_version_id")
    @classmethod
    def require_canonical_model_version_id(cls, value):
        try:
            canonical = str(uuid.UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("model_version_id must be a canonical UUID") from exc
        if value != canonical:
            raise ValueError("model_version_id must be a canonical UUID")
        return value


class AllCondition(_ExecutableConditionModel):
    all: Annotated[list[ExecutableCondition], Field(min_length=1)]


class AnyCondition(_ExecutableConditionModel):
    any: Annotated[list[ExecutableCondition], Field(min_length=1)]


class NoneCondition(_ExecutableConditionModel):
    none: Annotated[list[ExecutableCondition], Field(min_length=1)]


class VoteConditionBody(_ExecutableConditionModel):
    conditions: Annotated[list[ExecutableCondition], Field(min_length=1)]
    min_votes: Annotated[int, Field(strict=True, ge=1)] = 4

    @model_validator(mode="after")
    def require_reachable_threshold(self):
        if self.min_votes > len(self.conditions):
            raise ValueError("vote min_votes cannot exceed conditions")
        return self


class VoteCondition(_ExecutableConditionModel):
    vote: VoteConditionBody


ExecutableCondition = (
    AllCondition
    | AnyCondition
    | NoneCondition
    | VoteCondition
    | IndicatorThresholdCondition
    | PriceCrossesCondition
    | IndicatorCrossesCondition
    | PriceChangeCondition
    | VolumeSpikeCondition
    | BollingerCondition
    | IndicatorComparisonCondition
    | FeatureThresholdCondition
    | ModelPredictionCondition
)

for _recursive_model in (AllCondition, AnyCondition, NoneCondition, VoteConditionBody, VoteCondition):
    _recursive_model.model_rebuild(_types_namespace={"ExecutableCondition": ExecutableCondition})


class LiveFeaturePlanError(ValueError):
    """Frozen executable condition cannot be evaluated by the live feature path."""


def _condition_feature_specs(condition: ExecutableCondition) -> list[tuple[str, dict]]:
    if isinstance(condition, (AllCondition, AnyCondition, NoneCondition)):
        children = (
            condition.all
            if isinstance(condition, AllCondition)
            else (condition.any if isinstance(condition, AnyCondition) else condition.none)
        )
        return [spec for child in children for spec in _condition_feature_specs(child)]
    if isinstance(condition, VoteCondition):
        return [spec for child in condition.vote.conditions for spec in _condition_feature_specs(child)]
    if isinstance(condition, IndicatorThresholdCondition):
        params = condition.params.model_dump(exclude_none=True)
        indicator = "sma" if condition.indicator == "ma" else condition.indicator
        if indicator == "price_vs_ma":
            indicator = "sma"
        if indicator in {"sma", "ema", "rsi", "cum_return"}:
            period = params.get("period")
            if period is None:
                raise LiveFeaturePlanError(f"{condition.indicator} requires an explicit live period")
            return [(indicator, {"period": period})]
        if indicator in {"macd", "macd_histogram", "macd_line", "macd_signal"}:
            return [("macd", params)]
        raise LiveFeaturePlanError(f"indicator {condition.indicator} is not live-capable")
    if isinstance(condition, PriceCrossesCondition):
        params = condition.params.model_dump(exclude_none=True)
        indicator = "sma" if condition.indicator == "ma" else condition.indicator
        period = params.get("period")
        if indicator not in {"sma", "ema"} or period is None:
            raise LiveFeaturePlanError(f"price_crosses {condition.indicator} is not live-capable")
        return [(indicator, {"period": period})]
    if isinstance(condition, IndicatorCrossesCondition):
        indicator = "sma" if condition.indicator == "ma" else condition.indicator
        if indicator not in {"sma", "ema", "rsi"}:
            raise LiveFeaturePlanError(f"indicator_crosses {condition.indicator} is not live-capable")
        return [
            (indicator, {"period": condition.params.fast}),
            (indicator, {"period": condition.params.slow}),
        ]
    if isinstance(condition, PriceChangeCondition):
        return [("returns", {})]
    if isinstance(condition, VolumeSpikeCondition):
        return []
    if isinstance(condition, BollingerCondition):
        return [("bollinger", {"period": condition.params.period, "num_std": 2.0})]
    if isinstance(condition, IndicatorComparisonCondition):
        specs = []
        for indicator, period in (
            (condition.indicator_a, condition.params.period_a),
            (condition.indicator_b, condition.params.period_b),
        ):
            indicator = "sma" if indicator == "ma" else indicator
            if indicator not in {"sma", "ema", "rsi"}:
                raise LiveFeaturePlanError(f"indicator comparison {indicator} is not live-capable")
            specs.append((indicator, {"period": period}))
        return specs
    if isinstance(condition, FeatureThresholdCondition):
        if condition.column in {"open", "high", "low", "close", "volume"}:
            return []
        raise LiveFeaturePlanError(f"feature {condition.column} is not live-capable")
    if isinstance(condition, ModelPredictionCondition):
        raise LiveFeaturePlanError("model prediction features are not live-capable")
    raise LiveFeaturePlanError("condition is not live-capable")


class RuleExecutableEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    condition: ExecutableCondition
    action: Literal["long", "short", "close"]
    quantity_pct: Annotated[int | float, Field(ge=0.0, le=1.0)] | None = None
    confidence: Annotated[int | float, Field(ge=0.0, le=1.0)] = 1.0


class RuleExecutablePayload(BaseModel):
    """Frozen payload accepted by the live rule strategy constructor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: Annotated[str, Field(min_length=1)]
    description: str = ""
    rules: Annotated[list[RuleExecutableEntry], Field(min_length=1)]

    @field_validator("name")
    @classmethod
    def require_trimmed_name(cls, value):
        if value != value.strip():
            raise ValueError("rule name must be non-empty trimmed text")
        return value


class VotingExecutablePayload(BaseModel):
    """Frozen payload accepted by the live voting strategy constructor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: Annotated[str, Field(min_length=1)] = "voting_strategy"
    sub_signals: Annotated[list[ExecutableCondition], Field(min_length=1)]
    min_votes: Annotated[int, Field(strict=True, ge=1)] = 4
    position_pct: Annotated[float, Field(strict=True, gt=0.0, le=1.0)] = 0.08
    bear_sub_signals: list[ExecutableCondition] = Field(default_factory=list)
    bear_min_votes: Annotated[int, Field(strict=True, ge=1)] = 4
    bear_position_pct: Annotated[float, Field(strict=True, gt=0.0, le=1.0)] = 0.06
    cooldown_bars: Annotated[int, Field(strict=True, ge=0)] = 12
    conviction_gap: Annotated[int, Field(strict=True, ge=0)] = 2
    strategy_mode: Literal["bidirectional", "long_only", "regime_gated"] = "bidirectional"

    @model_validator(mode="after")
    def require_reachable_vote_thresholds(self):
        if self.min_votes > len(self.sub_signals):
            raise ValueError("min_votes cannot exceed sub_signals")
        if self.bear_sub_signals and self.bear_min_votes > len(self.bear_sub_signals):
            raise ValueError("bear_min_votes cannot exceed bear_sub_signals")
        return self


class DeferredExecutablePayload(BaseModel):
    """Frozen payload for model/backtest-only strategies not run by this evaluator."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


ExecutablePayload = RuleExecutablePayload | VotingExecutablePayload | DeferredExecutablePayload


class ProtectiveExitConfig(BaseModel):
    """Durable stop-loss for loop-opened tw_stock lots (Phase 101 D-23.12).

    Read verbatim by ``workers.cpu_tasks._durable_protective_triggers`` from the
    top level of ``StrategyVersion.config_json``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    stop_loss_pct: Annotated[float, Field(strict=True, gt=0.0, lt=1.0, allow_inf_nan=False)]


class ExecutableStrategyConfig(BaseModel):
    """Immutable live execution identity plus strategy-specific payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    strategy_type: Literal["rule", "voting", "liquidity_sweep"]
    symbol: Annotated[str, Field(min_length=1)]
    market: Literal["tw_stock", "crypto_spot", "crypto_perp"]
    instrument: Literal["spot", "perpetual"]
    interval: Annotated[str, Field(min_length=1)]
    payload: ExecutablePayload
    protective_exit: ProtectiveExitConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_strategy_payload(cls, value):
        if not isinstance(value, dict):
            return value
        payload_type = {
            "rule": RuleExecutablePayload,
            "voting": VotingExecutablePayload,
            "liquidity_sweep": DeferredExecutablePayload,
        }.get(value.get("strategy_type"))
        if payload_type is None:
            return value
        normalized = dict(value)
        normalized["payload"] = payload_type.model_validate(value.get("payload"))
        return normalized

    @field_validator("symbol", "interval")
    @classmethod
    def require_trimmed_identity(cls, value):
        if value != value.strip():
            raise ValueError("executable identity must be non-empty trimmed text")
        return value

    @model_validator(mode="after")
    def require_market_instrument_identity(self):
        expected = "perpetual" if self.market == "crypto_perp" else "spot"
        if self.instrument != expected:
            raise ValueError("executable instrument does not match market")
        return self

    @model_validator(mode="after")
    def require_tw_stock_protective_exit(self):
        if self.protective_exit is not None and self.market != "tw_stock":
            raise ValueError("protective_exit is defined only for tw_stock")
        return self

    def routing_matches(self, strategy) -> bool:
        return self.symbol == strategy.symbol and self.market == strategy.market and self.interval == strategy.interval

    def runtime_config(self) -> dict[str, object]:
        return {
            **self.payload.model_dump(mode="json", exclude_unset=True),
            "symbol": self.symbol,
            "market": self.market,
            "interval": self.interval,
        }

    def live_feature_specs(self) -> list[tuple[str, dict]]:
        conditions: list[ExecutableCondition]
        if isinstance(self.payload, RuleExecutablePayload):
            conditions = [rule.condition for rule in self.payload.rules]
        elif isinstance(self.payload, VotingExecutablePayload):
            conditions = [*self.payload.sub_signals, *self.payload.bear_sub_signals]
        else:
            return []

        specs: list[tuple[str, dict]] = []
        seen: set[tuple[str, tuple[tuple[str, object], ...]]] = set()
        for condition in conditions:
            for name, params in _condition_feature_specs(condition):
                key = (name, tuple(sorted(params.items())))
                if key not in seen:
                    seen.add(key)
                    specs.append((name, params))
        if isinstance(self.payload, VotingExecutablePayload):
            for name, params in (("atr", {"period": 14}), ("returns", {})):
                key = (name, tuple(sorted(params.items())))
                if key not in seen:
                    seen.add(key)
                    specs.append((name, params))
        return specs


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


def _stored_time(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


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

    def _policy_predicate(self, strategy, candidate):
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

    def _holdout_resolution(self, campaign, review) -> HoldoutResolution:
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
        resolution = self._holdout_resolution(campaign, review)
        evidence = {
            "holdout_use_id": None if resolution.holdout_use_id is None else str(resolution.holdout_use_id),
            "holdout_identity_sha256": resolution.holdout_identity_sha256,
        }
        if resolution.kind == "foreign_owner":
            evidence["owner_campaign_id"] = str(resolution.owner_campaign_id)
        state, reason = _HOLDOUT_OUTCOMES[resolution.kind]
        return _predicate("holdout_ownership", state, reason, **evidence)

    def _latest_reconciliation(self, account_scope, account_generation):
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
        latest = self._latest_reconciliation(reconciliation.account_scope, reconciliation.account_generation)
        if latest is not None and latest.id != reconciliation.id:
            evidence["newer_reconciliation_ids"] = [str(latest.id)]
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
            return _predicate(
                "release_authority", "unavailable", "release principal or policy is unavailable", **evidence
            )
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
        policy_predicate, policy = self._policy_predicate(strategy, candidate)
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
        from poseidon.decision_loop.execution import DecisionExecutionService

        DecisionExecutionService(self.session)._advisory_lock(
            policy.account_scope,
            policy.reconciliation.account_generation,
        )

    @staticmethod
    def _account_identity(policy):
        if policy.reconciliation is None:
            return None
        return policy.account_scope, policy.reconciliation.account_generation

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
                self._candidate_predicate(strategy, candidate, candidate_version_id),
                self._policy_predicate(strategy, candidate)[0],
                self._bootstrap_eligibility_predicate(strategy),
                self._protective_exit_predicate(candidate),
                self._account_fresh_predicate(policy),
                self._authority_predicate(principal, policy),
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

    def _rollback_gate(self, strategy, target, reconciliation, source_policy, principal, now):
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

        policy_predicate, policy = self._policy_predicate(strategy, target)
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
                and self._account_identity(policy) == self._account_identity(source_policy)
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
        authority = self._authority_predicate(principal, policy)
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
        if self._account_identity(target_policy) == self._account_identity(source_policy):
            self._lock_reconciliation_authority(target_policy)
        reconciliation = self.session.get(AccountReconciliation, reconciliation_id)
        gate, source, policy = self._rollback_gate(
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
        identity = self._account_identity(policy)
        if identity is None:
            return None, None
        latest = self._latest_reconciliation(*identity)
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
                resolution = self._holdout_resolution(campaign, review)
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
