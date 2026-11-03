"""Executable strategy config: the frozen DSL payload, condition models, and live feature plan of a version."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    Read verbatim by ``workers.cpu_tasks.execution._durable_protective_triggers`` from the
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
