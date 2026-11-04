"""Leaf condition evaluator functions and registry.

Each evaluator: (condition: dict, features: pd.DataFrame, row_idx: int) -> bool. Each is registered together with
the strict schema of the condition it evaluates, so the executable-config schema of a released strategy version
(``decision_loop.strategy_config``) is derived from this registry instead of listed by hand.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated, Literal, get_args

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

CONDITION_REGISTRY: dict[str, Callable] = {}
CONDITION_SCHEMAS: dict[str, type[BaseModel]] = {}

# Alias mapping for indicator names
_INDICATOR_ALIASES = {"ma": "sma"}

# Direct column name mappings (no period suffix needed)
_DIRECT_COLUMN_MAP = {
    "macd_histogram": "macd_histogram",
    "macd_signal": "macd_signal",
    "macd_line": "macd_line",
}


def register_condition(name: str, *, schema: type[BaseModel]):
    """Register a condition evaluator with the schema whose ``type`` literal names it."""
    if name not in get_args(schema.model_fields["type"].annotation):
        raise ValueError(f"condition schema {schema.__name__} does not accept type {name!r}")

    def decorator(fn):
        CONDITION_REGISTRY[name] = fn
        CONDITION_SCHEMAS[name] = schema
        return fn

    return decorator


def resolve_column_name(indicator: str, params: dict) -> str:
    """Map DSL indicator reference to FeatureEngine column name.

    Examples:
        ("rsi", {"period": 14}) -> "rsi_14"
        ("sma", {"period": 60}) -> "sma_60"
        ("ma", {"period": 60}) -> "sma_60"
        ("macd", {}) -> "macd_line"
    """
    # Apply aliases
    indicator = _INDICATOR_ALIASES.get(indicator, indicator)

    # Direct column mappings (e.g. macd_histogram, macd_signal, macd_line)
    if indicator in _DIRECT_COLUMN_MAP:
        return _DIRECT_COLUMN_MAP[indicator]

    # Cumulative return uses '{period}d' suffix convention
    if indicator == "cum_return":
        period = params.get("period")
        return f"cum_return_{period}d" if period else "cum_return"

    if indicator == "macd":
        return "macd_line"

    period = params.get("period")
    if period is not None:
        return f"{indicator}_{period}"

    return indicator


class ConditionModel(BaseModel):
    """Frozen strict base of the condition schemas: what a released strategy version may contain."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class IndicatorParameters(ConditionModel):
    period: Annotated[int, Field(strict=True, gt=0)] | None = None
    fast_period: Annotated[int, Field(strict=True, gt=0)] | None = None
    slow_period: Annotated[int, Field(strict=True, gt=0)] | None = None
    signal_period: Annotated[int, Field(strict=True, gt=0)] | None = None


class CrossParameters(ConditionModel):
    fast: Annotated[int, Field(strict=True, gt=0)]
    slow: Annotated[int, Field(strict=True, gt=0)]


class ComparisonParameters(ConditionModel):
    period_a: Annotated[int, Field(strict=True, gt=0)]
    period_b: Annotated[int, Field(strict=True, gt=0)]


class VolumeParameters(ConditionModel):
    period: Annotated[int, Field(strict=True, gt=0)] = 20
    multiplier: Annotated[int | float, Field(gt=0)] = 2.0


class BollingerParameters(ConditionModel):
    period: Annotated[int, Field(strict=True, gt=0)] = 20
    lookback: Annotated[int, Field(strict=True, gt=1)] = 168


class IndicatorThresholdCondition(ConditionModel):
    type: Literal["indicator_above", "indicator_below"]
    indicator: Annotated[str, Field(min_length=1)]
    params: IndicatorParameters = Field(default_factory=IndicatorParameters)
    threshold: int | float


class PriceCrossesCondition(ConditionModel):
    type: Literal["price_crosses"]
    indicator: Annotated[str, Field(min_length=1)]
    params: IndicatorParameters = Field(default_factory=IndicatorParameters)
    direction: Literal["up", "down"] = "up"


class IndicatorCrossesCondition(ConditionModel):
    type: Literal["indicator_crosses"]
    indicator: Annotated[str, Field(min_length=1)] = "sma"
    params: CrossParameters
    direction: Literal["up", "down"] = "up"


class PriceChangeCondition(ConditionModel):
    type: Literal["price_change_pct"]
    threshold: int | float
    direction: Literal["up", "down", "absolute"] = "up"


class VolumeSpikeCondition(ConditionModel):
    type: Literal["volume_spike"]
    params: VolumeParameters = Field(default_factory=VolumeParameters)


class BollingerCondition(ConditionModel):
    type: Literal["bollinger_width_percentile"]
    params: BollingerParameters = Field(default_factory=BollingerParameters)
    threshold: Annotated[int | float, Field(ge=0, le=1)] = 0.85


class IndicatorComparisonCondition(ConditionModel):
    type: Literal["indicator_comparison"]
    indicator_a: Annotated[str, Field(min_length=1)]
    indicator_b: Annotated[str, Field(min_length=1)]
    params: ComparisonParameters
    direction: Literal["above", "below"] = "above"


class FeatureThresholdCondition(ConditionModel):
    type: Literal["feature_above", "feature_below"]
    column: Annotated[str, Field(min_length=1)]
    threshold: int | float


class ModelPredictionCondition(ConditionModel):
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


@register_condition("indicator_above", schema=IndicatorThresholdCondition)
def eval_indicator_above(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if indicator value > threshold."""
    indicator = condition.get("indicator", "")
    params = condition.get("params", {})
    threshold = condition["threshold"]

    # Handle price_vs_ma: compare close to MA
    if indicator == "price_vs_ma":
        ma_col = resolve_column_name("sma", params)
        close = features.iloc[row_idx]["close"]
        ma_val = features.iloc[row_idx][ma_col]
        return float(close - ma_val) > threshold

    col = resolve_column_name(indicator, params)
    return float(features.iloc[row_idx][col]) > threshold


@register_condition("indicator_below", schema=IndicatorThresholdCondition)
def eval_indicator_below(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if indicator value < threshold."""
    indicator = condition.get("indicator", "")
    params = condition.get("params", {})
    threshold = condition["threshold"]

    if indicator == "price_vs_ma":
        ma_col = resolve_column_name("sma", params)
        close = features.iloc[row_idx]["close"]
        ma_val = features.iloc[row_idx][ma_col]
        return float(close - ma_val) < threshold

    col = resolve_column_name(indicator, params)
    return float(features.iloc[row_idx][col]) < threshold


@register_condition("price_crosses", schema=PriceCrossesCondition)
def eval_price_crosses(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if price crosses an indicator value."""
    if row_idx < 1:
        return False

    indicator = condition.get("indicator", "")
    params = condition.get("params", {})
    direction = condition.get("direction", "up")
    col = resolve_column_name(indicator, params)

    curr_close = float(features.iloc[row_idx]["close"])
    prev_close = float(features.iloc[row_idx - 1]["close"])
    curr_ind = float(features.iloc[row_idx][col])
    prev_ind = float(features.iloc[row_idx - 1][col])

    if direction == "up":
        return prev_close <= prev_ind and curr_close > curr_ind
    elif direction == "down":
        return prev_close >= prev_ind and curr_close < curr_ind
    return False


@register_condition("indicator_crosses", schema=IndicatorCrossesCondition)
def eval_indicator_crosses(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if fast indicator crosses slow indicator."""
    if row_idx < 1:
        return False

    params = condition.get("params", {})
    if params.get("fast") is None or params.get("slow") is None:
        raise ValueError("indicator_crosses requires params.fast and params.slow")

    indicator = condition.get("indicator", "sma")
    direction = condition.get("direction", "up")
    fast_col = resolve_column_name(indicator, {"period": params.get("fast")})
    slow_col = resolve_column_name(indicator, {"period": params.get("slow")})

    curr_fast = float(features.iloc[row_idx][fast_col])
    prev_fast = float(features.iloc[row_idx - 1][fast_col])
    curr_slow = float(features.iloc[row_idx][slow_col])
    prev_slow = float(features.iloc[row_idx - 1][slow_col])

    if direction == "up":
        return prev_fast <= prev_slow and curr_fast > curr_slow
    elif direction == "down":
        return prev_fast >= prev_slow and curr_fast < curr_slow
    return False


@register_condition("price_change_pct", schema=PriceChangeCondition)
def eval_price_change_pct(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if return exceeds threshold."""
    threshold = condition["threshold"]
    direction = condition.get("direction", "up")
    ret = float(features.iloc[row_idx]["return_1d"])

    if direction == "up":
        return ret > threshold
    elif direction == "down":
        return ret < -threshold
    return abs(ret) > threshold


@register_condition("volume_spike", schema=VolumeSpikeCondition)
def eval_volume_spike(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if volume exceeds multiplier * average."""
    params = condition.get("params", {})
    period = params.get("period", 20)
    multiplier = params.get("multiplier", 2.0)

    curr_vol = float(features.iloc[row_idx]["volume"])

    # Calculate average over available rows (up to period)
    start = max(0, row_idx - period)
    avg_vol = float(features.iloc[start:row_idx]["volume"].mean()) if row_idx > 0 else curr_vol

    return curr_vol > multiplier * avg_vol


@register_condition("bollinger_width_percentile", schema=BollingerCondition)
def eval_bollinger_width_percentile(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if Bollinger Band width is in a squeeze (below percentile threshold).

    Computes the percentile rank of current BB width within a lookback window.
    No look-ahead: only uses rows up to and including row_idx.
    """
    params = condition.get("params", {})
    period = params.get("period", 20)
    lookback = params.get("lookback", 168)  # 1 week of hourly bars
    threshold = condition.get("threshold", 0.85)  # Nunchi proven: below 85th percentile

    upper_col = f"bb_upper_{period}"
    lower_col = f"bb_lower_{period}"

    # Compute width series up to current row (no look-ahead)
    start = max(0, row_idx - lookback + 1)
    widths = features[upper_col].iloc[start : row_idx + 1] - features[lower_col].iloc[start : row_idx + 1]

    if len(widths) < 2:
        return False

    current_width = widths.iloc[-1]
    percentile = (widths < current_width).sum() / len(widths)
    return percentile < threshold


@register_condition("indicator_comparison", schema=IndicatorComparisonCondition)
def eval_indicator_comparison(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Compare two indicator values: indicator_a vs indicator_b.

    Parameters from condition dict:
        indicator_a, indicator_b: indicator names
        params.period_a, params.period_b: periods for each indicator
        direction: 'above' (a > b) or 'below' (a < b)
    """
    params = condition.get("params", {})
    indicator_a = condition.get("indicator_a", "")
    indicator_b = condition.get("indicator_b", "")
    col_a = resolve_column_name(indicator_a, {"period": params.get("period_a")})
    col_b = resolve_column_name(indicator_b, {"period": params.get("period_b")})
    direction = condition.get("direction", "above")

    val_a = float(features.iloc[row_idx][col_a])
    val_b = float(features.iloc[row_idx][col_b])

    if direction == "above":
        return val_a > val_b
    return val_a < val_b


@register_condition("feature_above", schema=FeatureThresholdCondition)
def eval_feature_above(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if a named feature column value > threshold.

    Unlike indicator_above, uses direct column name (no resolve_column_name).
    Designed for R2 features (institutional, fundamental, funding, macro).
    Returns False on NaN or missing column.
    """
    column = condition["column"]
    threshold = condition["threshold"]
    if column not in features.columns:
        return False
    val = features.iloc[row_idx][column]
    if pd.isna(val):
        return False
    return float(val) > threshold


@register_condition("feature_below", schema=FeatureThresholdCondition)
def eval_feature_below(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if a named feature column value < threshold.

    Returns False on NaN or missing column.
    """
    column = condition["column"]
    threshold = condition["threshold"]
    if column not in features.columns:
        return False
    val = features.iloc[row_idx][column]
    if pd.isna(val):
        return False
    return float(val) < threshold


@register_condition("ml_prediction", schema=ModelPredictionCondition)
def eval_ml_prediction(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Evaluate ML model prediction as a vote.

    Reads pre-computed qlib_prediction column from features DataFrame.
    Direction controls evaluation:
      - direction="long": prediction > threshold (bullish signal)
      - direction="short": prediction < -threshold (bearish signal)

    Returns False on NaN or missing column (consistent with feature_above).
    """
    threshold = condition.get("threshold", 0.5)
    direction = condition.get("direction", "long")

    column = "qlib_prediction"
    if column not in features.columns:
        return False
    val = features.iloc[row_idx][column]
    if pd.isna(val):
        return False

    if direction == "long":
        return float(val) > threshold
    elif direction == "short":
        return float(val) < -threshold
    return False
