"""Leaf condition evaluator functions and registry.

Each evaluator: (condition: dict, features: pd.DataFrame, row_idx: int) -> bool
"""

from collections.abc import Callable

import pandas as pd

CONDITION_REGISTRY: dict[str, Callable] = {}

# Alias mapping for indicator names
_INDICATOR_ALIASES = {"ma": "sma"}


def register_condition(name: str):
    """Decorator to register a condition evaluator."""

    def decorator(fn):
        CONDITION_REGISTRY[name] = fn
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

    if indicator == "macd":
        return "macd_line"

    period = params.get("period")
    if period is not None:
        return f"{indicator}_{period}"

    return indicator


@register_condition("indicator_above")
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


@register_condition("indicator_below")
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


@register_condition("price_crosses")
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


@register_condition("indicator_crosses")
def eval_indicator_crosses(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if fast indicator crosses slow indicator."""
    if row_idx < 1:
        return False

    params = condition.get("params", {})
    direction = condition.get("direction", "up")
    fast_col = resolve_column_name("sma", {"period": params.get("fast")})
    slow_col = resolve_column_name("sma", {"period": params.get("slow")})

    curr_fast = float(features.iloc[row_idx][fast_col])
    prev_fast = float(features.iloc[row_idx - 1][fast_col])
    curr_slow = float(features.iloc[row_idx][slow_col])
    prev_slow = float(features.iloc[row_idx - 1][slow_col])

    if direction == "up":
        return prev_fast <= prev_slow and curr_fast > curr_slow
    elif direction == "down":
        return prev_fast >= prev_slow and curr_fast < curr_slow
    return False


@register_condition("price_change_pct")
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


@register_condition("volume_spike")
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
