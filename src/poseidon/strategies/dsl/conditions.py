"""Leaf condition evaluator functions and registry.

Each evaluator: (condition: dict, features: pd.DataFrame, row_idx: int) -> bool
"""

from collections.abc import Callable

import pandas as pd

CONDITION_REGISTRY: dict[str, Callable] = {}

# Alias mapping for indicator names
_INDICATOR_ALIASES = {"ma": "sma"}

# Direct column name mappings (no period suffix needed)
_DIRECT_COLUMN_MAP = {
    "macd_histogram": "macd_histogram",
    "macd_signal": "macd_signal",
    "macd_line": "macd_line",
}


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


@register_condition("bollinger_width_percentile")
def eval_bollinger_width_percentile(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if Bollinger Band width is in a squeeze (below percentile threshold).

    Computes the percentile rank of current BB width within a lookback window.
    No look-ahead: only uses rows up to and including row_idx.
    """
    params = condition.get("params", {})
    period = params.get("period", 20)
    lookback = params.get("lookback", 168)  # 1 week of hourly bars
    threshold = condition.get("threshold", 0.2)  # below 20th percentile = squeeze

    upper_col = f"bb_upper_{period}"
    lower_col = f"bb_lower_{period}"

    # Compute width series up to current row (no look-ahead)
    start = max(0, row_idx - lookback + 1)
    widths = features[upper_col].iloc[start:row_idx + 1] - features[lower_col].iloc[start:row_idx + 1]

    if len(widths) < 2:
        return False

    current_width = widths.iloc[-1]
    percentile = (widths < current_width).sum() / len(widths)
    return percentile < threshold


@register_condition("indicator_comparison")
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
