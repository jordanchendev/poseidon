"""DSL v2.1 derived series: rolling (Donchian) channels and smoothed indicators.

Both are causal: the value at bar ``k`` only reads bars ``<= k`` (``< k`` for
a channel with ``exclude_current``), and every warm-up row is NaN rather than
a partial-window value, so a truncated frame reproduces the full frame's
prefix exactly.

- ``rolling_channel`` -> ``ch_{highest|lowest}_{source}_{bars}_{prev|incl}``:
  max/min of ``source`` over the ``bars`` bars before the current one
  (``prev``, ``exclude_current=True``) or ending at it (``incl``). Bar-count
  based, so the window spans session boundaries.
- ``ind_smooth`` -> ``{ind}_{period}_{sma|ema}_{smooth_period}``: SMA (or EMA
  with span ``smooth_period``, ``adjust=False``) of the ``ind`` series, NaN
  until ``smooth_period`` valid indicator values exist.
"""

from __future__ import annotations

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature
from poseidon.data.features.technical import ATR, EMA, RSI, SMA

CHANNEL_KINDS = ("highest", "lowest")
CHANNEL_SOURCES = ("high", "low", "close")
SMOOTH_TYPES = ("sma", "ema")
_BASE_INDICATORS = {"ema": EMA, "sma": SMA, "atr": ATR, "rsi": RSI}


def channel_column(channel: str, source: str, bars: int, exclude_current: bool = True) -> str:
    return f"ch_{channel}_{source}_{int(bars)}_{'prev' if exclude_current else 'incl'}"


def smoothed_column(ind: str, period: int, smooth: str, smooth_period: int) -> str:
    return f"{ind}_{int(period)}_{smooth}_{int(smooth_period)}"


def compute_rolling_channel(
    ohlcv: pd.DataFrame, channel: str, source: str, bars: int, exclude_current: bool = True
) -> pd.Series:
    if channel not in CHANNEL_KINDS:
        raise ValueError(f"channel must be one of {CHANNEL_KINDS}, got {channel!r}")
    if source not in CHANNEL_SOURCES:
        raise ValueError(f"source must be one of {CHANNEL_SOURCES}, got {source!r}")
    bars = int(bars)
    if bars < 1:
        raise ValueError("bars must be positive")
    window = ohlcv[source].astype(float).rolling(window=bars, min_periods=bars)
    result = window.max() if channel == "highest" else window.min()
    if exclude_current:
        result = result.shift(1)
    result.name = channel_column(channel, source, bars, exclude_current)
    return result


def compute_smoothed_indicator(
    ohlcv: pd.DataFrame, ind: str, period: int, smooth: str, smooth_period: int
) -> pd.Series:
    if ind not in _BASE_INDICATORS:
        raise ValueError(f"ind must be one of {sorted(_BASE_INDICATORS)}, got {ind!r}")
    if smooth not in SMOOTH_TYPES:
        raise ValueError(f"smooth must be one of {SMOOTH_TYPES}, got {smooth!r}")
    smooth_period = int(smooth_period)
    base = _BASE_INDICATORS[ind]().compute(ohlcv, period=int(period)).astype(float)
    if smooth == "sma":
        result = base.rolling(window=smooth_period, min_periods=smooth_period).mean()
    else:
        result = base.ewm(span=smooth_period, adjust=False, min_periods=smooth_period).mean()
    result.name = smoothed_column(ind, period, smooth, smooth_period)
    return result


@register_feature
class RollingChannel(BaseFeature):
    """Donchian channel: highest/lowest of high/low/close over N bars (before or ending at the bar)."""

    name = "rolling_channel"
    description = "Highest/lowest of a price over the previous N bars (DSL v2.1 channel operand)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        channel: str = "highest",
        source: str = "high",
        bars: int = 20,
        exclude_current: bool = True,
        **kwargs,
    ) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=channel_column(channel, source, bars, exclude_current))
        return compute_rolling_channel(ohlcv, channel, source, bars, exclude_current)


@register_feature
class SmoothedIndicator(BaseFeature):
    """Moving average (SMA/EMA) of an ema/sma/atr/rsi series."""

    name = "ind_smooth"
    description = "SMA/EMA of an indicator series, e.g. SMA(50) of ATR(14) (DSL v2.1 smooth)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        ind: str = "atr",
        period: int = 14,
        smooth: str = "sma",
        smooth_period: int = 50,
        **kwargs,
    ) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=smoothed_column(ind, period, smooth, smooth_period))
        return compute_smoothed_indicator(ohlcv, ind, period, smooth, smooth_period)
