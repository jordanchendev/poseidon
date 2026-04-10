"""Swing point and breakout distance features for liquidity sweep detection."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class SwingHigh(BaseFeature):
    """Rolling maximum of high prices over lookback period."""

    name = "swing_high"
    description = "Swing high (rolling max of high)"

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 24, **kwargs
    ) -> pd.Series:
        col = f"swing_high_{period}"
        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col)
        result = ohlcv["high"].rolling(window=period).max()
        result.name = col
        return result


@register_feature
class SwingLow(BaseFeature):
    """Rolling minimum of low prices over lookback period."""

    name = "swing_low"
    description = "Swing low (rolling min of low)"

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 24, **kwargs
    ) -> pd.Series:
        col = f"swing_low_{period}"
        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col)
        result = ohlcv["low"].rolling(window=period).min()
        result.name = col
        return result


@register_feature
class BreakoutDistance(BaseFeature):
    """Distance beyond swing high/low in ATR multiples.

    Positive = price broke past the level. Negative = still within range.
    """

    name = "breakout_distance"
    description = "Breakout distance past swing high/low in ATR multiples"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        period: int = 24,
        atr_period: int = 14,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv, min_rows=max(period, atr_period) + 1):
            return pd.DataFrame(dtype=float)
        swing_high = ohlcv["high"].rolling(window=period).max()
        swing_low = ohlcv["low"].rolling(window=period).min()
        # ATR inline
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close_prev = (ohlcv["high"] - ohlcv["close"].shift(1)).abs()
        low_close_prev = (ohlcv["low"] - ohlcv["close"].shift(1)).abs()
        true_range = pd.concat(
            [high_low, high_close_prev, low_close_prev], axis=1
        ).max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()
        atr_safe = atr.replace(0, float("nan"))

        breakout_up = (ohlcv["high"] - swing_high) / atr_safe
        breakout_down = (swing_low - ohlcv["low"]) / atr_safe

        return pd.DataFrame(
            {
                f"breakout_up_{period}": breakout_up,
                f"breakout_down_{period}": breakout_down,
            },
            index=ohlcv.index,
        )


@register_feature
class FibExtension(BaseFeature):
    """Fibonacci extension levels from swing range."""

    name = "fib_extension"
    description = "Fibonacci extension projections from swing high/low range"

    FIB_LEVELS = [0.618, 1.0, 1.618]

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 24, **kwargs
    ) -> pd.DataFrame:
        if not self._validate(ohlcv, min_rows=period):
            return pd.DataFrame(dtype=float)
        swing_high = ohlcv["high"].rolling(window=period).max()
        swing_low = ohlcv["low"].rolling(window=period).min()
        swing_range = swing_high - swing_low

        result = {}
        for level in self.FIB_LEVELS:
            level_str = str(level).replace(".", "_")
            result[f"fib_ext_up_{level_str}"] = swing_high + swing_range * level
            result[f"fib_ext_down_{level_str}"] = swing_low - swing_range * level

        return pd.DataFrame(result, index=ohlcv.index)
