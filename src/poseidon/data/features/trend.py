"""Trend strength and session features for liquidity sweep filtering."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class ADX(BaseFeature):
    """Average Directional Index -- quantifies trend strength.

    ADX > 25 = trending, ADX < 20 = ranging.
    Core filter for sweep strategy: strong trend suppresses reversal signals (R-02).
    """

    name = "adx"
    description = "Average Directional Index (trend strength 0-100)"

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 14, **kwargs
    ) -> pd.Series:
        col = f"adx_{period}"
        if not self._validate(ohlcv, min_rows=period * 2 + 1):
            return pd.Series(dtype=float, name=col)
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        # +DM / -DM
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=ohlcv.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=ohlcv.index,
        )

        # True Range
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()
        tr = pd.concat(
            [high_low, high_close_prev, low_close_prev], axis=1
        ).max(axis=1)

        # Smoothed with EMA (Wilder's smoothing = EMA with alpha=1/period)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = (
            100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        )

        # DX and ADX
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, float("nan"))
        dx = (plus_di - minus_di).abs() / di_sum * 100
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        adx.name = col
        return adx


@register_feature
class TrendStrength(BaseFeature):
    """Price position relative to long-term SMA, normalized by ATR.

    Positive = bullish trend, Negative = bearish trend.
    |value| > 2.0 = strong trend that should suppress reversal signals.
    """

    name = "trend_strength"
    description = "Price deviation from SMA in ATR multiples"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        long_period: int = 100,
        atr_period: int = 14,
        **kwargs,
    ) -> pd.Series:
        col = f"trend_strength_{long_period}"
        if not self._validate(ohlcv, min_rows=long_period + 1):
            return pd.Series(dtype=float, name=col)
        sma = ohlcv["close"].rolling(window=long_period).mean()
        # ATR inline
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close_prev = (ohlcv["high"] - ohlcv["close"].shift(1)).abs()
        low_close_prev = (ohlcv["low"] - ohlcv["close"].shift(1)).abs()
        tr = pd.concat(
            [high_low, high_close_prev, low_close_prev], axis=1
        ).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        result = (ohlcv["close"] - sma) / atr.replace(0, float("nan"))
        result.name = col
        return result


@register_feature
class HourOfDay(BaseFeature):
    """UTC hour of each bar -- crypto liquidations cluster around funding settlement times."""

    name = "hour_of_day"
    description = "UTC hour (0-23) for session-based filtering"

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        col = "hour_of_day"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col)
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            result = ohlcv.index.hour.to_series(index=ohlcv.index)
        elif "time" in ohlcv.columns:
            result = pd.to_datetime(ohlcv["time"]).dt.hour
            result.index = ohlcv.index
        else:
            result = pd.Series(float("nan"), index=ohlcv.index)
        result.name = col
        return result
