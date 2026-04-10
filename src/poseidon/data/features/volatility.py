"""Volatility estimators: standard, Parkinson, Garman-Klass."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class StandardVolatility(BaseFeature):
    """Close-to-close returns volatility (rolling standard deviation)."""

    name = "std_vol"
    description = "Standard volatility from close-to-close returns"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv, min_rows=2):
            return pd.Series(dtype=float, name=f"std_vol_{period}")
        returns = ohlcv["close"].pct_change()
        result = returns.rolling(window=period).std()
        result.name = f"std_vol_{period}"
        return result


@register_feature
class ParkinsonVolatility(BaseFeature):
    """Range-based volatility using high/low prices.

    More efficient than close-to-close volatility for capturing intraday moves.
    Best for: crypto 24h markets, intraday-sensitive analysis.
    """

    name = "parkinson_vol"
    description = "Parkinson volatility estimator (high-low range)"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=f"parkinson_vol_{period}")
        hl_ratio = np.log(ohlcv["high"] / ohlcv["low"])
        squared_log_hl = hl_ratio ** 2
        result = np.sqrt(
            squared_log_hl.rolling(window=period).sum() / (4 * period * np.log(2))
        )
        result.name = f"parkinson_vol_{period}"
        return result


@register_feature
class GarmanKlassVolatility(BaseFeature):
    """OHLC-based volatility estimator.

    Most accurate of the three — uses all four OHLC prices.
    Captures overnight gaps better than Parkinson.
    Best for: mixed-market portfolios, equities with significant overnight moves.
    """

    name = "garman_klass_vol"
    description = "Garman-Klass volatility estimator (OHLC)"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=f"garman_klass_vol_{period}")
        hl = np.log(ohlcv["high"] / ohlcv["low"]) ** 2
        co = np.log(ohlcv["close"] / ohlcv["open"]) ** 2
        gk = (0.5 * hl) - ((2 * np.log(2) - 1) * co)
        result = np.sqrt(gk.rolling(window=period).mean())
        result.name = f"garman_klass_vol_{period}"
        return result


@register_feature
class ATRPercentile(BaseFeature):
    """ATR percentile rank in historical distribution -- H-C core signal.

    Used to dynamically adjust ambush distance based on current volatility state.
    """

    name = "atr_percentile"
    description = "ATR percentile rank in rolling window"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        period: int = 14,
        lookback: int = 100,
        **kwargs,
    ) -> pd.Series:
        col = f"atr_percentile_{lookback}"
        if not self._validate(ohlcv, min_rows=lookback + period):
            return pd.Series(dtype=float, name=col)
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close_prev = (ohlcv["high"] - ohlcv["close"].shift(1)).abs()
        low_close_prev = (ohlcv["low"] - ohlcv["close"].shift(1)).abs()
        tr = pd.concat(
            [high_low, high_close_prev, low_close_prev], axis=1
        ).max(axis=1)
        atr = tr.rolling(window=period).mean()
        result = atr.rolling(window=lookback).rank(pct=True)
        result.name = col
        return result


@register_feature
class VolRegime(BaseFeature):
    """Volatility regime classification based on short/long vol ratio.

    0=low (<0.8), 1=normal (0.8-1.2), 2=high (1.2-2.0), 3=extreme (>2.0).
    Used to adjust ambush distance and position sizing.
    """

    name = "vol_regime"
    description = "Volatility regime (0=low, 1=normal, 2=high, 3=extreme)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        **kwargs,
    ) -> pd.Series:
        col = "vol_regime"
        if not self._validate(ohlcv, min_rows=long_period + 1):
            return pd.Series(dtype=float, name=col)
        returns = ohlcv["close"].pct_change()
        short_vol = returns.rolling(window=short_period).std()
        long_vol = returns.rolling(window=long_period).std()
        ratio = short_vol / long_vol.replace(0, float("nan"))

        conditions = [
            ratio < 0.8,
            (ratio >= 0.8) & (ratio < 1.2),
            (ratio >= 1.2) & (ratio < 2.0),
            ratio >= 2.0,
        ]
        choices = [0, 1, 2, 3]
        result = pd.Series(
            np.select(conditions, choices, default=np.nan),
            index=ohlcv.index,
            dtype=float,
        )
        result.name = col
        return result
