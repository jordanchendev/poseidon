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
