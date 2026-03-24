"""Regime detection features: volatility ratio, realized vol, vol-of-vol, return autocorrelation."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class VolatilityRatio(BaseFeature):
    """Ratio of short-term to long-term volatility for regime transitions."""

    name = "vol_ratio"
    description = "Ratio of short-term to long-term volatility"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        **kwargs,
    ) -> pd.Series:
        column_name = f"vol_ratio_{short_period}_{long_period}"
        if not self._validate(ohlcv, min_rows=long_period + 1):
            return pd.Series(dtype=float, name=column_name)
        returns = ohlcv["close"].pct_change()
        short_vol = returns.rolling(window=short_period).std()
        long_vol = returns.rolling(window=long_period).std()
        result = short_vol / long_vol
        result.name = column_name
        return result


@register_feature
class RealizedVolatility(BaseFeature):
    """Multi-period realized volatility for regime detection."""

    name = "realized_vol"
    description = "Multi-period realized volatility"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        period: int = 10,
        annualize: int = 252,
        **kwargs,
    ) -> pd.Series:
        column_name = f"realized_vol_{period}"
        if not self._validate(ohlcv, min_rows=period + 1):
            return pd.Series(dtype=float, name=column_name)
        log_returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        result = log_returns.rolling(window=period).std() * np.sqrt(annualize)
        result.name = column_name
        return result


@register_feature
class VolatilityOfVolatility(BaseFeature):
    """Volatility of volatility for measuring regime stability."""

    name = "vol_of_vol"
    description = "Volatility of volatility for regime stability"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        vol_period: int = 20,
        meta_period: int = 10,
        **kwargs,
    ) -> pd.Series:
        column_name = f"vol_of_vol_{vol_period}_{meta_period}"
        if not self._validate(ohlcv, min_rows=vol_period + meta_period):
            return pd.Series(dtype=float, name=column_name)
        returns = ohlcv["close"].pct_change()
        volatility = returns.rolling(window=vol_period).std()
        result = volatility.rolling(window=meta_period).std()
        result.name = column_name
        return result


@register_feature
class ReturnAutocorrelation(BaseFeature):
    """Rolling return autocorrelation for regime characterization."""

    name = "return_autocorr"
    description = "Rolling return autocorrelation"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        period: int = 20,
        lag: int = 1,
        **kwargs,
    ) -> pd.Series:
        column_name = f"return_autocorr_{period}"
        if not self._validate(ohlcv, min_rows=period + lag + 1):
            return pd.Series(dtype=float, name=column_name)
        returns = ohlcv["close"].pct_change()
        result = returns.rolling(window=period).corr(returns.shift(lag))
        result.name = column_name
        return result
