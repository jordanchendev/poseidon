"""Price momentum features -- simple returns over 3M/6M/12M windows.

Computes (P_t / P_{t-N}) - 1 using OHLCV close prices. Unlike nonprice
features (quality_factor, monthly_revenue), momentum routes to OHLCV data
directly -- is_nonprice_spec must return False for these names.

Trading day approximations:
  3M  = 63 days
  6M  = 126 days
  12M = 252 days

Per D-01/D-02/D-03 in Phase 71 CONTEXT.md.
"""

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature

_DAYS_3M = 63
_DAYS_6M = 126
_DAYS_12M = 252


@register_feature
class PriceMomentum3M(BaseFeature):
    """3-month price momentum (simple return from close price)."""

    name = "momentum_3m"
    description = "3-month price momentum (simple return)"
    supports_backtest = True
    bias_risk = ["look_ahead_ohlcv"]

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        col_name = "momentum_3m"
        if not self._validate(ohlcv, min_rows=_DAYS_3M + 1):
            return pd.Series(dtype=float, name=col_name)
        close = ohlcv["close"]
        result = close / close.shift(_DAYS_3M) - 1.0
        result.name = col_name
        return result


@register_feature
class PriceMomentum6M(BaseFeature):
    """6-month price momentum (simple return from close price)."""

    name = "momentum_6m"
    description = "6-month price momentum (simple return)"
    supports_backtest = True
    bias_risk = ["look_ahead_ohlcv"]

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        col_name = "momentum_6m"
        if not self._validate(ohlcv, min_rows=_DAYS_6M + 1):
            return pd.Series(dtype=float, name=col_name)
        close = ohlcv["close"]
        result = close / close.shift(_DAYS_6M) - 1.0
        result.name = col_name
        return result


@register_feature
class PriceMomentum12M(BaseFeature):
    """12-month price momentum (simple return from close price)."""

    name = "momentum_12m"
    description = "12-month price momentum (simple return)"
    supports_backtest = True
    bias_risk = ["look_ahead_ohlcv"]

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        col_name = "momentum_12m"
        if not self._validate(ohlcv, min_rows=_DAYS_12M + 1):
            return pd.Series(dtype=float, name=col_name)
        close = ohlcv["close"]
        result = close / close.shift(_DAYS_12M) - 1.0
        result.name = col_name
        return result
