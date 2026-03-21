"""Return features: daily return, log return, cumulative return."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class Returns(BaseFeature):
    """Daily and log returns.

    Returns DataFrame with columns: return_1d, log_return_1d.
    """

    name = "returns"
    description = "Daily simple return and log return"

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self._validate(ohlcv, min_rows=2):
            return pd.DataFrame(columns=["return_1d", "log_return_1d"])
        daily_return = ohlcv["close"].pct_change()
        log_return = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        return pd.DataFrame({
            "return_1d": daily_return,
            "log_return_1d": log_return,
        })


@register_feature
class CumulativeReturn(BaseFeature):
    """Cumulative return over N periods."""

    name = "cum_return"
    description = "Cumulative return over N periods"

    def compute(self, ohlcv: pd.DataFrame, period: int = 5, **kwargs) -> pd.Series:
        if not self._validate(ohlcv, min_rows=2):
            return pd.Series(dtype=float, name=f"cum_return_{period}d")
        result = ohlcv["close"].pct_change(periods=period)
        result.name = f"cum_return_{period}d"
        return result
