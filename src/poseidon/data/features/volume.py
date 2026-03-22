"""Volume-based features: volume SMA, volume ratio, OBV."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class VolumeSMA(BaseFeature):
    """Simple Moving Average of volume."""

    name = "volume_sma"
    description = "Simple Moving Average of volume"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=f"volume_sma_{period}")
        result = ohlcv["volume"].rolling(window=period).mean()
        result.name = f"volume_sma_{period}"
        return result


@register_feature
class VolumeRatio(BaseFeature):
    """Current volume relative to N-period average."""

    name = "volume_ratio"
    description = "Current volume relative to N-period average"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=f"volume_ratio_{period}")
        result = ohlcv["volume"] / ohlcv["volume"].rolling(window=period).mean()
        result.name = f"volume_ratio_{period}"
        return result


@register_feature
class OBV(BaseFeature):
    """On-Balance Volume (cumulative signed volume).

    Measures buying/selling pressure as a cumulative indicator.
    Positive close-to-close changes add volume, negative changes subtract.
    """

    name = "obv"
    description = "On-Balance Volume (cumulative signed volume)"

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        if not self._validate(ohlcv, min_rows=2):
            return pd.Series(dtype=float, name="obv")
        direction = np.sign(ohlcv["close"].diff())
        result = (ohlcv["volume"] * direction).cumsum()
        result.name = "obv"
        return result
