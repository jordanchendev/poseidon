"""CVD (Cumulative Volume Delta) feature.

Measures net buying/selling pressure using BVC classification.
Exposes rolling change (NOT raw cumulative) for stationarity.

Formula:
    delta = buy_volume - sell_volume          (per-bar net flow)
    cvd   = delta.cumsum()                    (cumulative)
    result = cvd[t] - cvd[t - period]         (rolling change)
"""

from __future__ import annotations

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class CVD(BaseFeature):
    """Cumulative Volume Delta -- net buying/selling pressure.

    Uses BVC to classify volume, then computes cumulative delta.
    Exposes rolling change (not raw cumulative) for stationarity (D-09).
    """

    name = "cvd"
    description = "Cumulative Volume Delta change over N bars"

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 20, **kwargs
    ) -> pd.Series:
        col = f"cvd_change_{period}"
        if not self._validate(ohlcv, min_rows=period + 1):
            return pd.Series(dtype=float, name=col)

        from poseidon.data.features.bvc import classify_volume

        bvc = classify_volume(ohlcv)
        delta = bvc["buy_volume"] - bvc["sell_volume"]
        cvd = delta.cumsum()
        # Rolling change for stationarity (D-09, Pitfall 4)
        result = cvd - cvd.shift(period)
        result.name = col
        return result
