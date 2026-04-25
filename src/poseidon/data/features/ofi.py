"""OFI (Order Flow Imbalance) feature.

Measures the imbalance in volume flow changes using BVC classification.
OFI = delta(buy_volume) - delta(sell_volume), smoothed via rolling sum.

Formula:
    ofi_raw = buy_volume.diff() - sell_volume.diff()
    result  = ofi_raw.rolling(window=period).sum()
"""

from __future__ import annotations

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class OFI(BaseFeature):
    """Order Flow Imbalance from BVC-derived buy/sell volume.

    OFI = delta(buy_volume) - delta(sell_volume) per bar.
    Smoothed via rolling sum over N bars (D-14, D-15).
    """

    name = "ofi"
    description = "Order Flow Imbalance (rolling N-bar sum)"

    def compute(self, ohlcv: pd.DataFrame, period: int = 5, **kwargs) -> pd.Series:
        col = f"ofi_{period}"
        if not self._validate(ohlcv, min_rows=period + 2):
            return pd.Series(dtype=float, name=col)

        from poseidon.data.features.bvc import classify_volume

        bvc = classify_volume(ohlcv)
        ofi_raw = bvc["buy_volume"].diff() - bvc["sell_volume"].diff()
        result = ofi_raw.rolling(window=period).sum()
        result.name = col
        return result
