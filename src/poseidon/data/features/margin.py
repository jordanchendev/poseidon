"""Margin transaction features for TW stock market.

Captures margin buying and short selling utilization rates as non-price
alpha signals. Data is injected via kwargs from the engine (no I/O in
feature classes).
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


def _align_to_index(
    source: pd.Series, target_index: pd.Index, method: str = "ffill"
) -> pd.Series:
    """Align a source series to a target index with forward-fill.

    Handles timezone mismatch: FinLab returns naive datetime, OHLCV uses UTC-aware.
    """
    if isinstance(source.index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if source.index.tz is None and target_index.tz is not None:
            source = source.copy()
            source.index = source.index.tz_localize("UTC")
    return source.reindex(target_index, method=method)


def _nan_series(index: pd.Index, name: str) -> pd.Series:
    """Return a NaN-filled Series with the given index and name."""
    return pd.Series(np.nan, index=index, name=name, dtype=float)


@register_feature
class MarginBuyRatio(BaseFeature):
    """Margin buy ratio (融資使用率), aligned from daily data.

    Higher values indicate more margin borrowing relative to the credit limit,
    suggesting leveraged bullish sentiment.
    """

    name = "margin_buy_ratio"
    description = "Margin purchase utilization rate (融資使用率)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        margin_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "margin_buy_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if margin_data is None or margin_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "margin_buy_ratio" not in margin_data.columns:
            return _nan_series(ohlcv.index, col_name)

        result = _align_to_index(margin_data["margin_buy_ratio"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class MarginSellRatio(BaseFeature):
    """Margin sell ratio (融券使用率), aligned from daily data.

    Higher values indicate more short selling relative to the short limit,
    suggesting bearish sentiment or hedging activity.
    """

    name = "margin_sell_ratio"
    description = "Short selling utilization rate (融券使用率)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        margin_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "margin_sell_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if margin_data is None or margin_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "margin_sell_ratio" not in margin_data.columns:
            return _nan_series(ohlcv.index, col_name)

        result = _align_to_index(margin_data["margin_sell_ratio"], ohlcv.index)
        result.name = col_name
        return result
