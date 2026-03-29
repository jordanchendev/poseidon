"""Trade structure features for TW stock market.

Captures market microstructure signals: average trade size and turnover ratio.
Data is injected via kwargs from the engine (no I/O in feature classes).
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


def _nan_series(index: pd.Index, name: str) -> pd.Series:
    """Return a NaN-filled Series with the given index and name."""
    return pd.Series(np.nan, index=index, name=name, dtype=float)


@register_feature
class AvgTradeSize(BaseFeature):
    """Average trade size: trade_value / trade_count.

    Larger average trade sizes may indicate institutional participation,
    while smaller sizes suggest retail-dominated activity.
    """

    name = "avg_trade_size"
    description = "Average trade size (value / count)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        trade_structure_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "avg_trade_size"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if trade_structure_data is None or trade_structure_data.empty:
            return _nan_series(ohlcv.index, col_name)

        required = {"trade_value", "trade_count"}
        if not required.issubset(trade_structure_data.columns):
            return _nan_series(ohlcv.index, col_name)

        # Normalize timezone for reindex compatibility
        ts = trade_structure_data
        if isinstance(ts.index, pd.DatetimeIndex) and isinstance(ohlcv.index, pd.DatetimeIndex):
            if ts.index.tz is None and ohlcv.index.tz is not None:
                ts = ts.copy()
                ts.index = ts.index.tz_localize("UTC")
        aligned = ts.reindex(ohlcv.index, method="ffill")
        result = aligned["trade_value"] / aligned["trade_count"]
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result


@register_feature
class TurnoverRatio(BaseFeature):
    """Turnover ratio proxy: trade_value / (close * volume).

    A proxy for true turnover ratio (which requires shares outstanding).
    Higher values indicate more active trading relative to price-volume.
    """

    name = "turnover_ratio"
    description = "Turnover ratio proxy (trade_value / close*volume)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        trade_structure_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "turnover_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if trade_structure_data is None or trade_structure_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "trade_value" not in trade_structure_data.columns:
            return _nan_series(ohlcv.index, col_name)

        trade_val = trade_structure_data["trade_value"]
        if isinstance(trade_val.index, pd.DatetimeIndex) and isinstance(ohlcv.index, pd.DatetimeIndex):
            if trade_val.index.tz is None and ohlcv.index.tz is not None:
                trade_val = trade_val.copy()
                trade_val.index = trade_val.index.tz_localize("UTC")
        aligned_value = trade_val.reindex(ohlcv.index, method="ffill")
        result = aligned_value / (ohlcv["close"] * ohlcv["volume"])
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result
