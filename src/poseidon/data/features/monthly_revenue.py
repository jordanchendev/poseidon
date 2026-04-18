"""Monthly revenue momentum features.

CumulativeRevenueGrowth: year-to-date cumulative revenue growth (from Thalassa).
RevenueAccelerationMonths: consecutive months of accelerating YoY revenue growth,
computed in Poseidon from Thalassa monthly_rev_yoy series.
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


def _ffill_to_index(source: pd.Series, target_index: pd.Index) -> pd.Series:
    """Forward-fill sparse series to dense target index (handles tz mismatch)."""
    clean = source.dropna()
    if isinstance(clean.index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if clean.index.tz is None and target_index.tz is not None:
            clean = clean.copy()
            clean.index = clean.index.tz_localize("UTC")
    return clean.reindex(target_index, method="ffill")


def _nan_series(index: pd.Index, name: str) -> pd.Series:
    """Return a NaN-filled Series with the given index and name."""
    return pd.Series(np.nan, index=index, name=name, dtype=float)


@register_feature
class CumulativeRevenueGrowth(BaseFeature):
    """Year-to-date cumulative revenue growth rate."""

    name = "cumulative_revenue_growth"
    description = "Year-to-date cumulative revenue growth rate (from Thalassa, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        monthly_revenue_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "cumulative_revenue_growth"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if monthly_revenue_data is None or monthly_revenue_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "cum_rev_yoy" not in monthly_revenue_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(monthly_revenue_data["cum_rev_yoy"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class RevenueAccelerationMonths(BaseFeature):
    """Consecutive months of accelerating YoY revenue growth."""

    name = "revenue_acceleration_months"
    description = "Number of consecutive months where YoY revenue growth is increasing"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        monthly_revenue_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "revenue_acceleration_months"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if monthly_revenue_data is None or monthly_revenue_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "monthly_rev_yoy" not in monthly_revenue_data.columns:
            return _nan_series(ohlcv.index, col_name)
        yoy = monthly_revenue_data["monthly_rev_yoy"].dropna()
        if yoy.empty:
            return _nan_series(ohlcv.index, col_name)
        # Diff of YoY: positive means accelerating
        yoy_diff = yoy.diff()
        is_positive = (yoy_diff > 0).astype(int)
        groups = (is_positive != is_positive.shift()).cumsum()
        consecutive = is_positive.groupby(groups).cumsum()
        # Forward-fill monthly to daily
        result = _ffill_to_index(consecutive, ohlcv.index)
        result.name = col_name
        return result
