"""Fundamental features for TW stock market.

Captures valuation metrics (P/E, P/B) and revenue growth (MoM, YoY)
by forward-filling quarterly/monthly data to daily frequency.
Data is injected via kwargs from the engine (no I/O in feature classes).
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


def _ffill_to_index(source: pd.Series, target_index: pd.Index) -> pd.Series:
    """Forward-fill a sparse series to a dense target index.

    First drops NaN from the source to get clean data points, then
    reindexes to the target with forward-fill. This handles cases where
    the source DataFrame has NaN in this column (e.g., combined quarterly
    + monthly data where not every row has every column).
    """
    clean = source.dropna()
    return clean.reindex(target_index, method="ffill")


def _nan_series(index: pd.Index, name: str) -> pd.Series:
    """Return a NaN-filled Series with the given index and name."""
    return pd.Series(np.nan, index=index, name=name, dtype=float)


@register_feature
class PERatio(BaseFeature):
    """Price-to-earnings ratio, forward-filled from quarterly data to daily."""

    name = "pe_ratio"
    description = "Price-to-earnings ratio (quarterly, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "pe_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if fundamental_data is None or fundamental_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "pe_ratio" not in fundamental_data.columns:
            return _nan_series(ohlcv.index, col_name)

        result = _ffill_to_index(fundamental_data["pe_ratio"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class PBRatio(BaseFeature):
    """Price-to-book ratio, forward-filled from quarterly data to daily."""

    name = "pb_ratio"
    description = "Price-to-book ratio (quarterly, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "pb_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if fundamental_data is None or fundamental_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "pb_ratio" not in fundamental_data.columns:
            return _nan_series(ohlcv.index, col_name)

        result = _ffill_to_index(fundamental_data["pb_ratio"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class RevenueMoM(BaseFeature):
    """Month-over-month revenue growth, forward-filled from monthly data.

    Computes pct_change on forward-filled monthly revenue, so each time
    the underlying monthly value changes, the MoM growth updates.
    """

    name = "revenue_mom"
    description = "Month-over-month revenue growth rate"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "revenue_mom"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if fundamental_data is None or fundamental_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "monthly_rev" not in fundamental_data.columns:
            return _nan_series(ohlcv.index, col_name)

        ffilled = _ffill_to_index(fundamental_data["monthly_rev"], ohlcv.index)
        result = ffilled.pct_change()
        result.name = col_name
        return result


@register_feature
class RevenueYoY(BaseFeature):
    """Year-over-year revenue growth, computed from monthly and prev-year revenue.

    YoY = (monthly_rev - prev_year_rev) / prev_year_rev
    Both series are forward-filled to daily before computation.
    """

    name = "revenue_yoy"
    description = "Year-over-year revenue growth rate"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "revenue_yoy"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if fundamental_data is None or fundamental_data.empty:
            return _nan_series(ohlcv.index, col_name)

        required = {"monthly_rev", "prev_year_rev"}
        if not required.issubset(fundamental_data.columns):
            return _nan_series(ohlcv.index, col_name)

        monthly = _ffill_to_index(fundamental_data["monthly_rev"], ohlcv.index)
        prev_year = _ffill_to_index(fundamental_data["prev_year_rev"], ohlcv.index)

        result = (monthly - prev_year) / prev_year
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result
