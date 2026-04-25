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
    # Normalize timezone: FinLab naive datetime vs OHLCV UTC-aware
    if (
        isinstance(clean.index, pd.DatetimeIndex)
        and isinstance(target_index, pd.DatetimeIndex)
        and clean.index.tz is None
        and target_index.tz is not None
    ):
        clean = clean.copy()
        clean.index = clean.index.tz_localize("UTC")
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


@register_feature
class ROE(BaseFeature):
    """Return on equity, forward-filled from quarterly data to daily."""

    name = "roe"
    description = "Return on equity (quarterly ROE稅後, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "roe"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_data is None or fundamental_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "roe" not in fundamental_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(fundamental_data["roe"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class ROA(BaseFeature):
    """Return on assets, forward-filled from quarterly data to daily."""

    name = "roa"
    description = "Return on assets (quarterly ROA稅後息前, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "roa"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_data is None or fundamental_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "roa" not in fundamental_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(fundamental_data["roa"], ohlcv.index)
        result.name = col_name
        return result


# --- Extended fundamental features (from Thalassa fundamental_extended endpoint) ---


@register_feature
class GrossMargin(BaseFeature):
    """Gross margin rate, forward-filled from quarterly data to daily."""

    name = "gross_margin"
    description = "Gross margin rate (quarterly, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_extended_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "gross_margin"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_extended_data is None or fundamental_extended_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if col_name not in fundamental_extended_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(fundamental_extended_data[col_name], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class OperatingMargin(BaseFeature):
    """Operating margin rate, forward-filled from quarterly data to daily."""

    name = "operating_margin"
    description = "Operating margin rate (quarterly, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_extended_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "operating_margin"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_extended_data is None or fundamental_extended_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if col_name not in fundamental_extended_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(fundamental_extended_data[col_name], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class DebtRatio(BaseFeature):
    """Debt ratio, forward-filled from quarterly data to daily."""

    name = "debt_ratio"
    description = "Debt ratio (quarterly, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_extended_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "debt_ratio"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_extended_data is None or fundamental_extended_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if col_name not in fundamental_extended_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(fundamental_extended_data[col_name], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class EPS(BaseFeature):
    """Earnings per share, forward-filled from quarterly data to daily."""

    name = "eps"
    description = "Earnings per share (quarterly, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_extended_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "eps"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_extended_data is None or fundamental_extended_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if col_name not in fundamental_extended_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(fundamental_extended_data[col_name], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class ROEGrowth(BaseFeature):
    """Quarter-over-quarter ROE growth rate."""

    name = "roe_growth"
    description = "Quarter-over-quarter ROE growth rate"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_extended_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "roe_growth"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_extended_data is None or fundamental_extended_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "roe" not in fundamental_extended_data.columns:
            return _nan_series(ohlcv.index, col_name)
        ffilled = _ffill_to_index(fundamental_extended_data["roe"], ohlcv.index)
        result = ffilled.pct_change()
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result


@register_feature
class ROAGrowth(BaseFeature):
    """Quarter-over-quarter ROA growth rate."""

    name = "roa_growth"
    description = "Quarter-over-quarter ROA growth rate"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fundamental_extended_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "roa_growth"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if fundamental_extended_data is None or fundamental_extended_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "roa" not in fundamental_extended_data.columns:
            return _nan_series(ohlcv.index, col_name)
        ffilled = _ffill_to_index(fundamental_extended_data["roa"], ohlcv.index)
        result = ffilled.pct_change()
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result
