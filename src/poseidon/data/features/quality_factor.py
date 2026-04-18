"""Quality factor features -- pre-computed Z-scores from Thalassa.

Profitability, growth, and safety Z-scores are computed cross-sectionally
in Thalassa and delivered as quarterly/monthly data. Feature classes only
forward-fill to daily frequency.
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
class QualityProfitabilityZ(BaseFeature):
    """Cross-sectional profitability Z-score (from Thalassa, forward-filled)."""

    name = "quality_profitability_z"
    description = "Cross-sectional profitability Z-score (from Thalassa, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        quality_factor_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "quality_profitability_z"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if quality_factor_data is None or quality_factor_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "profitability_z" not in quality_factor_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(quality_factor_data["profitability_z"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class QualityGrowthZ(BaseFeature):
    """Cross-sectional growth Z-score (from Thalassa, forward-filled)."""

    name = "quality_growth_z"
    description = "Cross-sectional growth Z-score (from Thalassa, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        quality_factor_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "quality_growth_z"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if quality_factor_data is None or quality_factor_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "growth_z" not in quality_factor_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(quality_factor_data["growth_z"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class QualitySafetyZ(BaseFeature):
    """Cross-sectional safety Z-score (from Thalassa, forward-filled)."""

    name = "quality_safety_z"
    description = "Cross-sectional safety Z-score (from Thalassa, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        quality_factor_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "quality_safety_z"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if quality_factor_data is None or quality_factor_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "safety_z" not in quality_factor_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(quality_factor_data["safety_z"], ohlcv.index)
        result.name = col_name
        return result
