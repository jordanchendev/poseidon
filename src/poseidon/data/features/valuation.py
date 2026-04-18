"""Valuation features -- PE/PBR/dividend yield and their rolling historical percentiles.

Raw ratios are forward-filled from Thalassa pe-pbr endpoint.
Percentile features compute rolling(window=252).rank(pct=True) in Poseidon,
giving strategy the position of current valuation within trailing 1-year history.
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
class DividendYield(BaseFeature):
    """Dividend yield, forward-filled from periodic data."""

    name = "dividend_yield"
    description = "Dividend yield (from Thalassa, forward-filled)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        pe_pbr_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "dividend_yield"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if pe_pbr_data is None or pe_pbr_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "dividend_yield" not in pe_pbr_data.columns:
            return _nan_series(ohlcv.index, col_name)
        result = _ffill_to_index(pe_pbr_data["dividend_yield"], ohlcv.index)
        result.name = col_name
        return result


@register_feature
class PEPercentile(BaseFeature):
    """PE ratio percentile rank over trailing 252 trading days."""

    name = "pe_percentile"
    description = "PE ratio percentile rank over trailing 252 trading days"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        pe_pbr_data: pd.DataFrame | None = None,
        window: int = 252,
        **kwargs,
    ) -> pd.Series:
        col_name = "pe_percentile"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if pe_pbr_data is None or pe_pbr_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "pe_ratio" not in pe_pbr_data.columns:
            return _nan_series(ohlcv.index, col_name)
        ffilled = _ffill_to_index(pe_pbr_data["pe_ratio"], ohlcv.index)
        result = ffilled.rolling(window, min_periods=1).rank(pct=True)
        result.name = col_name
        return result


@register_feature
class PBRPercentile(BaseFeature):
    """PBR (price-to-book ratio) percentile rank over trailing 252 trading days."""

    name = "pbr_percentile"
    description = "PBR percentile rank over trailing 252 trading days"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        pe_pbr_data: pd.DataFrame | None = None,
        window: int = 252,
        **kwargs,
    ) -> pd.Series:
        col_name = "pbr_percentile"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if pe_pbr_data is None or pe_pbr_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "pb_ratio" not in pe_pbr_data.columns:
            return _nan_series(ohlcv.index, col_name)
        ffilled = _ffill_to_index(pe_pbr_data["pb_ratio"], ohlcv.index)
        result = ffilled.rolling(window, min_periods=1).rank(pct=True)
        result.name = col_name
        return result


@register_feature
class DividendYieldPercentile(BaseFeature):
    """Dividend yield percentile rank over trailing 252 trading days."""

    name = "dividend_yield_percentile"
    description = "Dividend yield percentile rank over trailing 252 trading days"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        pe_pbr_data: pd.DataFrame | None = None,
        window: int = 252,
        **kwargs,
    ) -> pd.Series:
        col_name = "dividend_yield_percentile"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if pe_pbr_data is None or pe_pbr_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "dividend_yield" not in pe_pbr_data.columns:
            return _nan_series(ohlcv.index, col_name)
        ffilled = _ffill_to_index(pe_pbr_data["dividend_yield"], ohlcv.index)
        result = ffilled.rolling(window, min_periods=1).rank(pct=True)
        result.name = col_name
        return result
