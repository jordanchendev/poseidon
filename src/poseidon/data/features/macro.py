"""Macro index features: broad market condition signals.

VIX (fear gauge), DXY (dollar strength), TNX (10Y yield), TWDUSD (TWD/USD rate).
Data is injected via the ``macro_data`` kwarg by FeatureEngine.compute_with_companions().
"""

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


def _align_macro_column(
    ohlcv: pd.DataFrame,
    macro_data: pd.DataFrame | None,
    col: str,
) -> pd.Series:
    """Align a single macro column to *ohlcv* index with forward-fill.

    Returns NaN Series if data is unavailable.
    """
    if macro_data is None or (hasattr(macro_data, "empty") and macro_data.empty):
        return pd.Series(float("nan"), index=ohlcv.index, name=col)
    if col not in macro_data.columns:
        return pd.Series(float("nan"), index=ohlcv.index, name=col)
    series = macro_data[col]
    # Ensure both indexes are DatetimeIndex before reindex.
    # macro_data may arrive with int64 index (e.g. from DB or cache),
    # which causes "Cannot compare dtypes datetime64 and int64" on reindex.
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index, utc=True, errors="coerce")
        series = series[series.index.notna()].sort_index()
    # Normalize timezone: yfinance returns datetime64[s] (naive), OHLCV uses UTC-aware
    if isinstance(series.index, pd.DatetimeIndex) and isinstance(ohlcv.index, pd.DatetimeIndex):
        if series.index.tz is None and ohlcv.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_localize("UTC")
        elif series.index.tz is not None and ohlcv.index.tz is not None and series.index.tz != ohlcv.index.tz:
            series = series.copy()
            series.index = series.index.tz_convert(ohlcv.index.tz)
    aligned = series.reindex(ohlcv.index, method="ffill")
    aligned.name = col
    return aligned


@register_feature
class MacroVIX(BaseFeature):
    """CBOE Volatility Index — market fear gauge."""

    name = "macro_vix"
    description = "VIX volatility index (market fear gauge)"

    def compute(self, ohlcv: pd.DataFrame, macro_data: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=self.name)
        return _align_macro_column(ohlcv, macro_data, self.name)


@register_feature
class MacroDXY(BaseFeature):
    """US Dollar Index — dollar strength signal."""

    name = "macro_dxy"
    description = "DXY US dollar index (dollar strength)"

    def compute(self, ohlcv: pd.DataFrame, macro_data: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=self.name)
        return _align_macro_column(ohlcv, macro_data, self.name)


@register_feature
class MacroTNX(BaseFeature):
    """10-Year Treasury Note Yield — interest rate signal."""

    name = "macro_tnx"
    description = "TNX 10-year treasury yield"

    def compute(self, ohlcv: pd.DataFrame, macro_data: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=self.name)
        return _align_macro_column(ohlcv, macro_data, self.name)


@register_feature
class MacroTWDUSD(BaseFeature):
    """TWD/USD exchange rate — Taiwan dollar strength signal."""

    name = "macro_twdusd"
    description = "TWD/USD exchange rate"

    def compute(self, ohlcv: pd.DataFrame, macro_data: pd.DataFrame | None = None, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=self.name)
        return _align_macro_column(ohlcv, macro_data, self.name)
