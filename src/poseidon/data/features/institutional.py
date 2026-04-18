"""Institutional flow features for TW stock market.

Captures foreign investor, investment trust, and dealer trading behavior
as non-price alpha signals. Data is injected via kwargs from the engine
(no I/O in feature classes).
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
    # Normalize timezone: FinLab naive datetime vs OHLCV UTC-aware
    if isinstance(source.index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if source.index.tz is None and target_index.tz is not None:
            source = source.copy()
            source.index = source.index.tz_localize("UTC")
    return source.reindex(target_index, method=method)


def _ffill_to_index(source: pd.Series, target_index: pd.Index) -> pd.Series:
    """Forward-fill a sparse series to a dense target index.

    First drops NaN from the source to get clean data points, then
    reindexes to the target with forward-fill. Handles tz mismatch.
    """
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
class ForeignNetBuyRatio(BaseFeature):
    """Foreign investor net buy as a ratio of volume.

    Higher ratios indicate stronger foreign buying pressure relative to
    total market activity.
    """

    name = "foreign_net_buy_ratio"
    description = "Foreign investor net buy / volume ratio"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        institutional_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "foreign_net_buy_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if institutional_data is None or institutional_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "foreign" not in institutional_data.columns:
            return _nan_series(ohlcv.index, col_name)

        aligned = _align_to_index(institutional_data["foreign"], ohlcv.index)
        result = aligned / ohlcv["volume"]
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result


@register_feature
class ForeignNetBuyCum(BaseFeature):
    """Cumulative foreign investor net buy over a rolling window.

    Captures sustained foreign buying/selling pressure over multiple days.
    """

    name = "foreign_net_buy_cum"
    description = "Rolling cumulative foreign investor net buy"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        institutional_data: pd.DataFrame | None = None,
        period: int = 5,
        **kwargs,
    ) -> pd.Series:
        col_name = f"foreign_net_buy_cum_{period}"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if institutional_data is None or institutional_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "foreign" not in institutional_data.columns:
            return _nan_series(ohlcv.index, col_name)

        aligned = _align_to_index(institutional_data["foreign"], ohlcv.index)
        result = aligned.rolling(period).sum()
        result.name = col_name
        return result


@register_feature
class TrustNetBuyRatio(BaseFeature):
    """Investment trust net buy as a ratio of volume.

    Trusts represent domestic institutional sentiment. Higher ratios
    indicate stronger trust buying pressure.
    """

    name = "trust_net_buy_ratio"
    description = "Investment trust net buy / volume ratio"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        institutional_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "trust_net_buy_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if institutional_data is None or institutional_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "trust" not in institutional_data.columns:
            return _nan_series(ohlcv.index, col_name)

        aligned = _align_to_index(institutional_data["trust"], ohlcv.index)
        result = aligned / ohlcv["volume"]
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result


@register_feature
class TrustNetBuyCum(BaseFeature):
    """Cumulative investment trust net buy over a rolling window."""

    name = "trust_net_buy_cum"
    description = "Rolling cumulative investment trust net buy"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        institutional_data: pd.DataFrame | None = None,
        period: int = 5,
        **kwargs,
    ) -> pd.Series:
        col_name = f"trust_net_buy_cum_{period}"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if institutional_data is None or institutional_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "trust" not in institutional_data.columns:
            return _nan_series(ohlcv.index, col_name)

        aligned = _align_to_index(institutional_data["trust"], ohlcv.index)
        result = aligned.rolling(period).sum()
        result.name = col_name
        return result


@register_feature
class DealerNetBuyRatio(BaseFeature):
    """Dealer net buy as a ratio of volume.

    Dealers include proprietary trading desks. Their positioning often
    reflects hedging activity or directional bets.
    """

    name = "dealer_net_buy_ratio"
    description = "Dealer net buy / volume ratio"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        institutional_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "dealer_net_buy_ratio"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if institutional_data is None or institutional_data.empty:
            return _nan_series(ohlcv.index, col_name)

        if "dealer" not in institutional_data.columns:
            return _nan_series(ohlcv.index, col_name)

        aligned = _align_to_index(institutional_data["dealer"], ohlcv.index)
        result = aligned / ohlcv["volume"]
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result


# --- Extended institutional features ---


@register_feature
class ForeignHoldingChange(BaseFeature):
    """Change in foreign holding ratio (point-in-time daily diff)."""

    name = "foreign_holding_change"
    description = "Daily change in foreign investor holding ratio"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        foreign_holding_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "foreign_holding_change"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if foreign_holding_data is None or foreign_holding_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "foreign_holding_ratio" not in foreign_holding_data.columns:
            return _nan_series(ohlcv.index, col_name)
        ffilled = _ffill_to_index(foreign_holding_data["foreign_holding_ratio"], ohlcv.index)
        result = ffilled.diff()
        result = result.replace([np.inf, -np.inf], np.nan)
        result.name = col_name
        return result


@register_feature
class NetBuyConsecutiveDays(BaseFeature):
    """Consecutive days of positive foreign net buy."""

    name = "net_buy_consecutive_days"
    description = "Consecutive trading days of positive foreign investor net buy"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        institutional_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        col_name = "net_buy_consecutive_days"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)
        if institutional_data is None or institutional_data.empty:
            return _nan_series(ohlcv.index, col_name)
        if "foreign" not in institutional_data.columns:
            return _nan_series(ohlcv.index, col_name)
        aligned = _align_to_index(institutional_data["foreign"], ohlcv.index)
        is_buy = (aligned > 0).astype(int)
        groups = (is_buy != is_buy.shift()).cumsum()
        consecutive = is_buy.groupby(groups).cumsum()
        result = consecutive.astype(float)
        result.name = col_name
        return result
