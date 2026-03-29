"""Crypto funding rate feature: perpetual vs spot premium signal."""

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class FundingRateDaily(BaseFeature):
    """Daily funding rate for crypto perpetual contracts.

    Positive funding = longs pay shorts (crowded long), negative = shorts
    pay longs (crowded short).  Data is injected via the ``funding_data``
    kwarg by :pymethod:`FeatureEngine.compute_with_companions`.
    """

    name = "funding_rate_daily"
    description = "Daily aggregated crypto perpetual funding rate"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        funding_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """Return funding rate aligned to *ohlcv* index with forward-fill.

        Args:
            ohlcv: Primary symbol OHLCV DataFrame.
            funding_data: DataFrame with a ``funding_rate_daily`` column,
                typically produced by :class:`FundingRateLoader`.
            **kwargs: Ignored extra keyword arguments.

        Returns:
            Series named ``funding_rate_daily``.  NaN when data is missing.
        """
        col = "funding_rate_daily"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col)

        if funding_data is None or (hasattr(funding_data, "empty") and funding_data.empty):
            return pd.Series(float("nan"), index=ohlcv.index, name=col)

        if col not in funding_data.columns:
            return pd.Series(float("nan"), index=ohlcv.index, name=col)

        series = funding_data[col]
        # Normalize timezone: CCXT returns naive datetime, OHLCV uses UTC-aware
        if isinstance(series.index, pd.DatetimeIndex) and isinstance(ohlcv.index, pd.DatetimeIndex):
            if series.index.tz is None and ohlcv.index.tz is not None:
                series = series.copy()
                series.index = series.index.tz_localize("UTC")
        aligned = series.reindex(ohlcv.index, method="ffill")
        aligned.name = col
        return aligned
