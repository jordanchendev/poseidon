"""Crypto funding rate feature: perpetual vs spot premium signal."""

import numpy as np
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


@register_feature
class FundingRateExtreme(BaseFeature):
    """Funding rate z-score and extreme flag for entry filtering.

    Extreme funding = crowded one-side positioning = higher sweep reversal
    probability.  Addresses Codex Review R-06.
    """

    name = "funding_rate_extreme"
    description = "Funding rate z-score, extreme flag, and direction"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        funding_data: pd.DataFrame | None = None,
        period: int = 20,
        threshold: float = 2.0,
        **kwargs,
    ) -> pd.DataFrame:
        cols = ["funding_zscore", "funding_extreme", "funding_direction"]
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float, columns=cols)

        if funding_data is None or (
            hasattr(funding_data, "empty") and funding_data.empty
        ):
            return pd.DataFrame(
                {c: pd.Series(float("nan"), index=ohlcv.index) for c in cols}
            )

        # Get funding rate series
        fr_col = "funding_rate_daily"
        if fr_col not in funding_data.columns:
            return pd.DataFrame(
                {c: pd.Series(float("nan"), index=ohlcv.index) for c in cols}
            )

        series = funding_data[fr_col]
        # Timezone normalization (same as FundingRateDaily)
        if isinstance(series.index, pd.DatetimeIndex) and isinstance(
            ohlcv.index, pd.DatetimeIndex
        ):
            if series.index.tz is None and ohlcv.index.tz is not None:
                series = series.copy()
                series.index = series.index.tz_localize("UTC")
        aligned = series.reindex(ohlcv.index, method="ffill")

        rolling_mean = aligned.rolling(window=period).mean()
        rolling_std = aligned.rolling(window=period).std()
        zscore = (aligned - rolling_mean) / rolling_std.replace(
            0, float("nan")
        )
        extreme = (zscore.abs() > threshold).astype(float)
        direction = np.sign(aligned)

        return pd.DataFrame(
            {
                "funding_zscore": zscore,
                "funding_extreme": extreme,
                "funding_direction": direction,
            },
            index=ohlcv.index,
        )
