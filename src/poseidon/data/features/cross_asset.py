"""Cross-asset features: correlation between related instruments."""

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class CrossAssetCorrelation(BaseFeature):
    """Rolling correlation between primary symbol and a companion asset.

    Captures inter-market dynamics (e.g., BTC-ETH co-movement, TSMC-TX futures
    linkage) that single-symbol TA features cannot represent.

    Companion data is injected by FeatureEngine.compute_with_companions(),
    not loaded by the feature itself.
    """

    name = "cross_corr"
    description = "Rolling correlation with companion asset"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        period: int = 20,
        companion_data: pd.DataFrame | None = None,
        companion_symbol: str = "",
        **kwargs,
    ) -> pd.Series:
        """Compute rolling correlation between primary and companion close prices.

        Args:
            ohlcv: Primary symbol OHLCV DataFrame.
            period: Rolling window size for correlation.
            companion_data: Companion symbol DataFrame with at least a 'close' column.
                If None, returns NaN Series (graceful fallback).
            companion_symbol: Identifier for naming the output column.

        Returns:
            Series named 'cross_corr_{companion_symbol}_{period}'.
        """
        col_name = f"cross_corr_{companion_symbol}_{period}"

        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col_name)

        # Graceful fallback: no companion data -> NaN column
        if companion_data is None or companion_data.empty:
            return pd.Series(float("nan"), index=ohlcv.index, name=col_name)

        # Check companion has 'close' column
        if "close" not in companion_data.columns:
            return pd.Series(float("nan"), index=ohlcv.index, name=col_name)

        primary_ret = ohlcv["close"].pct_change()

        # Align companion to primary index using forward-fill for calendar differences
        aligned_close = companion_data["close"].reindex(ohlcv.index, method="ffill")
        companion_ret = aligned_close.pct_change()

        result = primary_ret.rolling(period).corr(companion_ret)
        result.name = col_name
        return result
