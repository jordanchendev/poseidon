"""Inter-market lead-lag features.

Captures cross-market relationships such as US-TW overnight gaps,
crypto-equity correlation shifts, and VIX proxy from ATR ratios.
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class OvernightGap(BaseFeature):
    """Overnight gap between a foreign market close and domestic market open.

    Measures the lead-lag relationship where, for example, the US market close
    influences the next-day Taiwan market open.  Companion data is injected by
    FeatureEngine.compute_with_companions().
    """

    name = "overnight_gap"
    description = "Overnight gap between foreign close and domestic open"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        companion_data: pd.DataFrame | None = None,
        companion_symbol: str = "",
        **kwargs,
    ) -> pd.Series:
        col_name = f"overnight_gap_{companion_symbol}"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        # Graceful fallback when no companion data
        if companion_data is None or companion_data.empty:
            return pd.Series(float("nan"), index=ohlcv.index, name=col_name)

        if "close" not in companion_data.columns:
            return pd.Series(float("nan"), index=ohlcv.index, name=col_name)

        # Shift companion close forward by 1 day so it aligns with the
        # *next* trading day of the primary symbol.
        companion_close = companion_data["close"].copy()
        companion_close.index = companion_close.index + pd.Timedelta(days=1)

        # Align to primary index with forward-fill for calendar differences
        aligned_close = companion_close.reindex(ohlcv.index, method="ffill")

        gap = (ohlcv["open"] - aligned_close) / aligned_close
        # Handle division by zero
        gap = gap.fillna(0.0)
        gap.name = col_name
        return gap


@register_feature
class CryptoEquityCorrShift(BaseFeature):
    """Rolling correlation change between crypto and equity returns.

    Measures how the correlation between two markets is shifting over time.
    A rising corr_shift means the two markets are becoming more correlated
    (risk-on/risk-off convergence), while falling means divergence.
    """

    name = "corr_shift"
    description = "Change in rolling correlation with companion asset"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        period: int = 20,
        shift_period: int = 5,
        companion_data: pd.DataFrame | None = None,
        companion_symbol: str = "",
        **kwargs,
    ) -> pd.Series:
        col_name = f"corr_shift_{companion_symbol}_{period}"

        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col_name)

        # Graceful fallback when no companion data
        if companion_data is None or companion_data.empty:
            return pd.Series(float("nan"), index=ohlcv.index, name=col_name)

        if "close" not in companion_data.columns:
            return pd.Series(float("nan"), index=ohlcv.index, name=col_name)

        # Align companion to primary index with forward-fill
        aligned_close = companion_data["close"].reindex(ohlcv.index, method="ffill")

        primary_ret = ohlcv["close"].pct_change()
        companion_ret = aligned_close.pct_change()

        rolling_corr = primary_ret.rolling(period).corr(companion_ret)
        corr_shift = rolling_corr - rolling_corr.shift(shift_period)
        corr_shift.name = col_name
        return corr_shift


@register_feature
class VIXProxy(BaseFeature):
    """VIX-like volatility proxy from ATR ratios.

    Uses the ratio of short-term to long-term Average True Range as a proxy
    for implied volatility.  Values > 1 indicate elevated short-term
    volatility (analogous to VIX spikes).  Single-symbol feature -- no
    companion data needed.
    """

    name = "vix_proxy"
    description = "VIX proxy from short/long ATR ratio"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        **kwargs,
    ) -> pd.Series:
        col_name = f"vix_proxy_{short_period}_{long_period}"

        if not self._validate(ohlcv, min_rows=long_period):
            return pd.Series(dtype=float, name=col_name)

        high = ohlcv["high"]
        low = ohlcv["low"]
        prev_close = ohlcv["close"].shift(1)

        # True Range: max of (H-L, |H-prev_close|, |L-prev_close|)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        short_atr = tr.rolling(short_period).mean()
        long_atr = tr.rolling(long_period).mean()

        # Ratio: >1 means short-term vol exceeds long-term (VIX-like spike)
        vix_proxy = short_atr / long_atr
        # Avoid inf from division by zero
        vix_proxy = vix_proxy.replace([np.inf, -np.inf], np.nan)
        vix_proxy.name = col_name
        return vix_proxy
