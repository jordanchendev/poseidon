"""Volume profile features: VWAP deviation, volume-at-price percentile, buy/sell volume ratio."""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class VWAPDeviation(BaseFeature):
    """Normalized deviation of close price from Volume-Weighted Average Price."""

    name = "vwap_dev"
    description = "Normalized deviation from VWAP"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        col_name = f"vwap_dev_{period}"
        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col_name)

        typical_price = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        tp_vol = typical_price * ohlcv["volume"]

        rolling_tp_vol = tp_vol.rolling(window=period).sum()
        rolling_vol = ohlcv["volume"].rolling(window=period).sum()

        vwap = rolling_tp_vol / rolling_vol
        result = (ohlcv["close"] - vwap) / vwap
        result.name = col_name
        return result


@register_feature
class VolumeAtPricePercentile(BaseFeature):
    """Current price position in recent volume distribution (volume-at-price percentile)."""

    name = "vap_pctile"
    description = "Volume-at-price percentile of current close"

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 20, n_bins: int = 10, **kwargs
    ) -> pd.Series:
        col_name = f"vap_pctile_{period}"
        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col_name)

        close = ohlcv["close"].values
        volume = ohlcv["volume"].values
        n = len(close)
        result = np.full(n, np.nan)

        for i in range(period - 1, n):
            window_close = close[i - period + 1 : i + 1]
            window_vol = volume[i - period + 1 : i + 1]

            price_min = window_close.min()
            price_max = window_close.max()
            if price_max == price_min:
                result[i] = 1.0 / n_bins
                continue

            # Bin each bar's close price and accumulate volume per bin
            bin_edges = np.linspace(price_min, price_max, n_bins + 1)
            # np.digitize returns 1-based bin indices; clamp to [1, n_bins]
            bin_idx = np.clip(np.digitize(window_close, bin_edges[1:-1]), 0, n_bins - 1)
            vol_per_bin = np.bincount(bin_idx, weights=window_vol, minlength=n_bins)

            total_vol = vol_per_bin.sum()
            if total_vol == 0:
                result[i] = 0.0
                continue

            current_bin = np.clip(
                np.digitize([close[i]], bin_edges[1:-1])[0], 0, n_bins - 1
            )
            result[i] = vol_per_bin[current_bin] / total_vol

        series = pd.Series(result, index=ohlcv.index, name=col_name)
        return series


@register_feature
class BuySellVolumeRatio(BaseFeature):
    """Rolling buy/sell volume ratio approximated from close vs open direction."""

    name = "bs_vol_ratio"
    description = "Buy/sell volume ratio (0=all sell, 0.5=balanced, 1=all buy)"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        col_name = f"bs_vol_ratio_{period}"
        if not self._validate(ohlcv, min_rows=period):
            return pd.Series(dtype=float, name=col_name)

        price_range = ohlcv["high"] - ohlcv["low"]
        buy_fraction = (ohlcv["close"] - ohlcv["low"]) / price_range
        # Handle zero range (high == low) by defaulting to 0.5
        buy_fraction = buy_fraction.fillna(0.5)

        buy_volume = ohlcv["volume"] * buy_fraction
        total_volume = ohlcv["volume"]

        rolling_buy = buy_volume.rolling(window=period).sum()
        rolling_total = total_volume.rolling(window=period).sum()

        result = rolling_buy / rolling_total
        result.name = col_name
        return result
