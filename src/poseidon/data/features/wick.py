"""Wick and range features for liquidity sweep detection."""

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class WickRatio(BaseFeature):
    """Upper/lower wick ratio relative to bar range."""

    name = "wick_ratio"
    description = "Wick ratios (upper, lower, total) relative to bar range"

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float)
        high = ohlcv["high"]
        low = ohlcv["low"]
        open_ = ohlcv["open"]
        close = ohlcv["close"]
        bar_range = high - low
        bar_range_safe = bar_range.replace(0, float("nan"))  # avoid div-by-zero

        upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
        body = (close - open_).abs()

        return pd.DataFrame(
            {
                "wick_ratio_upper": upper_wick / bar_range_safe,
                "wick_ratio_lower": lower_wick / bar_range_safe,
                "wick_ratio_total": 1 - body / bar_range_safe,
            },
            index=ohlcv.index,
        )


@register_feature
class RangeExpansion(BaseFeature):
    """Current bar range relative to ATR -- values > 2.0 indicate significant anomaly."""

    name = "range_expansion"
    description = "Bar range as multiple of ATR"

    def compute(self, ohlcv: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
        col = f"range_expansion_{period}"
        if not self._validate(ohlcv, min_rows=period + 1):
            return pd.Series(dtype=float, name=col)
        bar_range = ohlcv["high"] - ohlcv["low"]
        # Compute ATR inline (avoid circular import)
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close_prev = (ohlcv["high"] - ohlcv["close"].shift(1)).abs()
        low_close_prev = (ohlcv["low"] - ohlcv["close"].shift(1)).abs()
        true_range = pd.concat(
            [high_low, high_close_prev, low_close_prev], axis=1
        ).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        result = bar_range / atr.replace(0, float("nan"))
        result.name = col
        return result


@register_feature
class BodyRatio(BaseFeature):
    """Real body as fraction of total range -- tiny body + large range = long wick candle."""

    name = "body_ratio"
    description = "Body-to-range ratio (small = doji/pin bar)"

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.Series:
        col = "body_ratio"
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col)
        body = (ohlcv["close"] - ohlcv["open"]).abs()
        bar_range = ohlcv["high"] - ohlcv["low"] + 1e-10  # avoid div-by-zero
        result = body / bar_range
        result.name = col
        return result
