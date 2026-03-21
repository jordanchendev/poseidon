"""Technical indicators: moving averages, RSI, MACD, Bollinger Bands, ATR."""

import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class SMA(BaseFeature):
    """Simple Moving Average."""

    name = "sma"
    description = "Simple Moving Average of close price"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=f"sma_{period}")
        result = ohlcv["close"].rolling(window=period).mean()
        result.name = f"sma_{period}"
        return result


@register_feature
class EMA(BaseFeature):
    """Exponential Moving Average."""

    name = "ema"
    description = "Exponential Moving Average of close price"

    def compute(self, ohlcv: pd.DataFrame, period: int = 20, **kwargs) -> pd.Series:
        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=f"ema_{period}")
        result = ohlcv["close"].ewm(span=period, adjust=False).mean()
        result.name = f"ema_{period}"
        return result


@register_feature
class RSI(BaseFeature):
    """Relative Strength Index (Wilder's smoothing via rolling mean)."""

    name = "rsi"
    description = "Relative Strength Index"

    def compute(self, ohlcv: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
        if not self._validate(ohlcv, min_rows=2):
            return pd.Series(dtype=float, name=f"rsi_{period}")
        delta = ohlcv["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        result = 100 - (100 / (1 + rs))
        result.name = f"rsi_{period}"
        return result


@register_feature
class MACD(BaseFeature):
    """Moving Average Convergence Divergence.

    Returns DataFrame with columns: macd_line, macd_signal, macd_histogram.
    """

    name = "macd"
    description = "MACD line, signal, and histogram"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs,
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(columns=["macd_line", "macd_signal", "macd_histogram"])
        ema_fast = ohlcv["close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = ohlcv["close"].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        return pd.DataFrame({
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
        })


@register_feature
class BollingerBands(BaseFeature):
    """Bollinger Bands (upper, middle, lower).

    Returns DataFrame with 3 columns: bb_upper_{period}, bb_middle_{period}, bb_lower_{period}.
    """

    name = "bollinger"
    description = "Bollinger Bands (upper, middle, lower)"

    def compute(
        self, ohlcv: pd.DataFrame, period: int = 20, num_std: float = 2.0, **kwargs
    ) -> pd.DataFrame:
        if not self._validate(ohlcv):
            return pd.DataFrame(columns=[f"bb_upper_{period}", f"bb_middle_{period}", f"bb_lower_{period}"])
        middle = ohlcv["close"].rolling(window=period).mean()
        std = ohlcv["close"].rolling(window=period).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return pd.DataFrame({
            f"bb_upper_{period}": upper,
            f"bb_middle_{period}": middle,
            f"bb_lower_{period}": lower,
        })


@register_feature
class ATR(BaseFeature):
    """Average True Range."""

    name = "atr"
    description = "Average True Range (volatility based on high/low/close)"

    def compute(self, ohlcv: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
        if not self._validate(ohlcv, min_rows=2):
            return pd.Series(dtype=float, name=f"atr_{period}")
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close_prev = (ohlcv["high"] - ohlcv["close"].shift(1)).abs()
        low_close_prev = (ohlcv["low"] - ohlcv["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        result = true_range.rolling(window=period).mean()
        result.name = f"atr_{period}"
        return result
