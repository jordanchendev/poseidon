"""Pydantic config models for TW Futures strategies.

Three strategies share this config module:
- TrendFollowingConfig: EMA crossover + ATR trailing stop (D-09)
- MeanReversionConfig: Bollinger Band + RSI (D-10)
- VolatilityBreakoutConfig: Range breakout + ATR filter (D-11)

Config values loaded from YAML via yaml.safe_load -> Pydantic validation.
"""

from pydantic import BaseModel


class TrendFollowingConfig(BaseModel):
    """EMA crossover trend following on TX daily bars (D-09)."""

    strategy: str = "trend_following_tx"
    name: str = "Trend Following TX Daily"
    market: str = "tw_futures"
    symbol: str = "TX"
    interval: str = "1d"
    ema_fast: int = 20
    ema_slow: int = 60
    atr_period: int = 14
    atr_multiplier: float = 2.0
    lookback_days: int = 120


class MeanReversionConfig(BaseModel):
    """Bollinger Band + RSI mean reversion on TX 1H bars (D-10)."""

    strategy: str = "mean_reversion_tx"
    name: str = "Mean Reversion TX 1H"
    market: str = "tw_futures"
    symbol: str = "TX"
    interval: str = "1h"
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    lookback_days: int = 30


class VolatilityBreakoutConfig(BaseModel):
    """Range breakout with ATR expansion filter on TX 30M bars (D-11)."""

    strategy: str = "volatility_breakout_tx"
    name: str = "Volatility Breakout TX 30M"
    market: str = "tw_futures"
    symbol: str = "TX"
    interval: str = "30m"
    breakout_period: int = 20
    atr_period: int = 14
    atr_sma_period: int = 50
    trail_r: float = 1.5
    lookback_days: int = 14
