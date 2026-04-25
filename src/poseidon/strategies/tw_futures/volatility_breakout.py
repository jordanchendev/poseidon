"""VolatilityBreakoutStrategy -- N-bar breakout + ATR expansion for TX 30m.

Signal logic (D-11):
  - close > highest_high(prev N bars) AND ATR > ATR_SMA -> LONG entry
  - close < lowest_low(prev N bars) AND ATR > ATR_SMA -> SHORT entry
  - R-multiple trailing stop via HOLD signal with updated_stop_loss metadata

Stateful: tracks _position_side and _trailing_stop across evaluate() calls.
BacktestRunner reads HOLD signal metadata to update portfolio stop-loss price.
"""

import logging
import math
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy
from poseidon.strategies.tw_futures.configs import VolatilityBreakoutConfig

logger = logging.getLogger(__name__)


@register_strategy
class VolatilityBreakoutStrategy(BaseStrategy):
    """N-bar breakout with ATR expansion filter for TW futures."""

    name = "volatility_breakout_tx"
    strategy_type = StrategyType.RULE
    stateful = True
    market = "tw_futures"
    interval = "30m"
    supports_backtest = True
    supports_live = False

    def __init__(
        self,
        *,
        config: VolatilityBreakoutConfig | dict,
        instrument: InstrumentType = InstrumentType.FUTURES,
        strategy_id: UUID | None = None,
    ):
        if isinstance(config, dict):
            config = VolatilityBreakoutConfig(**config)
        self.config = config
        self.name = config.name
        self.symbol = config.symbol
        self.market = config.market
        self.interval = config.interval
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()

        # Stateful tracking (Pitfall 5)
        self._position_side: str | None = None
        self._trailing_stop: float | None = None

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate N-bar breakout + ATR expansion on latest bar.

        Args:
            features: DataFrame with atr_{period} column from FeatureEngine,
                      plus raw OHLCV (high, low, close) for breakout calc.

        Returns:
            List of Signal objects (LONG/SHORT/CLOSE/HOLD).
        """
        if features.empty:
            return []

        row = features.iloc[-1]
        atr_col = f"atr_{self.config.atr_period}"

        atr = row.get(atr_col)
        close = row.get("close")

        # NaN guard on close and ATR
        if atr is None or close is None:
            return []
        if math.isnan(atr) or math.isnan(close):
            return []

        # Compute highest_high / lowest_low of PREVIOUS N bars (shift(1) excludes current)
        shifted_high = features["high"].shift(1)
        shifted_low = features["low"].shift(1)
        high_n = shifted_high.rolling(self.config.breakout_period).max().iloc[-1]
        low_n = shifted_low.rolling(self.config.breakout_period).min().iloc[-1]

        # ATR SMA for expansion check
        atr_series = features[atr_col]
        atr_sma = atr_series.rolling(self.config.atr_sma_period).mean().iloc[-1]

        # NaN guard on computed values
        if math.isnan(high_n) or math.isnan(low_n) or math.isnan(atr_sma):
            return []

        atr_expanding = atr > atr_sma
        signal_time = self._extract_signal_time(features)
        signals: list[Signal] = []

        # --- Exit logic: R-multiple trailing stop via HOLD signal ---
        if self._position_side == "long" and self._trailing_stop is not None:
            new_sl = close - self.config.trail_r * atr
            new_sl = max(new_sl, self._trailing_stop)  # Only tighten, never loosen
            self._trailing_stop = new_sl

            # Check for new entry in opposite direction
            if close < low_n and atr_expanding:
                # CLOSE long + SHORT entry
                signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
                self._reset_state()

                stop_loss = close + self.config.trail_r * atr
                signals.append(self._make_signal(SignalAction.SHORT, close, signal_time, stop_loss_price=stop_loss))
                self._position_side = "short"
                self._trailing_stop = stop_loss
                return signals

            # Normal HOLD with updated trailing stop
            signals.append(
                self._make_signal(
                    SignalAction.HOLD,
                    close,
                    signal_time,
                    metadata={"updated_stop_loss": new_sl},
                )
            )
            return signals

        if self._position_side == "short" and self._trailing_stop is not None:
            new_sl = close + self.config.trail_r * atr
            new_sl = min(new_sl, self._trailing_stop)  # Only tighten, never loosen
            self._trailing_stop = new_sl

            # Check for new entry in opposite direction
            if close > high_n and atr_expanding:
                # CLOSE short + LONG entry
                signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
                self._reset_state()

                stop_loss = close - self.config.trail_r * atr
                signals.append(self._make_signal(SignalAction.LONG, close, signal_time, stop_loss_price=stop_loss))
                self._position_side = "long"
                self._trailing_stop = stop_loss
                return signals

            # Normal HOLD with updated trailing stop
            signals.append(
                self._make_signal(
                    SignalAction.HOLD,
                    close,
                    signal_time,
                    metadata={"updated_stop_loss": new_sl},
                )
            )
            return signals

        # --- Entry logic: N-bar breakout + ATR expansion (D-11) ---
        if close > high_n and atr_expanding and self._position_side != "long":
            stop_loss = close - self.config.trail_r * atr
            signals.append(self._make_signal(SignalAction.LONG, close, signal_time, stop_loss_price=stop_loss))
            self._position_side = "long"
            self._trailing_stop = stop_loss

        elif close < low_n and atr_expanding and self._position_side != "short":
            stop_loss = close + self.config.trail_r * atr
            signals.append(self._make_signal(SignalAction.SHORT, close, signal_time, stop_loss_price=stop_loss))
            self._position_side = "short"
            self._trailing_stop = stop_loss

        return signals

    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        if self.config.breakout_period <= 0:
            raise ValueError("breakout_period must be positive")
        if self.config.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.config.atr_sma_period <= 0:
            raise ValueError("atr_sma_period must be positive")
        if self.config.trail_r <= 0:
            raise ValueError("trail_r must be positive")
        return True

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        """Return feature specs required by FeatureEngine.

        Returns:
            List of (feature_name, params) tuples.
            Note: highest_high/lowest_low computed from raw OHLCV in evaluate(),
            ATR SMA computed from atr column via rolling mean in evaluate().
        """
        return [
            ("atr", {"period": self.config.atr_period}),
        ]

    def _reset_state(self) -> None:
        """Reset position tracking state."""
        self._position_side = None
        self._trailing_stop = None

    def _make_signal(
        self,
        action: SignalAction,
        close: float,
        signal_time: datetime,
        *,
        stop_loss_price: float | None = None,
        metadata: dict | None = None,
    ) -> Signal:
        """Create a Signal with common fields pre-filled."""
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=action,
            confidence=0.8,
            interval=self.interval,
            signal_time=signal_time,
            stop_loss_price=stop_loss_price,
            metadata=metadata or {},
        )

    @staticmethod
    def _extract_signal_time(features: pd.DataFrame) -> datetime:
        """Extract signal_time from DataFrame index."""
        ts = features.index[-1]
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        return datetime.now(UTC)
