"""TrendFollowingStrategy -- EMA crossover + ATR trailing stop for TX daily.

Signal logic (D-09):
  - EMA(20) crosses above EMA(60) -> LONG entry
  - EMA(20) crosses below EMA(60) -> SHORT entry
  - ATR(14) * 2.0 trailing stop -> HOLD signal with updated_stop_loss metadata

Stateful: tracks _position_side and _trailing_stop across evaluate() calls.
BacktestRunner reads HOLD signal metadata to update portfolio stop-loss price.
"""

import logging
import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy
from poseidon.strategies.tw_futures.configs import TrendFollowingConfig

logger = logging.getLogger(__name__)


@register_strategy
class TrendFollowingStrategy(BaseStrategy):
    """EMA crossover trend following with ATR trailing stop for TW futures."""

    name = "trend_following_tx"
    strategy_type = StrategyType.RULE
    stateful = True
    market = "tw_futures"
    interval = "1d"
    supports_backtest = True
    supports_live = False

    def __init__(
        self,
        *,
        config: TrendFollowingConfig | dict,
        instrument: InstrumentType = InstrumentType.FUTURES,
        strategy_id: UUID | None = None,
    ):
        if isinstance(config, dict):
            config = TrendFollowingConfig(**config)
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
        """Evaluate EMA crossover and ATR trailing stop on latest bar.

        Args:
            features: DataFrame with ema_{fast}, ema_{slow}, atr_{period}
                      columns computed by FeatureEngine.

        Returns:
            List of Signal objects (LONG/SHORT/CLOSE/HOLD).
        """
        if features.empty:
            return []

        row = features.iloc[-1]

        # Extract feature columns
        ema_fast_col = f"ema_{self.config.ema_fast}"
        ema_slow_col = f"ema_{self.config.ema_slow}"
        atr_col = f"atr_{self.config.atr_period}"

        ema_fast = row.get(ema_fast_col)
        ema_slow = row.get(ema_slow_col)
        atr = row.get(atr_col)

        # NaN guard
        if ema_fast is None or ema_slow is None or atr is None:
            return []
        if math.isnan(ema_fast) or math.isnan(ema_slow) or math.isnan(atr):
            return []

        close = row["close"]
        signal_time = self._extract_signal_time(features)
        signals: list[Signal] = []

        # --- Entry logic: EMA crossover (D-09) ---
        if ema_fast > ema_slow and self._position_side != "long":
            # Close existing short position first
            if self._position_side == "short":
                signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
                self._reset_state()

            # LONG entry
            stop_loss = close - self.config.atr_multiplier * atr
            signals.append(
                self._make_signal(
                    SignalAction.LONG, close, signal_time, stop_loss_price=stop_loss
                )
            )
            self._position_side = "long"
            self._trailing_stop = stop_loss

        elif ema_fast < ema_slow and self._position_side != "short":
            # Close existing long position first
            if self._position_side == "long":
                signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
                self._reset_state()

            # SHORT entry
            stop_loss = close + self.config.atr_multiplier * atr
            signals.append(
                self._make_signal(
                    SignalAction.SHORT, close, signal_time, stop_loss_price=stop_loss
                )
            )
            self._position_side = "short"
            self._trailing_stop = stop_loss

        # --- Exit logic: ATR trailing stop via HOLD signal ---
        elif self._position_side == "long" and self._trailing_stop is not None:
            new_sl = close - self.config.atr_multiplier * atr
            new_sl = max(new_sl, self._trailing_stop)  # Only tighten, never loosen
            self._trailing_stop = new_sl
            signals.append(
                self._make_signal(
                    SignalAction.HOLD,
                    close,
                    signal_time,
                    metadata={"updated_stop_loss": new_sl},
                )
            )

        elif self._position_side == "short" and self._trailing_stop is not None:
            new_sl = close + self.config.atr_multiplier * atr
            new_sl = min(new_sl, self._trailing_stop)  # Only tighten, never loosen
            self._trailing_stop = new_sl
            signals.append(
                self._make_signal(
                    SignalAction.HOLD,
                    close,
                    signal_time,
                    metadata={"updated_stop_loss": new_sl},
                )
            )

        return signals

    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        if self.config.ema_fast <= 0:
            raise ValueError("ema_fast must be positive")
        if self.config.ema_slow <= self.config.ema_fast:
            raise ValueError("ema_slow must be greater than ema_fast")
        if self.config.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.config.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")
        return True

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        """Return feature specs required by FeatureEngine.

        Returns:
            List of (feature_name, params) tuples.
        """
        return [
            ("ema", {"period": self.config.ema_fast}),
            ("ema", {"period": self.config.ema_slow}),
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
        return datetime.now(timezone.utc)
