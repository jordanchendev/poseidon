"""MeanReversionStrategy -- Bollinger Band + RSI for TX 1h bars.

Signal logic (D-10):
  - close < BB_lower AND RSI < oversold -> LONG entry
  - close > BB_upper AND RSI > overbought -> SHORT entry
  - Price returns to BB middle band -> CLOSE exit

Stateful: tracks _position_side across evaluate() calls to avoid double-entry
and correctly handle direction reversals (CLOSE then new entry).
"""

import logging
import math
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy
from poseidon.strategies.tw_futures.configs import MeanReversionConfig

logger = logging.getLogger(__name__)


@register_strategy
class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band + RSI mean reversion for TW futures."""

    name = "mean_reversion_tx"
    strategy_type = StrategyType.RULE
    stateful = True
    market = "tw_futures"
    interval = "1h"
    supports_backtest = True
    supports_live = False

    def __init__(
        self,
        *,
        config: MeanReversionConfig | dict,
        instrument: InstrumentType = InstrumentType.FUTURES,
        strategy_id: UUID | None = None,
    ):
        if isinstance(config, dict):
            config = MeanReversionConfig(**config)
        self.config = config
        self.name = config.name
        self.symbol = config.symbol
        self.market = config.market
        self.interval = config.interval
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()

        # Stateful tracking (Pitfall 5)
        self._position_side: str | None = None

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate BB + RSI conditions on latest bar.

        Args:
            features: DataFrame with bb_upper_{period}, bb_middle_{period},
                      bb_lower_{period}, rsi_{period} columns from FeatureEngine.

        Returns:
            List of Signal objects (LONG/SHORT/CLOSE).
        """
        if features.empty:
            return []

        row = features.iloc[-1]

        # Extract feature columns
        bb_upper_col = f"bb_upper_{self.config.bb_period}"
        bb_middle_col = f"bb_middle_{self.config.bb_period}"
        bb_lower_col = f"bb_lower_{self.config.bb_period}"
        rsi_col = f"rsi_{self.config.rsi_period}"

        bb_upper = row.get(bb_upper_col)
        bb_middle = row.get(bb_middle_col)
        bb_lower = row.get(bb_lower_col)
        rsi = row.get(rsi_col)

        # NaN guard
        if bb_upper is None or bb_middle is None or bb_lower is None or rsi is None:
            return []
        if math.isnan(bb_upper) or math.isnan(bb_middle) or math.isnan(bb_lower) or math.isnan(rsi):
            return []

        close = row["close"]
        if math.isnan(close):
            return []

        signal_time = self._extract_signal_time(features)
        signals: list[Signal] = []

        # --- Exit logic: price returns to BB middle band (D-10) ---
        if self._position_side == "long" and close >= bb_middle:
            signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
            self._reset_state()
            return signals

        if self._position_side == "short" and close <= bb_middle:
            signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
            self._reset_state()
            return signals

        # --- Entry logic: BB touch + RSI extreme (D-10) ---
        if close < bb_lower and rsi < self.config.rsi_oversold:
            if self._position_side != "long":
                # Close existing short first
                if self._position_side == "short":
                    signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
                    self._reset_state()

                # LONG entry
                signals.append(self._make_signal(SignalAction.LONG, close, signal_time))
                self._position_side = "long"

        elif close > bb_upper and rsi > self.config.rsi_overbought and self._position_side != "short":
            # Close existing long first
            if self._position_side == "long":
                signals.append(self._make_signal(SignalAction.CLOSE, close, signal_time))
                self._reset_state()

            # SHORT entry
            signals.append(self._make_signal(SignalAction.SHORT, close, signal_time))
            self._position_side = "short"

        return signals

    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        if self.config.bb_period <= 0:
            raise ValueError("bb_period must be positive")
        if self.config.bb_std <= 0:
            raise ValueError("bb_std must be positive")
        if self.config.rsi_period <= 0:
            raise ValueError("rsi_period must be positive")
        if not (0 < self.config.rsi_oversold < self.config.rsi_overbought < 100):
            raise ValueError("rsi_oversold and rsi_overbought must satisfy 0 < rsi_oversold < rsi_overbought < 100")
        return True

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        """Return feature specs required by FeatureEngine.

        Returns:
            List of (feature_name, params) tuples.
        """
        return [
            ("bollinger", {"period": self.config.bb_period, "num_std": self.config.bb_std}),
            ("rsi", {"period": self.config.rsi_period}),
        ]

    def _reset_state(self) -> None:
        """Reset position tracking state."""
        self._position_side = None

    def _make_signal(
        self,
        action: SignalAction,
        close: float,
        signal_time: datetime,
        *,
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
