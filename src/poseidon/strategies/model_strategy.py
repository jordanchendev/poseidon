"""ModelStrategy — wraps a BaseModel and converts predictions to Signals.

Data flow:
    FeatureEngine.compute() -> pd.DataFrame (OHLCV + features)
        -> BaseModel.predict(features) -> pd.DataFrame (prediction, confidence)
            -> ModelStrategy.evaluate() -> list[Signal]

Hold predictions are filtered out. Only actionable signals (long/short/close) are emitted.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pandas as pd

from poseidon.ml.base import BaseModel as MLBaseModel
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class ModelStrategy(BaseStrategy):
    """Strategy that wraps an ML model and converts predictions to Signals."""

    name = "model_strategy"
    strategy_type = StrategyType.MODEL
    # inherits defaults: supports_live=False, supports_backtest=True, stateful=False

    def __init__(
        self,
        *,
        name: str,
        model: MLBaseModel,
        symbol: str,
        market: str,
        interval: str = "1d",
        instrument: InstrumentType = InstrumentType.SPOT,
        strategy_id: UUID | None = None,
    ):
        self.name = name
        self.model = model
        self.symbol = symbol
        self.market = market
        self.interval = interval
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()
        self.model_id: UUID | None = None
        self._current_position: str | None = None  # Track: "long", "short", or None

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Call model.predict() on the last row and convert to entry/exit signals.

        Tracks current position to avoid duplicate entries and generates
        'close' signals when direction changes:
        - No position + long prediction → emit long
        - No position + short prediction → emit short
        - In long + short prediction → emit close (reverse next bar)
        - In long + long prediction → no signal (already in position)
        - In position + hold prediction → no signal
        """
        if features.empty:
            return []

        # Pass the full feature history — Transformer models need a lookback
        # window of multiple rows, not just the last bar. XGBoost handles
        # multi-row input fine too (predict returns one row per input row).
        predictions = self.model.predict(features)

        # Only act on the last prediction (current bar)
        last_pred = predictions.iloc[[-1]]

        signals: list[Signal] = []
        for idx, row in last_pred.iterrows():
            action_str = row["prediction"]
            confidence = float(row["confidence"])

            signal_time = (
                idx
                if isinstance(idx, datetime)
                else datetime.now(timezone.utc)
            )

            if action_str == "hold":
                continue

            # Determine actual signal based on current position
            if self._current_position is None:
                # No position → enter
                self._current_position = action_str
                signals.append(self._make_signal(
                    SignalAction(action_str), confidence, signal_time,
                ))
            elif self._current_position != action_str:
                # Direction changed → close first
                signals.append(self._make_signal(
                    SignalAction.CLOSE, confidence, signal_time,
                ))
                # Then enter new direction (if not just closing)
                if action_str in ("long", "short"):
                    self._current_position = action_str
                    signals.append(self._make_signal(
                        SignalAction(action_str), confidence, signal_time,
                    ))
                else:
                    self._current_position = None
            # else: same direction, already in position → no signal

        logger.info(
            "ModelStrategy '%s' pos=%s -> %d signals",
            self.name,
            self._current_position,
            len(signals),
        )
        return signals

    def _make_signal(self, action: SignalAction, confidence: float, signal_time: datetime) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            model_id=self.model_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=action,
            confidence=confidence,
            quantity_pct=None if action != SignalAction.CLOSE else 1.0,
            interval=self.interval,
            signal_time=signal_time,
        )

    def validate_config(self) -> bool:
        if self.model is None:
            raise ValueError("ModelStrategy requires a model instance")
        if not self.symbol:
            raise ValueError("ModelStrategy requires a symbol")
        if not self.market:
            raise ValueError("ModelStrategy requires a market")
        return True
