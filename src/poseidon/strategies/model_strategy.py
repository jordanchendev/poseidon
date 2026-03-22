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

logger = logging.getLogger(__name__)


class ModelStrategy(BaseStrategy):
    """Strategy that wraps an ML model and converts predictions to Signals."""

    strategy_type = StrategyType.MODEL

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

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Call model.predict() on the last row and convert to Signal if actionable."""
        if features.empty:
            return []

        # Only predict the last row — same pattern as RuleStrategy
        last_row = features.iloc[[-1]]
        predictions = self.model.predict(last_row)

        signals: list[Signal] = []
        for idx, row in predictions.iterrows():
            action_str = row["prediction"]
            if action_str == "hold":
                continue

            signal_time = (
                idx
                if isinstance(idx, datetime)
                else datetime.now(timezone.utc)
            )

            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    model_id=self.model_id,
                    symbol=self.symbol,
                    market=self.market,
                    instrument=self.instrument,
                    action=SignalAction(action_str),
                    confidence=float(row["confidence"]),
                    interval=self.interval,
                    signal_time=signal_time,
                )
            )

        logger.info(
            "ModelStrategy '%s' evaluated -> %d signals",
            self.name,
            len(signals),
        )
        return signals

    def validate_config(self) -> bool:
        if self.model is None:
            raise ValueError("ModelStrategy requires a model instance")
        if not self.symbol:
            raise ValueError("ModelStrategy requires a symbol")
        if not self.market:
            raise ValueError("ModelStrategy requires a market")
        return True
