"""Signal-driven long/cash strategy for a one-instrument Qlib backtest."""

from __future__ import annotations

import pandas as pd
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


class SingleInstrumentLongCashStrategy(WeightStrategyBase):
    """Hold one instrument at a fixed weight only when its prior score is positive.

    ``WeightStrategyBase`` obtains ``score`` using the preceding trade-calendar
    step, so this class deliberately has no current-bar data access.
    """

    def __init__(self, instrument: str = "TX", long_weight: float = 0.95, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.instrument = instrument
        self.long_weight = long_weight
        self.threshold = threshold

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """Return 95% long for positive prior score, otherwise all cash."""
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        value = score.get(self.instrument)
        if value is None or pd.isna(value) or float(value) <= self.threshold:
            return {}
        return {self.instrument: self.long_weight}
