"""BaseStrategy ABC — unified interface for all strategy types.

Both ModelStrategy and RuleStrategy implement this interface.
The backtest engine and signal pipeline treat all strategies equally
through this common interface.
"""

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

from poseidon.signals.schemas import Signal


class StrategyType(str, Enum):
    """Strategy type discriminator."""

    MODEL = "model"
    RULE = "rule"
    VOTING = "voting"


class BaseStrategy(ABC):
    """Abstract base class for all strategies.

    Subclasses must define:
        name: str -- unique strategy identifier
        strategy_type: StrategyType -- model or rule
        symbol: str -- target symbol (e.g., "2330", "BTCUSDT")
        market: str -- target market (e.g., "tw_stock", "crypto_spot")
        interval: str -- candle interval (e.g., "1d", "1h")

    The evaluate() method is the single entry point. It receives a feature
    DataFrame (from FeatureEngine) and returns a list of Signal objects.
    An empty list means no signal is generated for this evaluation.
    """

    name: str = ""
    strategy_type: StrategyType
    symbol: str = ""
    market: str = ""
    interval: str = "1d"

    @abstractmethod
    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate strategy against feature data and produce signals.

        Args:
            features: Wide DataFrame from FeatureEngine containing OHLCV
                      columns plus computed feature columns.

        Returns:
            List of Signal objects. Empty list if no signal triggered.
        """
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the strategy configuration is complete and correct.

        Returns:
            True if the strategy is properly configured and ready to evaluate.

        Raises:
            ValueError: If configuration is invalid.
        """
        ...
