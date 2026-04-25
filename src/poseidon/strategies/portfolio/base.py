"""PortfolioStrategy ABC -- universe-wide stock selection strategies.

Unlike BaseStrategy (single symbol -> Signal), PortfolioStrategy
receives the full universe and returns a target portfolio.

This is a separate hierarchy -- it does NOT extend BaseStrategy.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from poseidon.strategies.portfolio.schemas import TargetPosition


class PortfolioStrategy(ABC):
    """ABC for universe-wide stock selection strategies.

    Unlike BaseStrategy (single symbol -> Signal), PortfolioStrategy
    receives the full universe and returns a target portfolio.

    This is a separate hierarchy -- it does NOT extend BaseStrategy.
    """

    name: str = ""

    # Capability metadata (Phase 34)
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    @abstractmethod
    def select_stocks(self, universe_df: pd.DataFrame, as_of: date | None = None) -> list[TargetPosition]:
        """Select stocks from the full universe.

        Args:
            universe_df: Wide DataFrame with all symbols as columns,
                         date index. Contains price/volume/revenue data.
            as_of: Point-in-time date for selection. Defaults to latest.

        Returns:
            List of TargetPosition with symbol and weight.
        """
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate strategy configuration is complete."""
        ...
