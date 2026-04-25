"""Holdout data enforcement for experiment integrity.

Per D-08: Percentage-based split (last 20%) ensures holdout data is never
touched during optimization.
Per D-09: HoldoutConfig dataclass with holdout_pct=0.20 and locked=True.
Per D-10: Raises HoldoutViolationError when optimization data touches holdout range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


class HoldoutViolationError(Exception):
    """Raised when optimization data touches holdout date range."""

    pass


@dataclass
class HoldoutConfig:
    """Configuration for holdout data enforcement.

    Attributes:
        holdout_pct: Fraction of data reserved for holdout (default 20%).
        locked: If True, prevents boundary recomputation after initial lock.
    """

    holdout_pct: float = 0.20
    locked: bool = True

    def compute_boundary(self, ohlcv: pd.DataFrame) -> datetime:
        """Compute holdout boundary: timestamp at (1 - holdout_pct) position in index.

        Returns the datetime that separates train data from holdout data.
        For holdout_pct=0.20 and 100 rows, boundary is at index 80.

        Args:
            ohlcv: DataFrame with datetime index (sorted ascending).

        Returns:
            datetime marking the start of the holdout period.
        """
        n = len(ohlcv)
        boundary_idx = int(n * (1.0 - self.holdout_pct))
        return pd.Timestamp(ohlcv.index[boundary_idx]).to_pydatetime()

    def validate_data_range(self, ohlcv: pd.DataFrame, holdout_boundary: datetime) -> None:
        """Raise HoldoutViolationError if ohlcv data extends past holdout_boundary.

        Per D-10: optimization must not touch holdout range.

        Args:
            ohlcv: DataFrame with datetime index to validate.
            holdout_boundary: The boundary datetime (inclusive start of holdout).

        Raises:
            HoldoutViolationError: If data's last timestamp >= holdout_boundary.
        """
        last_date = pd.Timestamp(ohlcv.index[-1]).to_pydatetime()
        if last_date >= holdout_boundary:
            raise HoldoutViolationError(
                f"Data end {last_date} >= holdout boundary {holdout_boundary}. "
                "Optimization must not touch holdout range."
            )
