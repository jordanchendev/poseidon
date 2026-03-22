"""Base class for all data fetchers."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Abstract base class for market data fetchers.

    All fetchers must return DataFrames with columns:
    time (datetime, UTC, tz-aware), open, high, low, close, volume.
    """

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2  # seconds

    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Symbol identifier (e.g., "2330", "AAPL", "BTC/USDT")
            interval: Candle interval ("1d", "1h", "5m")
            start: Start date as ISO string "YYYY-MM-DD"
            end: End date as ISO string "YYYY-MM-DD"

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
            The 'time' column must be timezone-aware datetime in UTC.
            Returns empty DataFrame if no data available.
        """
        ...

    @abstractmethod
    def list_symbols(self) -> list[str]:
        """List available symbols for this fetcher."""
        ...

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute func with exponential backoff retry on failure.

        Retries up to MAX_RETRIES times with delays of 2s, 4s, 8s.
        """
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                        attempt + 1,
                        self.MAX_RETRIES,
                        func.__name__,
                        str(exc),
                        delay,
                    )
                    time.sleep(delay)
        raise last_exception
