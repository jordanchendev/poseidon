"""CCXT fetcher for crypto spot data (Binance)."""

import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd

from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# Max candles per CCXT request
MAX_CANDLES_PER_REQUEST = 1000


class CCXTFetcher(BaseFetcher):
    """Fetcher for crypto spot data from Binance via CCXT.

    Uses synchronous CCXT client (not async) because Celery tasks are synchronous.
    If POSEIDON_BINANCE_API_KEY is set, uses authenticated client (higher rate limits).
    Falls back to public API otherwise.
    """

    def __init__(self):
        from poseidon.core.config import settings

        config: dict = {"enableRateLimit": True}
        if settings.binance_api_key:
            config["apiKey"] = settings.binance_api_key
            logger.info("CCXTFetcher: using authenticated Binance API (read-only)")
        else:
            logger.info("CCXTFetcher: using public Binance API (no API key)")
        self.exchange = ccxt.binance(config)

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV from Binance via CCXT with automatic pagination.

        Args:
            symbol: CCXT symbol format (e.g., "BTC/USDT")
            interval: "1d" or "1h"
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: time (UTC tz-aware), open, high, low, close, volume
        """
        timeframe = interval  # CCXT uses "1h", "1d" directly

        since_ms = self.exchange.parse8601(f"{start}T00:00:00Z")
        until_ms = self.exchange.parse8601(f"{end}T23:59:59Z")

        all_data = self._retry_with_backoff(
            self._fetch_paginated, symbol, timeframe, since_ms, until_ms
        )

        if not all_data:
            logger.info("No data returned from CCXT for %s (%s - %s)", symbol, start, end)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        # CCXT returns: [[timestamp_ms, open, high, low, close, volume], ...]
        df = pd.DataFrame(all_data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

        # Convert millisecond timestamps to timezone-aware UTC datetimes
        df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp_ms"])

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Select final columns in correct order
        df = df[["time", "open", "high", "low", "close", "volume"]]

        # Remove duplicates (pagination overlap)
        df = df.drop_duplicates(subset=["time"], keep="last")
        df = df.sort_values("time").reset_index(drop=True)

        return df

    def list_symbols(self) -> list[str]:
        """List available trading pairs on Binance."""
        self.exchange.load_markets()
        return list(self.exchange.markets.keys())

    def _fetch_paginated(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> list[list]:
        """Paginate through historical kline data.

        CCXT returns max 1000 candles per request. For longer ranges,
        we paginate by advancing the 'since' parameter.
        """
        all_data = []
        current_since = since_ms

        while current_since < until_ms:
            batch = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=current_since, limit=MAX_CANDLES_PER_REQUEST
            )
            if not batch:
                break

            all_data.extend(batch)

            # Advance past the last candle timestamp (+1ms to avoid overlap)
            last_timestamp = batch[-1][0]
            current_since = last_timestamp + 1

            # Safety: if we got fewer than limit, we've reached the end
            if len(batch) < MAX_CANDLES_PER_REQUEST:
                break

        return all_data
