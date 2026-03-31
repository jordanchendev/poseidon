"""CCXT fetcher for crypto perpetual contract OHLCV data (Binance)."""

import logging

import ccxt
import pandas as pd

from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# Max candles per CCXT request
MAX_CANDLES_PER_REQUEST = 1000


class PerpFetcher(BaseFetcher):
    """Fetcher for perpetual contract OHLCV data from Binance via CCXT.

    Creates its own CCXT exchange instance configured for swap/perp markets
    (separate from CCXTFetcher's spot instance). Uses defaultType='swap'
    to route all requests to the futures API.

    Perp OHLCV data is stored in the existing ohlcv table with:
    - market = "crypto_perp"
    - instrument = "perpetual"
    """

    # Metadata for DB storage
    market = "crypto_perp"
    instrument = "perpetual"

    def __init__(self):
        from poseidon.core.config import settings

        config: dict = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",  # Route to futures/perp API
            },
        }
        if settings.binance_api_key:
            config["apiKey"] = settings.binance_api_key
            logger.info("PerpFetcher: using authenticated Binance API (read-only)")
        else:
            logger.info("PerpFetcher: using public Binance API (no API key)")
        self.exchange = ccxt.binance(config)

    def fetch_ohlcv(
        self, symbol: str, interval: str = "4h", start: str = "", end: str = ""
    ) -> pd.DataFrame:
        """Fetch perpetual contract OHLCV from Binance via CCXT with pagination.

        Args:
            symbol: CCXT perp symbol format (e.g., "BTC/USDT:USDT")
            interval: Candle interval, default "4h" for perp trading
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: time (UTC tz-aware), open, high, low, close, volume
        """
        timeframe = interval

        since_ms = self.exchange.parse8601(f"{start}T00:00:00Z")
        until_ms = self.exchange.parse8601(f"{end}T23:59:59Z")

        all_data = self._retry_with_backoff(
            self._fetch_paginated, symbol, timeframe, since_ms, until_ms
        )

        if not all_data:
            logger.info(
                "No perp data returned from CCXT for %s (%s - %s)",
                symbol, start, end,
            )
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "volume"]
            )

        # CCXT returns: [[timestamp_ms, open, high, low, close, volume], ...]
        df = pd.DataFrame(
            all_data,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )

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
        """List available perpetual contract trading pairs on Binance."""
        self.exchange.load_markets()
        return [
            symbol
            for symbol, market in self.exchange.markets.items()
            if market.get("swap", False)
        ]

    def _fetch_paginated(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> list[list]:
        """Paginate through historical perp kline data.

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
