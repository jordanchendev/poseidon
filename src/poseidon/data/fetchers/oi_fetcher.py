"""Open interest history fetcher for crypto perpetual contracts via CCXT."""

import logging
import time as time_module

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# Binance OI history API max records per request
MAX_RECORDS_PER_REQUEST = 500


class OpenInterestFetcher:
    """Fetch open interest history from Binance via CCXT.

    Does NOT inherit BaseFetcher -- OI is not OHLCV data.
    Uses fetchOpenInterestHistory with pagination and retry.
    """

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2

    def __init__(self):
        from poseidon.core.config import settings

        config: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        if settings.binance_api_key:
            config["apiKey"] = settings.binance_api_key
            logger.info("OpenInterestFetcher: using authenticated Binance API")
        else:
            logger.info("OpenInterestFetcher: using public Binance API (no API key)")
        self.exchange = ccxt.binance(config)

    def fetch_oi(
        self,
        symbol: str,
        interval: str = "1h",
        start: str = "",
        end: str = "",
    ) -> pd.DataFrame:
        """Fetch OI history for a perpetual contract symbol.

        Args:
            symbol: CCXT perp format (e.g. "BTC/USDT:USDT").
            interval: OI snapshot interval ("5m", "1h", "4h").
            start: Start date "YYYY-MM-DD".
            end: End date "YYYY-MM-DD" (optional).

        Returns:
            DataFrame with columns [time, open_interest, open_interest_value].
        """
        since_ms = self.exchange.parse8601(f"{start}T00:00:00Z") if start else None
        until_ms = self.exchange.parse8601(f"{end}T23:59:59Z") if end else None

        all_data = self._retry_with_backoff(
            self._fetch_paginated, symbol, interval, since_ms, until_ms
        )

        if not all_data:
            logger.info("No OI data returned for %s (%s)", symbol, interval)
            return pd.DataFrame(columns=["time", "open_interest", "open_interest_value"])

        df = pd.DataFrame(all_data)
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open_interest"] = pd.to_numeric(
            df.get("openInterestAmount", pd.Series(dtype=float)), errors="coerce"
        )
        df["open_interest_value"] = pd.to_numeric(
            df.get("openInterestValue", pd.Series(dtype=float)), errors="coerce"
        )
        df = df[["time", "open_interest", "open_interest_value"]]
        df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)
        return df

    def _fetch_paginated(
        self,
        symbol: str,
        interval: str,
        since_ms: int | None,
        until_ms: int | None,
    ) -> list[dict]:
        """Paginate through OI history, respecting Binance limits."""
        all_data: list[dict] = []
        current_since = since_ms

        while True:
            batch = self.exchange.fetchOpenInterestHistory(
                symbol, interval, since=current_since, limit=MAX_RECORDS_PER_REQUEST
            )
            if not batch:
                break

            all_data.extend(batch)

            last_ts = batch[-1].get("timestamp", 0)
            current_since = last_ts + 1

            if until_ms and current_since >= until_ms:
                break
            if len(batch) < MAX_RECORDS_PER_REQUEST:
                break

        return all_data

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute func with exponential backoff retry on failure."""
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "OI fetch attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt + 1,
                        self.MAX_RETRIES,
                        str(exc),
                        delay,
                    )
                    time_module.sleep(delay)
        raise last_exception
