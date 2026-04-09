"""Polygon.io-backed market data fetcher for US stocks.

Uses the Aggregates (Bars) REST API:
  GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}

Free tier: 5 API calls/minute.  For 21 daily US symbols this is plenty.
"""

from __future__ import annotations

import logging
import time as _time

import pandas as pd
import requests

from poseidon.core.config import settings
from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_EMPTY_OHLCV = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


class PolygonFetcher(BaseFetcher):
    """Fetch US stock OHLCV via Polygon.io Aggregates API."""

    # Free tier: 5 req/min — stay well under
    _MIN_REQUEST_INTERVAL = 13  # seconds between calls

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.polygon_api_key
        self._last_request_ts: float = 0.0

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        if interval != "1d":
            logger.warning("PolygonFetcher currently supports only 1d. Requested: %s", interval)
            return _EMPTY_OHLCV.copy()

        if not self._api_key:
            raise ValueError(
                "Polygon API key is not configured. "
                "Set POSEIDON_POLYGON_API_KEY."
            )

        url = (
            f"{_BASE_URL}/v2/aggs/ticker/{symbol}"
            f"/range/1/day/{start}/{end}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self._api_key,
        }

        self._throttle()
        resp = self._retry_with_backoff(requests.get, url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results")
        if not results:
            logger.info("No Polygon data for %s (%s ~ %s)", symbol, start, end)
            return _EMPTY_OHLCV.copy()

        df = pd.DataFrame(results)
        df = df.rename(columns={
            "t": "time",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def list_symbols(self) -> list[str]:
        """Use config/symbols.yaml as source of truth."""
        return []

    def _throttle(self) -> None:
        """Enforce minimum interval between API calls (free tier)."""
        elapsed = _time.monotonic() - self._last_request_ts
        if elapsed < self._MIN_REQUEST_INTERVAL:
            _time.sleep(self._MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_ts = _time.monotonic()
