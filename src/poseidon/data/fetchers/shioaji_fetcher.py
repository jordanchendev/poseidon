"""Shioaji-backed market data fetcher for Taiwan stocks and futures.

Supports sub-daily intervals (1m, 5m, 15m, 30m, 1h) by fetching 1-min kbars
from Shioaji and resampling to the target interval. Also supports 1d via
session-level aggregation.

Shioaji kbars API limitation: max ~180 days per request. This fetcher
automatically chunks long date ranges into 180-day windows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from poseidon.core.config import settings
from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

_EMPTY_OHLCV = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

# Shioaji kbars max date range per request (~6 months)
_MAX_CHUNK_DAYS = 180

_SUPPORTED_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "1d"}

# Resample rules for pandas: interval -> freq string
_RESAMPLE_FREQ = {
    "1m": None,  # no resample needed (raw data)
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
}


class ShioajiFetcher(BaseFetcher):
    """Fetch Taiwan stock and futures OHLCV via Shioaji historical K-bars."""

    def __init__(
        self,
        market: str = "tw_stock",
        api_key: str | None = None,
        secret_key: str | None = None,
        simulation: bool | None = None,
    ) -> None:
        self._market = market
        self._api_key = api_key or settings.shioaji_data_api_key
        self._secret_key = secret_key or settings.shioaji_data_secret_key
        self._simulation = (
            settings.shioaji_data_simulation if simulation is None else simulation
        )
        self._api = None

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV for a Taiwan stock or futures symbol.

        Fetches 1-min kbars from Shioaji and resamples to the target interval.
        Automatically chunks requests for date ranges > 180 days.
        """
        if interval not in _SUPPORTED_INTERVALS:
            logger.warning(
                "ShioajiFetcher does not support interval %s. Supported: %s",
                interval,
                _SUPPORTED_INTERVALS,
            )
            return _EMPTY_OHLCV.copy()

        api = self._ensure_api()
        contract = self._resolve_contract(api, symbol)

        # Chunk long date ranges into 180-day windows
        all_chunks: list[pd.DataFrame] = []
        chunk_start = datetime.fromisoformat(start) if isinstance(start, str) else start
        chunk_end = datetime.fromisoformat(end) if isinstance(end, str) else end

        while chunk_start < chunk_end:
            window_end = min(chunk_start + timedelta(days=_MAX_CHUNK_DAYS), chunk_end)
            start_str = chunk_start.strftime("%Y-%m-%d")
            end_str = window_end.strftime("%Y-%m-%d")

            try:
                kbars = self._retry_with_backoff(
                    api.kbars,
                    contract=contract,
                    start=start_str,
                    end=end_str,
                )
                raw = pd.DataFrame({**kbars})
                if not raw.empty:
                    all_chunks.append(raw)
            except Exception:
                logger.warning(
                    "Failed to fetch kbars for %s (%s - %s), skipping chunk",
                    symbol,
                    start_str,
                    end_str,
                    exc_info=True,
                )

            chunk_start = window_end + timedelta(days=1)

        if not all_chunks:
            logger.info("No kbars returned from Shioaji for %s (%s - %s)", symbol, start, end)
            return _EMPTY_OHLCV.copy()

        raw = pd.concat(all_chunks, ignore_index=True)
        df = self._normalize_columns(raw)

        if interval == "1d":
            return self._resample_daily(df)
        return self._resample_intraday(df, interval)

    def list_symbols(self) -> list[str]:
        """Use ``config/symbols.yaml`` as the source of truth."""
        return []

    # ── Resampling ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize Shioaji kbar columns to canonical names."""
        df = raw.rename(
            columns={
                "ts": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("time").reset_index(drop=True)

    @staticmethod
    def _resample_intraday(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample 1-min bars to the target intraday interval."""
        freq = _RESAMPLE_FREQ.get(interval)
        if freq is None:
            # 1m — return as-is
            return df[["time", "open", "high", "low", "close", "volume"]]

        resampled = (
            df.set_index("time")
            .resample(freq)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .dropna(subset=["open"])
            .reset_index()
        )
        return resampled[["time", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-min bars into daily OHLCV, anchored to TW close."""
        local_time = df["time"].dt.tz_convert("Asia/Taipei")
        df = df.copy()
        df["session_date"] = local_time.dt.normalize()
        daily = (
            df.sort_values("time")
            .groupby("session_date", as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
        )
        daily["time"] = daily["session_date"] + pd.Timedelta(hours=13, minutes=30)
        daily["time"] = daily["time"].dt.tz_convert("UTC")
        return daily[["time", "open", "high", "low", "close", "volume"]]

    # ── Contract resolution ─────────────────────────────────────────

    def _resolve_contract(self, api, symbol: str):
        """Resolve a Shioaji contract for either stocks or futures."""
        if self._market == "tw_futures":
            return self._resolve_futures_contract(api, symbol)
        return self._resolve_stock_contract(api, symbol)

    @staticmethod
    def _resolve_stock_contract(api, symbol: str):
        stocks = api.Contracts.Stocks
        try:
            return stocks[symbol]
        except Exception:
            pass
        for exchange in ("TSE", "OTC", "OES"):
            market_contracts = getattr(stocks, exchange, None)
            if market_contracts is None:
                continue
            try:
                return market_contracts[symbol]
            except Exception:
                continue
        raise KeyError(f"Could not resolve Shioaji stock contract for symbol: {symbol}")

    @staticmethod
    def _resolve_futures_contract(api, symbol: str):
        """Resolve a futures contract by product code (e.g. TX, MTX).

        Uses the first available contract (near-month) for the given product.
        """
        futures = api.Contracts.Futures
        product = getattr(futures, symbol, None)
        if product is not None:
            # Get the first (near-month) contract
            contracts = list(product)
            if contracts:
                return contracts[0]
        raise KeyError(f"Could not resolve Shioaji futures contract for symbol: {symbol}")

    # ── API lifecycle ───────────────────────────────────────────────

    def _ensure_api(self):
        if self._api is not None:
            return self._api

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Shioaji data credentials are not configured. "
                "Set POSEIDON_SHIOAJI_DATA_API_KEY and POSEIDON_SHIOAJI_DATA_SECRET_KEY."
            )

        import shioaji as sj

        api = sj.Shioaji(simulation=self._simulation)
        api.login(api_key=self._api_key, secret_key=self._secret_key)
        self._api = api
        return api
