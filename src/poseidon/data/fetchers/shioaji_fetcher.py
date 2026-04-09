"""Shioaji-backed market data fetcher for Taiwan stocks.

This fetcher is integrated into Poseidon's existing data layer. It lazily
imports ``shioaji`` so the codebase still imports cleanly on machines where the
SDK is unavailable. The fetch path is intentionally read-only: it uses a
dedicated data credential pair and defaults to ``simulation=True`` so Poseidon
can ingest market data without requiring an execution-capable API key.
"""

from __future__ import annotations

import logging

import pandas as pd

from poseidon.core.config import settings
from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

_EMPTY_OHLCV = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])


class ShioajiFetcher(BaseFetcher):
    """Fetch Taiwan stock OHLCV via Shioaji historical K-bars."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        simulation: bool | None = None,
    ) -> None:
        self._api_key = api_key or settings.shioaji_data_api_key
        self._secret_key = secret_key or settings.shioaji_data_secret_key
        self._simulation = (
            settings.shioaji_data_simulation if simulation is None else simulation
        )
        self._api = None

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV for a Taiwan stock symbol.

        Shioaji's historical ``kbars`` API returns intraday bars. Poseidon
        stores TW stocks as daily bars today, so this fetcher resamples the
        intraday response into a canonical 1d OHLCV frame anchored to the
        Taiwan cash-session close (13:30 Asia/Taipei).
        """
        if interval != "1d":
            logger.warning("ShioajiFetcher currently supports only 1d. Requested: %s", interval)
            return _EMPTY_OHLCV.copy()

        api = self._ensure_api()
        contract = self._resolve_stock_contract(api, symbol)
        kbars = self._retry_with_backoff(
            api.kbars,
            contract=contract,
            start=start,
            end=end,
        )
        raw = pd.DataFrame({**kbars})
        if raw.empty:
            logger.info("No kbars returned from Shioaji for %s (%s - %s)", symbol, start, end)
            return _EMPTY_OHLCV.copy()

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

        local_time = df["time"].dt.tz_convert("Asia/Taipei")
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

    def list_symbols(self) -> list[str]:
        """Use ``config/symbols.yaml`` as the source of truth."""
        return []

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
