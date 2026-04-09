"""FinLab fetcher for Taiwan stocks, Taiwan futures, and US stocks."""

import logging
import threading

import pandas as pd

from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

# Thread lock for FinLab's global set_market() state
_finlab_lock = threading.Lock()

# Dataset mapping per market
FINLAB_DATASETS = {
    "tw_stock": {
        "open": "price:開盤價",
        "high": "price:最高價",
        "low": "price:最低價",
        "close": "price:收盤價",
        "volume": "price:成交股數",
    },
    "tw_futures": {
        "open": "futures_price:開盤價",
        "high": "futures_price:最高價",
        "low": "futures_price:最低價",
        "close": "futures_price:收盤價",
        "volume": "futures_price:成交量",
    },
    "us_stock": {
        "open": "us_price:open",
        "high": "us_price:high",
        "low": "us_price:low",
        "close": "us_price:close",
        "volume": "us_price:volume",
    },
}

# FinLab market codes
FINLAB_MARKET_CODES = {
    "tw_stock": "tw",
    "tw_futures": "tw",
    "us_stock": "us",
}

# FinLab futures use "TX一般" format instead of plain "TX".
# Map Poseidon symbol codes to FinLab column names.
FINLAB_FUTURES_SYMBOL_MAP = {
    "TX": "TX一般",
    "MTX": "MTX一般",
    "TE": "TE一般",
    "TF": "TF一般",
}


class FinLabFetcher(BaseFetcher):
    """Fetcher using FinLab API for TW stocks, TW futures, and US stocks.

    FinLab returns wide DataFrames (all symbols at once). We cache the
    downloaded data per session to avoid redundant API calls when
    fetching multiple symbols from the same market.
    """

    def __init__(self, market: str, token: str | None = None):
        if market not in FINLAB_DATASETS:
            raise ValueError(f"FinLabFetcher does not support market: {market}")
        self.market = market
        self._token = token
        self._initialized = False
        # Cache: dataset_key -> wide DataFrame
        self._cache: dict[str, pd.DataFrame] = {}

    def _ensure_login(self):
        """Login to FinLab if not already done."""
        if self._initialized:
            return
        import finlab

        from poseidon.core.config import settings

        token = self._token or settings.finlab_api_token
        if token:
            finlab.login(token)
        self._initialized = True

    def _get_wide_df(self, dataset_key: str) -> pd.DataFrame:
        """Get a wide DataFrame from FinLab, using cache if available."""
        if dataset_key in self._cache:
            return self._cache[dataset_key]

        from finlab import data

        self._ensure_login()

        market_code = FINLAB_MARKET_CODES[self.market]
        with _finlab_lock:
            data.set_market(market_code)
            df = data.get(dataset_key)

        if df is not None and not df.empty:
            self._cache[dataset_key] = df
        else:
            df = pd.DataFrame()

        return df

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV from FinLab.

        Args:
            symbol: Stock code (e.g., "2330", "AAPL", "TX")
            interval: Must be "1d" (FinLab only provides daily data)
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: time (UTC tz-aware), open, high, low, close, volume
        """
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        if interval != "1d":
            logger.warning("FinLab only supports daily data. Requested: %s", interval)
            return empty

        # Map Poseidon symbol to FinLab column name (futures use "TX一般" format)
        finlab_symbol = symbol
        if self.market == "tw_futures":
            finlab_symbol = FINLAB_FUTURES_SYMBOL_MAP.get(symbol, f"{symbol}一般")

        datasets = FINLAB_DATASETS[self.market]
        ohlcv_parts = {}

        for col_name, dataset_key in datasets.items():
            wide_df = self._get_wide_df(dataset_key)
            if wide_df.empty or finlab_symbol not in wide_df.columns:
                logger.warning(
                    "Symbol %s (%s) not found in FinLab dataset %s",
                    symbol, finlab_symbol, dataset_key,
                )
                return empty
            ohlcv_parts[col_name] = wide_df[finlab_symbol]

        # Combine into single DataFrame
        df = pd.DataFrame(ohlcv_parts)
        df.index.name = "time"
        df = df.reset_index()

        # Filter date range
        df["time"] = pd.to_datetime(df["time"])
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]

        if df.empty:
            return empty

        # Ensure timezone-aware UTC
        if df["time"].dt.tz is None:
            # TW market: anchor to market close (13:30 Asia/Taipei)
            if self.market in ("tw_stock", "tw_futures"):
                df["time"] = df["time"] + pd.Timedelta(hours=13, minutes=30)
                df["time"] = df["time"].dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")
            else:
                df["time"] = df["time"].dt.tz_localize("UTC")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop NaN rows and select final columns
        df = df[["time", "open", "high", "low", "close", "volume"]].dropna()

        return df

    def list_symbols(self) -> list[str]:
        """List available symbols by checking the close price dataset."""
        datasets = FINLAB_DATASETS[self.market]
        wide_df = self._get_wide_df(datasets["close"])
        if wide_df.empty:
            return []
        return list(wide_df.columns)

    def clear_cache(self):
        """Clear cached data to free memory."""
        self._cache.clear()
