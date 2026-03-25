"""FinMind fetcher for Taiwan stocks and futures."""

import logging
import time

import pandas as pd
import requests

from poseidon.core.config import settings
from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# FinMind dataset names per market
FINMIND_DATASETS = {
    "tw_stock": "TaiwanStockPrice",
    "tw_futures": "TaiwanFuturesDaily",
}

# FinMind column name -> canonical OHLCV column name
FINMIND_TW_STOCK_COLUMN_MAP = {
    "date": "time",
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "Trading_Volume": "volume",
}

FINMIND_TW_FUTURES_COLUMN_MAP = {
    "date": "time",
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "volume": "volume",
}

# Delay between requests to be respectful to the free API
REQUEST_DELAY_SECONDS = 1.0


class FinMindFetcher(BaseFetcher):
    """Fetcher for Taiwan stock and futures data from FinMind API."""

    def __init__(self, token: str | None = None, market: str = "tw_stock"):
        self.token = token or settings.finmind_token
        self.market = market
        if market == "tw_stock":
            self.dataset = FINMIND_DATASETS["tw_stock"]
            self.column_map = FINMIND_TW_STOCK_COLUMN_MAP
        elif market == "tw_futures":
            self.dataset = FINMIND_DATASETS["tw_futures"]
            self.column_map = FINMIND_TW_FUTURES_COLUMN_MAP
        else:
            raise ValueError(f"FinMindFetcher does not support market: {market}")

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV from FinMind API.

        Args:
            symbol: Taiwan stock code (e.g., "2330") or futures code (e.g., "TX")
            interval: Must be "1d" (FinMind only provides daily data)
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: time (UTC tz-aware), open, high, low, close, volume
        """
        if interval != "1d":
            logger.warning("FinMind only supports daily data. Requested: %s", interval)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        params = {
            "dataset": self.dataset,
            "data_id": symbol,
            "start_date": start,
            "end_date": end,
            "token": self.token,
        }

        response_json = self._retry_with_backoff(self._make_request, params)

        data = response_json.get("data", [])
        if not data:
            logger.info("No data returned from FinMind for %s (%s - %s)", symbol, start, end)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data)

        # Futures-specific filtering: multiple rows per day (contract months + sessions)
        if self.market == "tw_futures":
            df = self._filter_futures(df)

        # Rename columns to canonical OHLCV names
        df = df.rename(columns=self.column_map)
        df = df[["time", "open", "high", "low", "close", "volume"]]

        # Convert to timezone-aware UTC datetimes
        df["time"] = pd.to_datetime(df["time"])
        if interval == "1d":
            # Daily: anchor to market close (13:30 Asia/Taipei) to avoid date shift
            df["time"] = df["time"] + pd.Timedelta(hours=13, minutes=30)
        df["time"] = df["time"].dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add delay between requests to respect rate limits
        time.sleep(REQUEST_DELAY_SECONDS)

        return df

    @staticmethod
    def _filter_futures(df: pd.DataFrame) -> pd.DataFrame:
        """Filter futures data to front-month contract and a single trading session.

        FinMind TaiwanFuturesDaily returns multiple rows per date:
        - Multiple contract months (contract_date: 202603, 202604, ...)
        - Multiple trading sessions (trading_session: 'after_market', etc.)

        We keep only the front-month (smallest contract_date per date)
        and the after_market session (contains the full-day settlement).
        """
        if "trading_session" in df.columns:
            df = df[df["trading_session"] == "after_market"]

        if "contract_date" in df.columns and "date" in df.columns:
            # Per date, keep only the nearest contract month
            front_month = df.groupby("date")["contract_date"].transform("min")
            df = df[df["contract_date"] == front_month]

        return df

    def list_symbols(self) -> list[str]:
        """List available symbols. Returns empty list (use config/symbols.yaml instead)."""
        return []

    def _make_request(self, params: dict) -> dict:
        """Make HTTP GET request to FinMind API."""
        response = requests.get(FINMIND_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
