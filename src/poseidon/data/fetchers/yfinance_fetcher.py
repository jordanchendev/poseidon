"""yfinance fetcher for US stocks."""

import logging
import time

import pandas as pd
import yfinance as yf

from poseidon.data.fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)

YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

# Delay between individual symbol requests
REQUEST_DELAY_SECONDS = 2.0


class YFinanceFetcher(BaseFetcher):
    """Fetcher for US stock data from Yahoo Finance via yfinance."""

    def fetch_ohlcv(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "MSFT")
            interval: Must be "1d" for Phase 1
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"

        Returns:
            DataFrame with columns: time (UTC tz-aware), open, high, low, close, volume
        """
        if interval != "1d":
            logger.warning("yfinance Phase 1 only supports daily data. Requested: %s", interval)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = self._retry_with_backoff(self._download, symbol, start, end)

        if df is None or df.empty:
            logger.info("No data returned from yfinance for %s (%s - %s)", symbol, start, end)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        # Rename columns to canonical names
        df = df.rename(columns=YFINANCE_COLUMN_MAP)

        # Handle multi-level columns if present (from group_by="ticker")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Select only the columns we need
        available_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if len(available_cols) < 5:
            logger.warning("Missing columns from yfinance for %s: have %s", symbol, available_cols)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = df[available_cols].copy()

        # Convert DatetimeIndex to 'time' column
        df = df.reset_index()
        # yfinance DatetimeIndex column is named "Date" for daily data
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "time"})

        # Ensure timezone-aware UTC
        # Note: yfinance daily bars return naive datetimes with no sub-day precision.
        # Localizing to UTC (not US/Eastern) avoids introducing a spurious 4-5 hour offset.
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Select final columns in correct order
        df = df[["time", "open", "high", "low", "close", "volume"]]

        # Drop any rows with NaN (holidays, missing data)
        df = df.dropna()

        time.sleep(REQUEST_DELAY_SECONDS)

        return df

    def fetch_ohlcv_bulk(
        self, symbols: list[str], interval: str, start: str, end: str
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple symbols in one call (more efficient).

        Returns dict mapping symbol -> DataFrame.
        """
        raw = self._retry_with_backoff(
            self._download_bulk, symbols, start, end
        )
        if raw is None or raw.empty:
            return {s: pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"]) for s in symbols}

        result = {}
        for symbol in symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    sdf = raw.xs(symbol, axis=1, level=1) if symbol in raw.columns.get_level_values(1) else pd.DataFrame()
                else:
                    sdf = raw.copy()

                if sdf.empty:
                    result[symbol] = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
                    continue

                sdf = sdf.rename(columns=YFINANCE_COLUMN_MAP)
                available_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in sdf.columns]
                sdf = sdf[available_cols].copy()
                sdf = sdf.reset_index()
                date_col = "Date" if "Date" in sdf.columns else sdf.columns[0]
                sdf = sdf.rename(columns={date_col: "time"})

                if sdf["time"].dt.tz is None:
                    sdf["time"] = sdf["time"].dt.tz_localize("UTC")
                else:
                    sdf["time"] = sdf["time"].dt.tz_convert("UTC")

                for col in ["open", "high", "low", "close", "volume"]:
                    if col in sdf.columns:
                        sdf[col] = pd.to_numeric(sdf[col], errors="coerce")

                sdf = sdf[["time", "open", "high", "low", "close", "volume"]].dropna()
                result[symbol] = sdf
            except Exception as exc:
                logger.warning("Failed to process yfinance data for %s: %s", symbol, exc)
                result[symbol] = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        return result

    def list_symbols(self) -> list[str]:
        """List available symbols. Returns empty list (use config/symbols.yaml instead)."""
        return []

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download data for a single symbol."""
        return yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            threads=False,
            progress=False,
        )

    def _download_bulk(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Download data for multiple symbols in one call."""
        return yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=False,
            progress=False,
        )
