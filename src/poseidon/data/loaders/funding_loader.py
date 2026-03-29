"""CCXT funding rate loader for crypto perpetual contracts."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FundingRateLoader:
    """Load perpetual funding rate history from CCXT and aggregate to daily.

    Uses Binance by default. Converts Poseidon spot symbols (e.g. "BTCUSDT")
    to CCXT perpetual format (e.g. "BTC/USDT:USDT").
    """

    def __init__(self, exchange_id: str = "binance"):
        self._exchange_id = exchange_id
        self._exchange = None

    def _get_exchange(self):
        """Lazy-initialize CCXT exchange."""
        if self._exchange is None:
            import ccxt

            exchange_cls = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_cls({"enableRateLimit": True})
        return self._exchange

    def _spot_to_perp(self, symbol: str) -> str:
        """Map Poseidon spot symbol to CCXT perpetual format.

        Examples:
            "BTCUSDT"  -> "BTC/USDT:USDT"
            "ETHUSDT"  -> "ETH/USDT:USDT"
        """
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT:USDT"
        # Fallback: return as-is (caller handles error)
        return symbol

    def get_daily_funding_rate(
        self,
        symbol: str,
        start: str = "2024-01-01",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch funding rate history and aggregate to daily sum.

        Args:
            symbol: Poseidon spot symbol (e.g. "BTCUSDT").
            start: Start date "YYYY-MM-DD".
            limit: Max records per API page.

        Returns:
            DataFrame indexed by date with column "funding_rate_daily".
            Returns empty DataFrame on error or no data.
        """
        exchange = self._get_exchange()
        perp_symbol = self._spot_to_perp(symbol)

        try:
            since = exchange.parse8601(f"{start}T00:00:00Z")
        except Exception:
            logger.exception("Failed to parse start date: %s", start)
            return pd.DataFrame()

        all_records: list[dict] = []

        try:
            while True:
                batch = exchange.fetch_funding_rate_history(
                    perp_symbol, since=since, limit=limit
                )
                if not batch:
                    break

                all_records.extend(batch)

                # Advance past last timestamp (+1ms)
                last_ts = batch[-1].get("timestamp", 0)
                since = last_ts + 1

                # If fewer than limit, we reached the end
                if len(batch) < limit:
                    break
        except Exception:
            logger.exception(
                "Failed to fetch funding rate for %s (%s)", symbol, perp_symbol
            )
            return pd.DataFrame()

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        if "timestamp" not in df.columns or "fundingRate" not in df.columns:
            logger.warning("Unexpected funding rate response format for %s", symbol)
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["date"] = df["datetime"].dt.date

        # Aggregate: sum of funding rates per day (typically 3x per day at 8h intervals)
        daily = df.groupby("date")["fundingRate"].sum().reset_index()
        daily.columns = ["date", "funding_rate_daily"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date")

        return daily
