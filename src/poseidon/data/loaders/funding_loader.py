"""CCXT funding rate loader for crypto perpetual contracts."""

import logging
from datetime import datetime, timezone

import pandas as pd

from poseidon.models.funding_rate import FundingRateRecord

logger = logging.getLogger(__name__)

# Funding rate settlement interval in seconds (8 hours)
_FUNDING_INTERVAL_SECS = 8 * 3600


class FundingRateLoader:
    """Load perpetual funding rate history from CCXT and aggregate to daily.

    Uses Binance by default. Converts Poseidon spot symbols (e.g. "BTCUSDT")
    to CCXT perpetual format (e.g. "BTC/USDT:USDT").

    Optionally persists raw funding rate records to the funding_rates DB table
    when a session_factory is provided.
    """

    def __init__(self, exchange_id: str = "binance", session_factory=None):
        self._exchange_id = exchange_id
        self._exchange = None
        self._session_factory = session_factory

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
            "BTCUSDT"      -> "BTC/USDT:USDT"
            "ETHUSDT"      -> "ETH/USDT:USDT"
            "BTC/USDT:USDT" -> "BTC/USDT:USDT"  (passthrough)
        """
        if ":" in symbol:
            return symbol  # Already in CCXT perp format
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

    def fetch_and_store(
        self,
        symbol: str,
        start: str = "2024-01-01",
        limit: int = 1000,
    ) -> int:
        """Fetch funding rate history and persist to funding_rates DB table.

        Args:
            symbol: CCXT perp format (e.g. "BTC/USDT:USDT") or spot format.
            start: Start date "YYYY-MM-DD".
            limit: Max records per API page.

        Returns:
            Number of records stored/upserted.

        Raises:
            RuntimeError: If no session_factory was provided.
        """
        if self._session_factory is None:
            raise RuntimeError(
                "FundingRateLoader requires session_factory for fetch_and_store()"
            )

        exchange = self._get_exchange()
        perp_symbol = self._spot_to_perp(symbol)

        try:
            since = exchange.parse8601(f"{start}T00:00:00Z")
        except Exception:
            logger.exception("Failed to parse start date: %s", start)
            return 0

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

                if len(batch) < limit:
                    break
        except Exception:
            logger.exception(
                "Failed to fetch funding rate for %s (%s)", symbol, perp_symbol
            )
            return 0

        if not all_records:
            return 0

        # Persist to DB
        count = 0
        session = self._session_factory()
        try:
            for record in all_records:
                ts_ms = record.get("timestamp")
                rate = record.get("fundingRate")
                if ts_ms is None or rate is None:
                    continue

                # Round timestamp to nearest 8h boundary to avoid duplicates
                rounded_ts = self._round_to_8h(ts_ms)

                obj = FundingRateRecord(
                    time=rounded_ts,
                    symbol=perp_symbol,
                    funding_rate=rate,
                    mark_price=record.get("markPrice"),
                    index_price=record.get("indexPrice"),
                )
                session.merge(obj)  # Upsert on (time, symbol) PK
                count += 1

            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to store funding rates for %s", perp_symbol)
            return 0
        finally:
            session.close()

        logger.info("Stored %d funding rate records for %s", count, perp_symbol)
        return count

    def get_daily_funding_rate_from_db(
        self,
        symbol: str,
        start: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Read funding rates from DB and aggregate to daily sum.

        Args:
            symbol: CCXT perp format or spot format.
            start: Start date "YYYY-MM-DD".

        Returns:
            DataFrame indexed by date with column "funding_rate_daily".
        """
        if self._session_factory is None:
            raise RuntimeError(
                "FundingRateLoader requires session_factory for get_daily_funding_rate_from_db()"
            )

        perp_symbol = self._spot_to_perp(symbol)
        start_dt = pd.Timestamp(start, tz="UTC")

        session = self._session_factory()
        try:
            rows = (
                session.query(FundingRateRecord)
                .filter(
                    FundingRateRecord.symbol == perp_symbol,
                    FundingRateRecord.time >= start_dt,
                )
                .order_by(FundingRateRecord.time)
                .all()
            )
        finally:
            session.close()

        if not rows:
            return pd.DataFrame()

        data = [
            {"datetime": r.time, "funding_rate": r.funding_rate}
            for r in rows
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["datetime"]).dt.date

        daily = df.groupby("date")["funding_rate"].sum().reset_index()
        daily.columns = ["date", "funding_rate_daily"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date")

        return daily

    @staticmethod
    def _round_to_8h(timestamp_ms: int) -> datetime:
        """Round millisecond timestamp to nearest 8h boundary.

        Funding settlements occur at 00:00, 08:00, 16:00 UTC.
        Rounding prevents duplicate records from slight timestamp drift.
        """
        ts_secs = timestamp_ms / 1000.0
        rounded_secs = round(ts_secs / _FUNDING_INTERVAL_SECS) * _FUNDING_INTERVAL_SECS
        return datetime.fromtimestamp(rounded_secs, tz=timezone.utc)
