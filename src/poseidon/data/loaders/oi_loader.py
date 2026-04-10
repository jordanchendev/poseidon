"""Open interest data loader: fetch from Binance and persist/read from DB."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class OpenInterestLoader:
    """Load open interest data from Binance and TimescaleDB.

    Follows the same pattern as FundingRateLoader:
    - fetch_and_store(): fetch from exchange API and persist to DB
    - get_oi_series(): read from DB for feature computation
    """

    def __init__(self, session_factory=None):
        self._session_factory = session_factory
        self._fetcher = None

    def _get_fetcher(self):
        """Lazy-initialize OpenInterestFetcher."""
        if self._fetcher is None:
            from poseidon.data.fetchers.oi_fetcher import OpenInterestFetcher

            self._fetcher = OpenInterestFetcher()
        return self._fetcher

    def fetch_and_store(
        self,
        symbol: str,
        interval: str = "1h",
        start: str = "",
    ) -> int:
        """Fetch OI from Binance and store in open_interest table.

        Args:
            symbol: CCXT perp format (e.g. "BTC/USDT:USDT").
            interval: OI snapshot interval ("5m", "1h", "4h").
            start: Start date "YYYY-MM-DD" (empty = fetcher decides).

        Returns:
            Number of records stored/upserted.

        Raises:
            RuntimeError: If no session_factory was provided.
        """
        if self._session_factory is None:
            raise RuntimeError(
                "OpenInterestLoader requires session_factory for fetch_and_store()"
            )

        fetcher = self._get_fetcher()
        df = fetcher.fetch_oi(symbol, interval, start)
        if df.empty:
            return 0

        from poseidon.models.open_interest import OpenInterest

        session = self._session_factory()
        count = 0
        try:
            for _, row in df.iterrows():
                obj = OpenInterest(
                    time=row["time"],
                    symbol=symbol,
                    market="crypto_perp",
                    interval=interval,
                    open_interest=row["open_interest"],
                    open_interest_value=row.get("open_interest_value"),
                )
                session.merge(obj)  # Upsert on PK
                count += 1
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to store OI data for %s", symbol)
            raise
        finally:
            session.close()

        logger.info("Stored %d OI records for %s (%s)", count, symbol, interval)
        return count

    def get_oi_series(
        self,
        symbol: str,
        interval: str = "1h",
        start: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Read OI from DB, return DataFrame indexed by time with 'open_interest' column.

        Args:
            symbol: CCXT perp format (e.g. "BTC/USDT:USDT").
            interval: OI snapshot interval.
            start: Start date "YYYY-MM-DD".

        Returns:
            DataFrame indexed by time with 'open_interest' column.
            Empty DataFrame if no data found.

        Raises:
            RuntimeError: If no session_factory was provided.
        """
        if self._session_factory is None:
            raise RuntimeError(
                "OpenInterestLoader requires session_factory for get_oi_series()"
            )

        from poseidon.models.open_interest import OpenInterest

        session = self._session_factory()
        try:
            rows = (
                session.query(OpenInterest)
                .filter(
                    OpenInterest.symbol == symbol,
                    OpenInterest.interval == interval,
                    OpenInterest.time >= pd.Timestamp(start, tz="UTC"),
                )
                .order_by(OpenInterest.time)
                .all()
            )
        finally:
            session.close()

        if not rows:
            return pd.DataFrame()

        data = [
            {"time": r.time, "open_interest": float(r.open_interest)}
            for r in rows
        ]
        df = pd.DataFrame(data).set_index("time")
        return df
