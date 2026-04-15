"""Unified data access repository for OHLCV, fundamentals, sentiment, and non-price data.

All data consumers (backtest, strategies, workers, GPU tasks, API endpoints)
MUST use this interface instead of importing storage.py or loader classes directly.

Design decisions:
- D-01: Single DataRepository class with method groups
- D-02: Session injection via constructor (matches SignalRepository/BacktestRepository)
- D-03: Lives in data/repository.py, co-located with storage.py and loaders
- D-04: Wraps existing loaders by composition (no logic rewrite)
- D-05: Loader instances created lazily inside methods (no eager init)
- D-10: Write operations do NOT auto-commit -- callers manage transactions explicitly
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DataRepository:
    """Unified data access for OHLCV, fundamentals, sentiment, and non-price data.

    Follows SignalRepository/BacktestRepository session injection pattern.
    Write operations do NOT auto-commit -- callers manage transactions explicitly.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def read_ohlcv(
        self,
        symbol: str,
        market: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read OHLCV data. Delegates to storage.read_ohlcv."""
        from poseidon.data.storage import read_ohlcv

        return read_ohlcv(self._session, symbol, market, interval, start, end)

    def upsert_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: str,
        instrument: str,
        interval: str,
    ) -> int:
        """Upsert OHLCV rows. Does NOT commit -- caller manages transaction."""
        from poseidon.data.storage import _upsert_ohlcv_core

        return _upsert_ohlcv_core(self._session, df, symbol, market, instrument, interval)

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def read_fundamentals(self, symbol: str, market: str):
        """Read all fundamentals rows for a symbol. Delegates to storage.read_fundamentals."""
        from poseidon.data.storage import read_fundamentals

        return read_fundamentals(self._session, symbol, market)

    def write_fundamentals(self, symbol: str, market: str, report_date: date, data: dict):
        """Write a fundamentals row. Does NOT commit -- caller manages transaction."""
        from poseidon.data.storage import _write_fundamentals_core

        return _write_fundamentals_core(self._session, symbol, market, report_date, data)

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def read_sentiment(self, symbol: str, market: str | None = None, limit: int = 100):
        """Read sentiment scores for a symbol. Delegates to storage.read_sentiment."""
        from poseidon.data.storage import read_sentiment

        return read_sentiment(self._session, symbol, market, limit)

    def write_sentiment(self, symbol: str, market: str, source_type: str, score: float):
        """Write a sentiment score. Does NOT commit -- caller manages transaction."""
        from poseidon.data.storage import _write_sentiment_core

        return _write_sentiment_core(self._session, symbol, market, source_type, score)

    # ------------------------------------------------------------------
    # Non-price data (DB reads -- FEAT-02 ingest-first pattern)
    # ------------------------------------------------------------------

    def read_funding_rates(self, symbol: str, start: str = "2024-01-01") -> pd.DataFrame:
        """Read daily funding rates from funding_rates DB table.

        Delegates to FundingRateLoader.get_daily_funding_rate_from_db() which
        queries the DB and aggregates to daily sums.
        """
        from poseidon.data.loaders import FundingRateLoader

        return FundingRateLoader(
            session_factory=lambda: self._session
        ).get_daily_funding_rate_from_db(symbol, start=start)

    def read_open_interest(
        self, symbol: str, interval: str = "1h", start: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Read OI series from DB. Passes SessionLocal as session_factory."""
        from poseidon.core.database import SessionLocal
        from poseidon.data.loaders import OpenInterestLoader

        return OpenInterestLoader(session_factory=SessionLocal).get_oi_series(
            symbol, interval, start
        )

    def read_macro(self, start: str = "2020-01-01") -> pd.DataFrame:
        """Read macro economic indices from macro_index DB table."""
        from poseidon.models.macro_index import MacroIndex

        start_date = pd.Timestamp(start).date()
        rows = (
            self._session.query(MacroIndex)
            .filter(MacroIndex.date >= start_date)
            .order_by(MacroIndex.date)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        data = [{"date": r.date, "indicator": r.indicator, "value": r.value} for r in rows]
        df = pd.DataFrame(data)
        result = df.pivot(index="date", columns="indicator", values="value")
        result.index = pd.to_datetime(result.index)
        result = result.ffill()
        return result

    def read_institutional_flow(self, symbol: str) -> pd.DataFrame:
        """Read institutional investor flow data from nonprice_timeseries DB table."""
        return self._read_nonprice_category(symbol, "institutional")

    def read_margin_transactions(self, symbol: str) -> pd.DataFrame:
        """Read margin transaction data from nonprice_timeseries DB table."""
        return self._read_nonprice_category(symbol, "margin")

    def read_fundamentals_df(self, symbol: str) -> pd.DataFrame:
        """Read fundamentals (PE, PB, revenue, ROE, ROA) from nonprice_timeseries DB table."""
        return self._read_nonprice_category(symbol, "fundamental")

    def read_trade_structure(self, symbol: str) -> pd.DataFrame:
        """Read trade structure data from nonprice_timeseries DB table."""
        return self._read_nonprice_category(symbol, "trade_structure")

    def _read_nonprice_category(self, symbol: str, category: str) -> pd.DataFrame:
        """Read a non-price data category from nonprice_timeseries DB table.

        Shared helper for institutional, fundamental, margin, trade_structure reads.
        Pivots the long-format rows into a wide DataFrame indexed by date.
        """
        from poseidon.models.nonprice_timeseries import NonpriceTimeseries

        rows = (
            self._session.query(NonpriceTimeseries)
            .filter(
                NonpriceTimeseries.symbol == symbol,
                NonpriceTimeseries.category == category,
            )
            .order_by(NonpriceTimeseries.date)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        data = [{"date": r.date, "indicator": r.indicator, "value": r.value} for r in rows]
        df = pd.DataFrame(data)
        result = df.pivot(index="date", columns="indicator", values="value")
        result.index = pd.to_datetime(result.index)
        return result

    # ------------------------------------------------------------------
    # Perpetual contract data (PerpDataLoader wrapping)
    # ------------------------------------------------------------------

    def read_perp_ohlcv(
        self, symbol: str, interval: str = "4h", lookback_days: int = 30
    ) -> pd.DataFrame:
        """Read perpetual contract OHLCV from DB."""
        from poseidon.core.database import SessionLocal
        from poseidon.data.loaders.perp_data_loader import PerpDataLoader

        return PerpDataLoader(SessionLocal).get_ohlcv(
            symbol, interval=interval, lookback_days=lookback_days
        )

    def read_latest_funding_rate(self, symbol: str) -> float | None:
        """Return the most recent funding rate for a symbol from DB."""
        from poseidon.core.database import SessionLocal
        from poseidon.data.loaders.perp_data_loader import PerpDataLoader

        return PerpDataLoader(SessionLocal).get_latest_funding_rate(symbol)

    def read_latest_price(self, symbol: str, interval: str = "4h") -> float | None:
        """Return the latest close price for a perpetual symbol."""
        from poseidon.core.database import SessionLocal
        from poseidon.data.loaders.perp_data_loader import PerpDataLoader

        return PerpDataLoader(SessionLocal).get_latest_price(symbol, interval=interval)
