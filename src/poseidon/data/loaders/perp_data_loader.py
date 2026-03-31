"""Data loader for perpetual contract OHLCV and funding rate from DB.

Symmetric with FinLabDataLoader -- strategy reads data through loader,
never directly queries session.
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy.orm import Session

from poseidon.models.funding_rate import FundingRateRecord
from poseidon.models.ohlcv import OHLCV

logger = logging.getLogger(__name__)


class PerpDataLoader:
    """Load perpetual contract OHLCV and funding rate from DB."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def _spot_to_perp(self, symbol: str) -> str:
        """Map spot symbol to CCXT perp format for DB queries.

        'BTCUSDT' -> 'BTC/USDT:USDT' (matches PerpFetcher storage format)
        """
        if ":" in symbol:
            return symbol
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT:USDT"
        return symbol

    def get_ohlcv(
        self, symbol: str, interval: str = "4h", lookback_days: int = 30
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for a perp symbol from DB.

        Columns: time, open, high, low, close, volume
        Filters by instrument='perpetual' with fallback to market='crypto_perp'.
        """
        perp_symbol = self._spot_to_perp(symbol)
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        session: Session = self._session_factory()
        try:
            rows = (
                session.query(OHLCV)
                .filter(
                    OHLCV.symbol == perp_symbol,
                    OHLCV.interval == interval,
                    OHLCV.time >= since,
                )
                .filter(
                    (OHLCV.instrument == "perpetual") | (OHLCV.market == "crypto_perp")
                )
                .order_by(OHLCV.time.asc())
                .all()
            )
        finally:
            session.close()

        if not rows:
            return pd.DataFrame()

        data = [
            {
                "time": r.time,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ]
        return pd.DataFrame(data)

    def get_latest_funding_rate(self, symbol: str) -> float | None:
        """Return the most recent funding rate for a symbol from DB."""
        perp_symbol = self._spot_to_perp(symbol)

        session: Session = self._session_factory()
        try:
            row = (
                session.query(FundingRateRecord)
                .filter(FundingRateRecord.symbol == perp_symbol)
                .order_by(FundingRateRecord.time.desc())
                .first()
            )
        finally:
            session.close()

        if row is None:
            return None
        return float(row.funding_rate)

    def get_latest_price(self, symbol: str, interval: str = "4h") -> float | None:
        """Return the latest close price for weight-to-shares conversion."""
        perp_symbol = self._spot_to_perp(symbol)

        session: Session = self._session_factory()
        try:
            row = (
                session.query(OHLCV.close)
                .filter(
                    OHLCV.symbol == perp_symbol,
                    OHLCV.interval == interval,
                )
                .filter(
                    (OHLCV.instrument == "perpetual") | (OHLCV.market == "crypto_perp")
                )
                .order_by(OHLCV.time.desc())
                .first()
            )
        finally:
            session.close()

        if row is None:
            return None
        return float(row.close)
