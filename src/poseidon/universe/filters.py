"""Concrete universe filter implementations.

Provides VolumeFilter, ListingAgeFilter, and BlacklistFilter for
screening symbols in the dynamic universe pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from poseidon.data.factory import get_data_repository
from poseidon.data.symbols import SymbolInfo
from poseidon.universe.base import UniverseFilter
from poseidon.universe.registry import register_filter

logger = logging.getLogger(__name__)


@register_filter
class VolumeFilter(UniverseFilter):
    """Remove symbols below minimum average daily volume.

    Queries average volume from OHLCV data over a lookback window.
    Symbols with no volume data are kept (conservative approach).
    """

    name = "volume"
    description = "Remove symbols below minimum average daily volume"

    def __init__(self, min_volume: float = 1_000_000, lookback_days: int = 30):
        self.min_volume = min_volume
        self.lookback_days = lookback_days

    def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
        if not symbols:
            return []
        avg_volumes = self._get_avg_volumes(symbols, market)
        result = []
        for s in symbols:
            vol = avg_volumes.get(s.id)
            if vol is None or vol >= self.min_volume:
                # No data -> keep (conservative)
                result.append(s)
        return result

    def _get_avg_volumes(
        self, symbols: list[SymbolInfo], market: str
    ) -> dict[str, float]:
        """Query average daily volume for each symbol over lookback window.

        Returns mapping of symbol_id -> avg_volume.
        Symbols with no OHLCV data are omitted from the result.
        """
        from poseidon.core.database import db_session

        # Pick the densest interval available for the market so "average
        # daily volume" is derived from the finest-grained data we have.
        # crypto has 4h / 1h data; stocks are 1d.
        interval = "4h" if market in {"crypto_spot", "crypto_perp"} else "1d"

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=self.lookback_days)
        volumes: dict[str, float] = {}
        with db_session() as db:
            repo = get_data_repository(db)
            for s in symbols:
                try:
                    df = repo.read_ohlcv(s.id, market, interval, start=start, end=now)
                    if df is not None and not df.empty and "volume" in df.columns:
                        volumes[s.id] = float(df["volume"].mean())
                except Exception:
                    logger.debug(
                        "No volume data for %s/%s", s.id, market, exc_info=True
                    )
        return volumes


@register_filter
class ListingAgeFilter(UniverseFilter):
    """Remove symbols with insufficient listing history.

    Checks the earliest OHLCV record to estimate listing age.
    Symbols with no data are kept (conservative approach).
    """

    name = "listing_age"
    description = "Remove symbols listed less than min_days ago"

    def __init__(self, min_days: int = 90):
        self.min_days = min_days

    def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
        if not symbols:
            return []
        ages = self._get_listing_ages(symbols, market)
        result = []
        for s in symbols:
            age = ages.get(s.id)
            if age is None or age >= self.min_days:
                result.append(s)
        return result

    def _get_listing_ages(
        self, symbols: list[SymbolInfo], market: str
    ) -> dict[str, int]:
        """Query listing age (days since earliest OHLCV record) per symbol.

        Returns mapping of symbol_id -> days_since_first_record.
        """
        from poseidon.core.database import db_session

        interval = "4h" if market in {"crypto_spot", "crypto_perp"} else "1d"

        now = datetime.now(timezone.utc)
        ages: dict[str, int] = {}
        with db_session() as db:
            repo = get_data_repository(db)
            for s in symbols:
                try:
                    # Unbounded start/end to catch the true earliest record
                    df = repo.read_ohlcv(s.id, market, interval)
                    if df is not None and not df.empty:
                        first_date = df.index.min()
                        if hasattr(first_date, "to_pydatetime"):
                            first_date = first_date.to_pydatetime()
                        if first_date.tzinfo is None:
                            first_date = first_date.replace(tzinfo=timezone.utc)
                        ages[s.id] = (now - first_date).days
                except Exception:
                    logger.debug(
                        "No listing data for %s/%s", s.id, market, exc_info=True
                    )
        return ages


@register_filter
class BlacklistFilter(UniverseFilter):
    """Remove explicitly blacklisted symbols.

    Simple set-membership check -- no DB queries required.
    """

    name = "blacklist"
    description = "Remove explicitly blacklisted symbols"

    def __init__(self, blacklist: set[str] | None = None):
        self.blacklist = blacklist or set()

    def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
        return [s for s in symbols if s.id not in self.blacklist]
