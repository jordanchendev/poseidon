"""DatasetBuilder: TimescaleDB-to-Qlib data pipeline.

Queries OHLCV data from Poseidon's TimescaleDB and produces Qlib-compatible
multi-index DataFrames (datetime x instrument) with $-prefixed columns.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from poseidon.data.storage import read_ohlcv
from poseidon.qlib.column_adapter import adapt_columns
from poseidon.universe.snapshot import get_snapshot_at

logger = logging.getLogger(__name__)

# Qlib-compatible column names for empty DataFrame construction
_QLIB_COLUMNS = ["$open", "$high", "$low", "$close", "$volume"]


class DatasetBuilder:
    """Queries OHLCV data from TimescaleDB and produces Qlib-compatible DataFrames.

    Capability metadata (per D-21):
        supports_backtest = True
        supports_live = False
        bias_risk = []
        stateful = False
    """

    # Capability metadata
    name: str = "qlib_dataset_builder"
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    def __init__(self, session: Session, market: str, interval: str) -> None:
        """Initialize DatasetBuilder.

        Args:
            session: SQLAlchemy session for TimescaleDB queries.
            market: Market identifier (e.g., "tw_stock", "crypto_perp").
            interval: OHLCV interval (e.g., "1d", "1h").
        """
        self.session = session
        self.market = market
        self.interval = interval

    def _make_empty_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with correct Qlib multi-index structure."""
        idx = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        return pd.DataFrame(index=idx, columns=_QLIB_COLUMNS, dtype=float)

    def build(
        self,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        snapshot_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Build a Qlib-compatible multi-index DataFrame from TimescaleDB.

        Args:
            symbols: Explicit list of symbol IDs. If None, uses snapshot_date
                to resolve the dynamic universe.
            start: Start of the date range (inclusive). None means no lower bound.
            end: End of the date range (inclusive). None means no upper bound.
            snapshot_date: Point-in-time for dynamic universe resolution via
                get_snapshot_at. Used only when symbols is None.

        Returns:
            DataFrame with MultiIndex (datetime, instrument) and columns
            $open, $high, $low, $close, $volume.

        Raises:
            ValueError: If neither symbols nor snapshot_date is provided.
        """
        resolved_symbols = self._resolve_symbols(symbols, snapshot_date)

        if not resolved_symbols:
            logger.warning("No symbols to build dataset for")
            return self._make_empty_df()

        frames: list[pd.DataFrame] = []
        for symbol in resolved_symbols:
            df = read_ohlcv(
                self.session, symbol, self.market, self.interval, start, end
            )
            if df.empty:
                logger.warning(
                    "No data for symbol %s in range %s to %s, skipping",
                    symbol,
                    start,
                    end,
                )
                continue

            df = df.copy()
            df["instrument"] = symbol
            frames.append(df)

        if not frames:
            logger.warning("No data found for any symbol, returning empty DataFrame")
            return self._make_empty_df()

        # Concatenate and build multi-index
        combined = pd.concat(frames)
        combined.index.name = "datetime"
        combined = combined.reset_index()
        combined = combined.set_index(["datetime", "instrument"])

        # Adapt columns from Poseidon convention to Qlib convention
        result = adapt_columns(combined)

        # Sort for deterministic output
        result = result.sort_index()

        n_symbols = result.index.get_level_values("instrument").nunique()
        n_rows = len(result)
        date_range_start = result.index.get_level_values("datetime").min()
        date_range_end = result.index.get_level_values("datetime").max()
        logger.info(
            "Built Qlib dataset: %d symbols, %d rows, %s to %s",
            n_symbols,
            n_rows,
            date_range_start,
            date_range_end,
        )

        return result

    def _resolve_symbols(
        self,
        symbols: list[str] | None,
        snapshot_date: datetime | None,
    ) -> list[str]:
        """Resolve symbol list from explicit list or dynamic universe snapshot.

        Args:
            symbols: Explicit symbol list (takes priority).
            snapshot_date: Point-in-time for snapshot lookup.

        Returns:
            List of symbol ID strings.

        Raises:
            ValueError: If neither symbols nor snapshot_date is provided.
        """
        if symbols is not None:
            return symbols

        if snapshot_date is None:
            raise ValueError(
                "Either symbols or snapshot_date must be provided"
            )

        snapshot = get_snapshot_at(self.session, self.market, snapshot_date)
        if snapshot is None:
            logger.warning(
                "No universe snapshot found for market=%s at %s",
                self.market,
                snapshot_date,
            )
            return []

        # snapshot.symbols is list[dict] with key "id"
        return [s["id"] for s in snapshot.symbols]
