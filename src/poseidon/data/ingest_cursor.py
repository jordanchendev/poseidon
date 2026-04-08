"""Ingest cursor helper — DATA-FOUND-01/02 per Phase 38 D-01..D-08.

Provides ``IngestCursorService`` for the self-healing ingest loop:

- ``get_or_bootstrap`` — return ``last_successful_ts`` for
  ``(symbol, market, interval)``. If no ``ingest_state`` row exists, bootstrap
  from ``MAX(time)`` in the ``ohlcv`` hypertable (D-02). Bootstrap uses
  ``INSERT ... ON CONFLICT DO NOTHING`` so concurrent workers racing on a fresh
  symbol cannot raise ``UniqueViolation`` (Pitfall 1).
- ``advance`` — called AFTER ``upsert_ohlcv`` commits so the cursor never
  jumps ahead of durable data (D-03, D-08).
- ``record_error`` — stamp ``last_attempt_ts`` + truncated ``last_error``
  without touching ``last_successful_ts``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from poseidon.models.ingest_state import IngestState


class IngestCursorService:
    """Per-(symbol, market, interval) cursor bookkeeping."""

    def get_or_bootstrap(
        self,
        session: Session,
        symbol: str,
        market: str,
        interval: str,
    ) -> datetime | None:
        """Return ``last_successful_ts``; bootstrap from ``MAX(time)`` if missing.

        Concurrent bootstrap-safe via ``ON CONFLICT DO NOTHING``.
        """
        row = (
            session.query(IngestState)
            .filter_by(symbol=symbol, market=market, interval=interval)
            .first()
        )
        if row is not None:
            return row.last_successful_ts

        max_ts = session.execute(
            text(
                "SELECT MAX(time) FROM ohlcv "
                "WHERE symbol = :symbol AND market = :market AND interval = :interval"
            ),
            {"symbol": symbol, "market": market, "interval": interval},
        ).scalar()

        stmt = (
            pg_insert(IngestState)
            .values(
                symbol=symbol,
                market=market,
                interval=interval,
                last_successful_ts=max_ts,
                first_backfill_done=(max_ts is not None),
            )
            .on_conflict_do_nothing(
                index_elements=["symbol", "market", "interval"]
            )
        )
        session.execute(stmt)
        session.commit()

        row = (
            session.query(IngestState)
            .filter_by(symbol=symbol, market=market, interval=interval)
            .one()
        )
        return row.last_successful_ts

    def advance(
        self,
        session: Session,
        symbol: str,
        market: str,
        interval: str,
        new_ts: datetime,
    ) -> None:
        """Advance cursor AFTER ``upsert_ohlcv`` commits (D-03, D-08)."""
        session.query(IngestState).filter_by(
            symbol=symbol, market=market, interval=interval
        ).update(
            {
                "last_successful_ts": new_ts,
                "last_attempt_ts": datetime.now(timezone.utc),
                "last_error": None,
            }
        )
        session.commit()

    def record_error(
        self,
        session: Session,
        symbol: str,
        market: str,
        interval: str,
        err: str,
    ) -> None:
        """Stamp last_attempt_ts + truncated last_error; leave cursor frozen."""
        session.query(IngestState).filter_by(
            symbol=symbol, market=market, interval=interval
        ).update(
            {
                "last_attempt_ts": datetime.now(timezone.utc),
                "last_error": (err or "")[:2000],
            }
        )
        session.commit()
