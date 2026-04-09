"""Gap detection helpers + idempotent writer for the Phase 40 audit task.

Three pure detection/healing helpers plus one writer:

- ``detect_gaps_for_tuple`` — SQL LAG window function scan per (market, symbol, interval)
- ``upsert_gaps`` — idempotent INSERT with ON CONFLICT DO NOTHING (Postgres) /
  INSERT OR IGNORE (SQLite)
- ``heal_resolved_gaps`` — stamps ``healed_at = now()`` on gap rows whose
  window is now fully populated

See .planning/phases/40-data-health-observability/40-CONTEXT.md D-04..D-09.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from poseidon.data.coverage import INTERVAL_SECONDS

# Per Phase 40 D-05: a (t_prev, t_curr) delta exceeds the expected
# interval by this multiplier before it counts as a gap. 1.5x absorbs
# minor clock drift without missing real gaps.
GAP_TOLERANCE_MULTIPLIER = 1.5


@dataclass(frozen=True)
class DetectedGap:
    market: str
    symbol: str
    interval: str
    gap_start: datetime
    gap_end: datetime
    missing_bars: int


def detect_gaps_for_tuple(
    session: Session,
    *,
    market: str,
    symbol: str,
    interval: str,
) -> list[DetectedGap]:
    """Return all gap windows for one (market, symbol, interval) tuple.

    Uses a SQL window function (LAG over ohlcv ordered by time) so the
    scan stays in the database and the per-tuple wire cost is one query.

    A gap is recorded when ``time - LAG(time) > interval_seconds *
    GAP_TOLERANCE_MULTIPLIER``. ``gap_start`` is the LAG (last present
    bar), ``gap_end`` is ``time`` (first present bar after the gap),
    ``missing_bars = floor((delta_seconds / interval_seconds)) - 1``.
    """
    interval_seconds = INTERVAL_SECONDS.get(interval)
    if not interval_seconds:
        return []

    threshold_seconds = interval_seconds * GAP_TOLERANCE_MULTIPLIER

    # Detect the dialect to handle timestamp arithmetic portably.
    dialect = session.bind.dialect.name if session.bind else "postgresql"

    if dialect == "sqlite":
        # SQLite: use julianday() for timestamp arithmetic (returns days).
        stmt = text(
            """
            WITH ordered AS (
                SELECT
                    time,
                    LAG(time) OVER (ORDER BY time) AS prev_time
                FROM ohlcv
                WHERE market = :market
                  AND symbol = :symbol
                  AND interval = :interval
            )
            SELECT prev_time, time
            FROM ordered
            WHERE prev_time IS NOT NULL
              AND (julianday(time) - julianday(prev_time)) * 86400.0 > :threshold
            ORDER BY prev_time
            """
        )
    else:
        # Postgres: use EXTRACT(EPOCH FROM ...) for clean seconds arithmetic.
        stmt = text(
            """
            WITH ordered AS (
                SELECT
                    time,
                    LAG(time) OVER (ORDER BY time) AS prev_time
                FROM ohlcv
                WHERE market = :market
                  AND symbol = :symbol
                  AND interval = :interval
            )
            SELECT prev_time, time
            FROM ordered
            WHERE prev_time IS NOT NULL
              AND EXTRACT(EPOCH FROM (time - prev_time)) > :threshold
            ORDER BY prev_time
            """
        )

    rows = session.execute(
        stmt,
        {
            "market": market,
            "symbol": symbol,
            "interval": interval,
            "threshold": threshold_seconds,
        },
    ).fetchall()

    gaps: list[DetectedGap] = []
    for row in rows:
        prev_time = row.prev_time
        curr_time = row.time

        # Normalize timestamps: SQLite may return strings.
        if isinstance(prev_time, str):
            prev_time = datetime.fromisoformat(prev_time)
        if isinstance(curr_time, str):
            curr_time = datetime.fromisoformat(curr_time)

        delta_seconds = (curr_time - prev_time).total_seconds()
        missing_bars = max(int(delta_seconds // interval_seconds) - 1, 1)
        gaps.append(
            DetectedGap(
                market=market,
                symbol=symbol,
                interval=interval,
                gap_start=prev_time,
                gap_end=curr_time,
                missing_bars=missing_bars,
            )
        )
    return gaps


def upsert_gaps(session: Session, gaps: Iterable[DetectedGap]) -> int:
    """Idempotently INSERT gap rows. Returns count of rows newly inserted.

    Uses Postgres ``INSERT ... ON CONFLICT (market, symbol, interval,
    gap_start) DO NOTHING`` against the unique index from migration 023
    (Phase 40 D-07). Re-runs of the audit on the same gap window are a
    no-op.

    On SQLite (unit tests), uses ``INSERT OR IGNORE`` which achieves the
    same idempotency semantics via the UNIQUE constraint on DataGap's
    table.
    """
    gap_list = list(gaps)
    if not gap_list:
        return 0

    dialect = session.bind.dialect.name if session.bind else "postgresql"

    if dialect == "sqlite":
        # SQLite path: INSERT OR IGNORE via raw SQL for idempotency.
        inserted = 0
        for g in gap_list:
            result = session.execute(
                text(
                    """
                    INSERT OR IGNORE INTO data_gaps
                        (gap_id, market, symbol, interval, gap_start, gap_end,
                         missing_bars, detected_at, healed_at)
                    VALUES
                        (:gid, :market, :symbol, :interval, :gap_start, :gap_end,
                         :missing_bars, :detected_at, NULL)
                    """
                ),
                {
                    "gid": str(__import__("uuid").uuid4()),
                    "market": g.market,
                    "symbol": g.symbol,
                    "interval": g.interval,
                    "gap_start": g.gap_start,
                    "gap_end": g.gap_end,
                    "missing_bars": g.missing_bars,
                    "detected_at": datetime.now(timezone.utc),
                },
            )
            inserted += result.rowcount or 0
        session.commit()
        return inserted
    else:
        # Postgres path: use the dialect-specific INSERT ... ON CONFLICT.
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from poseidon.models.data_gap import DataGap

        rows = [
            {
                "market": g.market,
                "symbol": g.symbol,
                "interval": g.interval,
                "gap_start": g.gap_start,
                "gap_end": g.gap_end,
                "missing_bars": g.missing_bars,
            }
            for g in gap_list
        ]
        stmt = pg_insert(DataGap).values(rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["market", "symbol", "interval", "gap_start"]
        )
        result = session.execute(stmt)
        session.commit()
        return result.rowcount or 0


def heal_resolved_gaps(session: Session) -> int:
    """Set healed_at = now() on gap rows whose window is now fully populated.

    For every still-open DataGap row (healed_at IS NULL), counts the
    ohlcv rows in the same tuple within (gap_start, gap_end). If that
    count is >= missing_bars, the audit considers the gap resolved and
    stamps healed_at = now() (Phase 40 D-07 lifecycle).

    Returns the number of rows healed.
    """
    # Query open gaps via raw SQL so we are portable across Postgres and
    # SQLite (avoids UUID column type mismatches during ORM flush on SQLite).
    open_rows = session.execute(
        text(
            """
            SELECT gap_id, market, symbol, interval, gap_start, gap_end, missing_bars
            FROM data_gaps
            WHERE healed_at IS NULL
            """
        )
    ).fetchall()

    if not open_rows:
        return 0

    healed = 0
    now = datetime.now(timezone.utc)
    for gap in open_rows:
        count_stmt = text(
            """
            SELECT COUNT(*) AS c
            FROM ohlcv
            WHERE market = :market
              AND symbol = :symbol
              AND interval = :interval
              AND time > :gap_start
              AND time < :gap_end
            """
        )
        row = session.execute(
            count_stmt,
            {
                "market": gap.market,
                "symbol": gap.symbol,
                "interval": gap.interval,
                "gap_start": gap.gap_start,
                "gap_end": gap.gap_end,
            },
        ).first()
        present = int(row.c if row else 0)
        if present >= gap.missing_bars:
            session.execute(
                text(
                    """
                    UPDATE data_gaps
                    SET healed_at = :now
                    WHERE gap_id = :gid
                    """
                ),
                {"now": now, "gid": gap.gap_id},
            )
            healed += 1
    session.commit()
    return healed
