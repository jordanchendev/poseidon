"""Data coverage arithmetic + refresh helper — Phase 39 plan 39-03.

This module is the single place the ``GET /api/data/coverage`` endpoint and
the Celery refresh task talk to the ``data_coverage_mv`` materialized view.

Design:

- ``INTERVAL_SECONDS`` maps each supported candle interval to its duration
  in seconds. The endpoint uses it to derive how many candles the observed
  ``(first_ts, last_ts)`` window should contain.
- ``compute_expected_count`` / ``compute_gap_count`` / ``compute_health``
  keep the arithmetic pure so the endpoint can be reasoned about without a
  DB fixture.
- ``refresh_data_coverage_mv`` executes
  ``REFRESH MATERIALIZED VIEW CONCURRENTLY data_coverage_mv`` against a
  SQLAlchemy session. The ``CONCURRENTLY`` form requires the unique index
  created in migration 022, but lets the refresh run without locking readers
  on ``GET /api/data/coverage``.

See:
- .planning/phases/39-backfill-api-coverage/39-CONTEXT.md D-10..D-12
- .planning/phases/39-backfill-api-coverage/39-RESEARCH.md Recommendation 4
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text


# Interval -> candle duration in seconds. Mirrors
# poseidon.workers.backfill_tasks._INTERVAL_SECONDS but is exposed at the
# module level because the API handler and the tests need it directly.
INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 4 * 3600,
    "1d": 86400,
}


def compute_expected_count(
    first_ts: datetime | None,
    last_ts: datetime | None,
    interval: str,
) -> int:
    """Return how many candles a complete ``(first_ts, last_ts)`` window has.

    Formula:
        expected_count = floor((last_ts - first_ts) / interval_seconds) + 1

    Both ``first_ts`` and ``last_ts`` are required. If either is missing the
    tuple has no usable coverage and we return 0 (which in turn makes
    ``completeness_pct`` degenerate to 0 for the ``/coverage`` response).

    If ``first_ts == last_ts``, there is exactly one candle, so the function
    returns 1.
    """
    if first_ts is None or last_ts is None:
        return 0
    interval_seconds = INTERVAL_SECONDS.get(interval)
    if not interval_seconds:
        # Unknown interval: be conservative and report a single candle
        # rather than raising, so the endpoint can still surface the row.
        return 1
    delta_seconds = (last_ts - first_ts).total_seconds()
    if delta_seconds < 0:
        return 0
    return int(delta_seconds // interval_seconds) + 1


def compute_gap_count(row_count: int, expected_count: int) -> int:
    """Return the number of missing candles; clamp to 0 if row_count exceeds expected.

    A row_count larger than the expected count is a data bug (duplicate
    rows or clock drift), not a negative gap, so we clamp. The endpoint
    still surfaces the raw ``row_count`` so operators can spot it.
    """
    if expected_count <= 0:
        return 0
    return max(0, expected_count - row_count)


def compute_health(
    *,
    completeness_pct: float,
    staleness_seconds: float,
    interval: str,
    gap_count: int,
) -> str:
    """Return the coverage health bucket: ``green``, ``yellow``, or ``red``.

    Thresholds (Phase 39 plan 39-03 Task 1):

    - **green**: ``gap_count == 0`` AND ``staleness_seconds <= interval_seconds * 2``
      (fully complete window, recent enough that the next scheduled fetch
      will refresh it).
    - **yellow**: ``completeness_pct >= 0.95`` AND
      ``staleness_seconds <= interval_seconds * 4``
      (minor gaps or slight staleness, likely self-healing).
    - **red**: everything else (significant gaps, or severely stale).

    For unknown intervals we default ``interval_seconds`` to 3600 so the
    thresholds still behave sensibly.
    """
    interval_seconds = INTERVAL_SECONDS.get(interval, 3600)
    if gap_count == 0 and staleness_seconds <= interval_seconds * 2:
        return "green"
    if completeness_pct >= 0.95 and staleness_seconds <= interval_seconds * 4:
        return "yellow"
    return "red"


def refresh_data_coverage_mv(session: Any) -> None:
    """Execute ``REFRESH MATERIALIZED VIEW CONCURRENTLY data_coverage_mv``.

    Called by:
    - the hourly ``coverage_view_refresh`` Celery beat task
    - the success path of ``backfill_chunk`` when a job transitions to
      ``succeeded`` (so operators see fresh coverage immediately after a
      manual backfill completes, not only on the next hourly tick)

    The ``CONCURRENTLY`` form requires the unique index created in
    migration 022 (``ix_data_coverage_mv_tuple``).
    """
    session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY data_coverage_mv"))
    session.commit()
