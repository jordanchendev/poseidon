"""Fail-closed check that a runner process points at the isolated database.

``python -m poseidon.pipeline.guard`` exits non-zero when
``POSEIDON_PIPELINE_EXPECTED_DATABASE`` is set and ``POSEIDON_DATABASE_URL``
names a different database (e.g. production ``poseidon``).
"""

from __future__ import annotations

import sys

from sqlalchemy.engine import make_url

from poseidon.core.config import settings


class PipelineIsolationError(RuntimeError):
    pass


def assert_isolated_database(database_url: str | None = None, expected: str | None = None) -> None:
    expected = settings.pipeline_expected_database if expected is None else expected
    if not expected:
        return
    actual = make_url(database_url or settings.database_url).database
    if actual != expected:
        raise PipelineIsolationError(f"pipeline runner expects database {expected!r}, configured {actual!r}")


def main() -> int:
    try:
        assert_isolated_database()
    except PipelineIsolationError as error:
        print(f"refusing to start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
