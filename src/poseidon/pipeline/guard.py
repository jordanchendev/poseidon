"""Fail-closed check that a runner process points at the isolated database.

The runner entry points (``poseidon.pipeline.app``, ``poseidon.pipeline.celery_app``
and ``python -m poseidon.pipeline.guard``) refuse to start unless
``POSEIDON_PIPELINE_EXPECTED_DATABASE`` is set to a non-empty name and
``POSEIDON_DATABASE_URL`` names exactly that database. An unset or empty
expectation is a refusal, never a pass.
"""

from __future__ import annotations

import sys

from sqlalchemy.engine import make_url

from poseidon.core.config import settings

EXPECTED_DATABASE_ENV = "POSEIDON_PIPELINE_EXPECTED_DATABASE"


class PipelineIsolationError(RuntimeError):
    pass


def assert_isolated_database(database_url: str | None = None, expected: str | None = None) -> None:
    expected = settings.pipeline_expected_database if expected is None else expected
    if not expected or not expected.strip():
        raise PipelineIsolationError(
            f"{EXPECTED_DATABASE_ENV} is unset or empty; set it to the isolated pipeline database "
            "(e.g. poseidon_pipeline) so the runner cannot start against production"
        )
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
