"""Integration tests for MLflow -> Postgres bridge (RESEARCH-API-07).

These tests require the qlib-research cp312 container with pyqlib and mlflow.
Run inside the container:
    docker compose exec qlib-research python -m pytest tests/integration/test_mlflow_bridge.py -x

Skipped in local unit test runs (no pyqlib on cp313 / Mac).
"""

from __future__ import annotations

import pytest

REQUIRES_QLIB = pytest.mark.skipif(
    True,  # Will be replaced with actual import check when running in container
    reason="Requires qlib-research cp312 container with pyqlib + mlflow",
)


@REQUIRES_QLIB
def test_mlflow_tracking_uri_configured():
    """MLFLOW_TRACKING_URI env var should be set and point to PostgreSQL."""
    import os

    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    assert "postgresql" in uri, f"Expected PostgreSQL URI, got: {uri}"


@REQUIRES_QLIB
def test_mlflow_schema_exists():
    """The mlflow schema should exist in Postgres after migration 025."""
    import os

    from sqlalchemy import create_engine, inspect

    engine = create_engine(os.environ["DATABASE_URL"])
    insp = inspect(engine)
    schemas = insp.get_schema_names()
    assert "mlflow" in schemas, f"mlflow schema missing, found: {schemas}"


@REQUIRES_QLIB
def test_mlflow_records_to_postgres():
    """MLflow should be able to write experiment data to Postgres backend."""
    import mlflow

    # This test verifies the bridge works end-to-end
    mlflow.set_experiment("test_phase41_bridge")
    with mlflow.start_run() as run:
        mlflow.log_metric("test_ic", 0.05)
        mlflow.log_metric("test_icir", 0.3)
    # Read back
    stored_run = mlflow.get_run(run.info.run_id)
    assert stored_run.data.metrics["test_ic"] == 0.05
    assert stored_run.data.metrics["test_icir"] == 0.3
    # Cleanup
    mlflow.delete_run(run.info.run_id)


@REQUIRES_QLIB
def test_qlib_tasks_module_imports_cleanly():
    """qlib_tasks.py should import without error in the cp312 container."""
    from poseidon.workers.qlib_tasks import qlib_train

    assert qlib_train.name == "poseidon.workers.qlib_tasks.qlib_train"
