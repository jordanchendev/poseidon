"""ASGI app for the isolated runner: ``/health`` + ``/api/v1/pipelines`` only.

The production ``poseidon.main:app`` mounts the same router behind
``POSEIDON_PIPELINE_API_ENABLED``; the runner serves this smaller app so none
of the production routers (or their Celery producers) are reachable on :8011.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from poseidon.api import pipelines
from poseidon.api.errors import (
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from poseidon.pipeline.guard import assert_isolated_database

assert_isolated_database()

app = FastAPI(title="Poseidon Strategy Pipeline", description="Isolated strategy-pipeline runner")
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.get("/health")
def health() -> dict:
    from poseidon.core.database import db_session

    with db_session() as session:
        version = session.execute(text("SELECT version_num FROM alembic_version_pipeline")).scalar_one_or_none()
    return {"status": "ok", "pipeline_schema": version}


app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
