"""Phase 91 W2 — Celery dispatch unit tests (RDAGENT-02)."""

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


def test_send_task_called_with_correct_queue():
    from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app
    from poseidon.workers.qlib_rdagent_tasks import qlib_rdagent_run

    assert qlib_rdagent_run.name == "poseidon.workers.qlib_tasks.qlib_rdagent_run"
    assert qlib_rdagent_run.queue == POSEIDON_QLIB_QUEUE
    assert celery_app.conf.task_routes["poseidon.workers.qlib_tasks.*"]["queue"] == POSEIDON_QLIB_QUEUE
