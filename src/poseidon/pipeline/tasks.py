"""Celery tasks of the strategy-pipeline runner (queue ``poseidon_pipeline``)."""

from __future__ import annotations

import uuid

from poseidon.pipeline.celery_app import PIPELINE_QUEUE, pipeline_celery_app


def enqueue_advance(run_id: uuid.UUID) -> None:
    advance_pipeline.apply_async(args=[str(run_id)], queue=PIPELINE_QUEUE)


def _dispatcher():
    from poseidon.pipeline.dispatcher import PipelineDispatcher

    return PipelineDispatcher(enqueue=enqueue_advance)


@pipeline_celery_app.task(name="poseidon.pipeline.tasks.advance_pipeline", max_retries=0)
def advance_pipeline(run_id: str) -> dict:
    return _dispatcher().advance(run_id)


@pipeline_celery_app.task(name="poseidon.pipeline.tasks.sweep_pipelines", max_retries=0)
def sweep_pipelines() -> dict:
    return _dispatcher().sweep()


@pipeline_celery_app.task(name="poseidon.pipeline.tasks.update_refdata_daily", max_retries=0)
def update_refdata_daily() -> dict:
    """Daily refdata update + freshness (``refdata.daily``). Not in the beat schedule yet: the owner
    adds ``refdata.daily.beat_entry()`` to ``celery_app.beat_schedule`` when the runner is upgraded."""

    from poseidon.core.config import settings
    from poseidon.pipeline.refdata.daily import run_daily

    return run_daily(code_version=settings.pipeline_code_version or "unknown")
