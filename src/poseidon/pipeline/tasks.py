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
    """Daily refdata update + freshness (``refdata.daily``), on the runner beat at 09:40Z
    (``celery_app.beat_schedule["update-refdata-daily"]``)."""

    from poseidon.core.config import settings
    from poseidon.pipeline.refdata import taifex_options, tw_corporate_actions, tw_stocks
    from poseidon.pipeline.refdata.daily import run_daily

    # Thalassa stock bars and corporate actions (a runner without Thalassa's address records an
    # "unconfigured" error), then the TXO quotes hedged runs read; each one fails alone (run_daily records
    # the error and goes on) and each one's datasets are in the freshness report.
    updaters = (tw_stocks.DAILY_UPDATER, tw_corporate_actions.DAILY_UPDATER, taifex_options.DAILY_UPDATER)
    return run_daily(code_version=settings.pipeline_code_version or "unknown", updaters=updaters)
