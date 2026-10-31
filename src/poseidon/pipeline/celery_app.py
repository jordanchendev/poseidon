"""Celery app for the isolated strategy-pipeline runner.

Deliberately separate from ``poseidon.workers.celery_app``: that app's beat
schedule carries decision-loop and portfolio producers that must never run in
the runner. This app imports only ``poseidon.pipeline.tasks`` and schedules
only ``sweep_pipelines`` and the daily refdata job (``update_refdata_daily``,
17:00 Asia/Taipei, ``refdata.daily.beat_entry``) that spec v3 daily runs and
their shadow paper read.
"""

from __future__ import annotations

from celery import Celery

from poseidon.core.config import settings
from poseidon.pipeline.guard import assert_isolated_database
from poseidon.pipeline.refdata.daily import beat_entry as refdata_daily_beat_entry

PIPELINE_QUEUE = "poseidon_pipeline"
SWEEP_INTERVAL_SECONDS = 300.0

assert_isolated_database()

pipeline_celery_app = Celery(
    "poseidon_pipeline",
    broker=settings.redis_celery_url,
    backend=settings.redis_celery_url,
)

pipeline_celery_app.conf.update(
    task_default_queue=PIPELINE_QUEUE,
    task_routes={"poseidon.pipeline.tasks.*": {"queue": PIPELINE_QUEUE}},
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_scheduler="redbeat.RedBeatScheduler",
    redbeat_redis_url=settings.redis_celery_url,
    redbeat_key_prefix="poseidon-pipeline:redbeat:",
    beat_schedule={
        "sweep-pipelines": {
            "task": "poseidon.pipeline.tasks.sweep_pipelines",
            "schedule": SWEEP_INTERVAL_SECONDS,
        },
        "update-refdata-daily": refdata_daily_beat_entry(),
    },
    imports=["poseidon.pipeline.tasks"],
)

# ``celery -A poseidon.pipeline.celery_app`` looks for ``app`` / ``celery``.
app = pipeline_celery_app
