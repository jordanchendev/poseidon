"""Celery application configuration with RedBeat persistent scheduler."""

from celery import Celery
from celery.schedules import crontab

from poseidon.core.config import settings

celery_app = Celery("poseidon", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    # Task routing
    task_routes={
        "poseidon.workers.gpu_tasks.*": {"queue": "gpu"},
        "poseidon.workers.cpu_tasks.*": {"queue": "cpu"},
    },

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # RedBeat scheduler (persists schedule state in Redis)
    beat_scheduler="redbeat.RedBeatScheduler",
    redbeat_redis_url=settings.redis_url,
    redbeat_key_prefix="poseidon:redbeat:",

    # Beat schedule
    beat_schedule={
        # Taiwan stocks/futures: daily after market close (13:45 UTC+8 = 05:45 UTC, buffer to 06:00)
        "fetch-tw-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=6, minute=0),
            "args": ["tw_stock", "1d"],
        },
        # Taiwan futures: daily after market close (same schedule as TW stocks)
        "fetch-tw-futures-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=6, minute=0),
            "args": ["tw_futures", "1d"],
        },
        # US stocks: daily after market close (16:15 US/Eastern ~ 21:15 UTC, buffer to 21:30)
        "fetch-us-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=21, minute=30),
            "args": ["us_stock", "1d"],
        },
        # Crypto spot: hourly at :05 past the hour
        "fetch-crypto-hourly": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute=5),
            "args": ["crypto_spot", "1h"],
        },
        # Crypto spot: daily at 00:15 UTC
        "fetch-crypto-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=0, minute=15),
            "args": ["crypto_spot", "1d"],
        },
        # VaR computation: covariance matrix update daily at 00:30 UTC (per D-05)
        "update-covariance-matrix": {
            "task": "poseidon.workers.cpu_tasks.update_covariance_matrix",
            "schedule": crontab(hour=0, minute=30),
        },
        # VaR Historical Simulation: hourly at :15 past the hour (per D-03)
        "compute-var-historical": {
            "task": "poseidon.workers.cpu_tasks.compute_var_snapshot",
            "schedule": crontab(minute=15),
            "args": ["historical"],
        },
        # VaR Parametric + Cornish-Fisher: daily at 01:00 UTC after covariance update
        "compute-var-parametric": {
            "task": "poseidon.workers.cpu_tasks.compute_var_snapshot",
            "schedule": crontab(hour=1, minute=0),
            "args": ["all"],
        },
    },

    # Auto-discover task modules
    imports=[
        "poseidon.workers.cpu_tasks",
        "poseidon.workers.gpu_tasks",
    ],
)
