"""Celery application configuration with RedBeat persistent scheduler."""

from celery import Celery
from celery.schedules import crontab

from poseidon.core.config import settings

celery_app = Celery("poseidon", broker=settings.redis_celery_url, backend=settings.redis_celery_url)

celery_app.conf.update(
    # Task routing
    task_routes={
        "poseidon.workers.gpu_tasks.*": {"queue": "gpu"},
        "poseidon.workers.cpu_tasks.*": {"queue": "cpu"},
        "poseidon.workers.qlib_tasks.*": {"queue": "qlib_queue"},
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
    redbeat_redis_url=settings.redis_celery_url,
    redbeat_key_prefix="poseidon:redbeat:",

    # Beat schedule — trading/risk/portfolio/perp tasks ONLY
    # Data ingest tasks (fetch, refresh-universe, ingest, quality, backfill,
    # coverage, gap-audit, freshness) moved to Thalassa in Phase 60.
    beat_schedule={
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
        # Monte Carlo VaR: daily at 01:30 UTC (30 min after covariance update, per D-07)
        "compute-mc-var-daily": {
            "task": "poseidon.workers.cpu_tasks.compute_mc_var",
            "schedule": crontab(hour=1, minute=30),
        },
        # Strategy evaluation: triggered after data fetch (PRED-03)
        # Each runs ~5-10 min after corresponding fetch to ensure fresh data
        "evaluate-strategies-tw-daily": {
            "task": "poseidon.workers.cpu_tasks.evaluate_active_strategies",
            "schedule": crontab(hour=6, minute=10),
            "kwargs": {"market": "tw_stock", "interval": "1d"},
        },
        "evaluate-strategies-tw-futures-daily": {
            "task": "poseidon.workers.cpu_tasks.evaluate_active_strategies",
            "schedule": crontab(hour=6, minute=10),
            "kwargs": {"market": "tw_futures", "interval": "1d"},
        },
        "evaluate-strategies-us-daily": {
            "task": "poseidon.workers.cpu_tasks.evaluate_active_strategies",
            "schedule": crontab(hour=21, minute=40),
            "kwargs": {"market": "us_stock", "interval": "1d"},
        },
        "evaluate-strategies-crypto-hourly": {
            "task": "poseidon.workers.cpu_tasks.evaluate_active_strategies",
            "schedule": crontab(minute=15),
            "kwargs": {"market": "crypto_spot", "interval": "1h"},
        },
        "evaluate-strategies-crypto-daily": {
            "task": "poseidon.workers.cpu_tasks.evaluate_active_strategies",
            "schedule": crontab(hour=0, minute=25),
            "kwargs": {"market": "crypto_spot", "interval": "1d"},
        },
        # Risk pipeline trigger: after strategy evaluation (PRED-04)
        "trigger-risk-after-crypto-eval": {
            "task": "poseidon.workers.cpu_tasks.trigger_risk_update",
            "schedule": crontab(minute=20),
        },
        # Portfolio monthly rebalance: 15th of month at 01:30 UTC (09:30 UTC+8, after TW market open)
        "portfolio-monthly-rebalance": {
            "task": "poseidon.workers.cpu_tasks.portfolio_monthly_rebalance",
            "schedule": crontab(day_of_month=15, hour=1, minute=30),
        },
        # Portfolio stop-loss monitor: every 5 min (task self-gates to TW trading hours 01:00-05:30 UTC)
        "portfolio-stop-loss-monitor": {
            "task": "poseidon.workers.cpu_tasks.portfolio_stop_loss_monitor",
            "schedule": crontab(minute="*/5"),
        },
        # Portfolio NAV snapshot: daily post-close at 06:00 UTC (14:00 UTC+8)
        "portfolio-nav-snapshot": {
            "task": "poseidon.workers.cpu_tasks.portfolio_nav_snapshot",
            "schedule": crontab(hour=6, minute=0),
        },
        # --- Phase 27: Perpetual contract 24/7 scheduling ---
        # 1. Liquidation monitor: every 1 minute, 24/7 (PRSK-01, D-01)
        "perp-liquidation-monitor": {
            "task": "poseidon.workers.cpu_tasks.perp_liquidation_monitor",
            "schedule": crontab(minute="*"),
        },
        # 3. Perp 4h rebalance: 5 min after fetch (D-04, D-05, PRSK-02)
        "perp-rebalance-4h": {
            "task": "poseidon.workers.cpu_tasks.perp_rebalance",
            "schedule": crontab(hour="0,4,8,12,16,20", minute=5),
        },
        # 4. Funding settlement: every 8h (D-06)
        "perp-funding-settlement": {
            "task": "poseidon.workers.cpu_tasks.perp_funding_settlement",
            "schedule": crontab(hour="0,8,16", minute=10),
        },
        # 5. Perp NAV snapshot: after each rebalance cycle
        "perp-nav-snapshot": {
            "task": "poseidon.workers.cpu_tasks.perp_nav_snapshot",
            "schedule": crontab(hour="0,4,8,12,16,20", minute=15),
        },
    },

    # Auto-discover task modules
    imports=[
        "poseidon.workers.cpu_tasks",
        "poseidon.workers.gpu_tasks",
    ],
)
