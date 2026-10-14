"""Celery application configuration with RedBeat persistent scheduler."""

from celery import Celery
from celery.schedules import crontab

from poseidon.core.config import settings

POSEIDON_CPU_QUEUE = "poseidon_cpu"
POSEIDON_GPU_QUEUE = "poseidon_gpu"
POSEIDON_QLIB_QUEUE = "poseidon_qlib"

celery_app = Celery("poseidon", broker=settings.redis_celery_url, backend=settings.redis_celery_url)

celery_app.conf.update(
    # Task routing
    task_routes={
        "poseidon.workers.gpu_tasks.*": {"queue": "poseidon_gpu"},
        "poseidon.workers.cpu_tasks.*": {"queue": "poseidon_cpu"},
        "poseidon.workers.qlib_tasks.*": {"queue": "poseidon_qlib"},
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
    # coverage, gap-audit, freshness) moved to Thalassa.
    beat_schedule={
        # VaR computation: covariance matrix update daily at 00:30 UTC
        "update-covariance-matrix": {
            "task": "poseidon.workers.cpu_tasks.update_covariance_matrix",
            "schedule": crontab(hour=0, minute=30),
        },
        # VaR Historical Simulation: hourly at :15 past the hour
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
        # Monte Carlo VaR: daily at 01:30 UTC (30 min after covariance update)
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
        # --- Perpetual contract 24/7 scheduling ---
        # 1. Liquidation monitor: every 1 minute, 24/7
        "perp-liquidation-monitor": {
            "task": "poseidon.workers.cpu_tasks.perp_liquidation_monitor",
            "schedule": crontab(minute="*"),
        },
        # 3. Perp 4h rebalance: 5 min after fetch
        "perp-rebalance-4h": {
            "task": "poseidon.workers.cpu_tasks.perp_rebalance",
            "schedule": crontab(hour="0,4,8,12,16,20", minute=5),
        },
        # 4. Funding settlement: every 8h
        "perp-funding-settlement": {
            "task": "poseidon.workers.cpu_tasks.perp_funding_settlement",
            "schedule": crontab(hour="0,8,16", minute=10),
        },
        # 5. Perp NAV snapshot: after each rebalance cycle
        "perp-nav-snapshot": {
            "task": "poseidon.workers.cpu_tasks.perp_nav_snapshot",
            "schedule": crontab(hour="0,4,8,12,16,20", minute=15),
        },
        # --- Decision loop (Phase 100-07) ---
        # Every task gates itself on decision_loop_execution_mode and returns
        # {"skipped": mode} in legacy/halted, so the legacy path is unchanged.
        "decision-loop-recover-execution": {
            "task": "poseidon.workers.cpu_tasks.scheduled_recover_decision_execution",
            "schedule": crontab(minute="*/15"),
        },
        # Approved paper account reconciliation: daily after the TW close and NAV snapshot.
        "decision-loop-reconcile-paper-account": {
            "task": "poseidon.workers.cpu_tasks.scheduled_reconcile_paper_account",
            "schedule": crontab(hour=6, minute=20),
        },
        "decision-loop-expire-decisions": {
            "task": "poseidon.workers.cpu_tasks.expire_due_decisions",
            "schedule": crontab(minute=5),
        },
        # Frozen evaluation runs: 30 minutes after evaluate-strategies-tw-daily (06:10).
        "decision-loop-produce-tw-daily": {
            "task": "poseidon.workers.cpu_tasks.produce_decision_runs_for_market",
            "schedule": crontab(hour=6, minute=40),
            "kwargs": {"market": "tw_stock", "interval": "1d"},
        },
        "decision-loop-label-outcomes": {
            "task": "poseidon.workers.cpu_tasks.label_decision_outcomes",
            "schedule": crontab(hour=7, minute=30),
        },
    },
    # Auto-discover task modules.
    # NB: poseidon.workers.qlib_tasks + poseidon.workers.qlib_rl_tasks must
    # only be imported by the qlib-research worker (cp312 + qlib stack).
    # api / cpu-worker / gpu-worker run on cp313 and would crash on a
    # module-level ``import qlib`` — both modules defer their qlib imports
    # to inside the task body so adding them here is safe (PATTERNS.md
    # §Deferred Qlib Import).
    imports=[
        "poseidon.workers.cpu_tasks",
        "poseidon.workers.gpu_tasks",
        "poseidon.workers.qlib_tasks",
        "poseidon.workers.qlib_rl_tasks",
        "poseidon.workers.qlib_rdagent_tasks",
        "poseidon.workers.qlib_ddg_da_tasks",
    ],
)
