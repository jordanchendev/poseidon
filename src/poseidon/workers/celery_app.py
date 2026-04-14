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
        # Phase 38 D-13: BackfillJob chunk processor lives on its own queue
        # so one-shot backfills never starve the periodic ingest path.
        "poseidon.workers.backfill_tasks.*": {"queue": "backfill"},
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

    # Beat schedule
    beat_schedule={
        # Taiwan stocks/futures: daily after market close (13:45 UTC+8 = 05:45 UTC, buffer to 06:00)
        "fetch-tw-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=6, minute=0),
            "args": ["tw_stock", "1d"],
        },
        # TW stock: 5m during trading hours (09:00-13:30 UTC+8 = 01:00-05:30 UTC)
        "fetch-tw-stock-5m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="*/5", hour="1,2,3,4,5"),
            "args": ["tw_stock", "5m"],
        },
        # TW stock: 15m during trading hours
        "fetch-tw-stock-15m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="1,16,31,46", hour="1,2,3,4,5"),
            "args": ["tw_stock", "15m"],
        },
        # TW stock: 30m during trading hours
        "fetch-tw-stock-30m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="2,32", hour="1,2,3,4,5"),
            "args": ["tw_stock", "30m"],
        },
        # TW stock: 1h during trading hours
        "fetch-tw-stock-1h": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute=5, hour="1,2,3,4,5"),
            "args": ["tw_stock", "1h"],
        },
        # Taiwan futures: daily after market close (same schedule as TW stocks)
        "fetch-tw-futures-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=6, minute=0),
            "args": ["tw_futures", "1d"],
        },
        # TW futures: 5m during trading hours (08:45-13:45 UTC+8 = 00:45-05:45 UTC)
        "fetch-tw-futures-5m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="*/5", hour="1,2,3,4,5"),
            "args": ["tw_futures", "5m"],
        },
        # TW futures: 15m during trading hours
        "fetch-tw-futures-15m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="1,16,31,46", hour="1,2,3,4,5"),
            "args": ["tw_futures", "15m"],
        },
        # TW futures: 30m during trading hours
        "fetch-tw-futures-30m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="2,32", hour="1,2,3,4,5"),
            "args": ["tw_futures", "30m"],
        },
        # TW futures: 1h during trading hours
        "fetch-tw-futures-1h": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute=5, hour="1,2,3,4,5"),
            "args": ["tw_futures", "1h"],
        },
        # US stocks: daily after market close (16:15 US/Eastern ~ 21:15 UTC, buffer to 21:30)
        "fetch-us-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=21, minute=30),
            "args": ["us_stock", "1d"],
        },
        # Crypto spot: every 5 min at :01 past (5m data)
        "fetch-crypto-5m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="*/5"),
            "args": ["crypto_spot", "5m"],
        },
        # Crypto spot: every 15 min at :02 past (15m data)
        "fetch-crypto-15m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="1,16,31,46"),
            "args": ["crypto_spot", "15m"],
        },
        # Crypto spot: every 30 min at :03 past (30m data)
        "fetch-crypto-30m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="2,32"),
            "args": ["crypto_spot", "30m"],
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
        # Monte Carlo VaR: daily at 01:30 UTC (30 min after covariance update, per D-07)
        "compute-mc-var-daily": {
            "task": "poseidon.workers.cpu_tasks.compute_mc_var",
            "schedule": crontab(hour=1, minute=30),
        },
        # Data quality scoring: daily at 02:00 UTC (per D-09, DVAL-05)
        "compute-quality-scores-daily": {
            "task": "poseidon.workers.cpu_tasks.compute_quality_scores",
            "schedule": crontab(hour=2, minute=0),
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
        # 2a. Perp 5m OHLCV fetch
        "fetch-crypto-perp-5m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="*/5"),
            "args": ["crypto_perp", "5m"],
        },
        # 2b. Perp 15m OHLCV fetch
        "fetch-crypto-perp-15m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="1,16,31,46"),
            "args": ["crypto_perp", "15m"],
        },
        # 2c. Perp 30m OHLCV fetch
        "fetch-crypto-perp-30m": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute="2,32"),
            "args": ["crypto_perp", "30m"],
        },
        # 2d. Perp 1h OHLCV fetch
        "fetch-crypto-perp-1h": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute=5),
            "args": ["crypto_perp", "1h"],
        },
        # 2e. Perp 4h OHLCV fetch (D-04, D-05)
        "fetch-crypto-perp-4h": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour="0,4,8,12,16,20", minute=0),
            "args": ["crypto_perp", "4h"],
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
        # --- Phase 35: Universe refresh scheduling ---
        # TW stocks: daily before market data fetch (05:30 UTC)
        "refresh-universe-tw-stock": {
            "task": "poseidon.workers.cpu_tasks.refresh_universe",
            "schedule": crontab(hour=5, minute=30),
            "args": ["tw_stock"],
        },
        # TW futures: daily before market data fetch (05:30 UTC)
        "refresh-universe-tw-futures": {
            "task": "poseidon.workers.cpu_tasks.refresh_universe",
            "schedule": crontab(hour=5, minute=30),
            "args": ["tw_futures"],
        },
        # US stocks: daily before market data fetch (21:00 UTC)
        "refresh-universe-us-stock": {
            "task": "poseidon.workers.cpu_tasks.refresh_universe",
            "schedule": crontab(hour=21, minute=0),
            "args": ["us_stock"],
        },
        # Crypto spot: every 4 hours (per D-13)
        "refresh-universe-crypto-spot": {
            "task": "poseidon.workers.cpu_tasks.refresh_universe",
            "schedule": crontab(hour="*/4", minute=0),
            "args": ["crypto_spot"],
        },
        # Crypto perp: every 4 hours (per D-13)
        "refresh-universe-crypto-perp": {
            "task": "poseidon.workers.cpu_tasks.refresh_universe",
            "schedule": crontab(hour="*/4", minute=0),
            "args": ["crypto_perp"],
        },
        # --- Phase 39 plan 39-03: data_coverage_mv hourly refresh (D-12) ---
        # Immediate-after-success refresh is triggered from backfill_chunk's
        # succeeded path; this beat tick is the "background safety net" so
        # the MV stays within at most one hour of the underlying ohlcv table.
        "refresh-data-coverage-mv": {
            "task": "poseidon.workers.backfill_tasks.coverage_view_refresh",
            "schedule": crontab(minute=0),
        },
        # --- Phase 39 plan 39-04: historical backfill dispatcher (BACKFILL-05) ---
        # Hourly tick that scans symbols.yaml for tuples without ingest
        # history and creates BackfillJob rows via the durable Phase 39
        # contract. The dispatcher runs on the cpu queue (NOT backfill)
        # because it only writes rows; the dedicated backfill-worker then
        # consumes the durable jobs from queue 'backfill'.
        "historical-backfill-dispatcher": {
            "task": "poseidon.workers.cpu_tasks.historical_backfill_dispatcher",
            "schedule": crontab(minute=0),
        },
        # --- Phase 40 plan 40-02: data_gap_audit daily (COVERAGE-03) ---
        # Daily at 03:00 UTC, after the 00:15 UTC crypto daily fetch and
        # well before the 06:00 UTC TW market-close fetch — so the audit
        # sees the freshest data from both cycles. Routed onto the cpu
        # queue via the existing poseidon.workers.cpu_tasks.* rule.
        "data-gap-audit": {
            "task": "poseidon.workers.cpu_tasks.data_gap_audit",
            "schedule": crontab(hour=3, minute=0),
        },
        # --- Phase 40 plan 40-03: ingest_freshness_watchdog every 15min (FRESH-01..04) ---
        # The watchdog IS the heartbeat -- Beat dying or watchdog dying
        # both produce silence on UPTIME_KUMA_PUSH_URL and Uptime Kuma
        # fires the dead-man's-switch alert automatically (D-12).
        # Routed onto the cpu queue via the existing
        # poseidon.workers.cpu_tasks.* rule.
        "ingest-freshness-watchdog": {
            "task": "poseidon.workers.cpu_tasks.ingest_freshness_watchdog",
            "schedule": crontab(minute="*/15"),
        },
    },

    # Auto-discover task modules
    imports=[
        "poseidon.workers.cpu_tasks",
        "poseidon.workers.gpu_tasks",
        "poseidon.workers.backfill_tasks",
    ],
)
