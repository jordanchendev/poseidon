"""Verify Poseidon Beat schedule contains only trading/research tasks.

After Phase 60 Beat split, data ingest tasks live in Thalassa.
No task should run on both services (per D-24).
"""


def test_poseidon_beat_has_no_data_ingest_tasks():
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    data_task_prefixes = [
        "fetch-",
        "refresh-universe-",
        "ingest-",
        "compute-quality-scores",
        "refresh-data-coverage",
        "historical-backfill",
        "data-gap-audit",
        "ingest-freshness-watchdog",
    ]
    for name in schedule:
        for prefix in data_task_prefixes:
            assert not name.startswith(prefix), f"Data task '{name}' should not be in Poseidon schedule"


def test_poseidon_beat_has_trading_tasks():
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    expected = [
        "update-covariance-matrix",
        "compute-var-historical",
        "compute-var-parametric",
        "compute-mc-var-daily",
        "evaluate-strategies-tw-daily",
        "evaluate-strategies-tw-futures-daily",
        "evaluate-strategies-us-daily",
        "evaluate-strategies-crypto-hourly",
        "evaluate-strategies-crypto-daily",
        "trigger-risk-after-crypto-eval",
        "portfolio-monthly-rebalance",
        "portfolio-stop-loss-monitor",
        "portfolio-nav-snapshot",
        "perp-liquidation-monitor",
        "perp-rebalance-4h",
        "perp-funding-settlement",
        "perp-nav-snapshot",
    ]
    for name in expected:
        assert name in schedule, f"Trading task '{name}' missing from Poseidon schedule"


def test_poseidon_beat_entry_count():
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert len(schedule) == 17, f"Expected 17 entries, got {len(schedule)}: {list(schedule.keys())}"
