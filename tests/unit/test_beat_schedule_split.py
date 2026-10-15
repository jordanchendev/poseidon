"""Verify Poseidon Beat schedule contains only trading/research tasks.

After the Beat split, data ingest tasks live in Thalassa.
No task should run on both services.
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
        "decision-loop-recover-execution",
        "decision-loop-reconcile-paper-account",
        "decision-loop-expire-decisions",
        "decision-loop-produce-tw-daily",
        "decision-loop-freeze-outcome-manifests",
        "decision-loop-label-outcomes",
    ]
    for name in expected:
        assert name in schedule, f"Trading task '{name}' missing from Poseidon schedule"


def test_poseidon_beat_entry_count():
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert len(schedule) == 23, f"Expected 23 entries, got {len(schedule)}: {list(schedule.keys())}"


def test_cutover_preserves_redbeat_names_and_recovery_protective_symbols():
    from poseidon.workers import cpu_tasks
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    for entry, task in [
        ("portfolio-monthly-rebalance", "portfolio_monthly_rebalance"),
        ("perp-rebalance-4h", "perp_rebalance"),
        ("portfolio-stop-loss-monitor", "portfolio_stop_loss_monitor"),
        ("perp-liquidation-monitor", "perp_liquidation_monitor"),
    ]:
        assert schedule[entry]["task"] == "poseidon.workers.cpu_tasks." + task
        assert callable(getattr(cpu_tasks, task).run)
    assert callable(cpu_tasks.reconcile_execution_claim.run)
    assert callable(cpu_tasks.reconcile_paper_account.run)
    assert callable(cpu_tasks.recover_decision_execution.run)


def test_portfolio_rebalance_weekday_claim_slot():
    """Phase 101 D-11: the ordinary claimer fires Mon-Fri 01:30 UTC under its original RedBeat name."""
    from poseidon.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule["portfolio-monthly-rebalance"]
    schedule = entry["schedule"]
    assert entry["task"] == "poseidon.workers.cpu_tasks.portfolio_monthly_rebalance"
    assert schedule.hour == {1}
    assert schedule.minute == {30}
    assert schedule.day_of_week == {1, 2, 3, 4, 5}
    assert schedule.day_of_month == set(range(1, 32))
