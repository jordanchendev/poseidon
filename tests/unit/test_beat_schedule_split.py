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
        "decision-loop-produce-tw-daily-retry-1",
        "decision-loop-produce-tw-daily-retry-2",
        "decision-loop-produce-tw-daily-retry-3",
        "decision-loop-freeze-outcome-manifests",
        "decision-loop-label-outcomes",
    ]
    for name in expected:
        assert name in schedule, f"Trading task '{name}' missing from Poseidon schedule"


def test_poseidon_beat_entry_count():
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert len(schedule) == 26, f"Expected 26 entries, got {len(schedule)}: {list(schedule.keys())}"


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


def test_producer_retry_slots_share_the_primary_task_inside_the_freshness_window():
    """Phase 101-17.2: three retry slots after 06:40Z, all before the 05:30Z bar goes stale (09:30Z)."""
    from poseidon.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    primary = schedule["decision-loop-produce-tw-daily"]
    slots = []
    for index in (1, 2, 3):
        entry = schedule[f"decision-loop-produce-tw-daily-retry-{index}"]
        assert entry["task"] == primary["task"]
        assert entry["kwargs"] == primary["kwargs"] == {"market": "tw_stock", "interval": "1d"}
        (hour,) = entry["schedule"].hour
        (minute,) = entry["schedule"].minute
        slots.append(hour * 60 + minute)
    assert slots == [6 * 60 + 55, 7 * 60 + 10, 7 * 60 + 25]
    assert all(6 * 60 + 40 < slot < 9 * 60 + 30 for slot in slots)
