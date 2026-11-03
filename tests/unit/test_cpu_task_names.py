"""CPU task names are a wire contract: the beat schedule, queue routes, and queued messages use them."""

import importlib
import pkgutil

import poseidon.workers.cpu_tasks as cpu_tasks
from poseidon.workers.celery_app import celery_app

CPU_TASK_NAMES = frozenset(
    f"poseidon.workers.cpu_tasks.{name}"
    for name in (
        "autoresearch_run",
        "compute_mc_var",
        "compute_var_snapshot",
        "evaluate_active_strategies",
        "expire_due_decisions",
        "factor_centrality_analysis",
        "factor_ic_analysis",
        "fetch_market_data",
        "freeze_market_manifest",
        "freeze_outcome_manifests",
        "label_decision_outcomes",
        "materialize_execution_claim",
        "perp_funding_settlement",
        "perp_liquidation_monitor",
        "perp_nav_snapshot",
        "perp_rebalance",
        "portfolio_monthly_rebalance",
        "portfolio_nav_snapshot",
        "portfolio_stop_loss_monitor",
        "produce_decision_run",
        "produce_decision_runs_for_market",
        "project_decision_fill",
        "reconcile_execution_claim",
        "reconcile_paper_account",
        "recover_decision_execution",
        "run_backtest_task",
        "run_dual_mode_task",
        "run_optimization_task",
        "run_review_campaign",
        "run_stress_test",
        "scheduled_reconcile_paper_account",
        "scheduled_recover_decision_execution",
        "submit_decision_order",
        "trigger_risk_update",
        "update_covariance_matrix",
    )
)


def test_worker_imports_register_exactly_the_cpu_task_names():
    for module in celery_app.conf.imports:
        if module.startswith("poseidon.workers.cpu_tasks."):
            importlib.import_module(module)
    registered = {name for name in celery_app.tasks if name.startswith("poseidon.workers.cpu_tasks.")}
    assert registered == CPU_TASK_NAMES


def test_every_module_defining_a_cpu_task_is_a_worker_import():
    for info in pkgutil.iter_modules(cpu_tasks.__path__):
        importlib.import_module(f"{cpu_tasks.__name__}.{info.name}")
    defining = {
        celery_app.tasks[name].__module__ for name in celery_app.tasks if name.startswith("poseidon.workers.cpu_tasks.")
    }
    assert defining <= set(celery_app.conf.imports)
