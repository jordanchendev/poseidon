"""Phase 101 D-21b/D-23.5: the halted kill switch stops new writers only.

In ``halted`` mode the scheduled reconciliation and recovery wrappers keep
running so the rollback drill proves what production does, while producers,
manifest freeze, expiry, labeling, review and the ordinary claimer
(``portfolio_monthly_rebalance``) stay skipped.
"""

import uuid
from types import SimpleNamespace

import pytest

from poseidon.workers.cpu_tasks import decision_runs as cpu_decision_runs
from poseidon.workers.cpu_tasks import execution as cpu_execution
from poseidon.workers.cpu_tasks import loop as cpu_loop
from poseidon.workers.cpu_tasks import portfolio as cpu_portfolio
from tests.cpu_tasks_support import cpu_task, patch_cpu_tasks

ACCOUNT = "paper:pilot"
RECOVER_SENTINEL = {"sentinel": "recover"}
RECONCILE_SENTINEL = {"sentinel": "reconcile"}


def _settings(mode, **changes):
    values = {
        "decision_loop_execution_mode": mode,
        "decision_loop_approved_account_scope": ACCOUNT,
        "decision_loop_approved_market": "tw_stock",
        "decision_loop_approved_account_generation": "generation-1",
        "decision_loop_worker_principal_id": "decision-worker:producer",
        "decision_loop_label_definition_version": "label-v1",
    }
    values.update(changes)
    return SimpleNamespace(**values)


class _AccountSession:
    def __init__(self, account_id):
        self.account_id = account_id

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def scalar(self, _statement):
        return self.account_id


def test_reconcile_modes_extend_producing_modes_only_by_halted():
    reconcile_modes = cpu_loop.LOOP_RECONCILE_MODES
    producing_modes = cpu_loop.LOOP_PRODUCING_MODES
    assert reconcile_modes == frozenset({"shadow", "decision", "halted"})
    assert producing_modes == frozenset({"shadow", "decision"})
    assert reconcile_modes - producing_modes == {"halted"}


@pytest.mark.parametrize("mode", ["shadow", "decision", "halted"])
def test_scheduled_recover_decision_execution_runs_in_reconcile_modes(monkeypatch, mode):
    calls = []
    patch_cpu_tasks(monkeypatch, "settings", _settings(mode))
    monkeypatch.setattr(
        cpu_execution.recover_decision_execution, "run", lambda: calls.append("recover") or RECOVER_SENTINEL
    )
    assert cpu_execution.scheduled_recover_decision_execution.run() is RECOVER_SENTINEL
    assert calls == ["recover"]


@pytest.mark.parametrize("mode", ["shadow", "decision", "halted"])
def test_scheduled_reconcile_paper_account_runs_in_reconcile_modes(monkeypatch, mode):
    account_id = uuid.uuid4()
    calls = []
    patch_cpu_tasks(monkeypatch, "settings", _settings(mode))
    patch_cpu_tasks(monkeypatch, "SessionLocal", lambda: _AccountSession(account_id))
    monkeypatch.setattr(
        cpu_execution.reconcile_paper_account, "run", lambda value: calls.append(value) or RECONCILE_SENTINEL
    )
    assert cpu_execution.scheduled_reconcile_paper_account.run() is RECONCILE_SENTINEL
    assert calls == [str(account_id)]


def test_scheduled_wrappers_skip_in_legacy_without_delegating(monkeypatch):
    patch_cpu_tasks(monkeypatch, "settings", _settings("legacy"))
    patch_cpu_tasks(monkeypatch, "SessionLocal", lambda: pytest.fail("legacy opened a session"))
    monkeypatch.setattr(cpu_execution.recover_decision_execution, "run", lambda: pytest.fail("recover ran in legacy"))
    monkeypatch.setattr(
        cpu_execution.reconcile_paper_account, "run", lambda *_args: pytest.fail("reconcile ran in legacy")
    )
    assert cpu_execution.scheduled_recover_decision_execution.run() == {"skipped": "legacy"}
    assert cpu_execution.scheduled_reconcile_paper_account.run() == {"skipped": "legacy"}


HALTED_WRITERS = {
    "produce_decision_runs_for_market": ("tw_stock", "1d"),
    "freeze_market_manifest": ("tw_stock", "1d"),
    "expire_due_decisions": (),
    "label_decision_outcomes": (),
}


@pytest.mark.parametrize("task_name", sorted(HALTED_WRITERS))
def test_halted_keeps_new_writers_skipped(monkeypatch, task_name):
    patch_cpu_tasks(monkeypatch, "settings", _settings("halted"))
    patch_cpu_tasks(monkeypatch, "SessionLocal", lambda: pytest.fail(f"{task_name} opened a session in halted"))
    monkeypatch.setattr(
        cpu_decision_runs.produce_decision_run,
        "delay",
        lambda *args, **kwargs: pytest.fail("producer queued in halted"),
    )
    assert cpu_task(task_name).run(*HALTED_WRITERS[task_name]) == {"skipped": "halted"}


def test_halted_keeps_ordinary_claimer_skipped(monkeypatch):
    patch_cpu_tasks(monkeypatch, "settings", _settings("halted"))
    patch_cpu_tasks(monkeypatch, "SessionLocal", lambda: pytest.fail("claimer opened a session in halted"))
    patch_cpu_tasks(monkeypatch, "_execute_approved_decision", lambda *_args: pytest.fail("claimer executed in halted"))
    assert cpu_portfolio.portfolio_monthly_rebalance.run() == {"skipped": "execution_halted"}
