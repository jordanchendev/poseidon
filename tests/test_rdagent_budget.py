from __future__ import annotations

import types

import pytest


class _Query:
    def __init__(self, run):
        self.run = run

    def filter_by(self, **_kwargs):
        return self

    def one(self):
        return self.run


class _Session:
    def __init__(self, run):
        self.run = run
        self.committed = False

    def query(self, _model):
        return _Query(self.run)

    def commit(self):
        self.committed = True

    def rollback(self):
        raise AssertionError("rollback should not be called")


def test_cap_breach_sets_cooperative_cancel(monkeypatch):
    from rdagent.oai.backend import litellm

    from poseidon.rdagent.budget_guard import BudgetGuard

    monkeypatch.setattr(litellm, "ACC_COST", 0.0)
    model = types.ModuleType("poseidon.models.rd_agent_run")

    class RDAgentRun:
        pass

    model.RDAgentRun = RDAgentRun
    monkeypatch.setitem(__import__("sys").modules, "poseidon.models.rd_agent_run", model)

    run = types.SimpleNamespace(cancel_requested=False, cancel_reason=None, token_cost_acc_usd=None)

    session = _Session(run)
    guard = BudgetGuard(session, "12345678-1234-5678-1234-567812345678", 1.0, 0.01)
    guard.start()
    litellm.ACC_COST = 2.0
    import time

    deadline = time.monotonic() + 0.5
    while not session.committed and time.monotonic() < deadline:
        time.sleep(0.01)
    guard.stop()
    assert run.cancel_requested is True
    assert "cost" in run.cancel_reason


def test_reserve_blocks_call_before_cap(monkeypatch):
    from rdagent.oai.backend import litellm

    from poseidon.rdagent.budget_guard import BudgetExceeded, BudgetGuard

    monkeypatch.setattr(litellm, "ACC_COST", 0.9)
    run = types.SimpleNamespace(cancel_requested=False, cancel_reason=None, token_cost_acc_usd=None)
    guard = BudgetGuard(_Session(run), "12345678-1234-5678-1234-567812345678", 1.0)
    guard._baseline_cost = 0.0
    with pytest.raises(BudgetExceeded, match="reserve exceeds cap"):
        guard.check_before_call(0.2)


@pytest.mark.parametrize("amount", [float("nan"), -0.01])
def test_non_finite_or_negative_cost_is_refused(monkeypatch, amount):
    from rdagent.oai.backend import litellm

    from poseidon.rdagent.budget_guard import BudgetExceeded, BudgetGuard

    monkeypatch.setattr(litellm, "ACC_COST", amount)
    run = types.SimpleNamespace(cancel_requested=False, cancel_reason=None, token_cost_acc_usd=None)
    guard = BudgetGuard(_Session(run), "12345678-1234-5678-1234-567812345678", 1.0)
    with pytest.raises(BudgetExceeded, match="non-finite or negative"):
        guard.check_before_call()


def test_embedding_reserves_accumulate_before_upstream_cost_updates(monkeypatch):
    from rdagent.oai.backend import litellm

    from poseidon.rdagent.budget_guard import BudgetExceeded, BudgetGuard

    monkeypatch.setattr(litellm, "ACC_COST", 0.0)
    run = types.SimpleNamespace(cancel_requested=False, cancel_reason=None, token_cost_acc_usd=None)
    guard = BudgetGuard(_Session(run), "12345678-1234-5678-1234-567812345678", 1.0)
    guard._baseline_cost = 0.0

    guard.check_before_call(0.45)
    guard.charge_reserve(0.45)
    guard.check_before_call(0.45)
    guard.charge_reserve(0.45)

    assert guard.current_cost() == pytest.approx(0.9)
    with pytest.raises(BudgetExceeded, match="reserve exceeds cap"):
        guard.check_before_call(0.2)
