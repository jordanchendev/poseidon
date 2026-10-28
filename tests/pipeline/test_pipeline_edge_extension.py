"""Grid-edge extension in the optimize stage (owner decision 2026-10-23).

The trials run the real DSL v2 strategy on synthetic bars, but the objective
is stubbed: ``sharpe_daily`` / ``total_pnl_twd`` are a known function of the
parameter values, so the search, the extensions and the neighbour round are
deterministic and their outcome is known in advance.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from poseidon.pipeline import optimization as opt
from poseidon.pipeline.gate_stages import build_gate_stages
from poseidon.pipeline.plan import default_config
from poseidon.pipeline.stages import BacktestStage, OptimizeStage, SpecValidateStage
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.trials import MemoryTrialLedger
from poseidon.strategies.dsl.v2 import validate_v2
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from tests.pipeline.conftest import SyntheticTxSource
from tests.pipeline.gate_support import StageChain

AS_OF = date(2026, 10, 22)


def stop_template(stop_pts: dict, *, tp_r: dict | None = None, stop_also_target: bool = False) -> dict:
    """ORB with a points stop ``$param`` (the run #3 shape), optionally an r-multiple target param."""

    doc = tx_orb30_day()
    doc["name"] = "tx_orb_stop_pts"
    for rule in doc["rules"]:
        rule["risk"]["stop_loss"] = {"type": "points", "value": {"$param": "stop_pts"}}
    doc["params"] = {"stop_pts": stop_pts}
    if stop_also_target:
        doc["risk"]["take_profit"] = {"type": "points", "value": {"$param": "stop_pts"}}
    if tp_r is not None:
        doc["risk"]["take_profit"] = {"type": "r_multiple", "value": {"$param": "tp_r"}}
        doc["params"]["tp_r"] = tp_r
    return doc


def int_param(low, high, step, default):
    return {"type": "int", "low": low, "high": high, "step": step, "default": default}


def values_of(strategy) -> dict:
    risk = strategy.config.rules[0].risk
    out = {"stop_pts": risk.stop_loss.value}
    take_profit = strategy.config.risk.take_profit
    if take_profit.type == "r_multiple":
        out["tp_r"] = take_profit.value
    return out


@pytest.fixture
def objective(monkeypatch):
    """``objective.set(fn)``: trials report ``sharpe_daily = fn(values)``, pnl = 1e5 x sharpe."""

    real = opt.run_backtest
    state = {"fn": None}

    def fake(strategy, bars, config, **kwargs):
        outcome = real(strategy, bars, config, **kwargs)
        values = values_of(strategy)
        sharpe = float(state["fn"](values))
        metrics = {**outcome.metrics, "sharpe_daily": sharpe, "total_pnl_twd": 1e5 * sharpe, "n_obs": 50}
        return dataclasses.replace(outcome, metrics=metrics)

    monkeypatch.setattr(opt, "run_backtest", fake)

    class Handle:
        @staticmethod
        def set(fn):
            state["fn"] = fn

    return Handle


def run_optimize(tmp_path, spec, *, frozen_without_edge_key=False, **optimize):
    config = default_config(stage_plan=None, as_of=AS_OF)
    if frozen_without_edge_key:  # a run frozen before param_edge_at_bound_pass (run #6, run #3)
        del config["thresholds"]["param_edge_at_bound_pass"]
    config["optimize"].update(optimize)
    source = SyntheticTxSource()
    chain = StageChain(tmp_path, spec, config, ledger=MemoryTrialLedger())
    out = chain.through([SpecValidateStage(), BacktestStage(source), OptimizeStage(source)])
    assert [result.verdict for _name, result in out] == [Verdict.PASS] * 3, out[-1][1].reason
    return chain


def gate(chain, name):
    return chain.run(build_gate_stages(data_source=SyntheticTxSource())[name])


def peak(at: float, width: float = 100.0):
    return lambda values: 1.0 - ((values["stop_pts"] - at) / width) ** 2


def by_source(report, source):
    return [row for row in report["edge_extension"]["steps"] if row["param"] == source]


# ------------------------------------------------------------------ the cases


def test_best_on_the_upper_edge_is_extended_and_every_extension_is_a_trial(tmp_path, objective):
    objective.set(peak(80))
    chain = run_optimize(tmp_path, stop_template(int_param(20, 60, 10, 40)))
    report = chain.outputs["optimize"].report
    steps = report["edge_extension"]["steps"]
    assert [(s["value"], s["became_best"]) for s in steps] == [(70, True), (80, True), (90, False)]
    assert all(s["direction"] == 1 and s["param"] == "stop_pts" for s in steps)
    stops = report["edge_extension"]["stops"]
    assert stops[-1]["stopped_because"] == "not_better" and stops[-1]["value"] == 90
    # Every extension is an ordinary ledger trial, numbered after the search.
    records = chain.ledger.records
    assert report["n_trials_by_source"] == {"base": 0, "search": 5, "edge_extension": 3, "neighbour": 0}
    assert len(records) == report["n_trials"] == 8
    assert {s["trial_no"] for s in steps} <= {r.trial_no for r in records}
    assert sorted(r.params["stop_pts"] for r in records) == [20, 30, 40, 50, 60, 70, 80, 90]
    # The best and the concrete best spec carry the extended value; the declared grid is unchanged.
    assert report["best"]["params"] == {"stop_pts": 80}
    assert report["best_spec"]["rules"][0]["risk"]["stop_loss"]["value"] == 80
    validate_v2(report["best_spec"])
    assert report["param_grid"]["stop_pts"]["grid"] == [20, 30, 40, 50, 60]
    assert chain.outputs["spec_validate"].report["template"]["params"]["stop_pts"]["high"] == 60
    effective = report["effective_range"]["stop_pts"]
    assert (effective["declared_low"], effective["declared_high"], effective["low"], effective["high"]) == (
        20,
        60,
        20,
        90,
    )
    assert effective["extended_above"] and not effective["extended_below"]
    # The +/-1 neighbourhood of the final best is complete (both are extension trials).
    rows = {row["direction"]: row for row in report["neighbours"]}
    assert rows[-1]["params"] == {"stop_pts": 70} and rows[1]["params"] == {"stop_pts": 90}
    assert all(row["in_grid"] and not row["on_declared_grid"] for row in rows.values())
    assert report["neighbourhood_settled"]
    assert report["edge_extension"]["better_beyond_cap"] is False
    assert report["edge_extension"]["better_beyond_cap_points"] == []
    stability = gate(chain, "param_stability")
    assert stability.verdict == Verdict.PASS
    assert stability.report["gate_report"]["evaluated"] == 2
    assert stability.report["inputs"]["effective_range"]["stop_pts"]["high"] == 90


def test_the_cap_is_absolute_a_better_point_beyond_it_is_only_a_neighbour(tmp_path, objective):
    objective.set(peak(100, width=200))  # keeps improving up to 100, one step beyond the cap
    chain = run_optimize(tmp_path, stop_template(int_param(20, 60, 10, 40)))
    report = chain.outputs["optimize"].report
    extension = report["edge_extension"]
    assert [s["value"] for s in extension["steps"]] == [70, 80, 90] and all(
        s["became_best"] for s in extension["steps"]
    )
    assert [s["step"] for s in extension["steps"]] == [1, 2, 3]
    assert {"stopped_because": "max_steps", "value": 100}.items() <= extension["stops"][0].items()
    # 100 is evaluated for the neighbourhood (a trial) but never becomes the best.
    assert report["best"]["params"] == {"stop_pts": 90}
    assert report["best_spec"]["rules"][0]["risk"]["stop_loss"]["value"] == 90
    assert report["n_trials_by_source"] == {"base": 0, "search": 5, "edge_extension": 3, "neighbour": 1}
    rows = {row["direction"]: row for row in report["neighbours"]}
    assert rows[1]["params"] == {"stop_pts": 100} and rows[1]["in_grid"] and rows[1]["beyond_cap"]
    assert rows[-1]["params"] == {"stop_pts": 80} and not rows[-1]["beyond_cap"]
    assert report["neighbourhood_settled"]
    # The owner sees that the grid was too narrow.
    assert extension["better_beyond_cap"] is True
    (point,) = extension["better_beyond_cap_points"]
    assert point["params"] == {"stop_pts": 100} and point["source"] == "neighbour"
    assert point["sharpe_daily"] > report["best"]["metrics"]["sharpe_daily"] == point["best_sharpe_daily_then"]
    effective = report["effective_range"]["stop_pts"]
    assert (effective["high"], effective["best_cap_low"], effective["best_cap_high"]) == (100, -10, 90)
    stability = gate(chain, "param_stability")
    assert stability.verdict == Verdict.PASS and stability.report["gate_report"]["evaluated"] == 2
    assert stability.report["inputs"]["better_beyond_cap"] is True
    # DSR N counts the beyond-cap neighbour like every other trial.
    assert gate(chain, "dsr").report["gate_report"]["n_trials"] == len(chain.ledger.records) == 9


def test_inside_the_cap_the_neighbour_round_still_moves_the_best():
    spec = stop_template(int_param(20, 60, 10, 40))
    grid = opt.ParamGrid(validate_v2(spec).params)
    # Search saw 20..50 only (60 never evaluated); the best 50 moves to 60, then extends to 90.
    runner = _StubRunner(lambda v: v["stop_pts"] / 100)
    for value in (20, 30, 40, 50):
        runner.evaluate({"stop_pts": value}, "search")
    extension = opt.EdgeExtension(admissible=opt.hard_bound_check(spec, grid))
    rows, settled = opt.complete_neighbourhood(runner, grid, 1, extension)
    assert runner.best.params == {"stop_pts": 90} and settled
    assert [r.source for r in runner.results[4:]] == [
        "neighbour",
        "edge_extension",
        "edge_extension",
        "edge_extension",
        "neighbour",
    ]
    assert [r.params["stop_pts"] for r in runner.results[4:]] == [60, 70, 80, 90, 100]
    assert extension.report()["better_beyond_cap"] and extension.better_beyond_cap[0]["params"] == {"stop_pts": 100}
    assert {row["params"]["stop_pts"]: row["beyond_cap"] for row in rows} == {80: False, 100: True}
    assert len(runner.ledger.records) == 9


def test_a_hard_bound_stops_the_extension_and_the_gate_rule_applies(tmp_path, objective):
    objective.set(lambda values: 1.0 - values["stop_pts"] / 100)  # best at the lowest stop
    chain = run_optimize(tmp_path, stop_template(int_param(5, 25, 10, 15)), frozen_without_edge_key=True)
    report = chain.outputs["optimize"].report
    assert report["best"]["params"] == {"stop_pts": 5}
    assert report["edge_extension"]["steps"] == []
    (stop,) = report["edge_extension"]["stops"]
    assert stop["stopped_because"] == "hard_bound" and stop["value"] == -5
    assert "stop_loss.points.value" in stop["detail"] or "stop_loss" in stop["detail"]
    rows = {row["direction"]: row for row in report["neighbours"]}
    assert not rows[-1]["in_grid"] and rows[-1]["params"] is None and rows[-1]["trial_no"] is None
    assert "stop_loss" in rows[-1]["excluded_because"]
    assert rows[1]["in_grid"] and rows[1]["params"] == {"stop_pts": 15}
    assert report["n_trials"] == 3  # no trial for the excluded neighbour
    stability = gate(chain, "param_stability")
    gate_report = stability.report["gate_report"]
    assert stability.verdict == Verdict.FAIL and stability.report["reason_code"] == "insufficient_neighbours"
    assert stability.report["reason"] == "1 evaluated neighbours < 2"  # the run #3 failure, unchanged
    assert (gate_report["evaluated"], gate_report["excluded"], gate_report["min_neighbours"]) == (1, 1, 2)
    assert stability.report["inputs"]["excluded_neighbours"][0]["direction"] == -1
    assert "excluded_kind" not in rows[-1] and "edge_at_bound" not in gate_report  # key absent: nothing tagged


def test_a_dsl_config_with_the_key_passes_the_same_edge_at_bound_case_on_its_in_range_neighbours(tmp_path, objective):
    objective.set(lambda values: 1.0 - values["stop_pts"] / 100)  # run #3's shape: best on the validator's lower bound
    chain = run_optimize(tmp_path, stop_template(int_param(5, 25, 10, 15)))
    report = chain.outputs["optimize"].report
    rows = {row["direction"]: row for row in report["neighbours"]}
    assert report["best"]["params"] == {"stop_pts": 5} and not rows[-1]["in_grid"]
    assert rows[-1]["excluded_kind"] == opt.EXCLUDED_COMPONENT_BOUND and "excluded_kind" not in rows[1]
    stability = gate(chain, "param_stability")
    assert stability.verdict == Verdict.PASS
    assert stability.report["gate_report"]["edge_at_bound"] == ["stop_pts"]
    assert (stability.report["gate_report"]["evaluated"], stability.report["gate_report"]["excluded"]) == (1, 1)
    assert "edge_at_bound: stop_pts" in stability.report["summary"]


def test_a_dsl_in_range_neighbour_that_loses_money_still_fails_with_the_key(tmp_path, objective):
    objective.set(lambda values: 1.0 if values["stop_pts"] == 5 else -1.0)  # the only in-range neighbour loses
    chain = run_optimize(tmp_path, stop_template(int_param(5, 25, 10, 15)))
    stability = gate(chain, "param_stability")
    assert stability.verdict == Verdict.FAIL and stability.report["reason_code"] == "unstable_neighbourhood"
    assert stability.report["gate_report"]["edge_at_bound"] == ["stop_pts"]


def test_a_dsl_owner_set_range_edge_is_not_a_hard_bound_with_the_key(tmp_path, objective):
    # best on the declared edge (60) well inside the validator's bounds: the extension evaluates past it, no tag
    objective.set(peak(60))
    chain = run_optimize(tmp_path, stop_template(int_param(20, 60, 10, 40)))
    report = chain.outputs["optimize"].report
    assert all("excluded_kind" not in row for row in report["neighbours"])
    assert "edge_at_bound" not in gate(chain, "param_stability").report["gate_report"]


def test_every_field_the_param_feeds_bounds_the_extension(tmp_path, objective):
    # stop_pts feeds the points stop (<= 500) and the points target (<= 1000): 500 is fine, 520 is not.
    objective.set(lambda values: values["stop_pts"] / 1000)
    chain = run_optimize(tmp_path, stop_template(int_param(440, 480, 20, 460), stop_also_target=True))
    report = chain.outputs["optimize"].report
    assert [s["value"] for s in report["edge_extension"]["steps"]] == [500]
    stop = report["edge_extension"]["stops"][0]
    assert (stop["stopped_because"], stop["value"]) == ("hard_bound", 520)
    assert "stop_loss" in stop["detail"] and "take_profit" not in stop["detail"]
    assert report["best"]["params"] == {"stop_pts": 500}
    assert report["best_spec"]["risk"]["take_profit"]["value"] == 500
    rows = {row["direction"]: row for row in report["neighbours"]}
    assert rows[-1]["in_grid"] and not rows[1]["in_grid"] and "stop_loss" in rows[1]["excluded_because"]


def test_both_params_on_edges_are_extended(tmp_path, objective):
    objective.set(lambda v: 2.0 - ((v["stop_pts"] - 80) / 100) ** 2 - (v["tp_r"] - 3.0) ** 2)
    spec = stop_template(
        int_param(20, 60, 20, 40), tp_r={"type": "float", "low": 1.0, "high": 2.0, "step": 0.5, "default": 1.5}
    )
    chain = run_optimize(tmp_path, spec)
    report = chain.outputs["optimize"].report
    steps = report["edge_extension"]["steps"]
    assert {(s["param"], s["value"]) for s in steps} == {
        ("stop_pts", 80),
        ("stop_pts", 100),
        ("tp_r", 2.5),
        ("tp_r", 3.0),
        ("tp_r", 3.5),
    }
    assert report["best"]["params"] == {"stop_pts": 80, "tp_r": 3.0}
    assert report["best_spec"]["rules"][0]["risk"]["stop_loss"]["value"] == 80
    assert report["best_spec"]["risk"]["take_profit"]["value"] == 3.0
    effective = report["effective_range"]
    assert (effective["stop_pts"]["high"], effective["tp_r"]["high"]) == (100, 3.5)
    assert all(row["in_grid"] for row in report["neighbours"]) and len(report["neighbours"]) == 4
    assert gate(chain, "param_stability").report["gate_report"]["evaluated"] == 4


def test_dsr_n_counts_the_extension_trials(tmp_path, objective):
    objective.set(peak(80))
    chain = run_optimize(tmp_path, stop_template(int_param(20, 60, 10, 40)))
    report = chain.outputs["optimize"].report
    assert report["n_trials_by_source"]["edge_extension"] == 3
    dsr = gate(chain, "dsr")
    assert dsr.report["gate_report"]["n_trials"] == len(chain.ledger.records) == 8


def test_the_stage_budget_stops_extending_and_records_why(tmp_path, objective):
    objective.set(peak(80))
    chain = run_optimize(tmp_path, stop_template(int_param(20, 60, 10, 60)), budget_seconds=1e-6)
    report = chain.outputs["optimize"].report
    assert report["search"]["stopped_because"] == "budget"
    assert report["edge_extension"]["steps"] == [] and report["n_trials_by_source"]["edge_extension"] == 0
    stops = report["edge_extension"]["stops"]
    assert stops and all(stop["stopped_because"] == "budget" and "budget" in stop["detail"] for stop in stops)
    # The neighbour round is reserved (not budget-checked, as before): it still completes the
    # neighbourhood, and a better neighbour inside the hard bounds still moves the best.
    assert report["n_trials_by_source"]["neighbour"] >= 2
    assert all(row["in_grid"] for row in report["neighbours"]) and report["neighbourhood_settled"]


class _StubRunner(opt.TrialRunner):
    """TrialRunner whose evaluation is a pure function of the params (no strategy, no bars)."""

    def __init__(self, fn):
        super().__init__(
            build=None,
            bars=None,
            backtest_config={},
            ledger=MemoryTrialLedger(),
            window_sha256="w" * 64,
            should_stop=lambda: False,
            check_stop=lambda: None,
        )
        self.fn = fn

    def evaluate(self, params, source):
        sharpe = self.fn(params)
        return self.record(params, source, "complete", {"sharpe_daily": sharpe, "total_pnl_twd": 1e5 * sharpe})


def test_extend_edges_stops_when_the_next_extension_would_overrun_the_budget():
    spec = stop_template(int_param(20, 60, 10, 40))
    grid = opt.ParamGrid(validate_v2(spec).params)
    runner = _StubRunner(lambda v: v["stop_pts"] / 100)  # monotone: every extension is better
    for value in (20, 30, 40, 50, 60):
        runner.evaluate({"stop_pts": value}, "search")
    # 10 s per trial; the check needs elapsed + 10 s x (1 + 2 reserved neighbours) <= 85 s.
    extension = opt.EdgeExtension(
        admissible=opt.hard_bound_check(spec, grid),
        budget_seconds=85,
        started=0.0,
        reserve_trials=2,
        clock=lambda: 10.0 * len(runner.results),
    )
    opt.extend_edges(runner, grid, extension)
    assert [s["value"] for s in extension.steps] == [70]  # 50 + 30 <= 85; then 60 + 30 > 85
    assert extension.stops[-1]["stopped_because"] == "budget" and extension.stops[-1]["value"] == 80
    assert len(runner.ledger.records) == 6 and runner.results[-1].source == "edge_extension"


def test_without_extension_the_neighbour_round_keeps_the_old_exclusion():
    spec = stop_template(int_param(20, 60, 10, 60))
    grid = opt.ParamGrid(validate_v2(spec).params)
    runner = _StubRunner(lambda v: v["stop_pts"] / 100)
    for value in (20, 30, 40, 50, 60):
        runner.evaluate({"stop_pts": value}, "search")
    rows = opt.evaluate_neighbours(runner, grid, 1)
    assert [(row["direction"], row["in_grid"], row["excluded_because"]) for row in rows] == [
        (-1, True, None),
        (1, False, "outside the declared grid"),
    ]


def test_off_grid_values_build_a_valid_strategy_without_touching_the_template():
    from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory

    spec = stop_template(int_param(20, 60, 10, 40))
    grid = opt.ParamGrid(validate_v2(spec).params)
    factory = DslV2StrategyFactory(spec=spec)
    build = opt.off_grid_builder(factory, spec, grid, lambda t: DslV2StrategyFactory(spec=t))
    assert values_of(build({"stop_pts": 90}))["stop_pts"] == 90
    assert values_of(build({"stop_pts": 40}))["stop_pts"] == 40
    assert spec["params"]["stop_pts"]["high"] == 60
    with pytest.raises(ValueError):
        factory.build({"stop_pts": 90})  # the declared factory still refuses off-grid values
    check = opt.hard_bound_check(spec, grid)
    assert check({"stop_pts": 500}) is None and "stop_loss" in check({"stop_pts": 510})
