"""param_stability when the best sits on a component's own hard bound (run #18, owner decision 2026-10-28).

Run #18 searched ``$lookback_days`` 125/250/375/500 of ``leverage.drawdown_ladder`` (hard range 20-500). The best
was 500, its +1 neighbour 625 is past the component's bound, and the only neighbour left (375) was profitable:
``insufficient_neighbours``. Runs frozen with ``param_edge_at_bound_pass`` judge the in-range neighbours instead.
"""

from __future__ import annotations

import copy
import json
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from poseidon.pipeline import optimization as opt
from poseidon.pipeline.daily_stages import DailySpecValidateStage, SpecRunner
from poseidon.pipeline.gate_stages import ParamStabilityStage
from poseidon.pipeline.gates import param_stability as gate
from poseidon.pipeline.plan import DAILY_THRESHOLDS, class_settings, effective_config
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.stages import StageContext
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.trials import MemoryTrialLedger
from poseidon.pipeline.v3.validation import validate_spec_v3

FIXTURES = Path(__file__).parent / "fixtures"
LADDER = json.loads((FIXTURES / "spec_v3_ladder_vix.json").read_text())
OLD = {"param_neighbourhood_profitable_min": 0.6, "param_min_neighbours": 2}
NEW = {**OLD, gate.EDGE_AT_BOUND_KEY: True}


def ladder_spec(low=125, high=500, step=125, default=250) -> dict:
    spec = copy.deepcopy(LADDER)
    spec["sleeves"][0]["leverage"].pop()  # the VIX level ladder
    spec["sleeves"][0]["leverage"][1]["params"]["signal"] = "core:adjusted"
    spec["params"]["lookback"] = {"type": "int", "low": low, "high": high, "step": step, "default": default}
    return spec


class StubTrials(opt.TrialRunner):
    def __init__(self, pnl_of):
        super().__init__(
            build=None,
            bars=None,
            backtest_config={},
            ledger=MemoryTrialLedger(),
            window_sha256="w" * 64,
            should_stop=lambda: False,
            check_stop=lambda: None,
        )
        self.pnl_of = pnl_of

    def evaluate(self, params, source):
        pnl = self.pnl_of(params["lookback"])
        return self.record(params, source, "complete", {"sharpe_daily": pnl / 1e6, "total_pnl_twd": pnl})


def optimized(pnl_of, thresholds, spec=None, extend=True) -> dict:
    """The optimize stage's report fields param_stability reads, from a real SpecRunner admissibility check."""

    spec = spec or ladder_spec()
    grid = opt.ParamGrid(validate_spec_v3(spec).params)
    runner = SpecRunner(template=spec, grid=grid, store=None, start=date(2015, 1, 1), end=date(2020, 1, 1), basis=None)
    trials = StubTrials(pnl_of)
    for value in grid.specs["lookback"].grid():
        trials.evaluate({"lookback": value}, "search")
    extension = opt.EdgeExtension(admissible=runner.admissible, tag_kind=gate.EDGE_AT_BOUND_KEY in thresholds)
    rows, _settled = opt.complete_neighbourhood(trials, grid, 1, extension if extend else None)
    best = trials.best
    return {"best": {"trial_no": best.trial_no, "params": best.params, "metrics": best.metrics}, "neighbours": rows}


def judge(optimize: dict, thresholds: dict):
    return ParamStabilityStage().evaluate(SimpleNamespace(config={"thresholds": thresholds}), optimize)


RISING = lambda lookback: 1e5 * lookback  # noqa: E731  best at 500, everything profitable


def test_run_18s_case_failed_insufficient_neighbours_and_now_passes_with_edge_at_bound():
    before = judge(optimized(RISING, OLD), OLD)
    assert before.verdict == Verdict.FAIL and before.report["reason_code"] == "insufficient_neighbours"
    assert before.report["reason"] == "1 evaluated neighbours < 2"
    assert "edge_at_bound" not in before.report["gate_report"]

    optimize = optimized(RISING, NEW)
    assert optimize["best"]["params"] == {"lookback": 500}
    high = next(row for row in optimize["neighbours"] if row["direction"] == 1)
    assert high["excluded_kind"] == opt.EXCLUDED_COMPONENT_BOUND and "lookback_days" in high["excluded_because"]
    after = judge(optimize, NEW)
    assert after.verdict == Verdict.PASS
    assert after.report["gate_report"]["edge_at_bound"] == ["lookback"]
    assert after.report["gate_report"]["evaluated"] == 1
    assert "edge_at_bound: lookback" in after.report["summary"]


def test_an_in_range_neighbour_that_loses_money_still_fails():
    optimize = optimized(lambda lookback: 5e5 if lookback == 500 else -1e5, NEW)
    assert optimize["best"]["params"] == {"lookback": 500}
    result = judge(optimize, NEW)
    assert result.verdict == Verdict.FAIL and result.report["reason_code"] == "unstable_neighbourhood"
    assert result.report["gate_report"]["edge_at_bound"] == ["lookback"]


def test_a_spec_declared_edge_inside_the_component_range_is_judged_exactly_as_before():
    spec = ladder_spec(high=375, default=250)  # the component would allow 500: the edge is the spec's own
    peak = lambda lookback: 1e5 * (1000 - abs(lookback - 375))  # noqa: E731
    old, new = optimized(peak, OLD, spec=spec), optimized(peak, NEW, spec=spec)
    assert new["best"]["params"] == {"lookback": 375}  # best on the declared edge; 500 is a trial the extension ran
    assert all(row["in_grid"] and "excluded_kind" not in row for row in new["neighbours"])
    before, after = judge(old, OLD), judge(new, NEW)
    assert after.verdict == before.verdict
    assert "edge_at_bound" not in after.report["gate_report"]
    assert after.report["gate_report"] == before.report["gate_report"]


def test_a_neighbour_outside_the_declared_grid_without_an_extension_is_tagged_declared_grid_and_not_relaxed():
    spec = ladder_spec(high=375, default=250)
    grid = opt.ParamGrid(validate_spec_v3(spec).params)
    trials = StubTrials(RISING)
    for value in (125, 250, 375):
        trials.evaluate({"lookback": value}, "search")
    rows = opt.evaluate_neighbours(trials, grid, 1, tag_kind=True)
    high = next(row for row in rows if row["direction"] == 1)
    assert high["excluded_because"] == "outside the declared grid" and high["excluded_kind"] == "declared_grid"
    best = {"params": trials.best.params, "metrics": trials.best.metrics}
    result = judge({"best": best, "neighbours": rows}, NEW)
    assert result.verdict == Verdict.FAIL and result.report["reason_code"] == "insufficient_neighbours"


def test_a_hard_bound_exclusion_without_the_key_is_not_tagged_and_judged_as_before():
    optimize = optimized(RISING, OLD)
    assert all("excluded_kind" not in row for row in optimize["neighbours"])


def nb(param, step, pnl, kind=None):
    return gate.NeighbourResult(param, step, {} if pnl is None else {param: step}, pnl, kind)


def evaluate(neighbours, thresholds=NEW):
    return gate.evaluate(gate.ParamStabilityInputs({"a": 1, "b": 1}, 100.0, neighbours), thresholds)


def test_the_relaxation_needs_a_component_bound_to_explain_the_shortfall():
    bound = gate.COMPONENT_BOUND
    assert evaluate([nb("a", -1, 5.0), nb("a", 1, None, bound)]).passed
    # another param cut off by the spec's own range: the shortfall is not the component's
    assert not evaluate([nb("a", -1, 5.0), nb("a", 1, None, "declared_grid")]).passed
    assert not evaluate([nb("a", -1, 5.0), nb("a", 1, None, "other")]).passed
    # nothing evaluated: never a pass
    assert not evaluate([nb("a", -1, None, bound), nb("a", 1, None, bound)]).passed
    # enough neighbours: today's 60 % share rule decides, the bound changes nothing
    three = [nb("a", -1, 5.0), nb("b", -1, 5.0), nb("b", 1, -1.0), nb("a", 1, None, bound)]
    assert evaluate(three).passed and evaluate(three, OLD).passed
    bad = [nb("a", -1, 5.0), nb("b", -1, -5.0), nb("b", 1, -1.0), nb("a", 1, None, bound)]
    assert evaluate(bad).reason_code == "unstable_neighbourhood"
    # key absent: the old rule, even with a tagged row
    assert evaluate([nb("a", -1, 5.0), nb("a", 1, None, bound)], OLD).reason_code == "insufficient_neighbours"


# ----------------------------------------------------------------------------- validator


def test_the_validator_warns_when_a_param_range_touches_a_component_bound_without_failing():
    result = validate_spec_v3(ladder_spec())
    assert result.ok and len(result.warnings) == 1
    assert "params.lookback" in result.warnings[0] and "high 500" in result.warnings[0]
    assert "lookback_days" in result.warnings[0] and "hard bound 500" in result.warnings[0]
    low = validate_spec_v3(ladder_spec(low=20, high=200, step=60, default=80))
    assert low.ok and len(low.warnings) == 1 and "low 20" in low.warnings[0]
    assert validate_spec_v3(ladder_spec(low=60, high=240, step=60, default=120)).warnings == ()


def _validate(tmp_path, config, spec):
    stage = DailySpecValidateStage(refdata=lambda: RefdataStore(tmp_path / "refdata"))
    ctx = StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="",
        input_spec_json=spec,
        config=config,
        config_sha256="0" * 64,
        stage="spec_validate",
        attempt=1,
        upstream={},
    )
    return stage.evaluate(ctx)


def test_spec_validate_surfaces_the_warning_only_for_runs_frozen_with_the_key(tmp_path, monkeypatch):
    from poseidon.pipeline import daily_stages

    real = daily_stages.validate_spec_v3
    monkeypatch.setattr(daily_stages, "validate_spec_v3", lambda spec, **kw: real(spec))  # no refdata here
    monkeypatch.setattr(daily_stages, "price_points", lambda *args: None)
    spec = ladder_spec()
    base = {"strategy_class": "daily_equity", "data": {"start": "2015-01-01", "end": "2020-01-01"}}
    base["data"].update({"holdout_start": "2019-01-01"})
    frozen_new = {**base, "thresholds": {gate.EDGE_AT_BOUND_KEY: True}}
    frozen_old = {**base, "thresholds": {}}
    new = _validate(tmp_path, frozen_new, spec)
    assert new.verdict == Verdict.PASS and "1 warning(s)" in new.report["summary"]
    assert new.report["warnings"] and "hard bound 500" in new.report["warnings"][0]
    old = _validate(tmp_path, frozen_old, spec)
    assert "warnings" not in old.report and "warning" not in old.report["summary"]


# ----------------------------------------------------------------------------- frozen runs


def test_new_daily_configs_freeze_the_key_and_the_gate_key_matches_the_table():
    assert DAILY_THRESHOLDS[gate.EDGE_AT_BOUND_KEY] is True
    assert class_settings("daily_equity")["thresholds"][gate.EDGE_AT_BOUND_KEY] is True


def _terminal_configs():
    runs = json.loads((FIXTURES / "v3_terminal_runs.json").read_text())["runs"]
    return {no: effective_config(run["config_json"], run["route"]) for no, run in runs.items()}


def test_terminal_runs_never_froze_the_key():
    for no, config in _terminal_configs().items():
        assert gate.EDGE_AT_BOUND_KEY not in config["thresholds"], no
    run6 = (FIXTURES / "run6_frozen_v4.json").read_text()
    assert gate.EDGE_AT_BOUND_KEY not in run6


@pytest.mark.parametrize("run_no", ["10", "16"])
def test_a_terminal_run_judges_param_stability_by_the_old_rule_even_on_a_tagged_row(run_no):
    thresholds = _terminal_configs()[run_no]["thresholds"]
    neighbours = [nb("a", -1, 5.0), nb("a", 1, None, gate.COMPONENT_BOUND)]
    result = gate.evaluate(gate.ParamStabilityInputs({"a": 1}, 1.0, neighbours), thresholds)
    assert result.reason_code == "insufficient_neighbours" and "edge_at_bound" not in result.report
