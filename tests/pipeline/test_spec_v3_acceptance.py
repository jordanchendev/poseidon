"""The §3.2 acceptance spec end to end on the real refdata store (design 2026-10-24 §10 step 3).

Runs where the store exists (stormtrooper: mount ``~/strategy-pipeline/refdata`` read-only and set
``POSEIDON_PIPELINE_REFDATA_ROOT``). The spec runs exactly as written (TMF priced from TX, both
signals on ``index:taiex``, ``execution.next_open``, floor) with its default ``$param`` values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from poseidon.pipeline.refdata.datasets import FUTURES
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec, resolve
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.registry import REGISTRY
from poseidon.pipeline.v3.runtime import RollCalendar
from poseidon.pipeline.v3.validation import validate_spec_v3
from poseidon.pipeline.v3.validation_types import FeasibilityContext

STORE = RefdataStore()
pytestmark = pytest.mark.skipif(not STORE.exists(FUTURES), reason="refdata store not available")
WINDOWS = {"2014-11..now": ("2014-11-03", None), "2000..2014": ("2000-01-04", "2014-10-31")}


@pytest.fixture(scope="module", params=list(WINDOWS))
def accepted(request):
    start, end = WINDOWS[request.param]
    compiled = compile_spec(resolve(tmf_dynamic_leverage_3x()), start=start, end=end, store=STORE)
    strategy = compiled.strategy()
    return compiled, strategy, compiled.run(strategy)


def test_validator_accepts_tmf_and_points_tx_to_the_smaller_contracts(accepted):
    compiled, _strategy, res = accepted
    closes = res.positions_daily["close"].astype(float)
    feasibility = FeasibilityContext(
        start=compiled.config.start.date(),
        end=res.equity.index[-1].date(),
        prices={"TX": (float(closes.iloc[0]), float(closes.max()))},
    )
    checked = validate_spec_v3(tmf_dynamic_leverage_3x(), feasibility=feasibility)
    if compiled.config.start < pd.Timestamp("2000-01-20"):
        # The validator's warm-up estimate (245 trading days a year, at the largest ma_days 250) needs TAIEX
        # from 1998-12; 1999 had Saturday sessions, so the compiler's exact count (266 rows) is enough.
        assert {issue.path for issue in checked.errors} == {f"sleeves.0.leverage.{i}.params.signal" for i in (1, 2)}
        assert "data starts 1999-01-05" in checked.errors[0].msg
        feasibility = FeasibilityContext(prices=feasibility.prices)
        checked = validate_spec_v3(tmf_dynamic_leverage_3x(), feasibility=feasibility)
    assert checked.ok, checked.errors
    assert compiled.plan.signal_indices == ("taiex",)
    assert any("TMF: priced with TX prices" in note for note in checked.notes)
    tx = tmf_dynamic_leverage_3x()
    tx["instruments"]["core"]["symbol"] = "TX"
    errors = [issue.text() for issue in validate_spec_v3(tx, feasibility=feasibility).errors]
    assert len(errors) == 1 and "sleeves.0.sizing" in errors[0] and "MTX" in errors[0] and "TMF" in errors[0]


def test_acceptance_spec_runs_end_to_end(accepted):
    compiled, strategy, res = accepted
    assert res.ruined is None
    assert not set(res.events["kind"]) & {"forced_liquidation", "margin_call", "ruin"}
    pos = res.positions_daily.set_index("date")
    assert pos.index[0] >= compiled.config.start and pos["proxy"].all()
    # floor sizing never asks for more than the target leverage at the decision close
    asked = pos.dropna(subset=["target_units"])
    assert (asked["implied_target_leverage"] <= asked["target_leverage"] + 1e-9).all()
    assert set(pos["target_leverage"].round(6)) - {0.5} and pos["target_leverage"].max() <= 3.0
    assert res.fills.loc[res.fills["reason"] == "rebalance", "session"].eq("open").all()
    assert len(strategy.state("main", "leverage.1")["switches"]) > 0


def test_engine_rolls_exactly_when_the_roll_component_says(accepted):
    compiled, _strategy, res = accepted
    held = res.positions_daily.set_index("date")["contract"]
    engine_rolls = list(held.index[held != held.shift(1)][1:])
    fut = compiled.futures["core"]
    calendar = RollCalendar(compiled.calendar, tuple(pd.Timestamp(e) for e in fut.expiries.values()))
    roll = REGISTRY["roll.before_settlement"]
    params = roll.parse({"days_before": fut.roll_days_before})
    due = [t for t in held.index[1:] if roll.roll_due(params, session_date=t, calendar=calendar)]
    assert len(engine_rolls) > 100 and due == engine_rolls


def test_compiled_acceptance_spec_passes_the_truncation_audit_at_20_cuts(accepted):
    compiled, _strategy, _res = accepted
    report = compiled.audit(n_cuts=20, seed=20261024)
    print("cuts:", [str(c.date()) for c in report.cuts], "mismatches:", report.mismatches)
    assert len(report.cuts) == 20 and report.ok
