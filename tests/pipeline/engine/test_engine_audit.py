"""Truncation audit (structural no-look-ahead proof) and determinism of the engine.

The strategies are compiled spec v3 documents (the §3.2 acceptance spec without the volatility
cap, MA 50) on synthetic per-contract data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import Decision, EngineConfig, FuturesInput, run, truncation_audit
from poseidon.pipeline.engine import simulator as sim
from poseidon.pipeline.v3.compiler import plan, strategy_factory
from poseidon.pipeline.v3.runtime import Conventions

from ..spec_support import resolved_spec
from .support import TMF, synthetic_contracts

FRAME, EXPIRIES, INDEX = synthetic_contracts()
FUTURES = {"core": FuturesInput(spec=TMF, bars=FRAME, expiries=EXPIRIES, roll_days_before=1)}
START = pd.Timestamp("2024-04-01")

adjusted_strategy = strategy_factory(plan(resolved_spec(ma_days=50, vol_k=None)))
# levtx-style: signal on the index aux series, one row lagged, round to nearest
aux_strategy = strategy_factory(
    plan(resolved_spec(ma_days=50, band=0.01, vol_k=None, rounding="nearest", signal="index:taiex")),
    Conventions(signal_lag=1),
)


def test_synthetic_fixture_exercises_rolls_and_regime_switches():
    res = run(adjusted_strategy(), FUTURES, EngineConfig(start=START))
    assert (res.events["kind"] == "roll").sum() >= 8
    assert res.fills["reason"].eq("roll").sum() >= 16
    levels = res.positions_daily["target_leverage"].dropna().unique()
    assert set(levels) == {3.0, 0.5}  # both regimes occur


@pytest.mark.parametrize("execution", ["next_open", "signal_close"])
@pytest.mark.parametrize("factory,aux", [(adjusted_strategy, {}), (aux_strategy, {"taiex": INDEX})])
def test_truncation_audit_passes_at_20_cuts(execution, factory, aux):
    report = truncation_audit(factory, FUTURES, EngineConfig(execution=execution, start=START), aux=aux, n_cuts=20)
    assert len(report.cuts) == 20
    assert report.mismatches == ()
    assert report.ok


def test_truncation_audit_catches_an_engine_that_leaks_the_next_close(monkeypatch):
    honest_feed = sim._Run.feed

    def leaky_feed(self, i, t):
        honest_feed(self, i, t)
        for k, p in self.pos.items():
            nxt = p.table.close[min(i + 1, len(p.table.dates) - 1), int(p.table.held[i])]
            if np.isfinite(nxt):
                self.series[k].columns["close"][self.series[k].n - 1] = nxt

    class CloseProbe:
        """Targets depend on the last close's units digit, so any leaked close changes a decision."""

        def on_bar(self, view, book):
            return Decision({"core": int(view["core"].close[-1]) % 5})

    assert truncation_audit(CloseProbe, FUTURES, EngineConfig(start=START), n_cuts=20).ok
    monkeypatch.setattr(sim._Run, "feed", leaky_feed)
    report = truncation_audit(CloseProbe, FUTURES, EngineConfig(start=START), n_cuts=20)
    assert not report.ok
    assert any("decisions" in m for m in report.mismatches)


@pytest.mark.parametrize("execution", ["next_open", "signal_close"])
def test_runs_are_deterministic(execution):
    cfg = EngineConfig(execution=execution, start=START)
    a = run(aux_strategy(), FUTURES, cfg, aux={"taiex": INDEX})
    b = run(aux_strategy(), FUTURES, cfg, aux={"taiex": INDEX})
    for name in ("fills", "positions_daily", "daily_pnl", "events", "decisions"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))


def test_write_csv_emits_the_four_outputs(tmp_path):
    res = run(adjusted_strategy(), FUTURES, EngineConfig(start=START))
    res.write_csv(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "daily_pnl.csv",
        "events.csv",
        "fills.csv",
        "positions_daily.csv",
    ]
    daily = pd.read_csv(tmp_path / "daily_pnl.csv")
    assert len(daily) == len(res.daily_pnl) and daily["equity"].iloc[-1] == pytest.approx(res.final_equity)
