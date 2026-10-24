"""Truncation audit (structural no-look-ahead proof) and determinism of the engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import EngineConfig, FuturesInput, run, truncation_audit
from poseidon.pipeline.engine import simulator as sim

from .support import TMF, MaBandLeverage, synthetic_contracts

FRAME, EXPIRIES, INDEX = synthetic_contracts()
FUTURES = {"TMF": FuturesInput(spec=TMF, bars=FRAME, expiries=EXPIRIES, roll_days_before=1)}
START = pd.Timestamp("2024-04-01")


def adjusted_strategy():
    return MaBandLeverage(START, ma_days=50, band=0.02, rounding="floor")


def aux_strategy():
    return MaBandLeverage(START, aux="index", ma_days=50, band=0.01)


def test_synthetic_fixture_exercises_rolls_and_regime_switches():
    res = run(adjusted_strategy(), FUTURES, EngineConfig(start=START))
    assert (res.events["kind"] == "roll").sum() >= 8
    assert res.fills["reason"].eq("roll").sum() >= 16
    levels = res.positions_daily["target_leverage"].dropna().unique()
    assert set(levels) == {3.0, 0.5}  # both regimes occur


@pytest.mark.parametrize("execution", ["next_open", "signal_close"])
@pytest.mark.parametrize("factory,aux", [(adjusted_strategy, {}), (aux_strategy, {"index": INDEX})])
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

    monkeypatch.setattr(sim._Run, "feed", leaky_feed)
    report = truncation_audit(adjusted_strategy, FUTURES, EngineConfig(start=START), n_cuts=20)
    assert not report.ok
    assert any("decisions" in m for m in report.mismatches)


@pytest.mark.parametrize("execution", ["next_open", "signal_close"])
def test_runs_are_deterministic(execution):
    cfg = EngineConfig(execution=execution, start=START)
    a = run(aux_strategy(), FUTURES, cfg, aux={"index": INDEX})
    b = run(aux_strategy(), FUTURES, cfg, aux={"index": INDEX})
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
