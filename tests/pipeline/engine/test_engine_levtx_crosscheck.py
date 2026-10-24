"""Cross-check against the offline backtest ``local_dev/scripts/leveraged-tx/levtx.py`` (design §10 step 3).

Runs only when both are available (stormtrooper):
  LEVTX_SCRIPT=/path/levtx.py  LEVTX_DATA=/path/levtx-data  (the directory holding ext/)
Optional: LEVTX_REPORT=/path/report.json to keep the numbers.

levtx's "3x daily + 200DMA ±2%" is rebuilt as a target strategy (``MaBandLeverage`` with the TAIEX
price index as an aux series) and run through the engine under levtx's conventions:
``signal_close`` execution with the regime lagged one TAIEX day (levtx decides from close t-1 and
trades at close t), round-to-nearest sizing, TMF specs, roll at the close of the day before final
settlement, expiries = each contract's last day-session row (alive contracts never expire).
Both read the same TAIFEX/TWSE files through levtx's own loaders.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.engine import EngineConfig, FuturesInput, run, truncation_audit

from .support import TMF, MaBandLeverage

SCRIPT = os.environ.get("LEVTX_SCRIPT")
DATA = os.environ.get("LEVTX_DATA")
pytestmark = pytest.mark.skipif(not (SCRIPT and DATA), reason="LEVTX_SCRIPT / LEVTX_DATA not set")

WINDOWS = [("2014-11-03", None), ("2000-01-04", "2014-10-31"), ("2020-01-02", None), ("2008-01-02", None)]
REPORT: dict = {}


@pytest.fixture(scope="module")
def levtx():
    spec = importlib.util.spec_from_file_location("levtx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["levtx"] = module  # its dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    module.DATA, module.EXT = DATA, os.path.join(DATA, "ext")
    return module


@pytest.fixture(scope="module")
def market(levtx):
    tx = levtx.load_taifex()
    mk = levtx.build_r1(tx)
    sig = levtx.load_twse("MI_5MINS_HIST", ["open", "high", "low", "close"])
    end = mk.index.intersection(sig.index)[-1]

    day = tx[tx["session"] == "一般"]
    night = tx[tx["session"] != "一般"]
    frame = day[["date", "cm", "open", "high", "low", "close", "settle"]].rename(columns={"cm": "contract"})
    ext = night.groupby(["date", "cm"]).agg(night_high=("high", "max"), night_low=("low", "min"))
    frame = frame.merge(ext, left_on=["date", "contract"], right_index=True, how="left")
    frame["high"] = np.fmax(frame["high"], frame["night_high"])
    frame["low"] = np.fmin(frame["low"], frame["night_low"])
    last_day = day.groupby("cm")["date"].max()
    expiries = last_day.where(last_day != day["date"].max(), pd.Timestamp("2100-01-01")).to_dict()
    futures = {
        "TMF": FuturesInput(
            spec=TMF, bars=frame.drop(columns=["night_high", "night_low"]), expiries=expiries, roll_days_before=1
        )
    }
    calendar = list(mk.index)
    return {
        "mk": mk,
        "sig": sig,
        "end": end,
        "futures": futures,
        "calendar": calendar,
        "aux": {"taiex": sig[["close"]]},
    }


def _levtx_run(levtx, market, start, end):
    spec = levtx.Spec("3x daily + 200DMA±2%", 3, "daily", ma=200, band=0.02, bear_lev=0.5)
    r = levtx.simulate(spec, market["mk"], market["sig"], start, end)
    _, state = levtx.leverage_path(spec, market["sig"], start, end)
    w = market["mk"].loc[start:end]
    units = (r["lev_post"] * r["eq"] / (w["p_new"] * 10.0)).round().astype(int)
    return r, w, units, state.loc[start:end]


def _engine_run(market, start, end, execution="signal_close"):
    strategy = MaBandLeverage(start, aux="taiex")
    res = run(
        strategy,
        market["futures"],
        EngineConfig(execution=execution, start=start, end=end),
        aux=market["aux"],
        calendar=market["calendar"],
    )
    return res, strategy


@pytest.mark.parametrize("start,end", WINDOWS)
def test_engine_matches_levtx(levtx, market, start, end):
    end = pd.Timestamp(end) if end else market["end"]
    r, w, lev_units, lev_state = _levtx_run(levtx, market, start, end)
    res, strategy = _engine_run(market, start, end)
    if r["ruined"]:
        pytest.skip(f"levtx ruined on {r['ruined']}")

    pos = res.positions_daily.set_index("date")
    assert list(pos.index) == list(w.index)
    assert (pos["contract"] == w["held"]).all(), "held contract schedule differs"
    unit_diff = pos["held_units"].astype(int) != lev_units
    rebal_engine = set(pos.index[pos["held_units"].diff().fillna(pos["held_units"]) != 0])
    rebal_levtx = set(w.index[lev_units.diff().fillna(lev_units) != 0])

    ours = pd.Series(dict(strategy.states))
    switches_engine = list(ours.index[ours.diff().fillna(0) != 0])
    switches_levtx = list(lev_state.index[lev_state.diff().fillna(0) != 0])

    eq_engine, eq_levtx = res.equity, r["eq"]
    rel = (eq_engine / eq_levtx - 1).abs()
    final_gap = float(eq_engine.iloc[-1] / eq_levtx.iloc[-1] - 1)

    next_open, _ = _engine_run(market, start, end, execution="next_open")
    REPORT[f"{start}..{end.date()}"] = {
        "days": len(w),
        "unit_mismatch_days": int(unit_diff.sum()),
        "rebalance_days_engine": len(rebal_engine),
        "rebalance_days_levtx": len(rebal_levtx),
        "switch_dates": [str(d.date()) for d in switches_levtx],
        "final_equity_levtx": float(eq_levtx.iloc[-1]),
        "final_equity_engine": float(eq_engine.iloc[-1]),
        "final_gap_pct": final_gap * 100,
        "max_daily_equity_gap_pct": float(rel.max() * 100),
        "engine_margin_events": res.events["kind"].value_counts().to_dict(),
        "levtx_margin_flags": len(r["margin_flags"]),
        "next_open_final_equity": next_open.final_equity,
        "next_open_vs_levtx_pct": (next_open.final_equity / float(eq_levtx.iloc[-1]) - 1) * 100,
    }
    print(json.dumps({f"{start}..{end.date()}": REPORT[f"{start}..{end.date()}"]}, indent=1, default=str))
    if os.environ.get("LEVTX_REPORT"):
        Path(os.environ["LEVTX_REPORT"]).write_text(json.dumps(REPORT, indent=1, default=str))

    assert switches_engine == switches_levtx
    assert rebal_engine == rebal_levtx
    assert int(unit_diff.sum()) == 0
    assert abs(final_gap) <= 0.005


def test_acceptance_strategy_passes_truncation_audit_on_real_data(market):
    start, end = "2014-11-03", market["end"]
    report = truncation_audit(
        lambda: MaBandLeverage(start, aux="taiex", rounding="floor"),
        market["futures"],
        EngineConfig(execution="next_open", start=start, end=end),
        aux=market["aux"],
        calendar=market["calendar"],
        n_cuts=20,
        seed=20261024,
    )
    print("truncation audit cuts:", [str(c.date()) for c in report.cuts], "mismatches:", report.mismatches)
    assert len(report.cuts) == 20 and report.ok
