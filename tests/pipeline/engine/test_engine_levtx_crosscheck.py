"""Cross-check against the offline backtest ``local_dev/scripts/leveraged-tx/levtx.py`` (design §10 step 3).

Runs only when both levtx and the refdata store are available (stormtrooper):
  LEVTX_SCRIPT=/path/levtx.py  LEVTX_DATA=/path/levtx-data  (the directory holding ext/)
  POSEIDON_PIPELINE_REFDATA_ROOT=/path/refdata              (read-only is fine)
Optional: LEVTX_REPORT=/path/report.json to keep the numbers.

The spec side is the §3.2 acceptance spec compiled from refdata by ``v3.compiler`` and run by the
engine, configured to levtx's conventions: ``rounding: nearest`` in the spec, both signals on
``index:taiex`` (as the acceptance spec itself reads), and
``Conventions(execution="signal_close", signal_lag=1)`` (levtx decides from the index close t-1 and
fills at close t). Two levtx
variants: "3x daily + 200DMA±2%" (the spec without ``risk.vol_cap``) and "3x + 200DMA + volcap75"
(the acceptance chain, k = 0.75 over 20 days). levtx reads its own copy of the TAIFEX/TWSE files.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.refdata.datasets import FUTURES
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3.compiler import compile_spec
from poseidon.pipeline.v3.runtime import Conventions

from ..spec_support import resolved_spec

SCRIPT = os.environ.get("LEVTX_SCRIPT")
DATA = os.environ.get("LEVTX_DATA")
STORE = RefdataStore()
pytestmark = pytest.mark.skipif(
    not (SCRIPT and DATA and STORE.exists(FUTURES)), reason="LEVTX_SCRIPT / LEVTX_DATA / refdata not available"
)

WINDOWS = [("2014-11-03", None), ("2000-01-04", "2014-10-31"), ("2020-01-02", None), ("2008-01-02", None)]
VARIANTS = {
    "3x+200DMA": (dict(ma=200, band=0.02, bear_lev=0.5), dict(vol_k=None)),
    "3x+200DMA+volcap75": (dict(ma=200, band=0.02, bear_lev=0.5, vol_target=0.75), dict(vol_k=0.75)),
}
LEVTX_CONVENTIONS = Conventions(execution="signal_close", signal_lag=1)
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
    mk = levtx.build_r1(levtx.load_taifex())
    sig = levtx.load_twse("MI_5MINS_HIST", ["open", "high", "low", "close"])
    sig["vol20"] = np.log(sig["close"]).diff().rolling(20).std() * math.sqrt(252)  # as levtx.main
    refdata_end = pd.Timestamp(STORE.manifest(FUTURES)["coverage"]["end"])
    end = min(mk.index.intersection(sig.index)[-1], refdata_end)
    return {"mk": mk, "sig": sig, "end": end}


def _levtx_run(levtx, market, variant, start, end):
    spec = levtx.Spec(variant, 3, "daily", **VARIANTS[variant][0])
    r = levtx.simulate(spec, market["mk"], market["sig"], start, end)
    _, state = levtx.leverage_path(spec, market["sig"], start, end)
    w = market["mk"].loc[start:end]
    units = (r["lev_post"] * r["eq"] / (w["p_new"] * 10.0)).round().astype(int)
    return r, w, units, state.loc[start:end]


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("start,end", WINDOWS)
def test_compiled_spec_matches_levtx(levtx, market, variant, start, end):
    end = pd.Timestamp(end) if end else market["end"]
    r, w, lev_units, lev_state = _levtx_run(levtx, market, variant, start, end)
    if r["ruined"]:
        pytest.skip(f"levtx ruined on {r['ruined']}")
    compiled = compile_spec(
        resolved_spec(rounding="nearest", signal="index:taiex", **VARIANTS[variant][1]),
        start=start,
        end=end,
        conventions=LEVTX_CONVENTIONS,
    )
    strategy = compiled.strategy()
    res = compiled.run(strategy)

    pos = res.positions_daily.set_index("date")
    assert list(pos.index) == list(w.index)
    assert (pos["contract"] == w["held"]).all(), "held contract schedule differs"
    unit_diff = pos["held_units"].astype(int) != lev_units
    rebal_engine = set(pos.index[pos["held_units"].diff().fillna(pos["held_units"]) != 0])
    rebal_levtx = set(w.index[lev_units.diff().fillna(lev_units) != 0])

    switches_engine = [pd.Timestamp(d) for d, _on in strategy.state("main", "leverage.1")["switches"]]
    switches_levtx = list(lev_state.index[lev_state.diff().fillna(0) != 0])

    eq_engine, eq_levtx = res.equity, r["eq"]
    rel = (eq_engine / eq_levtx - 1).abs()
    final_gap = float(eq_engine.iloc[-1] / eq_levtx.iloc[-1] - 1)
    key = f"{variant} {start}..{end.date()}"
    REPORT[key] = {
        "days": len(w),
        "unit_mismatch_days": int(unit_diff.sum()),
        "rebalance_days_engine": len(rebal_engine),
        "rebalance_days_levtx": len(rebal_levtx),
        "switch_dates_levtx": [str(d.date()) for d in switches_levtx],
        "switch_dates_engine": [str(d.date()) for d in switches_engine],
        "final_equity_levtx": float(eq_levtx.iloc[-1]),
        "final_equity_engine": float(eq_engine.iloc[-1]),
        "final_gap_pct": final_gap * 100,
        "max_daily_equity_gap_pct": float(rel.max() * 100),
        "engine_events": res.events["kind"].value_counts().to_dict(),
        "levtx_margin_flags": len(r["margin_flags"]),
        "refdata_inputs": compiled.inputs,
    }
    print(json.dumps({key: REPORT[key]}, indent=1, default=str))
    if os.environ.get("LEVTX_REPORT"):
        Path(os.environ["LEVTX_REPORT"]).write_text(json.dumps(REPORT, indent=1, default=str))

    assert switches_engine == switches_levtx
    assert rebal_engine == rebal_levtx
    assert int(unit_diff.sum()) == 0
    assert abs(final_gap) <= 0.005
