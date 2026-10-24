"""Reference spec v3 documents."""

from __future__ import annotations

import copy
from typing import Any

# Design 2026-10-24 §3.2: the M1 acceptance strategy (no put). Owner decision 2026-10-24 afternoon #1:
# both signals read the TAIEX price index, as the offline backtest (levtx) does.
_TMF_DYNAMIC_LEVERAGE_3X: dict[str, Any] = {
    "spec_version": 3,
    "name": "tmf_dynamic_leverage_3x",
    "frequency": "1d",
    "capital": {"initial_twd": 1000000},
    "instruments": {
        "core": {"symbol": "TMF", "roll": {"component": "roll.before_settlement", "params": {"days_before": 1}}}
    },
    "sleeves": [
        {
            "id": "main",
            "instrument": "core",
            "direction": {"component": "direction.constant", "params": {"side": "long"}},
            "leverage": [
                {"component": "leverage.base", "params": {"value": 3.0}},
                {
                    "component": "regime.ma_band",
                    "params": {
                        "signal": "index:taiex",
                        "ma_days": {"$param": "ma_days"},
                        "band_pct": {"$param": "band"},
                        "off_leverage": 0.5,
                    },
                },
                {
                    "component": "risk.vol_cap",
                    "params": {"signal": "index:taiex", "window_days": 20, "k": {"$param": "vol_k"}},
                },
            ],
            "sizing": {"component": "sizing.target_leverage", "params": {"rounding": "floor"}},
            "rebalance": {"component": "rebalance.daily", "params": {"min_change_units": 1}},
            "exits": [],
            "hedges": [],
        }
    ],
    "allocation": {"component": "allocation.single"},
    "execution": {"component": "execution.next_open"},
    "params": {
        "ma_days": {"type": "int", "low": 150, "high": 250, "step": 25, "default": 200},
        "band": {"type": "float", "low": 0.01, "high": 0.03, "step": 0.005, "default": 0.02},
        "vol_k": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.125, "default": 0.75},
    },
}


def tmf_dynamic_leverage_3x() -> dict[str, Any]:
    """The M1 acceptance spec (design §3.2): 3x TMF, 200-day MA band to 0.5x, volatility cap, both on
    the TAIEX price index."""

    return copy.deepcopy(_TMF_DYNAMIC_LEVERAGE_3X)
