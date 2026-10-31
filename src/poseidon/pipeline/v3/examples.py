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


# Step 8 (M3) acceptance: two ETF sleeves, fully invested in the leveraged ETF while the TAIEX is above
# its MA band, in the plain index ETF below it. Hand-written; the corporate actions (0050 1 -> 4 on
# 2025-06-18, 00631L 1 -> 22 on 2026-03-31) are applied by the engine from tw_corporate_actions.
_ETF_ROTATION_00631L_0050: dict[str, Any] = {
    "spec_version": 3,
    "name": "etf_rotation_00631l_0050",
    "description": "TAIEX above its MA band: 100 % in 00631L (2x); below: 100 % in 0050.",
    "frequency": "1d",
    "capital": {"initial_twd": 1000000},
    "instruments": {"lev": {"symbol": "00631L"}, "base": {"symbol": "0050"}},
    "sleeves": [
        {
            "id": "risk_on",
            "instrument": "lev",
            "direction": {"component": "direction.constant", "params": {"side": "long"}},
            "leverage": [],
            "sizing": {"component": "sizing.fixed_notional", "params": {"pct_of_equity": 100, "rounding": "floor"}},
            "rebalance": {"component": "rebalance.threshold_band", "params": {"band_pct": 0.1}},
            "exits": [],
            "hedges": [],
        },
        {
            "id": "risk_off",
            "instrument": "base",
            "direction": {"component": "direction.constant", "params": {"side": "long"}},
            "leverage": [],
            "sizing": {"component": "sizing.fixed_notional", "params": {"pct_of_equity": 100, "rounding": "floor"}},
            "rebalance": {"component": "rebalance.threshold_band", "params": {"band_pct": 0.1}},
            "exits": [],
            "hedges": [],
        },
    ],
    "allocation": {
        "component": "allocation.regime_switch",
        "params": {
            "signal": "index:taiex",
            "ma_days": {"$param": "ma_days"},
            "band_pct": {"$param": "band"},
            "on": "risk_on",
            "off": "risk_off",
        },
    },
    "execution": {"component": "execution.next_open"},
    "params": {
        "ma_days": {"type": "int", "low": 100, "high": 250, "step": 50, "default": 200},
        "band": {"type": "float", "low": 0.01, "high": 0.03, "step": 0.01, "default": 0.02},
    },
}


def etf_rotation_00631l_0050() -> dict[str, Any]:
    """The M3 acceptance spec: 00631L / 0050 rotation on the TAIEX MA band (step 8)."""

    return copy.deepcopy(_ETF_ROTATION_00631L_0050)


# Design §3.2 / §10 M4: the same strategy with the protective put of the design example (a 5 % out-of-the-money
# front-month TXO put, rolled 5 trading days before expiry, one put per five TMF).
PROTECTIVE_PUT: dict[str, Any] = {
    "component": "hedge.protective_put",
    "params": {"moneyness_pct": 0.05, "tenor": "front", "roll_days_before_expiry": 5, "cover_ratio": 1.0},
}


def tmf_dynamic_leverage_3x_put() -> dict[str, Any]:
    """The M4 acceptance spec: :func:`tmf_dynamic_leverage_3x` plus ``hedge.protective_put``."""

    doc = tmf_dynamic_leverage_3x()
    doc["name"] = "tmf_dynamic_leverage_3x_put"
    doc["sleeves"][0]["hedges"] = [copy.deepcopy(PROTECTIVE_PUT)]
    return doc
