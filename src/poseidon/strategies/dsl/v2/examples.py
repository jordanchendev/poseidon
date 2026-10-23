"""Reference DSL v2 documents (design doc examples), used by tests and prompts."""

from __future__ import annotations

import copy
from typing import Any

_TX_ORB30_DAY: dict[str, Any] = {
    "dsl_version": 2,
    "name": "tx_orb30_day",
    "symbol": "TX",
    "market": "tw_futures",
    "interval": "5m",
    "risk": {
        "take_profit": {"type": "r_multiple", "value": 2.0},
        "flat_before_session_end_minutes": 10,
        "max_trades_per_session": 1,
        "no_reentry_after_stop": True,
    },
    "rules": [
        {
            "id": "orb_long",
            "action": "long",
            "condition": {
                "all": [
                    {"type": "time_window", "session": "day", "start": "09:15", "end": "13:35"},
                    {
                        "type": "cross",
                        "a": {"price": "close"},
                        "direction": "above",
                        "b": {"session": "or_high", "minutes": 30},
                    },
                ]
            },
            "risk": {"stop_loss": {"type": "level", "level": {"session": "or_low", "minutes": 30}, "offset_points": 0}},
        },
        {
            "id": "orb_short",
            "action": "short",
            "condition": {
                "all": [
                    {"type": "time_window", "session": "day", "start": "09:15", "end": "13:35"},
                    {
                        "type": "cross",
                        "a": {"price": "close"},
                        "direction": "below",
                        "b": {"session": "or_low", "minutes": 30},
                    },
                ]
            },
            "risk": {
                "stop_loss": {"type": "level", "level": {"session": "or_high", "minutes": 30}, "offset_points": 0}
            },
        },
    ],
}


def tx_orb30_day() -> dict[str, Any]:
    """The 30-minute day-session opening-range breakout from the design doc."""
    return copy.deepcopy(_TX_ORB30_DAY)


def _squeeze_rule(side: str) -> dict[str, Any]:
    long = side == "long"
    return {
        "id": f"squeeze_{side}",
        "action": side,
        "condition": {
            "all": [
                {
                    "type": "compare",
                    "a": {"ind": "atr", "period": 14},
                    "op": "<",
                    "b": {"ind": "atr", "period": 14, "smooth": {"type": "sma", "period": 50}},
                    "b_mult": 0.8,
                },
                {
                    "type": "cross",
                    "a": {"price": "close"},
                    "direction": "above" if long else "below",
                    "b": {"channel": "highest" if long else "lowest", "source": "high" if long else "low", "bars": 20},
                },
            ]
        },
    }


_TX_ATR_SQUEEZE_BREAKOUT: dict[str, Any] = {
    "dsl_version": 2,
    "name": "tx_atr_squeeze_breakout",
    "description": "DSL v2.1: ATR(14) below 0.8 x SMA(50) of ATR(14), then a 20-bar Donchian breakout; ATR trail.",
    "symbol": "TX",
    "market": "tw_futures",
    "interval": "5m",
    "risk": {
        "stop_loss": {"type": "atr_mult", "value": 2.0, "atr_period": 14},
        "trailing": {"type": "atr_mult", "value": 2.5, "atr_period": 14},
        "flat_before_session_end_minutes": 10,
        "max_trades_per_session": 2,
    },
    "rules": [_squeeze_rule("long"), _squeeze_rule("short")],
}


def tx_atr_squeeze_breakout() -> dict[str, Any]:
    """DSL v2.1 example: ATR volatility contraction, 20-bar channel breakout, ATR trailing stop."""
    return copy.deepcopy(_TX_ATR_SQUEEZE_BREAKOUT)
