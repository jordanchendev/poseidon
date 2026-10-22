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
