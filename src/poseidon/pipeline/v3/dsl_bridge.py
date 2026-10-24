"""DSL v2 <-> spec v3 mapping (design 2026-10-24 §3.4).

``v2_to_v3`` is mechanical: one sleeve on the DSL's symbol, the DSL document
(minus ``symbol``/``market``/``interval``/``params``) as ``direction.dsl_rules``,
1 fixed contract, trade on signal change, fill at the signal bar's close,
``frequency`` = the DSL interval, the DSL ``params`` at spec level.

``v3_to_v2`` is the exact inverse on that shape, so every DSL v2 document
round-trips. The one normalisation: an empty or null ``params`` comes back
absent (the three mean the same to ``validate_v2``).
"""

from __future__ import annotations

import copy
from typing import Any

DSL_MARKETS = {"TX": "tw_futures", "MTX": "tw_futures_mtx"}
FAST_PATH_ALIAS = "core"
FAST_PATH_SLEEVE = "main"
_LIFTED = ("symbol", "market", "interval", "params")


class NotAFastPathSpec(ValueError):
    """The v3 spec is not the DSL v2 fast-path shape, so it has no DSL v2 form."""


def _fast_path_frame(name: Any, symbol: str, frequency: Any) -> dict[str, Any]:
    return {
        "spec_version": 3,
        "name": name,
        "frequency": frequency,
        "instruments": {FAST_PATH_ALIAS: {"symbol": symbol, "roll": {"component": "roll.at_settlement", "params": {}}}},
        "sleeves": [
            {
                "id": FAST_PATH_SLEEVE,
                "instrument": FAST_PATH_ALIAS,
                "direction": {"component": "direction.dsl_rules", "params": None},
                "leverage": [],
                "sizing": {"component": "sizing.fixed_contracts", "params": {"n": 1}},
                "rebalance": {"component": "rebalance.on_signal_change", "params": {}},
                "exits": [],
                "hedges": [],
            }
        ],
        "allocation": {"component": "allocation.single", "params": {}},
        "execution": {"component": "execution.signal_close", "params": {}},
    }


def v2_to_v3(doc: dict[str, Any]) -> dict[str, Any]:
    """Map a DSL v2 document onto spec v3. Raises ValueError when symbol and market disagree."""

    if not isinstance(doc, dict):
        raise ValueError("a DSL v2 document must be a JSON object")
    symbol = doc.get("symbol")
    if symbol not in DSL_MARKETS or doc.get("market") != DSL_MARKETS[symbol]:
        raise ValueError(f"DSL v2 symbol/market {symbol!r}/{doc.get('market')!r} has no spec v3 instrument")
    body = {key: copy.deepcopy(value) for key, value in doc.items() if key not in _LIFTED}
    spec = _fast_path_frame(doc.get("name"), symbol, doc.get("interval"))
    spec["sleeves"][0]["direction"]["params"] = body
    spec["params"] = copy.deepcopy(doc.get("params") or {})
    return spec


def v3_to_v2(spec: dict[str, Any]) -> dict[str, Any]:
    """The DSL v2 document a fast-path spec stands for (inverse of :func:`v2_to_v3`)."""

    try:
        symbol = spec["instruments"][FAST_PATH_ALIAS]["symbol"]
        body = spec["sleeves"][0]["direction"]["params"]
    except (KeyError, IndexError, TypeError):
        raise NotAFastPathSpec("not a DSL v2 fast-path spec") from None
    if not isinstance(body, dict):
        raise NotAFastPathSpec("not a DSL v2 fast-path spec")
    expected = _fast_path_frame(body.get("name"), symbol, spec.get("frequency"))
    expected["sleeves"][0]["direction"]["params"] = body
    shape = {key: value for key, value in spec.items() if key not in ("params", "description", "capital")}
    if shape != expected:
        raise NotAFastPathSpec("not a DSL v2 fast-path spec (one TX/MTX sleeve, dsl_rules, 1 contract, signal close)")
    if spec.get("capital") not in (None, {"initial_twd": 1_000_000}) or spec.get("description"):
        raise NotAFastPathSpec("a DSL v2 document has no capital or spec-level description")
    doc = copy.deepcopy(body)
    doc.update({"symbol": symbol, "market": DSL_MARKETS[symbol], "interval": spec["frequency"]})
    if spec.get("params"):
        doc["params"] = copy.deepcopy(spec["params"])
    return doc
