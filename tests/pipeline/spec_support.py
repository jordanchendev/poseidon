"""Resolved spec v3 documents for the compiler/engine tests (variants of the §3.2 acceptance spec)."""

from __future__ import annotations

from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.spec import substitute


def resolved_spec(
    *,
    ma_days: int = 200,
    band: float = 0.02,
    vol_k: float | None = 0.75,
    base: float = 3.0,
    off: float = 0.5,
    rounding: str = "floor",
    rebalance: tuple[str, dict] = ("rebalance.daily", {"min_change_units": 1}),
    symbol: str = "TMF",
    days_before: int = 1,
    signal: str = "core:adjusted",
) -> dict:
    """The acceptance spec with every ``$param`` bound; ``vol_k=None`` drops the volatility cap.

    ``signal`` is the series both links read. The acceptance spec itself reads ``index:taiex``; the
    default here is the futures' own adjusted series because most engine tests run on synthetic
    futures without an index aux series.
    """

    doc = substitute(tmf_dynamic_leverage_3x(), {"ma_days": ma_days, "band": band, "vol_k": vol_k or 0.75})
    doc["params"] = {}
    doc["instruments"]["core"]["symbol"] = symbol
    doc["instruments"]["core"]["roll"]["params"]["days_before"] = days_before
    sleeve = doc["sleeves"][0]
    sleeve["leverage"][0]["params"]["value"] = base
    sleeve["leverage"][1]["params"]["off_leverage"] = off
    for link in sleeve["leverage"][1:]:
        link["params"]["signal"] = signal
    if vol_k is None:
        del sleeve["leverage"][2]
    sleeve["sizing"]["params"]["rounding"] = rounding
    sleeve["rebalance"] = {"component": rebalance[0], "params": dict(rebalance[1])}
    return doc
