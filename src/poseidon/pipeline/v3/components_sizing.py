"""Spec v3 components: sizing."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field

from poseidon.pipeline.v3.component_common import ANY_FREQUENCY, DAILY
from poseidon.pipeline.v3.instruments import Instrument
from poseidon.pipeline.v3.registry import SizingComponent, StrictParams, fmt, register

ROUNDING_EPS = 1e-9  # relative: keeps exact ratios such as 3e6 / 2e5 from flooring to 14 on float noise


def round_units(raw: float, rounding: str) -> int:
    """``floor`` rounds the magnitude down (never above the target leverage, long or short);
    ``nearest`` rounds half to even (Python ``round``, as the offline levtx backtest does)."""

    if rounding == "floor":
        return math.trunc(raw + math.copysign(ROUNDING_EPS * max(1.0, abs(raw)), raw))
    if rounding == "nearest":
        return round(raw)
    raise ValueError(f"unknown rounding {rounding!r}")


@register
class TargetLeverageSizing(SizingComponent):
    """units = round(equity x d x L x weight / (price x point value x lot)), floor by default (§5.3).

    The one sizing formula: the validator's feasibility checks and the engine run both call
    :meth:`units`. Units are contracts (board lots for stocks). Equity <= 0 sizes to 0.
    """

    name = "sizing.target_leverage"
    frequencies = DAILY
    basis = "equity"
    capabilities = ("equity_sizing", "integer_contracts", "target_leverage")
    summary = "Equity-based integer contracts: units = equity x direction x leverage / (price x point value)."

    class Params(StrictParams):
        rounding: Literal["floor", "nearest"] = "floor"

    example = {"rounding": "floor"}

    def units(self, params, *, equity, direction, leverage, weight, price, instrument: Instrument):
        if equity <= 0:
            return 0
        raw = equity * direction * leverage * weight / instrument.notional_twd(price)
        return round_units(raw, params.rounding)

    def describe(self, params, ctx):
        lot = f" × {ctx.instrument.lot}" if ctx.instrument.lot != 1 else ""
        rounding = "無條件捨去" if params.rounding == "floor" else "四捨五入"
        return f"口數 = 權益 × 倍數 ÷ (價格 × {fmt(ctx.instrument.point_value_twd)}{lot})，{rounding}"


@register
class FixedNotionalSizing(SizingComponent):
    """units = floor(equity x d x weight x pct_of_equity / 100 / (price x point value x lot)): a fixed share
    of equity, no leverage chain. The stock / ETF sizing (100 % = fully invested); futures may use it too."""

    name = "sizing.fixed_notional"
    frequencies = DAILY
    basis = "equity"
    uses_leverage = False
    capabilities = ("equity_sizing", "integer_contracts", "fixed_notional")
    summary = (
        "A fixed share of equity as notional, no leverage chain: "
        "units = equity x weight x pct_of_equity% / (price x point value x lot)."
    )

    class Params(StrictParams):
        pct_of_equity: Annotated[int, Field(ge=1, le=300)] = 100
        rounding: Literal["floor", "nearest"] = "floor"

    tunable = frozenset({"pct_of_equity"})
    example = {"pct_of_equity": 100, "rounding": "floor"}

    def exposure(self, params, leverage):
        return params.pct_of_equity / 100

    def units(self, params, *, equity, direction, leverage, weight, price, instrument: Instrument):
        if equity <= 0:
            return 0
        raw = equity * direction * weight * params.pct_of_equity / 100 / instrument.notional_twd(price)
        return round_units(raw, params.rounding)

    def describe(self, params, ctx):
        unit = "股" if ctx.instrument.lot == 1 else ("張" if ctx.instrument.kind == "stock" else "口")
        rounding = "無條件捨去" if params.rounding == "floor" else "四捨五入"
        return f"部位名目金額 = 權益的 {params.pct_of_equity}%，換成整數{unit}數（{rounding}）"


@register
class FixedContractsSizing(SizingComponent):
    """units = d x n (the v1 behaviour; PnL judged in TWD)."""

    name = "sizing.fixed_contracts"
    frequencies = ANY_FREQUENCY
    basis = "fixed"
    capabilities = ("fixed_contracts",)
    summary = "A fixed number of contracts n whenever the direction is non-zero (v1 behaviour)."

    class Params(StrictParams):
        n: Annotated[int, Field(ge=1, le=20)] = 1

    tunable = frozenset({"n"})
    example = {"n": 1}

    def units(self, params, *, equity, direction, leverage, weight, price, instrument):
        return direction * params.n

    def describe(self, params, ctx):
        return f"口數固定 {params.n} 口"
