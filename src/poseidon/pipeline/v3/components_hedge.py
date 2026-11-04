"""Spec v3 components: hedges (protective put)."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field

from poseidon.pipeline.engine.options import HedgePolicy
from poseidon.pipeline.v3.component_common import DAILY
from poseidon.pipeline.v3.instruments import INSTRUMENTS, SAME_INDEX_FAMILY
from poseidon.pipeline.v3.registry import HedgeComponent, StrictParams, fmt, pct, register


@register
class ProtectivePut(HedgeComponent):
    """Bought TXO monthly puts against the sleeve's long futures position (design §5.7).

    At each decision the engine sizes the hedge from the sleeve's final target units: puts =
    floor(cover_ratio x units x futures point value / 50), so at cover 1 one put covers five TMF
    (a flat or short position holds none). It keeps the held put until it has
    ``roll_days_before_expiry`` or fewer trading days left, then sells it and buys a new one:
    the ``front`` (or ``next``) monthly expiry with more trading days left than that, at the
    listed strike nearest to the futures close x (1 - moneyness_pct) that has a settlement price.
    A nearest strike more than 5 % from that target is ``hedge_unavailable`` (no new put that day;
    over 5 % of the hedged days fails the run). Valued and filled at daily settlement prices plus
    max(1 tick, 5 % of premium) slippage; a put held to expiry is cash-settled at the final
    settlement price. Behaviour lives in ``engine.options``.
    """

    name = "hedge.protective_put"
    frequencies = DAILY
    option_symbol = "TXO"
    capabilities = ("protective_put", "option_hedge", "tail_hedge")
    summary = (
        "Buy TXO monthly puts against the long futures position: strike nearest close x (1 - moneyness), "
        "roll before expiry, cover_ratio 1 = 1 put per 5 TMF; valued at daily settlement prices."
    )

    class Params(StrictParams):
        moneyness_pct: Annotated[float, Field(ge=0.0, le=0.20)] = 0.05
        tenor: Literal["front", "next"] = "front"
        roll_days_before_expiry: Annotated[int, Field(ge=1, le=15)] = 5
        cover_ratio: Annotated[float, Field(ge=0.25, le=1.5)] = 1.0

    tunable = frozenset({"moneyness_pct", "roll_days_before_expiry", "cover_ratio"})
    example = {"moneyness_pct": 0.05, "tenor": "front", "roll_days_before_expiry": 5, "cover_ratio": 1.0}

    def check(self, params, ctx):
        problems = []
        if ctx.instrument is not None and ctx.instrument.symbol not in SAME_INDEX_FAMILY:
            family = ", ".join(SAME_INDEX_FAMILY)
            problems.append(("", f"hedge.protective_put buys TXO puts and only hedges {family} sleeves"))
        direction = (ctx.sleeve or {}).get("direction") or {}
        if (
            direction.get("component") == "direction.constant"
            and (direction.get("params") or {}).get("side") == "short"
        ):
            problems.append(("", "a protective put hedges a long position; this sleeve is always short"))
        return problems

    def policy(self, params, *, name, underlying, chain):
        return HedgePolicy(
            name=name,
            underlying=underlying,
            chain=chain,
            moneyness=params.moneyness_pct,
            roll_days=params.roll_days_before_expiry,
            cover_ratio=params.cover_ratio,
            tenor=0 if params.tenor == "front" else 1,
            cp="P",
        )

    def describe(self, params, ctx):
        option = INSTRUMENTS[self.option_symbol]
        source = ctx.instrument.price_proxy or ctx.instrument.symbol
        per_put = option.point_value_twd / (ctx.instrument.point_value_twd * ctx.instrument.lot * params.cover_ratio)
        tenor = "近月" if params.tenor == "front" else "次近月"
        return (
            f"買進{option.name_zh}月選賣權保護多單：履約價取 {source} 期貨收盤價 ×(1−{pct(params.moneyness_pct)}%) "
            f"最近、有結算價的序列，{tenor}（剩餘超過 {params.roll_days_before_expiry} 個交易日的到期），"
            f"到期前 {params.roll_days_before_expiry} 個交易日換到下一個到期；口數 = {ctx.instrument.symbol} 口數 × "
            f"{fmt(ctx.instrument.point_value_twd * ctx.instrument.lot)} × {fmt(params.cover_ratio)} ÷ "
            f"{fmt(option.point_value_twd)}，無條件捨去（每 {fmt(per_put)} 口 {ctx.instrument.symbol} 一口賣權）；"
            "以每日結算價估值和成交（另加滑價），到期以最後結算價現金結算"
        )


@register
class ProtectivePutV2(ProtectivePut):
    """Version 2 of ``hedge.protective_put``: version 1 plus an optional take-profit and a roll when the front
    month expires. Version 1 stays registered for runs frozen with it; its parameters mean the same here.

    ``take_profit_multiple`` (absent or null: off): a held put whose daily settlement reaches that multiple of
    its average fill price is sold at the next fill and a new put is bought by the same selection rule (the
    author's "漲到買價 3 倍就賣掉，再買一口新的"). Checked on the settlement, the only option price the engine
    has. A put due to roll on the same day is sold as a take-profit; a put cash-settled at that close has
    expired before the check. ``roll_when``: ``own_expiry`` rolls the held put at ``roll_days_before_expiry``
    or fewer days before its own expiry (version 1); ``front_expiry`` (tenor ``next`` only) rolls it once the
    front month has that many or fewer days left (0: on the front's expiry day), so the put held is always the
    second expiry. ``roll_days_before_expiry`` may be 0 (hold to expiry, then buy the next).
    """

    version = 2
    summary = (
        "Buy TXO monthly puts against the long futures position: strike nearest close x (1 - moneyness), front or "
        "next month, roll before its own expiry or when the front month expires, optional take-profit (sell at "
        "take_profit_multiple x the fill price, buy a new one); cover_ratio 1 = 1 put per 5 TMF; daily settlement prices."
    )

    class Params(StrictParams):
        moneyness_pct: Annotated[float, Field(ge=0.0, le=0.20)] = 0.05
        tenor: Literal["front", "next"] = "front"
        roll_days_before_expiry: Annotated[int, Field(ge=0, le=15)] = 5
        roll_when: Literal["own_expiry", "front_expiry"] = "own_expiry"
        cover_ratio: Annotated[float, Field(ge=0.25, le=1.5)] = 1.0
        take_profit_multiple: Annotated[float, Field(ge=1.5, le=20.0)] | None = None

    example = {
        "moneyness_pct": 0.05,
        "tenor": "next",
        "roll_days_before_expiry": 0,
        "roll_when": "front_expiry",
        "cover_ratio": 1.0,
        "take_profit_multiple": 3.0,
    }

    def check(self, params, ctx):
        problems = super().check(params, ctx)
        if params.roll_when == "front_expiry" and params.tenor != "next":
            problems.append(("roll_when", "roll_when front_expiry rolls a next-month put; set tenor next"))
        return problems

    def policy(self, params, *, name, underlying, chain):
        return replace(
            super().policy(params, name=name, underlying=underlying, chain=chain),
            take_profit=params.take_profit_multiple,
            roll_with_tenor=params.roll_when == "front_expiry",
        )

    def describe(self, params, ctx):
        option = INSTRUMENTS[self.option_symbol]
        source = ctx.instrument.price_proxy or ctx.instrument.symbol
        per_put = option.point_value_twd / (ctx.instrument.point_value_twd * ctx.instrument.lot * params.cover_ratio)
        days = params.roll_days_before_expiry
        if params.roll_when == "front_expiry":
            month = "次近月"
            roll = (
                "近月到期日當天換成新的次近月" if days == 0 else f"近月剩 {days} 個交易日以內時換成新的次近月"
            ) + "，隨時持有次近月"
        else:
            month = "近月" if params.tenor == "front" else "次近月"
            roll = "持有到到期，到期後買下一個" if days == 0 else f"到期前 {days} 個交易日換到下一個到期"
        take_profit = (
            ""
            if params.take_profit_multiple is None
            else f"；結算價漲到買進成交均價的 {fmt(params.take_profit_multiple)} 倍時賣出，依同一規則再買新的賣權"
        )
        return (
            f"買進{option.name_zh}月選賣權保護多單：履約價取 {source} 期貨收盤價 ×(1−{pct(params.moneyness_pct)}%) "
            f"最近、有結算價的序列，{month}（剩餘超過 {days} 個交易日的到期），{roll}{take_profit}；口數 = "
            f"{ctx.instrument.symbol} 口數 × {fmt(ctx.instrument.point_value_twd * ctx.instrument.lot)} × "
            f"{fmt(params.cover_ratio)} ÷ {fmt(option.point_value_twd)}，無條件捨去（每 {fmt(per_put)} 口 "
            f"{ctx.instrument.symbol} 一口賣權）；以每日結算價估值和成交（另加滑價），到期以最後結算價現金結算"
        )
