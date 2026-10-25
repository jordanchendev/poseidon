"""The fixed zh-TW narrative of a spec v3 (design 2026-10-24 §3.2, §4).

Owner, Kairos and JEV read the same text: every sentence comes from a
component's ``describe()``, joined in a fixed order (execution, then per
sleeve: instrument + roll, direction, leverage chain, sizing + rebalance, hedges).
``$param`` fields show their default; :func:`params_line` lists the ranges.
"""

from __future__ import annotations

from poseidon.pipeline.v3 import components as _components  # noqa: F401  (registers the components)
from poseidon.pipeline.v3.instruments import declared
from poseidon.pipeline.v3.registry import REGISTRY, DescribeContext, fmt
from poseidon.pipeline.v3.spec import substitute


def _render(ref: dict, ctx: DescribeContext) -> str:
    component = REGISTRY[ref["component"]]
    return component.describe(component.parse(ref.get("params") or {}), ctx)


def narrative(resolved: dict) -> str:
    """Sentences for a validated spec whose ``$param``s are already substituted."""

    frequency = resolved["frequency"]
    sleeves = resolved["sleeves"]
    first = declared(resolved["instruments"][sleeves[0]["instrument"]])
    sentences = [_render(resolved["execution"], DescribeContext(frequency, first))]
    if len(sleeves) > 1:
        sentences.append("配置：" + _render(resolved["allocation"], DescribeContext(frequency, first)))
    for sleeve in sleeves:
        prefix = f"{sleeve['id']}：" if len(sleeves) > 1 else ""
        decl = resolved["instruments"][sleeve["instrument"]]
        instrument = declared(decl)
        ctx = DescribeContext(frequency, instrument, alias=sleeve["instrument"])
        proxy = f"，價格以 {instrument.price_proxy} 代理" if instrument.price_proxy else ""
        text = f"{prefix}商品：{instrument.name_zh}（{instrument.symbol}{proxy}）"
        if decl.get("unit") == "odd_lot":
            text += "，以零股（1 股為單位）交易"
        if decl.get("roll"):
            text += "，" + _render(decl["roll"], ctx)
        sentences.append(text)
        sentences.append("方向：" + _render(sleeve["direction"], ctx))
        chain = sleeve.get("leverage") or []
        if chain:
            fragments = []
            leverage = 0.0
            for link in chain:
                component = REGISTRY[link["component"]]
                params = component.parse(link.get("params") or {})
                fragments.append(
                    component.describe(params, DescribeContext(frequency, instrument, leverage, sleeve["instrument"]))
                )
                leverage = component.nominal(params, leverage)
            sentences.append("槓桿：" + "；".join(fragments))
        sentences.append(_render(sleeve["sizing"], ctx) + "；" + _render(sleeve["rebalance"], ctx))
        sentences.extend("避險：" + _render(hedge, ctx) for hedge in sleeve.get("hedges") or [])
    return "。".join(sentences) + "。"


def params_line(params: dict) -> str:
    """``可調參數：name low–high（間隔 step，預設 default）、…。`` or ``""`` without $params."""

    if not params:
        return ""
    parts = []
    for name, spec in sorted(params.items()):
        parts.append(
            f"{name} {fmt(spec.cast(spec.low))}–{fmt(spec.cast(spec.high))}"
            f"（間隔 {fmt(spec.step)}，預設 {fmt(spec.cast(spec.default))}）"
        )
    return "可調參數：" + "、".join(parts) + "。"


def describe_spec(template: dict, params: dict) -> str:
    """Narrative at the defaults plus the tunable ranges (``params``: name -> ``ParamSpec``)."""

    values = {name: spec.cast(spec.default) for name, spec in params.items()}
    text = narrative(substitute(template, values))
    line = params_line(params)
    return f"{text}{line}" if line else text
