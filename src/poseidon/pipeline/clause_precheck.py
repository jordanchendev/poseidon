"""Numeric precheck of an idea clause against a spec v3, without JEV (design 2026-10-24 §4).

Modelled on the research ``claim_precheck`` strict mode. The clause's quantities -- multiples
(``3 倍``, ``3x``, ``三倍``), percentages (``±2%``), day counts (``200 日``, ``MA200``), plain numbers
(``0.75``) -- and its side (做多 / 做空 ...) are looked up in the spec: every numeric component
parameter and every ``$param`` range ``[low, high]``. Percentages also match a fraction (2 % = 0.02).

Only two kinds of miss decide the clause as ``contradicts`` (strict mode):

- ``multiple``: the clause names a leverage multiple, the spec has leverage values, and none of them
  (nor the 5x policy cap, nor a ``$param`` range on a leverage link, nor a ``*_multiple`` parameter such as
  ``take_profit_multiple``) equals it; ladder rungs (``steps``) count as spec values;
- ``sign``: the clause says short and every sleeve is constantly long, or the other way round.

Any other miss (days, percentages, plain numbers, a multiple when the spec has no leverage at all) is
reported as advisory and left to JEV, because a number can be legitimately elsewhere (a default, a
policy) or a paraphrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from poseidon.pipeline.v3.spec import PARAM_REF_KEY, component_slots
from poseidon.pipeline.v3.validation_types import LEVERAGE_CAP

PRECHECK_VERSION = 3  # 2: 做空間 (space planning) is not 做空; 3: ladder rungs and take_profit_multiple are spec values
DECIDING_KINDS: dict[str, frozenset[str]] = {
    "strict": frozenset({"multiple", "sign"}),
    "any": frozenset({"multiple", "sign", "percent", "days", "number"}),
    "off": frozenset(),
}
REL_TOL = 0.005

_CJK = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"  # 1,000,000 is one number
_SIGN = r"(?P<sign>[+\-−])?"
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "multiple",
        re.compile(rf"{_SIGN}\s*(?P<num>(?:{_NUM})|[一二兩三四五六七八九十半])\s*(?:倍|[xX×](?![a-zA-Z(（])|times\b)"),
    ),
    ("percent", re.compile(rf"(?P<pm>±|\+/-)?{_SIGN}\s*(?P<num>(?:{_NUM}))\s*(?:%|％|percent\b|個百分點|百分點)")),
    ("days", re.compile(r"(?:D?MA|SMA|EMA)\s*(?P<num>\d+)\b|(?P<num2>\d+)\s*(?:D?MA|SMA|EMA)\b")),
    (
        "days",
        re.compile(r"(?P<num>\d+|[一二兩三四五六七八九十]+)\s*(?:個)?\s*(?:交易)?\s*(?:日|天|days?\b|trading days?\b)"),
    ),
    ("number", re.compile(rf"(?<![A-Za-z0-9.,])(?P<num>(?:{_NUM}))(?![A-Za-z0-9])")),
)
_NEGATION = re.compile(r"(?:不|不會|不要|不能|沒有|禁止|never|no|not|don't)\s*$", re.I)
_LONG = re.compile(r"做多|多單|買進期貨|(?<!as )\blong\b(?! as)", re.I)
_SHORT = re.compile(r"做空(?!間)|放空|空單|反向|\bshort\b(?![- ]term)", re.I)


@dataclass(frozen=True)
class Quantity:
    kind: str  # multiple / percent / days / number
    raw: str
    value: float

    def as_report(self) -> dict:
        return {"kind": self.kind, "raw": self.raw, "value": self.value}


def _value(token: str) -> float | None:
    """``"2.5"`` -> 2.5; ``"三"`` -> 3; ``"二十"`` -> 20, ``"十五"`` -> 15; anything else -> None."""

    if token[0].isdigit():
        return float(token.replace(",", ""))
    if "十" not in token:
        return float(_CJK[token]) if len(token) == 1 else None
    tens, _, ones = token.partition("十")
    if len(tens) > 1 or len(ones) > 1:
        return None
    return float((_CJK[tens] if tens else 1) * 10 + (_CJK[ones] if ones else 0))


def extract_quantities(text: str) -> list[Quantity]:
    """Quantities of ``text`` in pattern order; a span consumed by one pattern is not read again."""

    taken: list[tuple[int, int]] = []
    found: list[tuple[int, Quantity]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < b and a < end for a, b in taken):
                continue
            groups = match.groupdict()
            token = groups.get("num") or groups.get("num2")
            if token is None:
                continue
            value = _value(token)
            if value is None:
                continue
            if groups.get("sign") in ("-", "−"):
                value = -value
            taken.append((start, end))
            found.append((start, Quantity(kind, match.group(0).strip(), value)))
    return [quantity for _start, quantity in sorted(found, key=lambda item: item[0])]


def _side_mentions(pattern: re.Pattern, text: str) -> bool:
    return any(not _NEGATION.search(text[max(0, m.start() - 6) : m.start()]) for m in pattern.finditer(text))


def clause_side(text: str, quantities: list[Quantity]) -> int | None:
    """+1 (long), -1 (short) or None (neither or both) as the clause states it."""

    long_ = _side_mentions(_LONG, text)
    short = _side_mentions(_SHORT, text) or any(q.kind == "multiple" and q.value < 0 for q in quantities)
    if long_ == short:
        return None
    return 1 if long_ else -1


# --------------------------------------------------------------------------- spec values


@dataclass(frozen=True)
class SpecValue:
    path: str
    slot: str
    key: str
    low: float
    high: float  # == low for a fixed value
    source: str  # "value" / "$param:<name>" / "policy"

    def matches(self, value: float) -> bool:
        tol = 1e-9 + REL_TOL * max(abs(self.low), abs(self.high))
        return self.low - tol <= value <= self.high + tol

    def as_report(self) -> dict:
        return {"path": self.path, "low": self.low, "high": self.high, "source": self.source}


def spec_values(template: dict) -> list[SpecValue]:
    """Every numeric component parameter of ``template`` (a ``$param`` reference as its declared range).

    A parameter may be a list of rung objects (``leverage.drawdown_ladder`` ``steps``): each number in a
    rung is a value under its own key (``steps.0.leverage``), so a ladder's rungs are not lost.
    """

    declared = template.get("params") or {}
    out: list[SpecValue] = []

    def visit(path: tuple, slot: str, key: str, raw: Any) -> None:
        where = ".".join(str(part) for part in path)
        if isinstance(raw, dict) and PARAM_REF_KEY in raw:
            spec = declared.get(raw[PARAM_REF_KEY])
            if isinstance(spec, dict) and _is_number(spec.get("low")) and _is_number(spec.get("high")):
                name = raw[PARAM_REF_KEY]
                out.append(SpecValue(where, slot, key, float(spec["low"]), float(spec["high"]), f"$param:{name}"))
        elif isinstance(raw, dict):
            for child_key, child in raw.items():
                visit((*path, child_key), slot, child_key, child)
        elif isinstance(raw, list):
            for index, child in enumerate(raw):
                visit((*path, index), slot, key, child)
        elif _is_number(raw):
            out.append(SpecValue(where, slot, key, float(raw), float(raw), "value"))

    for path, slot, ref in component_slots(template):
        for key, raw in (ref.get("params") or {}).items():
            visit((*path, "params", key), slot, key, raw)
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_leverage_value(value: SpecValue) -> bool:
    """A multiple of equity on a leverage link (``value``, ``off_leverage``, ``max_leverage``, ``cut_to``),
    not a window, a band or a volatility constant such as ``risk.vol_cap.k``."""

    return value.slot == "leverage" and (value.key in ("value", "cut_to") or "leverage" in value.key)


def spec_sides(template: dict) -> set[str]:
    """``{"long"}`` / ``{"short"}`` for constant-direction sleeves; ``"rules"`` for any other direction."""

    sides = set()
    for sleeve in template.get("sleeves") or []:
        direction = sleeve.get("direction") or {}
        if direction.get("component") == "direction.constant":
            sides.add((direction.get("params") or {}).get("side"))
        else:
            sides.add("rules")
    return sides


# --------------------------------------------------------------------------- precheck


def precheck(clause: str, template: dict, *, mode: str = "strict") -> dict:
    """The clause's quantities and side checked against ``template`` (see module docstring)."""

    deciding = DECIDING_KINDS[mode]
    quantities = extract_quantities(clause)
    values = spec_values(template)
    leverage = [value for value in values if _is_leverage_value(value)]
    if leverage:
        leverage.append(
            SpecValue("policy.leverage_cap", "leverage", "leverage_cap", LEVERAGE_CAP, LEVERAGE_CAP, "policy")
        )
    multiples = [*leverage, *(value for value in values if value.key.endswith("_multiple"))]
    matched, unmatched = [], []
    for quantity in quantities:
        magnitude = abs(quantity.value)
        if quantity.kind == "multiple":
            kind = "multiple" if leverage else "multiple_no_leverage"
            hits = [value for value in multiples if value.matches(magnitude)]
        elif quantity.kind == "percent":
            kind = "percent"
            hits = [value for value in values if value.matches(magnitude / 100) or value.matches(magnitude)]
        else:
            kind = quantity.kind
            hits = [value for value in values if value.matches(magnitude)]
        if hits:
            matched.append({**quantity.as_report(), "spec": [hit.as_report() for hit in hits[:3]]})
        else:
            unmatched.append({**quantity.as_report(), "kind": kind})
    side = clause_side(clause, quantities)
    sides = spec_sides(template)
    sign_mismatch = side is not None and "rules" not in sides and sides == {"short" if side > 0 else "long"}
    if sign_mismatch:
        unmatched.append({"kind": "sign", "raw": "long" if side > 0 else "short", "spec_sides": sorted(sides)})
    deciders = [item for item in unmatched if item["kind"] in deciding]
    return {
        "version": PRECHECK_VERSION,
        "mode": mode,
        "quantities": [quantity.as_report() for quantity in quantities],
        "clause_side": side,
        "matched": matched,
        "unmatched": unmatched,
        "decides": bool(deciders),
        "decided_label": "contradicts" if deciders else None,
        "deciding_kinds": sorted({item["kind"] for item in deciders}),
    }
