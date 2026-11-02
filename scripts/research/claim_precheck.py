#!/usr/bin/env python3
"""Deterministic claim prechecks that run before (or instead of) an LLM citation judge.

Python 3.13 with the Poseidon package on the path: the tokenizer and lexicons it shares with the server's
poseidon.decision_loop.claim_lint live in poseidon.decision_loop.claim_quantities (one copy); what differs from the
server (X ± Y after a comma / 加減 / 正負, the derivation scale check, A+B and the inference rules) is here.
Importable and usable as a CLI:

    python claim_precheck.py extract "TEXT"
    python claim_precheck.py check --claim "CLAIM" --excerpt "EXCERPT"
    python claim_precheck.py lint "CLAIM"

Three pieces:

1. ``extract_quantities(text)`` -- numeric/unit normaliser for zh-TW and English financial text.
   Each quantity is canonicalised to ``(value_in_base_unit, unit_class, currency, period_token)``:
     * scales 千/萬/百萬/千萬/億/兆, thousand/million/billion/trillion (bn/mn) multiply the value;
       Chinese compounds such as ``1 兆 4,942.4 億`` are summed (= 1.49424e12).
     * money: prefixes NT$/US$/$/新台幣/新臺幣/¥, suffixes 美元/日圓/日元/円/元/(US|NT) dollars/yen.
       A bare ``元`` gives currency None, which is a wildcard when matching.
     * percent (%, ％, percent) -> class ``percent``; 個百分點/百分點/percentage point(s)/pp -> class ``pp``;
       bp/bps/basis point(s)/基點/個基點 -> class ``pp`` scaled by 0.01 (210 個基點 == 2.1 pp).
     * multiples: N 倍 / N times / N-fold / Nx / doubled / tripled / fourfold / 翻倍 -> class ``multiple``.
       ``一倍`` right after 增/成長/增加 is read as "doubled" (value 2); every other ``N 倍`` is read as N.
     * other units: nm/奈米/-nanometer, GB/TB/MB/PB (bytes), MT/s / GT/s, days/天; everything else is ``count``.
     * ranges (``52.3–54.3``, ``US$52.3 billion to US$54.3 billion``, ``between 65% and 67%``, ``65% 到 67%``,
       ``600 億至 640 億美元``) become one quantity with ``range=(lo, hi)``; the left end inherits a missing
       scale/currency/unit from the right end. ``X ± Y`` / ``X plus or minus Y`` / ``X 加減 Y`` / ``X 正負 Y``,
       also after a comma (``X，加減 Y``), becomes one quantity with ``pm``.
     * sign: explicit +/-/− or the nearest direction word in the same clause (increase/up/rose/grew/增/上升 ...
       vs decrease/down/fell/declined/減/下降 ...); a number right after "to"/"至"/"為"/"達"/"at" is a level (no sign).
     * qualifier: ``>`` (more than/over/surpassed/超過/逾/以上), ``<`` (up to/less than/不到/以下),
       ``<~`` (nearly/接近/將近), ``~`` (approximately/about/around/約).
     * period_token: the nearest growth token in the same sentence (<= 80 chars): yoy (year-over-year, year
       ago, same period last year, 年增/年減, 較去年同期), qoq (sequentially, prior/previous quarter, from the
       <n> quarter, 季增/季減, 較上季), mom (month-over-month, 月增/月減, 較上月).
     * NOT quantities: years (bare 1900-2100), bare integers >= 1000 without separator/unit (tickers such as
       2330/0050, product numbers such as 7600), numbers glued to letters (3Q26, FQ1, Q4, LPDDR6, HBM4E, 1H,
       low-20s, 1-gamma), ``N-N`` small-number dates (10-15), ISO dates, ``N 年/月/日/號/季``, ``第 N``,
       month-name dates (Oct. 15), fractions (1/2) and footnote markers such as ``net(2)``.
     * tables: after a ``(Unit: NT$ million ...)`` / ``(單位：新台幣百萬元)`` header, a bare number is flagged
       ``table=True`` and may match any unit class by its raw value or as money in the declared unit.
   ``extract_periods(text)`` lists period mentions (Q3, FQ4, 3Q26, 1H, FY2026, 2026 年, 第三季, 民國 116 年, a bare
   2027, ...) separately, each with the AD ``years`` it names and ``at_least`` for a lower bound (至少到 2028 年).

2. ``check_numbers(claim, excerpt)`` -- every claim quantity needs an excerpt counterpart with the same unit class
   (and compatible currency) within 0.5 % after scale normalisation (``direct``), or a ``derived`` one:
   range endpoint / midpoint, X ± Y endpoints, a qualifier bound (``>``: excerpt in [v, 1.5v]; ``<~``: [0.9v, v];
   ``~``: within 5 %; ``<``: (0, v]), A/B, A/B*100, (A/B-1)*100 for percent claims, A-B, percent-difference for pp
   claims, A/B and 1+p/100 for multiple claims. Ranges in the claim need both ends matched.
   A direct counterpart whose sign is the opposite one does not count. Unmatched quantities carry a ``kind``:
     * ``sign``  -- the same magnitude exists but with the opposite direction (17.6 % decrease vs increase);
     * ``scale`` -- the same digits exist at another power of ten (133.19 億 vs 1,331.9 億; 210 pp vs 210 bp),
                    also against an A-B (A+B with allow_sum) derivation of two excerpt quantities (12.9 vs 4.68-3.39);
     * ``value`` -- the excerpt has quantities of the same class, none equal;
     * ``absent``-- the excerpt has no quantity of that class at all.
   ``status`` is ``mismatch`` when anything is unmatched, ``ok`` when all matched, ``no_numbers`` when the claim
   has no quantity. ``period_mismatch`` lists matched growth-like claim quantities (percent/pp/multiple or signed)
   whose period token (yoy/qoq/mom) differs from every counterpart's token while some counterpart has one. It is
   reported only; deciding the label for it is policy.

   Narrative gates (owner decision 2026-10-29), run before JEV on a narrative claim:
     * ``check_periods(claim, excerpt)`` -- ``not_covered`` when the excerpt names years but not every year the claim
       names (a lower bound needs the excerpt to reach it);
     * ``quoted_media(claim, excerpt)`` -- ``restated`` when the excerpt has clarification-notice wording
       (MEDIA_MARKERS) and every number of the claim.

3. ``lint_atomic(claim)`` -- heuristic, NOT a parser. The claim is split on sentence delimiters (。；;!?. ),
   commas (，,), 、, conjunctions (且/並且/同時/以及/而且/但是/然而, and/while/but/whereas/although/however) and
   causal connectors (driven by/due to/because/as a result of/thanks to/led by, 因為/由於/主要受/受惠於/帶動/
   歸因於/源於). Connectors inside a quantity span (``between 65% and 67%``, ``523 億到 543 億``) are not split.
   A fragment counts as a clause when it contains a quantity, or a predicate from a small verb lexicon
   (English + zh-TW), or follows a causal connector. ``ok`` is False when >= 2 clauses remain.
   Honest limits: one clause holding two metrics without a delimiter ("revenue of $X with margin of Y%") passes;
   coordinated subjects ("Hon Hai and Mitsubishi plan ...") pass because the subject fragment has no verb, but a
   verb outside the lexicon makes a real second clause invisible; evaluative add-ons in the same clause
   ("66.4%, the highest in history" without a verb after the comma) can pass; "X compared with Y" is flagged
   because both sides carry a quantity, even when the comparison is the single fact. "compared with/to" is not a
   predicate, so a leading "Compared with the second quarter, X rose 17.6%" stays one clause.
   ``lint_atomic(claim, kind="inference")``: the multi-quantity rule does not refuse (a derivation names its inputs);
   causal clauses still refuse, and so do >= 2 predicate clauses once derivation-basis clauses (推算/依…/根據…) are set
   aside. ``has_inference_basis(text)`` is the runner's basis-marker gate for inference claims.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from itertools import permutations

from poseidon.decision_loop.claim_quantities import (
    _PRED_RE,
    _SPLIT_RE,
    CJK_SCALE,
    GROWTH_RE,
    UNITS,
    Quantity,
    _annotate,
    _bound_ok,
    _close,
    _compatible,
    _growth_like,
    _points,
    _raw_tokens,
    _word_tokens,
)

# --------------------------------------------------------------------------------------------- lexicons
PERIOD_RE = re.compile(
    r"(?P<roc>(?:民國\s*)?(?<![\d.])1[0-3]\d\s*年)|"
    r"(?P<span>(?<![\d.])20\d{2}\s*(?:[-–~～至到]|to|through)\s*20\d{2}(?![\d%]|\.\d))|"
    r"\b(?:F?Q[1-4](?:[-'’]?\d{2,4})?|[1-4]Q\d{2}|[12]H(?:\d{2})?|H[12]|FY\s?\d{2,4}|CY\s?\d{2,4})\b|"
    r"(?:fiscal|calendar)\s+(?:Q[1-4]\s+)?(?:19|20)\d{2}|(?:first|second|third|fourth)[- ]quarter|"
    r"(?:first|second)[- ]half|(?:19|20)\d{2}\s*年|第[一二三四1-4]季|[上下]半年|會計年度|"
    r"(?<![\d.,$/A-Za-z])20\d{2}(?![\d%]|[.,]\d)"
)
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
ROC_EPOCH = 1911  # 民國 116 年 = 2027
AT_LEAST_BEFORE = re.compile(
    r"(?:至少|最少)\s*(?:延續|持續)?\s*(?:到|至)?\s*$|at least\s+(?:through|until|to|into|in)?\s*$|through at least\s*$",
    re.I,
)
AT_LEAST_AFTER = re.compile(r"^\s*(?:以後|之後|或更久|or later|and beyond)", re.I)


# --------------------------------------------------------------------------------------------- extraction


def _merge(text, toks):
    toks = sorted(toks, key=lambda q: q.span[0])
    # drop overlaps (numeric tokens win over word tokens because they come first after sort by start)
    clean = []
    for q in toks:
        if clean and q.span[0] < clean[-1].span[1]:
            continue
        clean.append(q)
    toks = clean
    out = []
    i = 0
    while i < len(toks):
        a = toks[i]
        b = toks[i + 1] if i + 1 < len(toks) else None
        if b is not None:
            between = text[a.span[1] : b.span[0]]
            # Chinese compound: 1 兆 4,942.4 億
            if (
                between.strip() == ""
                and a.scale_tok in CJK_SCALE
                and b.scale_tok in CJK_SCALE
                and CJK_SCALE[b.scale_tok] < CJK_SCALE[a.scale_tok]
                and a.raw.endswith(a.scale_tok)
            ):
                b2 = Quantity(
                    **{
                        **asdict(b),
                        "raw": text[a.span[0] : b.span[1]],
                        "span": (a.span[0], b.span[1]),
                        "value": a.value + b.value,
                        "currency": b.currency or a.currency,
                    }
                )
                b2.unit_class = "money" if b2.currency or b.unit_class == "money" else b.unit_class
                toks[i + 1] = b2
                i += 1
                continue
            is_and = re.fullmatch(r"\s*and\s*", between, re.I) and re.search(
                r"(?:between|介於)\s*\S*\s*$", text[max(0, a.span[0] - 14) : a.span[0]], re.I
            )
            is_rng = re.fullmatch(r"\s*(?:to|–|—|~|～|至|到|-)\s*", between, re.I) or is_and
            # a comma may open the ± part ("$61.5 billion, plus or minus $1.5 billion", 「$38.15，加減 $1.00」); never a
            # range's "to": "rose 5%, to 30%" names a level
            is_pm = re.fullmatch(r"\s*[,，]?\s*(?:±|\+/-|\+/−|plus or minus|加減|正負)\s*", between, re.I)
            if is_rng or is_pm:
                if not a.has_unit and b.has_unit:  # 52.3–54.3 billion / 600 億至 640 億美元
                    a.value = a.value / a.scale * (b.scale if a.scale == 1.0 else a.scale)
                    if b.unit_class != "money" and b.unit_class != "count":
                        a.value *= next((mu for t, c, mu in UNITS if c == b.unit_class and t in b.raw), 1.0)
                    a.unit_class, a.currency = b.unit_class, b.currency
                elif not b.has_unit and a.has_unit:
                    b.value = b.value / b.scale * (a.scale if b.scale == 1.0 else b.scale)
                    b.unit_class, b.currency = a.unit_class, a.currency
                elif not a.has_unit and b.scale_tok and not a.scale_tok:
                    a.value *= b.scale
                same_cls = a.unit_class == b.unit_class
                cur_ok = a.currency is None or b.currency is None or a.currency == b.currency
                if same_cls and cur_ok and (is_pm or a.value <= b.value):
                    m = Quantity(
                        raw=text[a.span[0] : b.span[1]],
                        span=(a.span[0], b.span[1]),
                        value=a.value,
                        unit_class=a.unit_class,
                        currency=a.currency or b.currency,
                        qualifier=a.qualifier,
                        scale=b.scale,
                        scale_tok=b.scale_tok,
                        has_unit=True,
                    )
                    if is_pm:
                        m.pm = b.value
                    else:
                        m.range = (a.value, b.value)
                    out.append(m)
                    i += 2
                    continue
        out.append(a)
        i += 1
    return out


def extract_quantities(text: str) -> list[Quantity]:
    """All quantities in `text` (see module docstring), in order of appearance."""
    return _annotate(text, _merge(text, _raw_tokens(text) + _word_tokens(text)))


def _period_years(m) -> list[int]:
    """AD years a PERIOD_RE match names: 民國 years + 1911, a 2026-2028 span in full, 4-digit years, the 2-digit year
    of FY27 / CY27 / 3Q28 / 2H26 / Q4'26 as 20xx; [] for a period without a year (Q3, 第三季, first half)."""
    if m.group("roc"):
        return [int(re.search(r"1[0-3]\d", m.group("roc")).group(0)) + ROC_EPOCH]
    full = [int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", m.group(0))]
    if m.group("span"):
        return list(range(full[0], full[1] + 1))
    if full:
        return full
    short = re.search(r"(?:FY|CY|Q[1-4][-'’]?|[1-4]Q|[12]H)\s?(\d{2})$", m.group(0))
    return [2000 + int(short.group(1))] if short else []


def extract_periods(text: str) -> list[dict]:
    """Period mentions (not quantities) plus growth tokens, for display/audit and the narrative period gate.
    A period carries `years` (AD, see _period_years; full-width digits read as ASCII) and `at_least`, true for a lower
    bound such as 「至少到 2028 年」, "at least through 2028" or "2028 or later"."""
    text = text.translate(FULLWIDTH_DIGITS)
    out = [
        {
            "text": m.group(0),
            "span": m.span(),
            "kind": "period",
            "years": _period_years(m),
            "at_least": bool(
                AT_LEAST_BEFORE.search(text[max(0, m.start() - 24) : m.start()])
                or AT_LEAST_AFTER.match(text[m.end() : m.end() + 12])
            ),
        }
        for m in PERIOD_RE.finditer(text)
    ]
    out += [
        {"text": m.group(0), "span": m.span(), "kind": next(k for k, v in m.groupdict().items() if v)}
        for m in GROWTH_RE.finditer(text)
    ]
    return sorted(out, key=lambda d: d["span"])


# --------------------------------------------------------------------------------------------- numeric check


def _match_value(v, cq, pts, allow_sum=False):
    """Returns (direct_matches, derived_match_or_None). allow_sum adds A+B over same-class pairs (inference claims)."""
    direct = [
        p
        for p in pts
        if p[3].startswith("direct") and _compatible(cq.unit_class, cq.currency, p[1], p[2]) and _close(v, p[0])
    ]
    if direct:
        return direct, None
    for p in pts:
        if p[3].startswith("derived") and _compatible(cq.unit_class, cq.currency, p[1], p[2]) and _close(v, p[0]):
            return [], {"how": p[3].split(":", 1)[1], "from": [p[4].raw]}
    if cq.qualifier:
        for p in pts:
            if _compatible(cq.unit_class, cq.currency, p[1], p[2]) and p[1] != "*" and _bound_ok(cq.qualifier, v, p[0]):
                return [], {"how": f"bound{cq.qualifier}", "from": [p[4].raw]}
    # unit-bearing claim vs bare excerpt digits: tables carry the unit in the header ("(thousand NTD): 129,770,564"),
    # so compare the claim's unscaled digits with bare excerpt numbers, directly and as A-B over bare pairs.
    if cq.has_unit or cq.scale != 1:
        vr = v / cq.scale
        bare = [
            p
            for p in pts
            if p[3] == "direct" and p[1] == "count" and not p[4].has_unit and p[4].scale == 1 and p[0] != 0
        ]
        hit = [p for p in bare if _close(vr, p[0])]
        if hit:
            return hit, None
        for a, b in permutations(bare, 2):
            x = a[0] - b[0]
            if x > 0 and _close(vr, x):
                return [], {"how": "A-B (bare digits)", "from": [a[4].raw, b[4].raw], "value": round(x, 6)}
    # arithmetic over pairs of plain excerpt points
    plain = [p for p in pts if p[3] == "direct" and p[0] != 0]
    for a, b in permutations(plain, 2):
        same = a[1] == b[1] and (a[1] != "money" or not a[2] or not b[2] or a[2] == b[2])
        cands = []
        if cq.unit_class == "percent" and same:
            cands += [(a[0] / b[0] * 100, "A/B*100"), ((a[0] / b[0] - 1) * 100, "(A/B-1)*100")]
        if cq.unit_class == "pp" and a[1] == b[1] == "percent":
            cands.append((a[0] - b[0], "A-B (pct points)"))
        if cq.unit_class == "multiple" and same and a[1] not in ("percent", "pp", "multiple"):
            cands.append((a[0] / b[0], "A/B"))
        if (
            cq.unit_class in ("money", "count", "days")
            and same
            and a[1] == cq.unit_class
            and _compatible(cq.unit_class, cq.currency, a[1], a[2])
        ):
            cands.append((a[0] - b[0], "A-B"))
            if allow_sum:
                cands.append((a[0] + b[0], "A+B"))
        for x, how in cands:
            if x > 0 and (_close(v, x) or (cq.qualifier and _bound_ok(cq.qualifier, v, x))):
                return [], {"how": how, "from": [a[4].raw, b[4].raw], "value": round(x, 6)}
    if cq.unit_class == "multiple":
        for p in plain:
            if p[1] == "percent" and p[4].sign >= 0:
                x = 1 + p[0] / 100
                if _close(v, x) or (cq.qualifier and _bound_ok(cq.qualifier, v, x)):
                    return [], {"how": "1+p/100", "from": [p[4].raw], "value": round(x, 6)}
    return [], None


def _scale_off(v, x):
    if not (v and x):
        return None
    r = math.log10(abs(v / x))
    return round(r) if abs(r - round(r)) < 0.002 and round(r) != 0 else None


def _miss_kind(v, cq, pts, allow_sum=False):
    same = [p for p in pts if p[1] != "*" and _compatible(cq.unit_class, cq.currency, p[1], p[2])]
    for p in same:
        k = _scale_off(v, p[0])
        if k is not None:
            return "scale", {"excerpt": p[4].raw, "factor": f"1e{k}"}
    # a derivation (A-B, and A+B when allowed) off by a power of ten is a scale slip too
    if cq.unit_class in ("money", "count", "days"):
        plain = [p for p in same if p[3] == "direct" and p[1] == cq.unit_class and p[0] != 0]
        for a, b in permutations(plain, 2):
            for x, how in ((a[0] - b[0], "A-B"), *(((a[0] + b[0], "A+B"),) if allow_sum else ())):
                k = _scale_off(v, x) if x > 0 else None
                if k is not None:
                    return "scale", {"excerpt": [a[4].raw, b[4].raw], "derived": how, "factor": f"1e{k}"}
    return ("value", None) if same else ("absent", None)


def check_numbers(claim: str, excerpt: str, allow_sum: bool = False) -> dict:
    """Claim-vs-excerpt numeric check (see module docstring). allow_sum=True also accepts A+B of two excerpt
    quantities of the claim's class (used for inference claims only, so a fact claim never matches a sum)."""
    cqs, eqs = extract_quantities(claim), extract_quantities(excerpt)
    if not cqs:
        return {"status": "no_numbers", "unmatched": [], "matched": [], "period_mismatch": []}
    pts = [p for q in eqs for p in _points(q)]
    matched, unmatched, period_mm = [], [], []
    for cq in cqs:
        values = list(cq.range) if cq.range else [cq.value]
        res, ok = [], True
        for v in values:
            direct, derived = _match_value(v, cq, pts, allow_sum)
            if direct:
                good = [p for p in direct if not (cq.sign and p[4].sign and cq.sign != p[4].sign)]
                if good:
                    res.append({"value": v, "match": "direct", "counterparts": [p[4].raw for p in good], "_pts": good})
                    continue
                ok = False
                res.append(
                    {
                        "value": v,
                        "kind": "sign",
                        "detail": {"excerpt": [p[4].raw for p in direct], "claim_sign": cq.sign},
                    }
                )
                continue
            if derived:
                res.append({"value": v, "match": "derived", **derived})
                continue
            ok = False
            kind, det = _miss_kind(v, cq, pts, allow_sum)
            res.append({"value": v, "kind": kind, **({"detail": det} if det else {})})
        entry = {
            "claim": cq.raw,
            "canonical": list(cq.canonical()),
            **({"range": list(cq.range)} if cq.range else {}),
            **({"qualifier": cq.qualifier} if cq.qualifier else {}),
            **({"sign": cq.sign} if cq.sign else {}),
        }
        if ok:
            kinds = {r["match"] for r in res}
            entry["match"] = "derived" if "derived" in kinds else "direct"
            entry["via"] = [{k: v for k, v in r.items() if k not in ("_pts",)} for r in res]
            matched.append(entry)
            if cq.period and _growth_like(cq):
                cps = [p[4] for r in res for p in r.get("_pts", [])]
                periods = {p.period for p in cps if p.period}
                if periods and cq.period not in periods:
                    period_mm.append(
                        {
                            "claim": cq.raw,
                            "claim_period": cq.period,
                            "excerpt_periods": sorted(periods),
                            "counterparts": [p.raw for p in cps],
                        }
                    )
        else:
            bad = [r for r in res if "kind" in r]
            entry["kind"] = bad[0]["kind"]
            entry["detail"] = [{k: v for k, v in r.items() if k not in ("_pts",)} for r in res]
            unmatched.append(entry)
    return {
        "status": "mismatch" if unmatched else "ok",
        "unmatched": unmatched,
        "matched": matched,
        "period_mismatch": period_mm,
    }


# --------------------------------------------------------------------------------------------- narrative gates
# A two-digit pair of consecutive years such as 「27/28」, read on the excerpt side only (acceptance 2026-10-30, N5):
# a date (10/27), a non-consecutive pair (27/29) or a number that goes on (1.27/28, 27/28%) is not one.
SLASH_YEARS = re.compile(r"(?<![\d/.])([2-3]\d)/([2-3]\d)(?![\d/.%])")


def slash_years(text: str) -> set[int]:
    return {2000 + int(y) for m in SLASH_YEARS.finditer(text) if int(m[2]) == int(m[1]) + 1 for y in m.groups()}


def check_periods(claim: str, excerpt: str) -> dict:
    """Does the excerpt cover every year the claim names? A claim year is covered when the excerpt names it; a lower
    bound (「至少到 2028 年」) is covered when the excerpt's latest year reaches it. An excerpt that names no year proves
    nothing either way (no_excerpt_years). A year inside an excerpt "through 2028" span is covered only when named.
    The excerpt also names the years of a slash pair (slash_years); the claim never does."""
    claim_p = [p for p in extract_periods(claim) if p["kind"] == "period" and p["years"]]
    have = sorted(
        {y for p in extract_periods(excerpt) if p["kind"] == "period" for y in p["years"]} | slash_years(excerpt)
    )
    missing = sorted(
        {y for p in claim_p for y in p["years"] if have and (max(have) < y if p["at_least"] else y not in have)}
    )
    status = "not_covered" if missing else "covered"
    if not have:
        status = "no_excerpt_years"
    if not claim_p:
        status = "no_claim_years"
    return {
        "status": status,
        "claim_years": sorted({y for p in claim_p for y in p["years"]}),
        "at_least": sorted({y for p in claim_p if p["at_least"] for y in p["years"]}),
        "excerpt_years": have,
        "missing": missing,
    }


# TWSE/MOPS clarification-notice wording: the company restates a media report it neither confirms nor denies.
# Matched case-insensitively with all whitespace removed (PDF text spaces CJK characters apart).
MEDIA_MARKERS = (
    ("報導內容", "quote"),
    ("媒體報導", "quote"),
    ("報載", "quote"),
    ("content of the report", "quote"),
    ("media report", "quote"),
    ("press report", "quote"),
    ("news report", "quote"),
    ("不予評論", "no_comment"),
    ("不評論", "no_comment"),
    ("does not comment", "no_comment"),
    ("市場傳聞", "rumor"),
    ("market rumor", "rumor"),
    ("speculative report", "rumor"),
)


def quoted_media(claim: str, excerpt: str) -> dict:
    """restated: the excerpt carries a MEDIA_MARKERS marker and every number of the claim (check_numbers ok)."""
    flat = re.sub(r"\s+", "", excerpt).lower()
    hits = [{"marker": m, "kind": k} for m, k in MEDIA_MARKERS if re.sub(r"\s+", "", m) in flat]
    numbers = check_numbers(claim, excerpt)["status"]
    return {"restated": bool(hits) and numbers == "ok", "markers": hits, "numbers": numbers}


# --------------------------------------------------------------------------------------------- atomic lint


# Inference basis markers (owner decision 2026-10-21): an inference claim must say what it rests on or that it is
# an expectation, so a reader never takes it for an observed fact. Used by the runner gate (has_inference_basis);
# lint_atomic sets derivation-basis clauses ("由 A 與 B 推算", "依 X 判斷") aside for inference claims.
_BASIS_RE = re.compile(
    r"推算|估算|推估|推論|估計|判斷|研判|預期|預估|可能|若|如果|假設|換算|計算|相減|相加|合計|依(?![然舊])|根據|基於|"
    r"意味|暗示|隱含|有望|或將|恐將|\bbased\s+on\b|\bimpl(?:y|ies|ied)\b|\bsuggests?\b|\blikely\b|\bprobabl[ey]\b|"
    r"\bmay\b|\bmight\b|\bcould\b|\bexpect(?:s|ed)?\b|\bestimat\w*|\bif\b|\bassum\w*|\bderived\b|\bcomputed\b|"
    r"\bcalculated\b",
    re.I,
)
# narrower: a clause that names the inputs of a derivation ("由上半年 4.68 元與第二季 3.39 元推算", "依 X 判斷")
_DERIVATION_BASIS_RE = re.compile(
    r"推算|估算|推估|換算|計算|相減|相加|依(?![然舊])|根據|基於|\bbased\s+on\b|\bderived\s+from\b|\bcomputed\s+from\b",
    re.I,
)


def has_inference_basis(text: str) -> bool:
    """True when an inference claim's text carries a basis / expectation marker (_BASIS_RE)."""
    return bool(_BASIS_RE.search(text or ""))


def lint_atomic(claim: str, kind: str | None = None) -> dict:
    """Heuristic atomicity lint (see module docstring for the rules and their blind spots).

    kind="inference" (owner decision 2026-10-21): the multi-quantity rule does not refuse, because a derived inference
    names its inputs ("由上半年累計 4.68 元與第二季 3.39 元推算，第一季每股盈餘約為 1.29 元"). A causal/driver clause still
    refuses, and so do >= 2 clauses with their own predicate once derivation-basis clauses are set aside."""
    qspans = [q.span for q in extract_quantities(claim)]
    start = 0
    pieces = []
    causal_next = False
    for m in _SPLIT_RE.finditer(claim):
        if any(a < m.start() < b for a, b in qspans):
            continue  # connector inside a range / ± quantity
        pieces.append((start, m.start(), causal_next))
        causal_next = bool(m.group("causal"))
        start = m.end()
    pieces.append((start, len(claim), causal_next))
    clauses = []
    for s, e, causal in pieces:
        frag = claim[s:e].strip(" \t\n。.;；,，、")
        if not frag:
            continue
        has_q = any(s <= a < e for a, b in qspans)
        has_pred = bool(_PRED_RE.search(frag))
        content = len(re.findall(r"[A-Za-z]{2,}", frag)) >= 2 or len(re.findall(r"[一-鿿]", frag)) >= 2
        if has_q or has_pred or (causal and content):
            clauses.append({"text": frag, "quantity": has_q, "predicate": has_pred, "causal": causal})
    if kind == "inference" and len(clauses) >= 2 and not any(c["causal"] for c in clauses):
        asserted = [c for c in clauses if c["predicate"] and not _DERIVATION_BASIS_RE.search(c["text"])]
        if len(asserted) >= 2:
            return {
                "ok": False,
                "clauses": [c["text"] for c in clauses],
                "reason": "non-atomic: %d clauses with separate predicates" % len(asserted),
            }
        return {
            "ok": True,
            "clauses": [c["text"] for c in clauses],
            "reason": "inference: multi-quantity derivation allowed",
        }
    if len(clauses) >= 2:
        why = []
        if sum(c["quantity"] for c in clauses) >= 2:
            why.append("%d clauses carry their own quantity" % sum(c["quantity"] for c in clauses))
        if any(c["causal"] for c in clauses):
            why.append("causal/driver clause attached")
        if not why:
            why.append("%d clauses with separate predicates" % len(clauses))
        return {"ok": False, "clauses": [c["text"] for c in clauses], "reason": "non-atomic: " + "; ".join(why)}
    return {"ok": True, "clauses": [c["text"] for c in clauses] or [claim.strip()], "reason": "single clause"}


# --------------------------------------------------------------------------------------------- evidence tier
OFFICIAL_TIERS = ("primary filing", "company release")


def evidence_tier(e: dict) -> str:
    """Source tier of a frozen evidence item, as render_theme.py discloses it and the narrative [check] gate reads it.
    A declared payload source_tier wins; a secondary item never falls through to a document default."""
    p = e.get("payload") or {}
    if p.get("source_tier"):
        return p["source_tier"]
    k, uri = e.get("kind"), e.get("source_uri", "")
    if k == "secondary":
        return "secondary (no tier)"
    if k in ("price_facts", "narrative_snapshot") or uri.startswith("derived://"):
        return "derived"
    if k in ("ohlcv", "kpi_series"):
        return "exchange feed" + ("（經 Thalassa）" if k == "ohlcv" else "")
    if any(h in uri for h in ("sec.gov", "mops.twse.com.tw", "mopsov.twse.com.tw")):
        return "primary filing"
    return "company release"


# claim_theme_links.evidence_tier is the highest tier a claim cites (design §2.5), ranked in this order
TIER_ORDER = (*OFFICIAL_TIERS, "exchange feed", "derived", "sell-side", "media", "kol")


def _tier_rank(tier):
    return next((n for n, t in enumerate(TIER_ORDER) if tier.startswith(t)), len(TIER_ORDER))


def best_evidence(c, by_id):
    """The claim's highest-tier cited evidence (the latest of equals): claim_theme_links' evidence_tier and event_time."""
    cited = [by_id[e] for e in c["evidence_ids"] if e in by_id]
    return max(cited, key=lambda e: (-_tier_rank(evidence_tier(e)), e["event_time"]))


def precheck(claim: str, excerpt: str | None) -> dict:
    """Both checks; `excerpt=None` skips the numeric check."""
    num = check_numbers(claim, excerpt) if excerpt is not None else {"status": "skipped"}
    return {"numbers": num, "atomic": lint_atomic(claim)}


# --------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("text")
    p = sub.add_parser("check")
    p.add_argument("--claim", required=True)
    p.add_argument("--excerpt", required=True)
    p = sub.add_parser("lint")
    p.add_argument("claim")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        out = {"quantities": [q.to_dict() for q in extract_quantities(a.text)], "periods": extract_periods(a.text)}
    elif a.cmd == "check":
        out = check_numbers(a.claim, a.excerpt)
    else:
        out = lint_atomic(a.claim)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
