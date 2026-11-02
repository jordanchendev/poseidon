"""Deterministic claim prechecks for the research contract (SEED-002 C2).

Ported from local_dev/scripts/research/claim_precheck.py (offline stopgap tooling); stdlib only. The tokenizer and
lexicons it shares with scripts/research/claim_precheck.py live in ``claim_quantities`` (one copy). The rules and
blind spots below are the stopgap's, unchanged. ``ResearchService.complete_human_research`` runs
``lint_atomic`` on every claim and ``check_numbers`` against the cited evidence excerpts
(see ``poseidon.decision_loop.research.precheck_claims``).

Three pieces (``extract_periods`` and the CLI of the stopgap are not ported):

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
       scale/currency/unit from the right end. ``X ± Y`` / ``X plus or minus Y`` becomes one quantity with ``pm``.
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

2. ``check_numbers(claim, excerpt)`` -- every claim quantity needs an excerpt counterpart with the same unit class
   (and compatible currency) within 0.5 % after scale normalisation (``direct``), or a ``derived`` one:
   range endpoint / midpoint, X ± Y endpoints, a qualifier bound (``>``: excerpt in [v, 1.5v]; ``<~``: [0.9v, v];
   ``~``: within 5 %; ``<``: (0, v]), A/B, A/B*100, (A/B-1)*100 for percent claims, A-B, percent-difference for pp
   claims, A/B and 1+p/100 for multiple claims. Ranges in the claim need both ends matched.
   A direct counterpart whose sign is the opposite one does not count. Unmatched quantities carry a ``kind``:
     * ``sign``  -- the same magnitude exists but with the opposite direction (17.6 % decrease vs increase);
     * ``scale`` -- the same digits exist at another power of ten (133.19 億 vs 1,331.9 億; 210 pp vs 210 bp);
     * ``value`` -- the excerpt has quantities of the same class, none equal;
     * ``absent``-- the excerpt has no quantity of that class at all.
   ``status`` is ``mismatch`` when anything is unmatched, ``ok`` when all matched, ``no_numbers`` when the claim
   has no quantity. ``period_mismatch`` lists matched growth-like claim quantities (percent/pp/multiple or signed)
   whose period token (yoy/qoq/mom) differs from every counterpart's token while some counterpart has one. It is
   reported only; deciding the label for it is policy.

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
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from itertools import permutations

from poseidon.decision_loop.claim_quantities import (
    _PRED_RE,
    _SPLIT_RE,
    CJK_SCALE,
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
            is_pm = re.fullmatch(r"\s*(?:±|\+/-|\+/−|plus or minus)\s*", between, re.I)
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


def _match_value(v, cq, pts):
    """Returns (direct_matches, derived_match_or_None)."""
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


def _miss_kind(v, cq, pts):
    same = [p for p in pts if p[1] != "*" and _compatible(cq.unit_class, cq.currency, p[1], p[2])]
    for p in same:
        if p[0] and v:
            r = math.log10(abs(v / p[0]))
            if abs(r - round(r)) < 0.002 and round(r) != 0:
                return "scale", {"excerpt": p[4].raw, "factor": f"1e{round(r)}"}
    return ("value", None) if same else ("absent", None)


def check_numbers(claim: str, excerpt: str) -> dict:
    """Claim-vs-excerpt numeric check (see module docstring)."""
    cqs, eqs = extract_quantities(claim), extract_quantities(excerpt)
    if not cqs:
        return {"status": "no_numbers", "unmatched": [], "matched": [], "period_mismatch": []}
    pts = [p for q in eqs for p in _points(q)]
    matched, unmatched, period_mm = [], [], []
    for cq in cqs:
        values = list(cq.range) if cq.range else [cq.value]
        res, ok = [], True
        for v in values:
            direct, derived = _match_value(v, cq, pts)
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
            kind, det = _miss_kind(v, cq, pts)
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


def lint_atomic(claim: str) -> dict:
    """Heuristic atomicity lint (see module docstring for the rules and their blind spots)."""
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
    if len(clauses) >= 2:
        why = []
        if sum(c["quantity"] for c in clauses) >= 2:
            why.append(f"{sum(c['quantity'] for c in clauses)} clauses carry their own quantity")
        if any(c["causal"] for c in clauses):
            why.append("causal/driver clause attached")
        if not why:
            why.append(f"{len(clauses)} clauses with separate predicates")
        return {"ok": False, "clauses": [c["text"] for c in clauses], "reason": "non-atomic: " + "; ".join(why)}
    return {"ok": True, "clauses": [c["text"] for c in clauses] or [claim.strip()], "reason": "single clause"}
