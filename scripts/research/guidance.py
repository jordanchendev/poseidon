"""Document guidance and actuals (gap B2), source tiers and the narrative_snapshot evidence item for research_theme.py:
guidance[] / actuals[] checked number by number against their excerpts, the guidance bridge to the nearest
predecessor, and the quantity rendering consensus.py and kpi_series.py share. Moved out of research_theme.py unchanged;
see its module docstring for the contract.
"""

import re
import sys

from claim_precheck import check_numbers, extract_quantities
from refusal import ResearchRefusal

from poseidon.core.canonical import content_sha256

# ------------------------------------------------------------------------------------------------ guidance (gap B2)
SOURCE_TIERS = ("primary filing", "company release", "exchange feed", "derived", "sell-side", "media", "kol")


SECONDARY_TIERS = ("sell-side", "media", "kol")


SNAPSHOT_KEYS = ("id", "kind", "source_uri", "event_time", "available_at", "payload")


def snapshot_evidence(snap, as_of):
    """cfg narrative_snapshot -> frozen evidence item. The selection and fold are assemble.py's; this checks the shape."""
    if sorted(snap) != sorted(SNAPSHOT_KEYS) or snap["id"] != "narrative_snapshot" or snap["kind"] != snap["id"]:
        raise ResearchRefusal(
            "narrative_snapshot must be {%s} with id and kind narrative_snapshot" % ", ".join(SNAPSHOT_KEYS)
        )
    p = snap["payload"]
    if sorted(p) != ["algorithm", "not_shown", "records", "themes"] or not (p["records"] or p["themes"]):
        raise ResearchRefusal(
            "narrative_snapshot payload needs algorithm, themes, records and not_shown, and a theme or record"
        )
    return {**snap, "recorded_at": as_of, "content_sha256": content_sha256(p)}


GUIDANCE_BASES = ("GAAP", "non-GAAP", "IFRS", "n/a")


_UNIT_RE = re.compile(r"^(USD|US\$|NTD|TWD|NT\$|JPY)(?:\s+(thousand|million|billion|trillion))?$", re.I)


_CUR_PREFIX = {"usd": "US$", "us$": "US$", "ntd": "NT$", "twd": "NT$", "nt$": "NT$", "jpy": "¥"}


def _num_text(v):
    v = float(v)
    return f"{int(v):,}" if v.is_integer() else repr(v)


def quantity_text(value, unit):
    """Render (value, unit) as text claim_precheck parses back to the same canonical quantity.
    Units: % | pp | bp | x | USD/NTD/TWD/JPY [thousand|million|billion|trillion] | anything else (count with that word)."""
    sign, mag = ("-" if float(value) < 0 else ""), _num_text(abs(float(value)))
    u = (unit or "").strip()
    m = _UNIT_RE.match(u)
    if u in ("%", "percent"):
        return f"{sign}{mag}%"
    if u == "pp":
        return f"{sign}{mag} percentage points"
    if u == "bp":
        return f"{sign}{mag} basis points"
    if u in ("x", "times"):
        return f"{mag} times"
    if m:
        return f"{_CUR_PREFIX[m.group(1).lower()]}{sign}{mag}" + (f" {m.group(2).lower()}" if m.group(2) else "")
    return f"{sign}{mag} {u}".rstrip()


def _excerpt_by_locator(item, locator, what):
    hits = [ex["text"] for ex in item["excerpts"] if ex.get("locator") == locator]
    if not hits:
        raise ResearchRefusal("%s cites excerpt_locator %r which is not an excerpt of %s" % (what, locator, item["id"]))
    return hits[0]


def _numbers_in_excerpt(texts, excerpt, what, strict):
    """check_numbers every text against the excerpt. scale/sign mismatches always refuse; other mismatches refuse when
    strict, else warn on stderr. Returns 'ok' or 'warn'."""
    status = "ok"
    for t in texts:
        r = check_numbers(t, excerpt)
        neg = [q for q in extract_quantities(t) if q.sign == -1]
        if (
            r["status"] == "ok"
            and neg
            and not all(
                any(
                    e.sign == -1 and abs(e.value - q.value) <= 0.005 * max(abs(q.value), 1e-12)
                    for e in extract_quantities(excerpt)
                )
                for q in neg
            )
        ):
            # check_numbers lets a signed claim match an unsigned excerpt number; a negative structured value must not.
            r = {"status": "mismatch", "unmatched": [{"kind": "sign"}]}
        if r["status"] == "ok":
            continue
        kinds = sorted({u["kind"] for u in r["unmatched"]}) or [r["status"]]
        msg = "%s: %r not found in the cited excerpt (%s)" % (what, t, ", ".join(kinds))
        if strict or {"scale", "sign"} & set(kinds):
            raise ResearchRefusal("refusing to freeze: " + msg)
        print("warning: " + msg, file=sys.stderr)
        status = "warn"
    return status


def validate_guidance(item):
    """documents[].guidance[]: {metric, period, low+high | value, unit, basis, fx_assumption?, excerpt_locator}.
    Every number must appear in the referenced excerpt (claim_precheck.check_numbers): refuse on scale/sign mismatch,
    warn otherwise. Writes no state; the guidance list is hashed as part of the document payload."""
    if not isinstance(item["guidance"], list):
        raise ResearchRefusal("document %s guidance must be a list" % item["id"])
    for n, g in enumerate(item["guidance"]):
        what = "%s guidance[%d]" % (item["id"], n)
        for k in ("metric", "period", "unit", "excerpt_locator"):
            if not isinstance(g.get(k), str) or not g[k].strip():
                raise ResearchRefusal("%s.%s must be a non-empty string" % (what, k))
        if g.get("basis") not in GUIDANCE_BASES:
            raise ResearchRefusal(
                "%s.basis must be one of %s, got %r" % (what, "|".join(GUIDANCE_BASES), g.get("basis"))
            )
        has_range, has_value = "low" in g or "high" in g, "value" in g
        if has_range == has_value or (has_range and not ("low" in g and "high" in g)):
            raise ResearchRefusal("%s needs either low and high, or value" % what)
        nums = [g[k] for k in ("low", "high", "value") if k in g]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in nums):
            raise ResearchRefusal("%s low/high/value must be numbers" % what)
        if has_range and g["low"] > g["high"]:
            raise ResearchRefusal("%s low > high" % what)
        excerpt = _excerpt_by_locator(item, g["excerpt_locator"], what)
        texts = [quantity_text(x, g["unit"]) for x in nums]
        fx = g.get("fx_assumption")
        if fx is not None:
            if not isinstance(fx, str) or not fx.strip():
                raise ResearchRefusal("%s.fx_assumption must be a non-empty string when present" % what)
            texts.append(fx)
        _numbers_in_excerpt(texts, excerpt, what, strict=False)


def validate_actuals(item):
    """actuals attached to a document (from config actuals[]): {metric, period, value, unit, excerpt_locator}.
    Actuals are facts, so ANY numeric mismatch with the cited excerpt refuses the freeze."""
    for n, a in enumerate(item["actuals"]):
        what = "actuals for %s [%d]" % (item["id"], n)
        for k in ("metric", "period", "unit", "excerpt_locator"):
            if not isinstance(a.get(k), str) or not a[k].strip():
                raise ResearchRefusal("%s.%s must be a non-empty string" % (what, k))
        if isinstance(a.get("value"), bool) or not isinstance(a.get("value"), (int, float)):
            raise ResearchRefusal("%s.value must be a number" % what)
        _numbers_in_excerpt(
            [quantity_text(a["value"], a["unit"])],
            _excerpt_by_locator(item, a["excerpt_locator"], what),
            what,
            strict=True,
        )


def attach_actuals(documents, actuals):
    """Config actuals[] {metric, period, value, unit, evidence_id, excerpt_locator} -> documents[].actuals (copies)."""
    by_id = {d["id"]: dict(d) for d in documents}
    for n, a in enumerate(actuals or []):
        if a.get("evidence_id") not in by_id:
            raise ResearchRefusal(
                "actuals[%d].evidence_id %r is not a config document; an actual must cite a document excerpt"
                % (n, a.get("evidence_id"))
            )
        doc = by_id[a["evidence_id"]]
        doc["actuals"] = list(doc.get("actuals", [])) + [{k: v for k, v in a.items() if k != "evidence_id"}]
    return [by_id[d["id"]] for d in documents]


def _items_from(evidence, key):
    return [dict(x, document_id=e["id"]) for e in evidence for x in (e.get("payload") or {}).get(key, []) or []]


def narrative_actuals(narrative):
    """Optional claims[].actual {metric, period, value, unit} on fact claims; the number must be in the claim text
    (the claim text itself is prechecked against its cited excerpts)."""
    out = []
    for n, c in enumerate(narrative.get("claims", [])):
        a = c.get("actual")
        if a is None:
            continue
        what = "claims[%d].actual" % n
        if c.get("kind") != "fact":
            raise ResearchRefusal("%s is only allowed on fact claims" % what)
        if (
            not isinstance(a, dict)
            or any(not isinstance(a.get(k), str) or not a[k].strip() for k in ("metric", "period", "unit"))
            or isinstance(a.get("value"), bool)
            or not isinstance(a.get("value"), (int, float))
        ):
            raise ResearchRefusal("%s must be {metric, period, value (number), unit}" % what)
        _numbers_in_excerpt([quantity_text(a["value"], a["unit"])], c["text"], what + " vs its claim text", strict=True)
        out.append(dict(a, source="claims[%d]" % n, evidence_ids=c.get("evidence_ids", [])))
    return out


def _key(s):
    return " ".join(str(s).lower().split())


def predecessor_guidance(db, rev, periods, max_depth=20):
    """Walk the completed predecessor chain; return (revision_id, guidance items for any of `periods`) from the nearest
    predecessor whose frozen document payloads carry guidance for one of those periods, else (None, [])."""
    from poseidon.models.data_manifest import DataManifest
    from poseidon.models.research_revision import ResearchRevision

    want, prev_id = {_key(p) for p in periods}, rev.previous_revision_id
    for _ in range(max_depth):
        if not prev_id:
            break
        prev = db.get(ResearchRevision, prev_id)
        if prev is None:
            break
        man = db.get(DataManifest, prev.manifest_id)
        items = [g for g in _items_from(man.payload_json.get("evidence", []), "guidance") if _key(g["period"]) in want]
        if items:
            return str(prev.id), items
        prev_id = prev.previous_revision_id
    return None, []


def _span(g):
    return (g["low"], g["high"]) if "low" in g else (g["value"], g["value"])


def guidance_bridge(prior_rev_id, prior, actuals, current, reported_periods):
    """Rows: metric | prior guidance | actual | new guidance. vs_prior compares the actual to the prior span only when
    units are identical strings."""
    rows = []
    for g in prior:
        act = next(
            (a for a in actuals if _key(a["metric"]) == _key(g["metric"]) and _key(a["period"]) == _key(g["period"])),
            None,
        )
        new = [
            c
            for c in current
            if _key(c["metric"]) == _key(g["metric"]) and _key(c["period"]) not in {_key(p) for p in reported_periods}
        ]
        vs = "n/a"
        if act is not None and act["unit"] == g["unit"]:
            lo, hi = _span(g)
            vs = "above" if act["value"] > hi else "below" if act["value"] < lo else "within"
        elif act is not None:
            vs = "n/a (unit differs)"
        rows.append(
            {
                "metric": g["metric"],
                "period": g["period"],
                "unit": g["unit"],
                "basis": g.get("basis"),
                "prior": {k: g[k] for k in ("low", "high", "value") if k in g},
                "prior_document_id": g["document_id"],
                "actual": (
                    {"value": act["value"], "unit": act["unit"], "source": act.get("document_id") or act.get("source")}
                    if act
                    else None
                ),
                "vs_prior": vs,
                "new": [{k: c[k] for k in ("period", "low", "high", "value", "unit", "basis") if k in c} for c in new],
            }
        )
    return {"predecessor_revision_id": prior_rev_id, "reported_periods": sorted(set(reported_periods)), "rows": rows}
