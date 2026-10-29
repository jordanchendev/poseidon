#!/usr/bin/env python3
"""Offline research revision for a themed scope: document (fundamental) evidence + Thalassa OHLCV evidence.

Same contract path as research_report_2330.py (Poseidon 483c363 ManifestService -> ResearchService) on a local
SQLite ledger. Config JSON supplies scope/evidence; no production host is touched.

price_facts algorithm `price-facts-v2` (v1 fields are unchanged; v2 adds market-reaction fields). Everything is derived
from the frozen ohlcv payloads only, so it replays from the manifest.

Event window semantics:
  - base      = last close with bar time <= event_time ("last close before the event"; this is `close_at_event`).
  - event bar = first bar with time > event_time. T+n = close of the n-th bar with time > event_time, so T+1 is the
                event bar itself and T+5 is four bars later.
  - ret_t1_pct / ret_t5_pct = (T+n close / base - 1) * 100, rounded to 2 dp; null when fewer than n bars follow the event.
  - ret_since_event_pct (v1) = last close vs base.
Relative fields (symbol minus benchmark, percentage points, same windows, computed on unrounded returns):
  - rel_ret_t1_pct, rel_ret_t5_pct, rel_ret_since_event_pct.
  - benchmark = cfg["benchmark"] (default "0050"); it must be one of cfg["symbols"] so its ohlcv is frozen in the same
    manifest. Otherwise (or when either side is null) the relative fields are null. `benchmark` is recorded in the payload.
Narrative extras (not part of Poseidon's contract; validated here before completion, rendered by render_theme.py):
  - catalysts: [{date YYYY-MM-DD, type in CATALYST_TYPES, expected_signal, source}]; next_review_at must then start with
    the earliest catalyst date >= the as_of date.
  - tl_dr: one sentence, treated as inference.
  - event_status: pre_event|post_event (+ optional event_date YYYY-MM-DD), decided by the research runner, never the
    model; pre_event requires a non-empty catalysts list. Kept in research_json (Poseidon keeps extra narrative keys).
Guidance (gap B2): documents[].guidance[] {metric, period, low+high|value, unit, basis GAAP|non-GAAP|IFRS|n/a, fx_assumption?,
  excerpt_locator} is checked number-by-number against the referenced excerpt with check_numbers (scale/sign mismatch
  refuses the freeze, value/absent warns) and stored in the document payload (hashed). Config actuals[] {metric, period,
  value, unit, evidence_id, excerpt_locator} must match their excerpt exactly (any mismatch refuses) and are stored in
  that document's payload; fact claims may carry `actual` {metric, period, value, unit} whose number must be in the claim
  text. The reported period is cfg["reported_period"] or the actual periods; the nearest completed predecessor (walking
  previous_revision_id) whose frozen documents carry guidance for that period feeds identity.guidance_bridge.
KPI series (gap B3): --kpi-json <file> (offline) or cfg["kpi_series"]: true (live TWSE OpenAPI t187ap05_L) adds
  kpi_series:<symbol> (unit NTD thousand; last 12 months visible at as_of by 出表日期; trailing_3m_yoy_pct when the
  last three months are consecutive). Claims citing it are numerically prechecked against its points.
Read-across: cfg["read_across"] is refused (design 2026-10-29 §2.7); a claim names the companies it is about. Old
  revisions keep identity.read_across and render_theme.py still renders it.
Consensus (gap B1): --consensus-json <file> (repeatable; raw_path must hash to raw_sha256) and/or --consensus-live /
  cfg["consensus_live"] (best-effort Yahoo Finance via cfg["adr_ticker"]; any failure is recorded in
  identity.consensus_status and never fails the run) add consensus:<symbol> (kind consensus, tier "third-party
  aggregate"). A snapshot observed after as_of is skipped. Surprise: surprise:<symbol> (algorithm surprise-v1, see
  surprise_rows) from the consensus + document actuals + narrative claim actuals; replay_surprise() recomputes it from
  the manifest. A consensus observed after the actual was published gives status "post-event", never a surprise number.
Variant perception (B5): narrative variant_perception[] validated by validate_variant_perception before completion.
Secondary documents (design 2026-10-29 §6.1): documents[] of kind "secondary" need source_tier sell-side|media|kol and
  declare the secondary capability; claim gates keep fact claims off them.
Narrative snapshot (design 2026-10-29 §2.6): cfg["narrative_snapshot"] (built by assemble.py from the runner ledger) is
  frozen as evidence narrative_snapshot (kind narrative_snapshot, capability narrative_snapshot), after every other item.
  A config without it freezes to the same manifest as before.
Disclosure: cfg["pilot_position"] (long|short|none|undisclosed or {position, size_note}) -> research_json.disclosure.
Supersede: narrative supersedes_revision_id (or supersedes) must be a completed revision in the same ledger.
Claim precheck (claim_precheck.py, run before complete_human_research on every narrative claim):
  - numeric: claim quantities must match the cited evidence's excerpt texts (document kinds with payload.excerpts, and
    kpi_series points rendered as text). Fact claims: any mismatch refuses. Inference claims: only scale/sign mismatches
    refuse; other mismatches are recorded as numbers="mismatch-recorded" with the unmatched items. Inference claims may
    also match A+B of two cited quantities; a derivation precheck recomputed is recorded as `derived` [{claim, how, from}].
    Claims citing price_facts/ohlcv are not numerically checked (their numbers come from derived facts, not excerpts);
    claims citing no excerpt-bearing evidence are not numerically checked either. See the fact/inference split above.
  - atomic: a non-atomic claim refuses completion unless cfg["allow_non_atomic"] is true. For inference claims the
    multi-quantity rule does not refuse (a derivation names its inputs); causal clauses still do (lint_atomic kind).
"""

import argparse
import csv
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_precheck import check_numbers, extract_quantities, lint_atomic
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from poseidon.api.auth import AuthPrincipal
from poseidon.decision_loop.manifest import ManifestService, content_sha256, iso_time
from poseidon.decision_loop.research import ResearchService
from poseidon.models.base import Base


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


LOOKBACK, FIELDS = 400, ("open", "high", "low", "close", "volume")


def load_bars(csv_path):
    frames = {}
    with open(csv_path) as fh:
        for t, sym, o, h, l, c, v in csv.reader(fh):
            frames.setdefault(sym, []).append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
    out = {}
    for sym, rows in frames.items():
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for f in FIELDS:
            df[f] = df[f].astype(float)
        out[sym] = df.set_index("time").sort_index()
    return out


def ohlcv_evidence(symbol, df, as_of, base_url):
    cutoff = pd.Timestamp(as_of)
    df = df[df.index <= cutoff].tail(LOOKBACK)
    bars = [
        {"time": iso_time(ts.to_pydatetime(), "bar.time"), **{f: float(r[f]) for f in FIELDS}}
        for ts, r in df.iterrows()
    ]
    payload = {"symbol": symbol, "market": "tw_stock", "interval": "1d", "bars": bars}
    params = {
        "symbol": symbol,
        "market": "tw_stock",
        "interval": "1d",
        "start": (cutoff - pd.Timedelta(days=LOOKBACK)).isoformat(),
        "end": cutoff.isoformat(),
    }
    return {
        "id": f"ohlcv:{symbol}",
        "kind": "ohlcv",
        "source_uri": f"{base_url}/api/v1/ohlcv?{urlencode(params)}",
        "event_time": bars[-1]["time"],
        "available_at": bars[-1]["time"],
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def _normalize(text):
    """Canonical form for excerpt matching: collapse whitespace (incl. form feeds) and drop bare table pipes."""
    return " ".join(tok for tok in text.replace("\f", " ").split() if tok != "|")


def verify_excerpts(item):
    """Every excerpt must be a verbatim substring of the normalized raw text file it claims to come from.

    This is the host-side check that the 2026-10-20 dry run lacked: a bad line cut (pdftotext form feeds
    vs str.splitlines) produced excerpts that hashed fine but were not in the source.
    """
    raw_text_path = item.get("raw_text_path")
    if not raw_text_path:
        raise SystemExit("document %s has no raw_text_path to verify excerpts against" % item["id"])
    raw = _normalize(Path(raw_text_path).read_text(encoding="utf-8", errors="replace"))
    for ex in item["excerpts"]:
        text = _normalize(ex["text"])
        if not text or text not in raw:
            raise SystemExit(
                "excerpt %s of %s is not a substring of %s; refusing to freeze"
                % (ex.get("locator"), item["id"], raw_text_path)
            )
    digest = hashlib.sha256(Path(raw_text_path).read_bytes()).hexdigest()
    declared = item.get("raw_files", {})
    if digest not in declared.values():
        raise SystemExit("raw_text_path sha256 %s is not declared in raw_files of %s" % (digest[:12], item["id"]))


def document_evidence(item, as_of):
    verify_excerpts(item)
    payload = {k: item[k] for k in ("title", "publisher", "raw_files", "excerpts")}
    # Optional keys are added only when present, so configs without them freeze to the same payload hash as before.
    if item.get("guidance"):
        validate_guidance(item)
        payload["guidance"] = item["guidance"]
    if item.get("actuals"):
        validate_actuals(item)
        payload["actuals"] = item["actuals"]
    if item.get("source_tier"):
        if item["source_tier"] not in SOURCE_TIERS:
            raise SystemExit("document %s source_tier must be one of %s" % (item["id"], "|".join(SOURCE_TIERS)))
        payload["source_tier"] = item["source_tier"]
    if item["kind"] == "secondary" and payload.get("source_tier") not in SECONDARY_TIERS:
        raise SystemExit("secondary document %s needs source_tier %s" % (item["id"], "|".join(SECONDARY_TIERS)))
    if item["kind"] not in ("fundamental", "secondary"):
        raise SystemExit("document %s kind must be fundamental or secondary" % item["id"])
    return {
        "id": item["id"],
        "kind": item["kind"],
        "source_uri": item["source_uri"],
        "event_time": item["event_time"],
        "available_at": item["available_at"],
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


PRICE_FACTS_ALGO = "price-facts-v2"  # v1 fields + T+1/T+5 and benchmark-relative event returns (see module docstring)
CATALYST_TYPES = ("monthly_revenue", "earnings_call", "agm", "investor_day", "product_event", "regulatory", "other")
EVENT_STATUSES = ("pre_event", "post_event")  # narrative.event_status, set by the runner (never by the model)


def _frame(ohlcv_item):
    df = pd.DataFrame(ohlcv_item["payload"]["bars"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def _event_returns(c, since):
    """Fractional returns vs the last close <= since: {t1, t5, since}; None where there are not enough bars."""
    ts = pd.Timestamp(since)
    base = c[c.index <= ts].iloc[-1]
    after = c[c.index > ts]
    f = lambda n: float(after.iloc[n - 1] / base - 1) if len(after) >= n else None
    return {"t1": f(1), "t5": f(5), "since": float(c.iloc[-1] / base - 1)}


def price_facts_evidence(ohlcv_item, event_time, as_of, bench_item=None):
    """Derived facts computed ONLY from the frozen ohlcv evidence payload, so a reader can replay them from the manifest."""
    df = _frame(ohlcv_item)
    facts = stats(df, event_time, _frame(bench_item) if bench_item else None)
    payload = {
        "symbol": ohlcv_item["payload"]["symbol"],
        "algorithm": PRICE_FACTS_ALGO,
        "source_evidence_id": ohlcv_item["id"],
        "source_content_sha256": ohlcv_item["content_sha256"],
        "event_time": event_time,
        "facts": facts,
        "benchmark": bench_item["payload"]["symbol"] if bench_item else None,
        "benchmark_evidence_id": bench_item["id"] if bench_item else None,
        "benchmark_content_sha256": bench_item["content_sha256"] if bench_item else None,
    }
    return {
        "id": "price_facts:%s" % ohlcv_item["payload"]["symbol"],
        "kind": "price_facts",
        "source_uri": "derived://%s/%s" % (PRICE_FACTS_ALGO, ohlcv_item["id"]),
        "event_time": ohlcv_item["event_time"],
        "available_at": ohlcv_item["available_at"],
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def stats(df, since, bench=None):
    c = df["close"]
    er = _event_returns(c, since)
    br = _event_returns(bench["close"], since) if bench is not None else None
    pct2 = lambda x: None if x is None else round(x * 100, 2)
    rel = lambda k: pct2(er[k] - br[k]) if br is not None and er[k] is not None and br[k] is not None else None
    sma5, sma20 = c.rolling(5).mean(), c.rolling(20).mean()
    diff = sma5 - sma20
    sign = diff.apply(lambda x: 1 if x > 0 else -1)
    last_cross = sign[sign != sign.shift(1)].index[-1]
    hi60 = c.tail(60).max()
    base = c[c.index <= pd.Timestamp(since)].iloc[-1]
    return {
        "last_bar": c.index[-1].isoformat(),
        "close": float(c.iloc[-1]),
        "sma5": round(float(sma5.iloc[-1]), 2),
        "sma20": round(float(sma20.iloc[-1]), 2),
        "sma_state": "fast_above_slow" if diff.iloc[-1] > 0 else "fast_below_slow",
        "last_cross": {
            "date": last_cross.isoformat(),
            "direction": "up" if sign.loc[last_cross] > 0 else "down",
            "close_then": float(c.loc[last_cross]),
        },
        "ret_5d_pct": round(float(c.iloc[-1] / c.iloc[-6] - 1) * 100, 2),
        "ret_20d_pct": round(float(c.iloc[-1] / c.iloc[-21] - 1) * 100, 2),
        "ret_since_event_pct": round(float(c.iloc[-1] / base - 1) * 100, 2),
        "close_at_event": float(base),
        "ret_t1_pct": pct2(er["t1"]),
        "ret_t5_pct": pct2(er["t5"]),
        "rel_ret_t1_pct": rel("t1"),
        "rel_ret_t5_pct": rel("t5"),
        "rel_ret_since_event_pct": rel("since"),
        "high_60d": float(hi60),
        "high_60d_date": c.tail(60).idxmax().isoformat(),
        "drawdown_from_60d_high_pct": round(float(c.iloc[-1] / hi60 - 1) * 100, 2),
        "vol_last": float(df["volume"].iloc[-1]),
        "vol_avg20": round(float(df["volume"].tail(20).mean()), 0),
        "range_avg20_pct": round(float(((df["high"] - df["low"]) / df["close"]).tail(20).mean()) * 100, 2),
    }


def validate_narrative_extras(narrative, as_of):
    """Offline shape checks for optional narrative fields; raises SystemExit with a clear message."""
    if "tl_dr" in narrative and not (isinstance(narrative["tl_dr"], str) and narrative["tl_dr"].strip()):
        raise SystemExit("tl_dr must be a non-empty string")
    status = narrative.get("event_status")
    if status is not None and status not in EVENT_STATUSES:
        raise SystemExit("event_status must be one of %s, got %r" % ("|".join(EVENT_STATUSES), status))
    if narrative.get("event_date") is not None:
        try:
            datetime.date.fromisoformat(str(narrative["event_date"]))
        except ValueError:
            raise SystemExit("event_date must be an ISO date YYYY-MM-DD, got %r" % narrative["event_date"])
    cats = narrative.get("catalysts")
    if status == "pre_event" and not (isinstance(cats, list) and cats):
        raise SystemExit("event_status pre_event requires a non-empty catalysts list (include the event date)")
    if cats is None:
        return
    if not isinstance(cats, list) or not cats:
        raise SystemExit("catalysts must be a non-empty list when present")
    for n, c in enumerate(cats):
        if not isinstance(c, dict):
            raise SystemExit("catalysts[%d] must be an object" % n)
        try:
            datetime.date.fromisoformat(c.get("date", ""))
        except (TypeError, ValueError):
            raise SystemExit("catalysts[%d].date must be an ISO date YYYY-MM-DD, got %r" % (n, c.get("date")))
        if c.get("type") not in CATALYST_TYPES:
            raise SystemExit(
                "catalysts[%d].type must be one of %s, got %r" % (n, "|".join(CATALYST_TYPES), c.get("type"))
            )
        for k in ("expected_signal", "source"):
            if not isinstance(c.get(k), str) or not c[k].strip():
                raise SystemExit("catalysts[%d].%s must be a non-empty string" % (n, k))
    today = as_of[:10]
    future = sorted(c["date"] for c in cats if c["date"] >= today)
    if not future:
        print("note: no catalyst on/after as_of date %s; next_review_at not cross-checked" % today, file=sys.stderr)
        return
    if not str(narrative.get("next_review_at", "")).startswith(future[0]):
        raise SystemExit(
            "next_review_at %r must start with the earliest future catalyst date %s"
            % (narrative.get("next_review_at"), future[0])
        )


_BASIS_METRIC_RE = re.compile(
    r"EPS|每股盈餘|earnings per share|gross margin|毛利率|net income|淨利|operating (?:margin|income)|營業利益|diluted",
    re.I,
)
_BASIS_WORD_RE = re.compile(r"non[- ]?GAAP|非\s*GAAP|GAAP", re.I)


def check_basis(claim, excerpts):
    """GAAP/non-GAAP basis lint (gold-set label_policy v1 'basis'): a claim about EPS/margin/net income that cites excerpts
    carrying both GAAP and non-GAAP figures must name its basis. Returns 'ok', 'missing' or 'n/a'."""
    if not _BASIS_METRIC_RE.search(claim):
        return "n/a"
    blob = "\n".join(excerpts)
    both = re.search(r"non[- ]?GAAP", blob, re.I) and re.search(r"(?<![nN][oO][nN][- ])GAAP", blob)
    if not both:
        return "n/a"
    return "ok" if _BASIS_WORD_RE.search(claim) else "missing"


# ------------------------------------------------------------------------------------------------ guidance (gap B2)
SOURCE_TIERS = ("primary filing", "company release", "exchange feed", "derived", "sell-side", "media", "kol")
SECONDARY_TIERS = ("sell-side", "media", "kol")
SNAPSHOT_KEYS = ("id", "kind", "source_uri", "event_time", "available_at", "payload")


def snapshot_evidence(snap, as_of):
    """cfg narrative_snapshot -> frozen evidence item. The selection and fold are assemble.py's; this checks the shape."""
    if sorted(snap) != sorted(SNAPSHOT_KEYS) or snap["id"] != "narrative_snapshot" or snap["kind"] != snap["id"]:
        raise SystemExit(
            "narrative_snapshot must be {%s} with id and kind narrative_snapshot" % ", ".join(SNAPSHOT_KEYS)
        )
    p = snap["payload"]
    if sorted(p) != ["algorithm", "not_shown", "records", "themes"] or not p["records"]:
        raise SystemExit("narrative_snapshot payload needs algorithm, themes, records (non-empty) and not_shown")
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
        raise SystemExit("%s cites excerpt_locator %r which is not an excerpt of %s" % (what, locator, item["id"]))
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
            raise SystemExit("refusing to freeze: " + msg)
        print("warning: " + msg, file=sys.stderr)
        status = "warn"
    return status


def validate_guidance(item):
    """documents[].guidance[]: {metric, period, low+high | value, unit, basis, fx_assumption?, excerpt_locator}.
    Every number must appear in the referenced excerpt (claim_precheck.check_numbers): refuse on scale/sign mismatch,
    warn otherwise. Writes no state; the guidance list is hashed as part of the document payload."""
    if not isinstance(item["guidance"], list):
        raise SystemExit("document %s guidance must be a list" % item["id"])
    for n, g in enumerate(item["guidance"]):
        what = "%s guidance[%d]" % (item["id"], n)
        for k in ("metric", "period", "unit", "excerpt_locator"):
            if not isinstance(g.get(k), str) or not g[k].strip():
                raise SystemExit("%s.%s must be a non-empty string" % (what, k))
        if g.get("basis") not in GUIDANCE_BASES:
            raise SystemExit("%s.basis must be one of %s, got %r" % (what, "|".join(GUIDANCE_BASES), g.get("basis")))
        has_range, has_value = "low" in g or "high" in g, "value" in g
        if has_range == has_value or (has_range and not ("low" in g and "high" in g)):
            raise SystemExit("%s needs either low and high, or value" % what)
        nums = [g[k] for k in ("low", "high", "value") if k in g]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in nums):
            raise SystemExit("%s low/high/value must be numbers" % what)
        if has_range and g["low"] > g["high"]:
            raise SystemExit("%s low > high" % what)
        excerpt = _excerpt_by_locator(item, g["excerpt_locator"], what)
        texts = [quantity_text(x, g["unit"]) for x in nums]
        fx = g.get("fx_assumption")
        if fx is not None:
            if not isinstance(fx, str) or not fx.strip():
                raise SystemExit("%s.fx_assumption must be a non-empty string when present" % what)
            texts.append(fx)
        _numbers_in_excerpt(texts, excerpt, what, strict=False)


def validate_actuals(item):
    """actuals attached to a document (from config actuals[]): {metric, period, value, unit, excerpt_locator}.
    Actuals are facts, so ANY numeric mismatch with the cited excerpt refuses the freeze."""
    for n, a in enumerate(item["actuals"]):
        what = "actuals for %s [%d]" % (item["id"], n)
        for k in ("metric", "period", "unit", "excerpt_locator"):
            if not isinstance(a.get(k), str) or not a[k].strip():
                raise SystemExit("%s.%s must be a non-empty string" % (what, k))
        if isinstance(a.get("value"), bool) or not isinstance(a.get("value"), (int, float)):
            raise SystemExit("%s.value must be a number" % what)
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
            raise SystemExit(
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
            raise SystemExit("%s is only allowed on fact claims" % what)
        if (
            not isinstance(a, dict)
            or any(not isinstance(a.get(k), str) or not a[k].strip() for k in ("metric", "period", "unit"))
            or isinstance(a.get("value"), bool)
            or not isinstance(a.get("value"), (int, float))
        ):
            raise SystemExit("%s must be {metric, period, value (number), unit}" % what)
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


# ------------------------------------------------------------------------------------------------ KPI series (gap B3)
TWSE_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
KPI_POINTS = 12


def _roc_ym(s):
    s = str(s or "").strip()
    return "%04d-%s" % (int(s[:-2]) + 1911, s[-2:]) if re.fullmatch(r"\d{4,5}", s) and 1 <= int(s[-2:]) <= 12 else None


def _roc_date_end_utc(s):
    """ROC 'yyyMMdd' -> end of that Asia/Taipei day in UTC (conservative availability), or None."""
    s = str(s or "").strip()
    if not re.fullmatch(r"\d{6,7}", s):
        return None
    try:
        d = datetime.date(int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:]))
    except ValueError:
        return None
    return datetime.datetime.combine(d, datetime.time(15, 59, 59), tzinfo=datetime.UTC)


def _f(v):
    try:
        x = float(str(v).replace(",", "").strip())
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def load_kpi_rows(path=None):
    """Return (rows, source_uri, fetched_at). File form: a list of raw t187ap05_L rows (fetched_at unknown -> None) or
    {"rows": [...], "source_uri": ..., "fetched_at": ...} (e.g. accumulated monthly snapshots). No path -> live fetch."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data, TWSE_REVENUE_URL, None
        return data["rows"], data.get("source_uri", TWSE_REVENUE_URL), data.get("fetched_at")
    import urllib.request

    req = urllib.request.Request(
        TWSE_REVENUE_URL, headers={"User-Agent": "aquarium-research", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.loads(r.read().decode("utf-8-sig"))
    return rows, TWSE_REVENUE_URL, datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prev_month(ym):
    y, m = int(ym[:4]), int(ym[5:])
    return "%04d-%02d" % ((y, m - 1) if m > 1 else (y - 1, 12))


def kpi_series_evidence(symbol, rows, source_uri, fetched_at, as_of):
    """kpi_series:<symbol> from TWSE t187ap05_L rows (unit: NTD thousand). Point-in-time: a row counts only when its
    出表日期 (end of that Taipei day) <= as_of; per 資料年月 the latest such row wins. Returns None if no row."""
    cutoff = pd.Timestamp(as_of).to_pydatetime()
    best = {}
    for raw in rows:
        r = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
        if str(r.get("公司代號", "")).strip() != symbol:
            continue
        ym, pub = _roc_ym(r.get("資料年月")), _roc_date_end_utc(r.get("出表日期"))
        if not ym or not pub or pub > cutoff or _f(r.get("營業收入-當月營收")) is None:
            continue
        if ym not in best or pub > best[ym][0]:
            best[ym] = (pub, r)
    if not best:
        return None
    used = [best[k] for k in sorted(best)][-KPI_POINTS:]
    r2 = lambda x: None if x is None else round(x, 2)
    points = [
        {
            "period": _roc_ym(r["資料年月"]),
            "value": _f(r["營業收入-當月營收"]),
            "yoy_pct": r2(_f(r.get("營業收入-去年同月增減(%)"))),
            "mom_pct": r2(_f(r.get("營業收入-上月比較增減(%)"))),
            "prior_year_value": _f(r.get("營業收入-去年當月營收")),
        }
        for _, r in used
    ]
    last3 = points[-3:]
    consecutive = (
        len(last3) == 3
        and _prev_month(last3[2]["period"]) == last3[1]["period"]
        and _prev_month(last3[1]["period"]) == last3[0]["period"]
    )
    t3 = None
    if consecutive and all(p["prior_year_value"] for p in last3):
        t3 = round((sum(p["value"] for p in last3) / sum(p["prior_year_value"] for p in last3) - 1) * 100, 2)
    raw_used = [r for _, r in used]
    payload = {
        "symbol": symbol,
        "metric": "monthly_revenue",
        "unit": "NTD thousand",
        "points": points,
        "trailing_3m_yoy_pct": t3,
        "source_uri": source_uri,
        "fetched_at": fetched_at,
        "content_sha256": content_sha256(raw_used),
    }
    avail = iso_time(max(p for p, _ in used), "kpi.available_at")
    return {
        "id": "kpi_series:%s" % symbol,
        "kind": "kpi_series",
        "source_uri": source_uri,
        "event_time": avail,
        "available_at": avail,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def kpi_text(payload):
    """kpi_series payload as plain sentences for claim_precheck.check_numbers."""
    out = []
    for p in payload["points"]:
        s = "%s monthly revenue NT$%s thousand" % (p["period"], _num_text(p["value"]))
        if p.get("yoy_pct") is not None:
            s += ", %s %s%% year-over-year" % ("up" if p["yoy_pct"] >= 0 else "down", _num_text(abs(p["yoy_pct"])))
        if p.get("mom_pct") is not None:
            s += "; %s %s%% month-over-month" % ("up" if p["mom_pct"] >= 0 else "down", _num_text(abs(p["mom_pct"])))
        out.append(s + ".")
    t3 = payload.get("trailing_3m_yoy_pct")
    if t3 is not None:
        out.append(
            "Trailing three-month revenue %s %s%% year-over-year." % ("up" if t3 >= 0 else "down", _num_text(abs(t3)))
        )
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ consensus + surprise (gap B1)
CONSENSUS_TIER = "third-party aggregate"
SURPRISE_ALGO = "surprise-v1"
_SCALE = {"": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_CUR_CODE = {"usd": "USD", "us$": "USD", "ntd": "TWD", "twd": "TWD", "nt$": "TWD", "jpy": "JPY"}
_PERIOD_OK = re.compile(r"^(?:[1-4]Q\d{2}|[12]H\d{2}|FY\d{2})$")


def norm_period(p):
    """'3Q26' | 'Q3 2026' | '2026Q3' | '3Q2026' | "Q3'26" -> '3Q26'; 'FY2026' | 'FY26' -> 'FY26'; '1H26' kept.
    Anything else -> upper-cased, whitespace removed (so 'Q4' still matches 'Q4')."""
    s = re.sub(r"[\s'’]", "", str(p or "")).upper()
    for pat, f in (
        (r"([1-4])Q(\d{2}|\d{4})", lambda q, y: "%sQ%s" % (q, y[-2:])),
        (r"Q([1-4])(\d{2}|\d{4})", lambda q, y: "%sQ%s" % (q, y[-2:])),
        (r"(\d{4})Q([1-4])", lambda y, q: "%sQ%s" % (q, y[-2:])),
        (r"([12])H(\d{2}|\d{4})", lambda h, y: "%sH%s" % (h, y[-2:])),
        (r"FY(\d{2}|\d{4})", lambda y: "FY%s" % y[-2:]),
    ):
        m = re.fullmatch(pat, s)
        if m:
            return f(*m.groups())
    return s


def metric_key(m):
    return " ".join(str(m).replace("_", " ").lower().split())


def unit_base(unit):
    """(class, multiplier): money units -> ('money:TWD', 1e9) etc; '%' -> ('%', 1); anything else is its own count class."""
    u = (unit or "").strip()
    m = _UNIT_RE.match(u)
    if m:
        return "money:" + _CUR_CODE[m.group(1).lower()], _SCALE[(m.group(2) or "").lower()]
    if u in ("%", "percent"):
        return "%", 1.0
    if u == "pp":
        return "pp", 1.0
    return "count:" + u.lower(), 1.0


def convert(value, from_unit, to_unit):
    """value in from_unit expressed in to_unit, or None when the unit classes differ (no FX conversion, ever)."""
    (fc, fm), (tc, tm) = unit_base(from_unit), unit_base(to_unit)
    return None if fc != tc else float(value) * fm / tm


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_consensus_items(items, what):
    if not isinstance(items, list) or not items:
        raise SystemExit("%s.items must be a non-empty list" % what)
    out = []
    for n, it in enumerate(items):
        w = "%s.items[%d]" % (what, n)
        if not isinstance(it, dict):
            raise SystemExit("%s must be an object" % w)
        for k in ("metric", "period", "unit", "basis"):
            if not isinstance(it.get(k), str) or not it[k].strip():
                raise SystemExit("%s.%s must be a non-empty string" % (w, k))
        if it["basis"] not in GUIDANCE_BASES:
            raise SystemExit("%s.basis must be one of %s, got %r" % (w, "|".join(GUIDANCE_BASES), it["basis"]))
        period = norm_period(it["period"])
        if not _PERIOD_OK.match(period):
            raise SystemExit("%s.period %r does not normalise to 3Q26 / 1H26 / FY26" % (w, it["period"]))
        for k in ("mean", "low", "high"):
            if not _is_num(it.get(k)):
                raise SystemExit("%s.%s must be a number" % (w, k))
        if "median" in it and it["median"] is not None and not _is_num(it["median"]):
            raise SystemExit("%s.median must be a number when present" % w)
        if not it["low"] <= it["mean"] <= it["high"]:
            raise SystemExit("%s needs low <= mean <= high" % w)
        if isinstance(it.get("n_analysts"), bool) or not isinstance(it.get("n_analysts"), int) or it["n_analysts"] < 1:
            raise SystemExit("%s.n_analysts must be an integer >= 1" % w)
        cls, _ = unit_base(it["unit"])
        cur = it.get("currency")
        if cls.startswith("money:") and cur != cls[6:]:
            raise SystemExit("%s.currency %r does not match unit %r" % (w, cur, it["unit"]))
        if not cls.startswith("money:") and cur not in (None, ""):
            raise SystemExit("%s.currency must be null for a non-money unit %r" % (w, it["unit"]))
        out.append(dict(it, period=period, currency=cur or None))
    return out


def consensus_evidence(data, as_of, source="consensus-json"):
    """consensus:<symbol> from the manual/offline JSON (see README 'Consensus JSON') or a live fetch result.
    The captured raw file must hash to raw_sha256 (like documents' raw_files). Returns (item, note); item None + note when
    the snapshot was observed after as_of (not point-in-time)."""
    what = "%s %s" % (source, data.get("symbol"))
    for k in ("symbol", "provider", "as_of", "source_uri", "raw_sha256", "fetched_at"):
        if not isinstance(data.get(k), str) or not data[k].strip():
            raise SystemExit("%s: %s must be a non-empty string" % (what, k))
    raw = data.get("raw_path")
    if not raw or not Path(raw).is_file():
        raise SystemExit("%s: raw_path must point at the captured raw file (got %r)" % (what, raw))
    digest = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    if digest != data["raw_sha256"]:
        raise SystemExit(
            "%s: raw_path sha256 %s != declared raw_sha256 %s; refusing to freeze"
            % (what, digest[:12], data["raw_sha256"][:12])
        )
    observed = iso_time(data["as_of"], "consensus.as_of")
    if pd.Timestamp(observed) > pd.Timestamp(as_of):
        return None, "consensus for %s observed %s, after as_of %s: not point-in-time, skipped" % (
            data["symbol"],
            observed,
            as_of,
        )
    items = validate_consensus_items(data.get("items"), what)
    payload = {
        "symbol": data["symbol"],
        "provider": data["provider"],
        "as_of": observed,
        "items": items,
        "source_uri": data["source_uri"],
        "raw_files": {Path(raw).name: digest},
        "raw_sha256": digest,
        "fetched_at": data["fetched_at"],
        "source_tier": CONSENSUS_TIER,
        **({"instrument": data["instrument"]} if data.get("instrument") else {}),
        **({"note": data["note"]} if data.get("note") else {}),
    }
    return {
        "id": "consensus:%s" % data["symbol"],
        "kind": "consensus",
        "source_uri": data["source_uri"],
        "event_time": observed,
        "available_at": observed,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }, None


# -- best-effort live fetch (Yahoo Finance; US tickers / ADRs). Never raises; returns (data|None, status dict).
YAHOO_ANALYSIS_URL = "https://finance.yahoo.com/quote/{t}/analysis/"
YAHOO_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{t}?modules=earningsTrend"
_UA = "aquarium-research/1.0 (offline research tooling)"


def _http_get(url, opener, timeout=20):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/html,application/json"})
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # network down, TLS, timeout
        return None, repr(e).encode()


def _period_from_end(end, kind):
    d = datetime.date.fromisoformat(str(end)[:10])
    return ("%dQ%02d" % ((d.month - 1) // 3 + 1, d.year % 100)) if kind == "q" else "FY%02d" % (d.year % 100)


def yahoo_items(summary_result, adr):
    """quoteSummary result -> consensus items. EPS from an ADR quote is per ADR (metric eps_per_adr) so it never matches a
    per-share actual by accident. Periods: calendar quarter/year of trend endDate (assumes calendar fiscal year)."""
    raw = lambda v: v.get("raw") if isinstance(v, dict) else v
    items = []
    for t in (summary_result.get("earningsTrend") or {}).get("trend", []):
        per = str(t.get("period", ""))
        if not re.fullmatch(r"[0+]\d?[qy]", per) or not t.get("endDate"):
            continue
        period = _period_from_end(t["endDate"], per[-1])
        for key, metric in (("earningsEstimate", "eps_per_adr" if adr else "eps"), ("revenueEstimate", "revenue")):
            e = {k: raw(v) for k, v in (t.get(key) or {}).items()}
            if (
                not all(_is_num(e.get(k)) for k in ("avg", "low", "high"))
                or not isinstance(e.get("numberOfAnalysts"), int)
                or e["numberOfAnalysts"] < 1
            ):
                continue
            cur = e.get("earningsCurrency") or e.get("revenueCurrency")
            cur = {"NTD": "TWD"}.get(cur, cur)
            if cur not in ("USD", "TWD", "JPY"):
                continue
            items.append(
                {
                    "metric": metric,
                    "period": period,
                    "mean": e["avg"],
                    "low": e["low"],
                    "high": e["high"],
                    "n_analysts": e["numberOfAnalysts"],
                    "unit": cur,
                    "currency": cur,
                    "basis": "n/a",
                    "provider_period": per,
                }
            )
    return items


def fetch_yahoo_consensus(symbol, ticker, raw_dir, adr=True, opener=None):
    """Probe 1: the public analysis page (server-rendered quoteSummary JSON); probe 2: the quoteSummary API with a crumb.
    Cloudflare challenge pages are refused. Every attempt is recorded {url, status, sha256}; crumbs are stripped."""
    import html as _html
    import http.cookiejar
    import urllib.request

    opener = opener or urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    attempts, now = [], datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def attempt(url):
        st, body = _http_get(url, opener)
        attempts.append({"url": url, "status": st, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        return st, body

    def done(result, body, url, reason=None):
        if result is None:
            return None, {
                "provider": "yahoo",
                "ticker": ticker,
                "status": "unavailable",
                "reason": reason,
                "attempts": attempts,
                "at": now,
            }
        items = yahoo_items(result, adr)
        if not items:
            return None, {
                "provider": "yahoo",
                "ticker": ticker,
                "status": "unavailable",
                "reason": "earningsTrend present but no usable estimates",
                "attempts": attempts,
                "at": now,
            }
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        raw = Path(raw_dir) / (
            "yahoo-%s-%s.%s" % (ticker, now.replace(":", ""), "html" if url.endswith("/analysis/") else "json")
        )
        raw.write_bytes(body)
        data = {
            "symbol": symbol,
            "provider": "yahoo-finance",
            "instrument": "%s (%s)" % (ticker, "ADR" if adr else "listing"),
            "as_of": now,
            "source_uri": url,
            "raw_path": str(raw),
            "raw_sha256": hashlib.sha256(body).hexdigest(),
            "fetched_at": now,
            "items": items,
            "note": "Yahoo earningsTrend; EPS per %s; periods = calendar quarter/year of endDate"
            % ("ADR" if adr else "share"),
        }
        return data, {
            "provider": "yahoo",
            "ticker": ticker,
            "status": "ok",
            "items": len(items),
            "attempts": attempts,
            "at": now,
        }

    url = YAHOO_ANALYSIS_URL.format(t=ticker)
    st, body = attempt(url)
    text = body.decode("utf-8", "replace")
    if st == 200 and ("cf-chl" in text or "Just a moment..." in text):
        return done(None, None, url, "Cloudflare challenge page; refused")
    if st == 200:
        for m in re.finditer(
            r'<script type="application/json" data-sveltekit-fetched data-url="([^"]+)"[^>]*>(.*?)</script>', text, re.S
        ):
            if "quoteSummary" not in m.group(1) or "earningsTrend" not in _html.unescape(m.group(1)):
                continue
            try:
                res = json.loads(json.loads(m.group(2))["body"])["quoteSummary"]["result"][0]
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            return done(res, body, url)
    # probe 2: API (needs the A3 cookie from fc.yahoo.com and a crumb)
    attempt("https://fc.yahoo.com/")
    cst, crumb = _http_get("https://query2.finance.yahoo.com/v1/test/getcrumb", opener)
    attempts.append(
        {
            "url": "https://query2.finance.yahoo.com/v1/test/getcrumb",
            "status": cst,
            "bytes": len(crumb),
            "sha256": "(not recorded: session crumb)",
        }
    )
    if cst != 200 or not crumb.strip():
        return done(
            None,
            None,
            url,
            "analysis page had no server-rendered earningsTrend (HTTP %s); getcrumb HTTP %s" % (st, cst),
        )
    from urllib.parse import quote

    api = YAHOO_SUMMARY_URL.format(t=ticker)
    st2, body2 = _http_get(api + "&crumb=" + quote(crumb.decode().strip()), opener)
    attempts.append({"url": api, "status": st2, "bytes": len(body2), "sha256": hashlib.sha256(body2).hexdigest()})
    try:
        res = json.loads(body2)["quoteSummary"]["result"][0] if st2 == 200 else None
    except (ValueError, KeyError, IndexError, TypeError):
        res = None
    return done(res, body2, api, None if res else "quoteSummary API HTTP %s" % st2)


def adr_ticker_for(cfg, symbol):
    t = cfg.get("adr_ticker")
    if isinstance(t, dict):
        return t.get(symbol)
    if isinstance(t, str) and symbol == cfg.get("symbol"):
        return t
    return None if symbol.isdigit() else symbol


def consensus_text(payload):
    """consensus payload as sentences for claim_precheck (claims citing consensus:<symbol>)."""
    return "\n".join(
        "%s %s consensus mean %s, low %s, high %s, %d analysts."
        % (
            it["period"],
            it["metric"].replace("_", " "),
            quantity_text(it["mean"], it["unit"]),
            quantity_text(it["low"], it["unit"]),
            quantity_text(it["high"], it["unit"]),
            it["n_analysts"],
        )
        for it in payload["items"]
    )


def _actual_inputs(doc_items, narrative_acts, evidence_by_id):
    """Actuals usable for surprise: document actuals (payload, hashed) + narrative claim actuals, each with the time it
    became public (the cited document's available_at)."""
    out = []
    for e in doc_items:
        for a in e["payload"].get("actuals") or []:
            out.append(
                {
                    "metric": a["metric"],
                    "period": a["period"],
                    "value": a["value"],
                    "unit": a["unit"],
                    "source": e["id"],
                    "source_content_sha256": e["content_sha256"],
                    "published_at": e["available_at"],
                }
            )
    for a in narrative_acts:
        cited = [
            evidence_by_id[i]
            for i in a.get("evidence_ids", [])
            if i in evidence_by_id and evidence_by_id[i]["kind"] not in ("consensus", "surprise")
        ]
        out.append(
            {
                "metric": a["metric"],
                "period": a["period"],
                "value": a["value"],
                "unit": a["unit"],
                "source": a["source"],
                "cited": [e["id"] for e in cited],
                "published_at": max((e["available_at"] for e in cited), default=None),
            }
        )
    return out


def surprise_rows(consensus_payload, actuals):
    """surprise-v1: for every consensus item with a matching actual (metric_key + norm_period):
    surprise = actual - consensus mean in the consensus unit (actual converted by unit scale only, never FX),
    surprise_pct = surprise / |mean| * 100 (2 dp), vs_range = below | within | above the [low, high] analyst range.
    Only a consensus observed at or before the actual was published is a baseline; otherwise status 'post-event'."""
    rows = []
    for it in consensus_payload["items"]:
        act = next(
            (
                a
                for a in actuals
                if metric_key(a["metric"]) == metric_key(it["metric"]) and norm_period(a["period"]) == it["period"]
            ),
            None,
        )
        if act is None:
            continue
        row = {
            "metric": it["metric"],
            "period": it["period"],
            "unit": it["unit"],
            "basis": it["basis"],
            "consensus": {k: it[k] for k in ("mean", "median", "low", "high", "n_analysts") if it.get(k) is not None},
            "actual": {k: act[k] for k in ("value", "unit", "source", "published_at") if k in act},
            "surprise_abs": None,
            "surprise_pct": None,
            "vs_range": None,
        }
        val = convert(act["value"], act["unit"], it["unit"])
        if val is None:
            row["status"] = "n/a (unit differs: actual %s vs consensus %s)" % (act["unit"], it["unit"])
        elif not act.get("published_at"):
            row["status"] = "n/a (actual publication time unknown)"
        elif pd.Timestamp(consensus_payload["as_of"]) > pd.Timestamp(act["published_at"]):
            row["status"] = "post-event (consensus observed %s, after the actual was published %s)" % (
                consensus_payload["as_of"],
                act["published_at"],
            )
        else:
            s = val - it["mean"]
            row.update(
                status="ok",
                surprise_abs=round(s, 6),
                surprise_pct=(round(s / abs(it["mean"]) * 100, 2) if it["mean"] else None),
                vs_range="above" if val > it["high"] else "below" if val < it["low"] else "within",
                actual_in_consensus_unit=round(val, 6),
            )
        rows.append(row)
    return rows


def surprise_evidence(consensus_item, actuals, as_of):
    """Derived surprise:<symbol>; inputs are listed in the payload so it replays from the manifest (replay_surprise)."""
    rows = surprise_rows(consensus_item["payload"], actuals)
    if not rows:
        return None
    used = [
        a
        for a in actuals
        if any(
            metric_key(a["metric"]) == metric_key(r["metric"]) and norm_period(a["period"]) == r["period"] for r in rows
        )
    ]
    sym = consensus_item["payload"]["symbol"]
    payload = {
        "symbol": sym,
        "algorithm": SURPRISE_ALGO,
        "consensus_evidence_id": consensus_item["id"],
        "consensus_content_sha256": consensus_item["content_sha256"],
        "actual_inputs": used,
        "rows": rows,
    }
    times = [consensus_item["available_at"]] + [a["published_at"] for a in used if a.get("published_at")]
    t = max(times, key=pd.Timestamp)
    return {
        "id": "surprise:%s" % sym,
        "kind": "surprise",
        "source_uri": "derived://%s/%s" % (SURPRISE_ALGO, consensus_item["id"]),
        "event_time": t,
        "available_at": t,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def replay_surprise(manifest_payload):
    """Recompute every surprise:<symbol> from the manifest alone; returns {id: bool}. Document-sourced inputs must equal
    the frozen document actuals; claim-sourced inputs are taken from the payload (they also live in research_json claims)."""
    ev = {e["id"]: e for e in manifest_payload.get("evidence", [])}
    out = {}
    for e in ev.values():
        if e["kind"] != "surprise":
            continue
        p = e["payload"]
        c = ev[p["consensus_evidence_id"]]
        ok = c["content_sha256"] == p["consensus_content_sha256"] == content_sha256(c["payload"])
        for a in p["actual_inputs"]:
            if a["source"] in ev:
                d = ev[a["source"]]
                ok = ok and any(
                    all(x.get(k) == a[k] for k in ("metric", "period", "value", "unit"))
                    for x in d["payload"].get("actuals", [])
                )
        out[e["id"]] = ok and surprise_rows(c["payload"], p["actual_inputs"]) == p["rows"]
    return out


def surprise_text(payload):
    return "\n".join(
        "%s %s actual %s versus consensus mean %s, surprise %s (%s%%)."
        % (
            r["period"],
            r["metric"].replace("_", " "),
            quantity_text(r["actual_in_consensus_unit"], r["unit"]),
            quantity_text(r["consensus"]["mean"], r["unit"]),
            quantity_text(r["surprise_abs"], r["unit"]),
            _num_text(r["surprise_pct"]) if r["surprise_pct"] is not None else "n/a",
        )
        for r in payload["rows"]
        if r.get("status") == "ok"
    )


# ------------------------------------------------------------------------------------------------ variant perception (B5) + disclosure
VP_DIRECTIONS = ("above", "below", "in_line")
PILOT_POSITIONS = ("long", "short", "none", "undisclosed")


def validate_variant_perception(narrative, evidence):
    """variant_perception[]: {metric, period, consensus_ref, our_view (number|string), direction above|below|in_line, why}.
    consensus_ref must be a frozen consensus evidence carrying that metric/period; a numeric our_view must agree with the
    direction vs the consensus mean (in_line = inside [low, high]); our_view is in the consensus item's unit."""
    vp = narrative.get("variant_perception")
    if vp is None:
        return
    if not isinstance(vp, list) or not vp:
        raise SystemExit("variant_perception must be a non-empty list when present")
    by_id = {e["id"]: e for e in evidence}
    for n, v in enumerate(vp):
        w = "variant_perception[%d]" % n
        if not isinstance(v, dict):
            raise SystemExit("%s must be an object" % w)
        for k in ("metric", "period", "consensus_ref", "why"):
            if not isinstance(v.get(k), str) or not v[k].strip():
                raise SystemExit("%s.%s must be a non-empty string" % (w, k))
        if v.get("direction") not in VP_DIRECTIONS:
            raise SystemExit(
                "%s.direction must be one of %s, got %r" % (w, "|".join(VP_DIRECTIONS), v.get("direction"))
            )
        ref = by_id.get(v["consensus_ref"])
        if ref is None or ref["kind"] != "consensus":
            raise SystemExit("%s.consensus_ref %r is not a frozen consensus evidence" % (w, v["consensus_ref"]))
        v["period"] = norm_period(v["period"])  # stored normalised so the renderer matches it to the consensus item
        it = next(
            (
                i
                for i in ref["payload"]["items"]
                if metric_key(i["metric"]) == metric_key(v["metric"]) and i["period"] == v["period"]
            ),
            None,
        )
        if it is None:
            raise SystemExit("%s: %s has no %s %s item" % (w, v["consensus_ref"], v["metric"], v["period"]))
        ov = v.get("our_view")
        if _is_num(ov):
            ok = {"above": ov > it["mean"], "below": ov < it["mean"], "in_line": it["low"] <= ov <= it["high"]}[
                v["direction"]
            ]
            if not ok:
                raise SystemExit(
                    "%s: our_view %s is inconsistent with direction %r vs consensus mean %s (range %s–%s)"
                    % (w, ov, v["direction"], it["mean"], it["low"], it["high"])
                )
        elif not isinstance(ov, str) or not ov.strip():
            raise SystemExit("%s.our_view must be a number or a non-empty string" % w)


def pilot_disclosure(cfg):
    """cfg pilot_position: "long|short|none|undisclosed" or {position, size_note?}. Returns the disclosure dict or None."""
    pp = cfg.get("pilot_position")
    if pp is None:
        return None
    pp = {"position": pp} if isinstance(pp, str) else pp
    if not isinstance(pp, dict) or pp.get("position") not in PILOT_POSITIONS:
        raise SystemExit("pilot_position must be one of %s (or {position, size_note})" % "|".join(PILOT_POSITIONS))
    if "size_note" in pp and (not isinstance(pp["size_note"], str) or not pp["size_note"].strip()):
        raise SystemExit("pilot_position.size_note must be a non-empty string when present")
    return {
        "pilot_position": pp["position"],
        **({"size_note": pp["size_note"]} if pp.get("size_note") else {}),
        "as_of": cfg["as_of"],
        "recorded_from": "config",
    }


def validate_supersedes(db, narrative):
    """Optional narrative supersedes_revision_id / supersedes: must name a completed revision in this ledger."""
    from poseidon.models.research_revision import ResearchRevision

    sid = narrative.get("supersedes_revision_id") or narrative.get("supersedes")
    if sid is None:
        return None
    import uuid

    try:
        u = uuid.UUID(str(sid))
    except ValueError:
        raise SystemExit("supersedes_revision_id %r is not a revision id" % sid)
    prev = db.get(ResearchRevision, u)
    if prev is None or prev.status != "completed":
        raise SystemExit("supersedes_revision_id %s is not a completed revision in this ledger" % sid)
    return str(u)


def precheck_claims(narrative, evidence, allow_non_atomic=False):
    """Numeric + atomic precheck for every claim; raises SystemExit listing every problem. Returns per-claim results."""
    by_id = {e["id"]: e for e in evidence}
    problems, results = [], []
    for n, claim in enumerate(narrative.get("claims", [])):
        text, ids = claim.get("text", ""), claim.get("evidence_ids", [])
        cited = [by_id[i] for i in ids if i in by_id]
        if any(e["kind"] in ("price_facts", "ohlcv") for e in cited):
            num = {"status": "skipped", "reason": "cites price_facts/ohlcv"}
        else:
            excerpts = [ex["text"] for e in cited for ex in e["payload"].get("excerpts", [])]
            excerpts += [kpi_text(e["payload"]) for e in cited if e["kind"] == "kpi_series"]
            excerpts += [consensus_text(e["payload"]) for e in cited if e["kind"] == "consensus"]
            excerpts += [surprise_text(e["payload"]) for e in cited if e["kind"] == "surprise"]
            num = (
                check_numbers(text, "\n".join(excerpts), allow_sum=claim.get("kind") == "inference")
                if excerpts
                else {"status": "skipped", "reason": "no cited excerpts"}
            )
        atom = lint_atomic(text, claim.get("kind"))
        basis = check_basis(text, [ex["text"] for e in cited for ex in e["payload"].get("excerpts", [])])
        numbers = num["status"]
        # a stated derivation (A-B, A+B, A/B, ...) that precheck recomputed from the cited numbers is recorded
        derived = [
            {"claim": m["claim"], **{k: v for k, v in via.items() if k in ("how", "from", "value")}}
            for m in num.get("matched", [])
            if m.get("match") == "derived"
            for via in m["via"]
            if via.get("match") == "derived"
        ]
        if (
            numbers == "mismatch"
            and claim.get("kind") == "inference"
            and not {u["kind"] for u in num["unmatched"]} & {"scale", "sign"}
        ):
            numbers = "mismatch-recorded"  # inference may carry computed numbers absent from excerpts; scale/sign still refuse
        results.append(
            {
                "claim": n,
                "numbers": numbers,
                "atomic": atom["ok"],
                "basis": basis,
                **(
                    {"unmatched": ["%s (%s)" % (u["claim"], u["kind"]) for u in num["unmatched"]]}
                    if numbers == "mismatch-recorded"
                    else {}
                ),
                **({"derived": derived} if derived else {}),
            }
        )
        if basis == "missing":
            problems.append(
                "claims[%d] cites a document that reports both GAAP and non-GAAP figures but does not say which basis it uses"
                % n
            )
        if numbers == "mismatch":
            det = "; ".join("%s (%s)" % (u["claim"], u["kind"]) for u in num["unmatched"])
            problems.append("claims[%d] numeric mismatch vs cited excerpts %s: %s" % (n, ids, det))
        if not atom["ok"] and not allow_non_atomic:
            problems.append(
                "claims[%d] is non-atomic (%s): split into %s, or set allow_non_atomic in the config"
                % (n, atom["reason"], atom["clauses"])
            )
    if problems:
        raise SystemExit("claim precheck refused completion:\n  " + "\n  ".join(problems))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--narrative")
    ap.add_argument("--out")
    ap.add_argument(
        "--kpi-json", help="offline TWSE t187ap05_L rows (list, or {rows, source_uri, fetched_at}); enables kpi_series"
    )
    ap.add_argument(
        "--consensus-json",
        action="append",
        default=[],
        help="manual/offline consensus snapshot (README 'Consensus JSON'); repeatable",
    )
    ap.add_argument(
        "--consensus-live",
        action="store_true",
        help="best-effort Yahoo Finance fetch (US ticker or cfg adr_ticker); never fails the run",
    )
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    as_of = cfg["as_of"]
    bars = load_bars(a.csv)
    if "read_across" in cfg:
        raise SystemExit(
            "config read_across is no longer accepted; name the companies a claim is about in that claim's "
            "about_symbols instead (design 2026-10-29 §2.7)"
        )
    ohlcv_items = [ohlcv_evidence(s, bars[s], as_of, cfg["thalassa_base_url"]) for s in cfg["symbols"]]
    bench_sym = cfg.get("benchmark", "0050")
    bench_item = next((i for i in ohlcv_items if i["payload"]["symbol"] == bench_sym), None)
    derived = (
        [price_facts_evidence(item, cfg["event_time"], as_of, bench_item) for item in ohlcv_items]
        if cfg.get("price_facts", True)
        else []
    )
    documents = attach_actuals(cfg["documents"], cfg.get("actuals"))
    doc_items = [document_evidence(d, as_of) for d in documents]
    kpi_items = []
    if a.kpi_json or cfg.get("kpi_series"):
        rows, uri, fetched = load_kpi_rows(a.kpi_json)
        for sym in cfg.get("kpi_symbols", cfg["symbols"]):
            item = kpi_series_evidence(sym, rows, uri, fetched, as_of)
            if item is None:
                print("note: no t187ap05_L row visible at as_of for %s; kpi_series skipped" % sym, file=sys.stderr)
            else:
                kpi_items.append(item)
        if not kpi_items:
            raise SystemExit("kpi_series requested but no symbol has a monthly revenue row visible at as_of")
    narrative = json.load(open(a.narrative)) if a.narrative else None
    narrative_acts = narrative_actuals(narrative) if narrative else []
    # consensus (B1): manual files first, then the optional live probe; a failed probe is recorded, never fatal
    cons_items, cons_status = [], []
    for path in a.consensus_json:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        item, note = consensus_evidence(data, as_of)
        if note:
            print("note: " + note, file=sys.stderr)
            cons_status.append({"source": path, "status": "skipped", "reason": note})
        else:
            cons_items.append(item)
            cons_status.append({"source": path, "status": "ok", "items": len(item["payload"]["items"])})
    if a.consensus_live or cfg.get("consensus_live"):
        for sym in cfg.get("consensus_symbols", [cfg["symbol"]] if cfg.get("symbol") else cfg["symbols"]):
            if any(i["payload"]["symbol"] == sym for i in cons_items):
                continue
            ticker = adr_ticker_for(cfg, sym)
            if not ticker:
                cons_status.append(
                    {"symbol": sym, "status": "unavailable", "reason": "no adr_ticker mapping for a TW symbol"}
                )
                continue
            data, st = fetch_yahoo_consensus(
                sym,
                ticker,
                cfg.get("consensus_raw_dir") or str(Path(a.ledger).resolve().parent / "consensus_raw"),
                adr=sym != ticker,
            )
            st["symbol"] = sym
            if data:
                item, note = consensus_evidence(data, as_of, source="yahoo")
                if note:
                    st.update(status="skipped", reason=note)
                else:
                    cons_items.append(item)
            print(
                "note: consensus %s via %s: %s%s"
                % (sym, ticker, st["status"], (" (%s)" % st["reason"]) if st.get("reason") else ""),
                file=sys.stderr,
            )
            cons_status.append(st)
    surprise_items = []
    if cons_items:
        actual_inputs = _actual_inputs(doc_items, narrative_acts, {e["id"]: e for e in doc_items + kpi_items})
        for c in cons_items:
            if cfg.get("symbol") and c["payload"]["symbol"] != cfg["symbol"]:
                continue  # actuals in a single-symbol scope belong to that symbol only
            s_item = surprise_evidence(c, actual_inputs, as_of)
            if s_item:
                surprise_items.append(s_item)
    snap_items = [snapshot_evidence(cfg["narrative_snapshot"], as_of)] if cfg.get("narrative_snapshot") else []
    evidence = doc_items + ohlcv_items + derived + kpi_items + cons_items + surprise_items + snap_items
    capability = {"fundamental": True, "ohlcv": True}
    for kind in sorted({d["kind"] for d in doc_items} - {"fundamental"}):
        capability[kind] = True
    if snap_items:
        capability["narrative_snapshot"] = True
    if derived:
        capability["price_facts"] = True
    if kpi_items:
        capability["kpi_series"] = True
    if cons_items:
        capability["consensus"] = True
    if surprise_items:
        capability["surprise"] = True
    request = {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": as_of,
        "account_scope": cfg["account_scope"],
        "universe_id": cfg["universe_id"],
        **({"symbol": cfg["symbol"]} if cfg.get("symbol") else {}),
        "capability_json": capability,
        "required_data": cfg["required_data"],
        "sources_json": cfg.get(
            "sources_json", [{"source": "thalassa", "base_url": cfg["thalassa_base_url"]}, {"source": "sec-edgar"}]
        ),
        "evidence": evidence,
    }
    engine = create_engine(f"sqlite:///{a.ledger}")
    Base.metadata.create_all(engine, tables=[Base.metadata.tables[n] for n in ("data_manifests", "research_revisions")])
    runtime_digest = "script:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    ident = dict(
        policy_version=cfg["policy_version"],
        provider=cfg["provider"],
        model=cfg["model"],
        runtime_digest=runtime_digest,
    )
    with Session(engine) as db:
        manifest = ManifestService(db).freeze(request)
        svc = ResearchService(db)
        rev = svc.prepare(cfg["scope_key"], manifest.id, **ident)
        db.commit()
        info = {
            "scope_key": cfg["scope_key"],
            "manifest_id": str(manifest.id),
            "manifest_sha256": manifest.content_sha256,
            "as_of": as_of,
            "evidence": [
                {
                    "id": e["id"],
                    "kind": e["kind"],
                    "event_time": e["event_time"],
                    "sha256": e["content_sha256"],
                    "bars": len(e["payload"].get("bars", [])),
                    "excerpts": len(e["payload"].get("excerpts", [])),
                    **({"points": len(e["payload"]["points"])} if e["kind"] == "kpi_series" else {}),
                    **(
                        {
                            "items": len(e["payload"]["items"]),
                            "provider": e["payload"]["provider"],
                            "observed_at": e["payload"]["as_of"],
                        }
                        if e["kind"] == "consensus"
                        else {}
                    ),
                    **(
                        {"algorithm": e["payload"]["algorithm"]}
                        if e["kind"] in ("price_facts", "surprise", "narrative_snapshot")
                        else {}
                    ),
                    **(
                        {"records": len(e["payload"]["records"]), "not_shown": len(e["payload"]["not_shown"])}
                        if e["kind"] == "narrative_snapshot"
                        else {}
                    ),
                }
                for e in evidence
            ],
            "revision_id": str(rev.id),
            "request_sha256": rev.request_sha256,
            "status": rev.status,
            "previous_revision_id": (str(rev.previous_revision_id) if rev.previous_revision_id else None),
            "runtime_digest": runtime_digest,
            **ident,
            "stats": {d["payload"]["symbol"]: d["payload"]["facts"] for d in derived}
            if derived
            else {
                s: stats(
                    bars[s].loc[: pd.Timestamp(as_of)],
                    cfg["event_time"],
                    bars[bench_sym].loc[: pd.Timestamp(as_of)] if bench_item else None,
                )
                for s in cfg["symbols"]
            },
        }
        n_ex = sum(len(e["payload"].get("excerpts", [])) for e in doc_items)
        info["verification"] = {
            "excerpts_verified": n_ex,
            "excerpts_total": n_ex,
        }  # freeze refuses on any unverified excerpt
        if cons_status:
            info["consensus_status"] = cons_status
        current_guidance, actuals = _items_from(evidence, "guidance"), _items_from(evidence, "actuals")
        if a.narrative:
            narrative.update(
                {
                    "scope_key": cfg["scope_key"],
                    "as_of": manifest.payload_json["as_of"],
                    **({"symbol": cfg["symbol"]} if cfg.get("symbol") else {}),
                }
            )
            actuals = actuals + narrative_acts
        reported = [cfg["reported_period"]] if cfg.get("reported_period") else sorted({x["period"] for x in actuals})
        prior_id, prior = predecessor_guidance(db, rev, reported) if reported else (None, [])
        info["guidance"] = current_guidance
        info["guidance_bridge"] = (
            guidance_bridge(prior_id, prior, actuals, current_guidance, reported) if prior else None
        )
        if not a.narrative:
            print(json.dumps(info, ensure_ascii=False, indent=1))
            return
        validate_narrative_extras(narrative, as_of)
        validate_variant_perception(narrative, evidence)
        sup = validate_supersedes(db, narrative)
        if sup:
            narrative["supersedes_revision_id"] = sup
            narrative.pop("supersedes", None)
        disc = pilot_disclosure(cfg)
        if disc:
            if narrative.get("disclosure") not in (None, disc):
                raise SystemExit("narrative.disclosure and config pilot_position both set; keep one")
            narrative["disclosure"] = disc
        info["claim_precheck"] = precheck_claims(narrative, evidence, bool(cfg.get("allow_non_atomic")))
        principal = AuthPrincipal(
            "claude-code:aquarium-ef", frozenset({"researcher"}), frozenset({cfg["account_scope"]})
        )
        done = svc.complete_human_research(
            rev.id,
            narrative,
            principal=principal,
            account_scope=cfg["account_scope"],
            request_sha256=rev.request_sha256,
        )
        db.commit()
        again = svc.prepare(cfg["scope_key"], manifest.id, **ident)
        assert again.id == done.id and again.content_sha256 == done.content_sha256
        try:
            svc.complete_human_research(
                rev.id,
                dict(narrative, thesis="changed"),
                principal=principal,
                account_scope=cfg["account_scope"],
                request_sha256=rev.request_sha256,
            )
            raise SystemExit("immutability check FAILED")
        except Exception as e:
            immut = type(e).__name__ + ": " + str(e)
        db.rollback()
        info.update(
            {
                "status": done.status,
                "content_sha256": done.content_sha256,
                "replay_same_revision": True,
                "mutation_refused": immut,
            }
        )
        Path(a.out).write_text(
            json.dumps(
                {"identity": info, "research": done.research_json, "manifest_payload": manifest.payload_json},
                ensure_ascii=False,
                indent=1,
            )
        )
        print(
            json.dumps(
                {
                    k: info[k]
                    for k in (
                        "revision_id",
                        "request_sha256",
                        "content_sha256",
                        "status",
                        "replay_same_revision",
                        "mutation_refused",
                    )
                },
                ensure_ascii=False,
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
