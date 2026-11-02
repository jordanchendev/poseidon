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
  - atomic: a non-atomic claim always refuses completion. For inference claims the
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
from claim_precheck import check_numbers, lint_atomic
from consensus import (
    _actual_inputs,
    _is_num,
    adr_ticker_for,
    consensus_evidence,
    consensus_text,
    fetch_yahoo_consensus,
    metric_key,
    norm_period,
    surprise_evidence,
    surprise_text,
)
from guidance import (
    SECONDARY_TIERS,
    SOURCE_TIERS,
    _items_from,
    attach_actuals,
    guidance_bridge,
    narrative_actuals,
    predecessor_guidance,
    snapshot_evidence,
    validate_actuals,
    validate_guidance,
)
from kpi_series import kpi_series_evidence, kpi_text, load_kpi_rows
from refusal import ResearchRefusal
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from poseidon.core.authz import AuthPrincipal
from poseidon.core.canonical import content_sha256
from poseidon.core.errors import ValidationError
from poseidon.core.timeutil import iso_time
from poseidon.decision_loop.manifest import ManifestService
from poseidon.decision_loop.research import ResearchService
from poseidon.models.base import Base


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


LOOKBACK, FIELDS = 400, ("open", "high", "low", "close", "volume")
HERE = Path(__file__).resolve().parent
# The code a revision's runtime_digest names: this script and the modules split out of it (2026-11-02), in this order,
# so the digest still covers everything that used to be research_theme.py.
RUNTIME_FILES = ("research_theme.py", "guidance.py", "kpi_series.py", "consensus.py")


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
        raise ResearchRefusal("document %s has no raw_text_path to verify excerpts against" % item["id"])
    raw = _normalize(Path(raw_text_path).read_text(encoding="utf-8", errors="replace"))
    for ex in item["excerpts"]:
        text = _normalize(ex["text"])
        if not text or text not in raw:
            raise ResearchRefusal(
                "excerpt %s of %s is not a substring of %s; refusing to freeze"
                % (ex.get("locator"), item["id"], raw_text_path)
            )
    digest = hashlib.sha256(Path(raw_text_path).read_bytes()).hexdigest()
    declared = item.get("raw_files", {})
    if digest not in declared.values():
        raise ResearchRefusal("raw_text_path sha256 %s is not declared in raw_files of %s" % (digest[:12], item["id"]))


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
    if item.get("issuer"):  # the company that issued the document (assemble.issuer_of); the §5.2 official tier
        if not isinstance(item["issuer"], str) or not item["issuer"].strip():
            raise ResearchRefusal("document %s issuer must be a symbol" % item["id"])
        payload["issuer"] = item["issuer"]
    if item.get("source_tier"):
        if item["source_tier"] not in SOURCE_TIERS:
            raise ResearchRefusal("document %s source_tier must be one of %s" % (item["id"], "|".join(SOURCE_TIERS)))
        payload["source_tier"] = item["source_tier"]
    if item["kind"] == "secondary" and payload.get("source_tier") not in SECONDARY_TIERS:
        raise ResearchRefusal("secondary document %s needs source_tier %s" % (item["id"], "|".join(SECONDARY_TIERS)))
    if item["kind"] not in ("fundamental", "secondary"):
        raise ResearchRefusal("document %s kind must be fundamental or secondary" % item["id"])
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
    """Offline shape checks for optional narrative fields; raises ResearchRefusal with a clear message."""
    if "tl_dr" in narrative and not (isinstance(narrative["tl_dr"], str) and narrative["tl_dr"].strip()):
        raise ResearchRefusal("tl_dr must be a non-empty string")
    status = narrative.get("event_status")
    if status is not None and status not in EVENT_STATUSES:
        raise ResearchRefusal("event_status must be one of %s, got %r" % ("|".join(EVENT_STATUSES), status))
    if narrative.get("event_date") is not None:
        try:
            datetime.date.fromisoformat(str(narrative["event_date"]))
        except ValueError:
            raise ResearchRefusal("event_date must be an ISO date YYYY-MM-DD, got %r" % narrative["event_date"])
    cats = narrative.get("catalysts")
    if status == "pre_event" and not (isinstance(cats, list) and cats):
        raise ResearchRefusal("event_status pre_event requires a non-empty catalysts list (include the event date)")
    if cats is None:
        return
    if not isinstance(cats, list) or not cats:
        raise ResearchRefusal("catalysts must be a non-empty list when present")
    for n, c in enumerate(cats):
        if not isinstance(c, dict):
            raise ResearchRefusal("catalysts[%d] must be an object" % n)
        try:
            datetime.date.fromisoformat(c.get("date", ""))
        except (TypeError, ValueError):
            raise ResearchRefusal("catalysts[%d].date must be an ISO date YYYY-MM-DD, got %r" % (n, c.get("date")))
        if c.get("type") not in CATALYST_TYPES:
            raise ResearchRefusal(
                "catalysts[%d].type must be one of %s, got %r" % (n, "|".join(CATALYST_TYPES), c.get("type"))
            )
        for k in ("expected_signal", "source"):
            if not isinstance(c.get(k), str) or not c[k].strip():
                raise ResearchRefusal("catalysts[%d].%s must be a non-empty string" % (n, k))
    today = as_of[:10]
    future = sorted(c["date"] for c in cats if c["date"] >= today)
    if not future:
        print("note: no catalyst on/after as_of date %s; next_review_at not cross-checked" % today, file=sys.stderr)
        return
    if not str(narrative.get("next_review_at", "")).startswith(future[0]):
        raise ResearchRefusal(
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
        raise ResearchRefusal("variant_perception must be a non-empty list when present")
    by_id = {e["id"]: e for e in evidence}
    for n, v in enumerate(vp):
        w = "variant_perception[%d]" % n
        if not isinstance(v, dict):
            raise ResearchRefusal("%s must be an object" % w)
        for k in ("metric", "period", "consensus_ref", "why"):
            if not isinstance(v.get(k), str) or not v[k].strip():
                raise ResearchRefusal("%s.%s must be a non-empty string" % (w, k))
        if v.get("direction") not in VP_DIRECTIONS:
            raise ResearchRefusal(
                "%s.direction must be one of %s, got %r" % (w, "|".join(VP_DIRECTIONS), v.get("direction"))
            )
        ref = by_id.get(v["consensus_ref"])
        if ref is None or ref["kind"] != "consensus":
            raise ResearchRefusal("%s.consensus_ref %r is not a frozen consensus evidence" % (w, v["consensus_ref"]))
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
            raise ResearchRefusal("%s: %s has no %s %s item" % (w, v["consensus_ref"], v["metric"], v["period"]))
        ov = v.get("our_view")
        if _is_num(ov):
            ok = {"above": ov > it["mean"], "below": ov < it["mean"], "in_line": it["low"] <= ov <= it["high"]}[
                v["direction"]
            ]
            if not ok:
                raise ResearchRefusal(
                    "%s: our_view %s is inconsistent with direction %r vs consensus mean %s (range %s–%s)"
                    % (w, ov, v["direction"], it["mean"], it["low"], it["high"])
                )
        elif not isinstance(ov, str) or not ov.strip():
            raise ResearchRefusal("%s.our_view must be a number or a non-empty string" % w)


def pilot_disclosure(cfg):
    """cfg pilot_position: "long|short|none|undisclosed" or {position, size_note?}. Returns the disclosure dict or None."""
    pp = cfg.get("pilot_position")
    if pp is None:
        return None
    pp = {"position": pp} if isinstance(pp, str) else pp
    if not isinstance(pp, dict) or pp.get("position") not in PILOT_POSITIONS:
        raise ResearchRefusal("pilot_position must be one of %s (or {position, size_note})" % "|".join(PILOT_POSITIONS))
    if "size_note" in pp and (not isinstance(pp["size_note"], str) or not pp["size_note"].strip()):
        raise ResearchRefusal("pilot_position.size_note must be a non-empty string when present")
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
        raise ResearchRefusal("supersedes_revision_id %r is not a revision id" % sid)
    prev = db.get(ResearchRevision, u)
    if prev is None or prev.status != "completed":
        raise ResearchRefusal("supersedes_revision_id %s is not a completed revision in this ledger" % sid)
    return str(u)


def precheck_claims(narrative, evidence):
    """Numeric + atomic precheck for every claim; raises ResearchRefusal listing every problem. Returns per-claim results."""
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
        if not atom["ok"]:
            problems.append("claims[%d] is non-atomic (%s): split into %s" % (n, atom["reason"], atom["clauses"]))
    if problems:
        raise ResearchRefusal("claim precheck refused completion:", problems)
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
        raise ResearchRefusal(
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
            raise ResearchRefusal("kpi_series requested but no symbol has a monthly revenue row visible at as_of")
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
    runtime_digest = (
        "script:" + hashlib.sha256(b"".join((HERE / name).read_bytes() for name in RUNTIME_FILES)).hexdigest()[:16]
    )
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
                raise ResearchRefusal("narrative.disclosure and config pilot_position both set; keep one")
            narrative["disclosure"] = disc
        info["claim_precheck"] = precheck_claims(narrative, evidence)
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
        except ValidationError as e:
            immut = type(e).__name__ + ": " + str(e)
        else:
            raise ResearchRefusal("immutability check FAILED")
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
    try:
        main()
    except ResearchRefusal as e:
        sys.exit(str(e))
