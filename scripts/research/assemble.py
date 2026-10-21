#!/usr/bin/env python3
"""Deterministic research_theme.py config builder for one research-agenda row (no LLM).

Steps (all inputs are recorded under <run_dir>, nothing is chosen by a model):
  1. Fetch the row's primary_source_url (fall back to mirror_url on a fetch error or when no excerpt matched) with
     fetch_source.build_document into <run_dir>/sources; the template follows event_type
     (earnings / investor_conference -> earnings, press_release -> press_release, anything else -> generic) and every
     pre-verified excerpt proposal is kept, in document order, up to MAX_EXCERPTS. When no candidate matches the
     template (e.g. a MOPS notice of a future 法說會 fetched for an earnings topic), the first fetched document falls
     back to the generic template (paragraphs >= FALLBACK_MIN_CHARS, the longest MAX_EXCERPTS kept in document order,
     still pre-verified verbatim) instead of failing; only an unfetchable source fails.
  1b. event_status (owner decision 2026-10-21), decided here and never by the model: "pre_event" when event_date is
     after as_of, or the document dates the event (事實發生日 / 召開…日期 / 舉行…日期 …) after as_of, or no template
     excerpt matched (the outcome is not in the evidence); otherwise "post_event". Stored as cfg["event"].
  2. Daily OHLCV for the Taiwan-listed symbols plus the benchmark (0050) from Thalassa, using Poseidon's
     RemoteDataRepository with POSEIDON_THALASSA_BASE_URL / POSEIDON_THALASSA_API_KEY from the process environment;
     written to <run_dir>/bars.csv in research_theme.py's CSV format.
  3. TWSE t187ap05_L monthly revenue for the numeric non-benchmark symbols, saved as <run_dir>/sources/twse-t187ap05_L.json
     ({rows, source_uri, fetched_at}) and passed to research_theme.py as --kpi-json. A TWSE failure only drops
     kpi_series (recorded in notes); it never fails the run.
  4. Config of the shape research_theme.py reads, plus runner-only keys (runner_inputs, agenda, draft, event, assemble_notes)
     that research_theme.py ignores.

Any failure that leaves no usable evidence raises AssembleError(reason); the runner turns it into needs_review.
Usage (manual): assemble.py --row row.json --run-dir DIR [--as-of 2026-10-21T04:20:00Z]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_source

MAX_EXCERPTS = 40
FALLBACK_MIN_CHARS = 20  # notices are short; generic fallback keeps shorter paragraphs than the template default (80)
BENCHMARK = "0050"
LOOKBACK_DAYS = 600  # research_theme keeps the last 400 bars; 600 calendar days covers that
ACCOUNT_SCOPE = "paper:tw-stock-pilot"
UNIVERSE_ID = "thalassa-tw_stock-2026-10-15"
POLICY_VERSION = "tw-stock-pilot-2026-10/research-runner-v0"
DEFAULT_DRAFT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
# Freshness rules checked by ManifestService.freeze. ohlcv: 5 days so a Monday-morning run still sees Friday's bar
# (the producer's 1-day rule would refuse every run before Monday's close); fundamental: 90 days (as the UMC note).
REQUIRED_DATA = {"fundamental": {"max_age_seconds": 90 * 86400}, "ohlcv": {"max_age_seconds": 5 * 86400}}
TEMPLATE_BY_EVENT = {"earnings": "earnings", "investor_conference": "earnings", "press_release": "press_release"}
TW_SYMBOL = re.compile(r"^\d{4,6}[A-Z]?$")
TWSE_KPI_FILE = "twse-t187ap05_L.json"


class AssembleError(Exception):
    """Evidence could not be assembled; str(e) is the needs_review reason."""


def iso_z(dt):
    return dt.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def template_for(event_type):
    return TEMPLATE_BY_EVENT.get(event_type, "generic")


def scope_for(symbols):
    """(scope_key, symbol-or-None): a single Taiwan symbol is a spot scope; several form a basket scope."""
    names = [s for s in symbols if s != BENCHMARK] or list(symbols)
    if len(names) == 1:
        return f"tw_stock:spot:{names[0]}", names[0]
    return "tw_stock:basket:{}".format("-".join(sorted(names))), None


# ------------------------------------------------------------------------------------------------ source document
def fetch_document(url, *, doc_id, title, publisher, event_time, out_dir, template):
    """fetch_source.build_document with keep=all; network/conversion failures become AssembleError."""
    try:
        doc, proposals, _ = fetch_source.build_document(
            url,
            doc_id=doc_id,
            title=title,
            publisher=publisher,
            event_time=event_time,
            out_dir=out_dir,
            template=template,
            keep="all",
        )
    except SystemExit as e:  # fetch_source exits on a Cloudflare challenge page
        raise AssembleError(f"source fetch refused for {url}: {e}") from None
    except urllib.error.HTTPError as e:
        raise AssembleError(f"source fetch failed for {url}: HTTP {e.code}") from None
    except urllib.error.URLError as e:
        raise AssembleError(f"source fetch failed for {url}: {e.reason}") from None
    except FileNotFoundError as e:
        if "pdftotext" in str(e):
            raise AssembleError(f"PDF source {url} needs pdftotext (poppler-utils), not installed") from None
        raise AssembleError(f"source conversion failed for {url}: {e}") from None
    except (subprocess.CalledProcessError, OSError, TimeoutError, ValueError) as e:
        raise AssembleError(f"source fetch/convert failed for {url}: {e}") from None
    total = len(doc["excerpts"])
    doc["excerpts"] = doc["excerpts"][:MAX_EXCERPTS]
    stats = {"proposals": len(proposals), "verified": total, "kept": len(doc["excerpts"])}
    return doc, stats


def generic_fallback(doc):
    """Re-propose excerpts of an already fetched document with the generic template (no refetch). The longest
    MAX_EXCERPTS verified paragraphs are kept, in document order. Returns the number of verified proposals."""
    verified = [
        p for p in fetch_source.propose(doc["raw_text_path"], "generic", min_chars=FALLBACK_MIN_CHARS) if p["verified"]
    ]
    longest = sorted(range(len(verified)), key=lambda n: -len(verified[n]["text"]))[:MAX_EXCERPTS]
    doc["excerpts"] = [{"locator": verified[n]["locator"], "text": verified[n]["text"]} for n in sorted(longest)]
    return len(verified)


def source_document(row, run_dir, event_time, fetch=fetch_document):
    """Primary URL first; mirror on a fetch error or when the primary yields no template excerpt; if no fetched
    candidate matches the template, the first fetched one falls back to the generic template.
    Returns (doc, notes, template_matched)."""
    template = template_for(row.get("event_type"))
    notes, errors, fetched = [], [], []
    candidates = [("primary", row.get("primary_source_url"))]
    if row.get("mirror_url"):
        candidates.append(("mirror", row["mirror_url"]))
    for label, url in candidates:
        if not url:
            continue
        host = urllib.parse.urlsplit(url).hostname or "unknown"
        try:
            doc, stats = fetch(
                url,
                doc_id="fundamental:agenda-{}".format(row["id"]),
                title=row["topic"],
                publisher=host,
                event_time=event_time,
                out_dir=str(Path(run_dir) / "sources" / label),
                template=template,
            )
        except AssembleError as e:
            errors.append(str(e))
            continue
        if not doc["excerpts"]:
            errors.append(f"{label} source {url}: no verified excerpt matched the {template} template")
            fetched.append((label, url, doc))
            continue
        notes.append(f"document from {label} url ({url}); excerpts {stats}")
        return doc, notes, True
    for label, url, doc in fetched:
        if template != "generic" and doc.get("raw_text_path") and generic_fallback(doc):
            notes += errors
            notes.append(
                f"document from {label} url ({url}); no {template} excerpt matched, generic fallback kept "
                f"{len(doc['excerpts'])} verified paragraphs"
            )
            return doc, notes, False
        errors.append(f"{label} source {url}: no usable paragraph (generic fallback empty)")
    raise AssembleError("; ".join(errors) or "agenda row has no source url")


# ------------------------------------------------------------------------------------------------ event status
_DATE = r"(\d{2,4})\s*(?:[/.-]|年)\s*(\d{1,2})\s*(?:[/.-]|月)\s*(\d{1,2})\s*日?"
_EVENT_DATE_RE = re.compile(
    r"(事實發生日|(?:召開|舉行|開會|會議|說明會|法說會)[^\n:：|]{0,12}?(?:日期|時間))\s*[:：|]?\s*" + _DATE
)


def notice_event_dates(text):
    """Dates the document gives for the event: values after 事實發生日 / 召開…日期 / 舉行…時間 ... (ROC years +1911)."""
    out = []
    for m in _EVENT_DATE_RE.finditer(text or ""):
        y, mo, d = (int(g) for g in m.group(2, 3, 4))
        y = y + 1911 if y < 1000 else y
        try:
            out.append((m.group(1), datetime.date(y, mo, d)))
        except ValueError:
            continue
    return out


def event_status(row, as_of_dt, doc, template_matched):
    """Deterministic pre_event / post_event decision; returns {status, event_date, why[]}."""
    as_of_day = as_of_dt.astimezone(datetime.UTC).date()
    row_day = datetime.date.fromisoformat(str(row["event_date"])[:10])
    why, event_day = [], row_day
    if datetime.datetime.combine(row_day, datetime.time(), datetime.UTC) > as_of_dt:
        why.append(f"event_date {row_day} is after as_of")
    try:
        text = Path(doc["raw_text_path"]).read_text(encoding="utf-8", errors="replace")
    except (KeyError, OSError):
        text = "\n".join(ex["text"] for ex in doc.get("excerpts", []))
    future = [(label, d) for label, d in notice_event_dates(text) if d > as_of_day]
    if future:
        label, event_day = max(future, key=lambda x: x[1])
        why.append(f"document dates the event after as_of ({label} {event_day})")
    if not template_matched:
        why.append("no template excerpt matched (the event outcome is not in the evidence)")
    return {"status": "pre_event" if why else "post_event", "event_date": event_day.isoformat(), "why": why}


# ------------------------------------------------------------------------------------------------ OHLCV (Thalassa)
def thalassa_env():
    base, key = os.environ.get("POSEIDON_THALASSA_BASE_URL", ""), os.environ.get("POSEIDON_THALASSA_API_KEY", "")
    missing = [n for n, v in (("POSEIDON_THALASSA_BASE_URL", base), ("POSEIDON_THALASSA_API_KEY", key)) if not v]
    if missing:
        raise AssembleError("Thalassa env missing: {}".format(", ".join(missing)))
    return base.rstrip("/"), key


def thalassa_daily_bars(symbol, start, end):
    """Daily tw_stock bars from Thalassa via Poseidon's RemoteDataRepository -> list of (iso time, o, h, l, c, v)."""
    import math

    import pandas as pd

    from poseidon.data.remote_repository import RemoteDataRepository

    base, key = thalassa_env()
    repo = RemoteDataRepository(base_url=base, api_key=key)
    try:
        frame = repo.read_ohlcv(symbol, "tw_stock", "1d", start, end)
    finally:
        repo.close()
    out = []
    for index, row in frame.iterrows():
        vals = [row.get(f) for f in ("open", "high", "low", "close", "volume")]
        if any(v is None or not math.isfinite(float(v)) for v in vals):
            continue
        ts = pd.Timestamp(index)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if ts.to_pydatetime() > end:
            continue
        out.append((ts.isoformat(), *(float(v) for v in vals)))
    return out


def write_bars(symbols, as_of_dt, csv_path, fetch_bars=thalassa_daily_bars):
    start = as_of_dt - datetime.timedelta(days=LOOKBACK_DAYS)
    lines = []
    for sym in symbols:
        try:
            bars = fetch_bars(sym, start, as_of_dt)
        except AssembleError:
            raise
        except Exception as e:  # httpx / circuit breaker / parse errors; never includes the API key
            raise AssembleError(f"Thalassa OHLCV fetch failed for {sym}: {type(e).__name__}: {e}") from None
        if len(bars) < 21:
            raise AssembleError(f"Thalassa returned {len(bars)} daily bars for {sym} (need >= 21)")
        lines += [f"{t},{sym},{o!r},{h!r},{low!r},{c!r},{v!r}" for t, o, h, low, c, v in bars]
    Path(csv_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ TWSE kpi_series
def twse_rows():
    import research_theme

    return research_theme.load_kpi_rows(None)


def kpi_inputs(symbols, as_of, out_path, fetch_rows=twse_rows):
    """(kpi_symbols, notes). Saves only the rows for `symbols`; symbols with no row visible at as_of are dropped."""
    import research_theme

    cands = [s for s in symbols if s != BENCHMARK and re.fullmatch(r"\d{4}", s)]
    if not cands:
        return [], ["kpi_series: no TWSE numeric symbol"]
    try:
        rows, uri, fetched = fetch_rows()
    except Exception as e:
        return [], [f"kpi_series skipped: TWSE t187ap05_L fetch failed ({type(e).__name__}: {e})"]
    keep = [r for r in rows if str(r.get("公司代號", "")).strip() in cands]
    usable = [s for s in cands if research_theme.kpi_series_evidence(s, keep, uri, fetched, as_of) is not None]
    notes = [f"kpi_series: no t187ap05_L row visible at as_of for {s}" for s in cands if s not in usable]
    if usable:
        Path(out_path).write_text(
            json.dumps({"rows": keep, "source_uri": uri, "fetched_at": fetched}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return usable, notes


# ------------------------------------------------------------------------------------------------ config
def assemble(
    row,
    run_dir,
    *,
    as_of_dt,
    thalassa_base_url=None,
    draft=None,
    fetch=fetch_document,
    fetch_bars=thalassa_daily_bars,
    fetch_rows=twse_rows,
):
    """Build <run_dir>/config.json (+ bars.csv, sources/) for one agenda row; returns the config dict."""
    run_dir = Path(run_dir)
    (run_dir / "sources").mkdir(parents=True, exist_ok=True)
    as_of = iso_z(as_of_dt)
    syms = [str(s).strip().upper() for s in row.get("symbols") or []]
    tw = [s for s in syms if TW_SYMBOL.match(s)]
    notes = [f"symbol {s} is not a Taiwan listing; no OHLCV evidence" for s in syms if s not in tw]
    if not tw:
        raise AssembleError(f"no Taiwan-listed symbol in {syms}")
    try:
        event_dt = datetime.datetime.fromisoformat(str(row["event_date"])[:10]).replace(tzinfo=datetime.UTC)
    except (KeyError, ValueError):
        raise AssembleError("agenda row has no valid event_date: {!r}".format(row.get("event_date"))) from None
    if event_dt > as_of_dt:
        notes.append("event_date {} is after as_of; document/event time clamped to as_of".format(row["event_date"]))
        event_dt = as_of_dt
    event_time = iso_z(event_dt)
    doc, doc_notes, template_matched = source_document(row, run_dir, event_time, fetch=fetch)
    notes += doc_notes
    event = event_status(row, as_of_dt, doc, template_matched)
    notes.append("event_status {} ({})".format(event["status"], "; ".join(event["why"]) or "event on/before as_of"))
    symbols = tw + ([BENCHMARK] if BENCHMARK not in tw else [])
    write_bars(symbols, as_of_dt, run_dir / "bars.csv", fetch_bars=fetch_bars)
    kpi_path = run_dir / "sources" / TWSE_KPI_FILE
    kpi_symbols, kpi_notes = kpi_inputs(tw, as_of, kpi_path, fetch_rows=fetch_rows)
    notes += kpi_notes
    scope_key, symbol = scope_for(tw)
    base = thalassa_base_url or thalassa_env()[0]
    draft = {"model": DEFAULT_DRAFT_MODEL, "reasoning": True, "max_tokens": 16000, **(draft or {})}
    cfg = {
        "scope_key": scope_key,
        **({"symbol": symbol} if symbol else {}),
        "as_of": as_of,
        "event_time": event_time,
        "account_scope": ACCOUNT_SCOPE,
        "universe_id": UNIVERSE_ID,
        "thalassa_base_url": base,
        "symbols": symbols,
        "benchmark": BENCHMARK,
        "required_data": REQUIRED_DATA,
        "sources_json": [
            {"source": "thalassa", "base_url": base},
            {"source": "research-agenda", "agenda_id": row["id"], "origin": row.get("source") or row.get("origin")},
            {"source": "document", "url": doc["source_uri"]},
        ]
        + ([{"source": "twse-openapi", "dataset": "t187ap05_L"}] if kpi_symbols else []),
        "policy_version": POLICY_VERSION,
        "provider": "openrouter",
        "model": draft["model"],
        "documents": [doc],
        **({"kpi_symbols": kpi_symbols} if kpi_symbols else {}),
        # runner-only keys (research_theme.py ignores them)
        "runner_inputs": {
            "csv": str(run_dir / "bars.csv"),
            "kpi_json": str(kpi_path) if kpi_symbols else None,
        },
        "agenda": {k: row.get(k) for k in ("id", "topic", "symbols", "event_date", "event_type", "why_now", "source")},
        "draft": draft,
        "event": event,
        "assemble_notes": notes,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Build a research_theme.py config from one agenda row (no LLM).")
    ap.add_argument("--row", required=True, help="agenda row JSON file")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--as-of", help="ISO UTC cutoff (default: now, truncated to the minute)")
    a = ap.parse_args()
    row = json.loads(Path(a.row).read_text(encoding="utf-8"))
    now = datetime.datetime.now(datetime.UTC).replace(second=0, microsecond=0)
    as_of = datetime.datetime.fromisoformat(a.as_of.replace("Z", "+00:00")) if a.as_of else now
    try:
        cfg = assemble(row, a.run_dir, as_of_dt=as_of)
    except AssembleError as e:
        sys.exit(f"assemble failed: {e}")
    print(json.dumps({k: cfg[k] for k in ("scope_key", "as_of", "symbols", "assemble_notes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
