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
  4. Config of the shape research_theme.py reads, plus runner-only keys (runner_inputs, agenda, draft, event, assemble_notes,
     subject_symbols, theme) that research_theme.py ignores.

Symbols (design 2026-10-29 §2.1): a bare code ('2303') is Taiwan-listed and gets OHLCV; '<EXCHANGE>:<TICKER>'
('NASDAQ:MU') is a subject the research is about and gets no price evidence (cfg subject_symbols). A bare code that is the
ticker of a non-Taiwan theme member ('3110' while the ledger knows 'TSE:3110') is refused, not sent to Thalassa.
Theme scope (§3.3): a row with theme_id researches tw_stock:theme:<id>; its OHLCV basket is the theme's Taiwan-listed
members (approved by as_of) plus 0050, never the row's own picks.
Narrative snapshot (§2.6, §4.1): when the run is a theme scope, one of its symbols is an approved member of an active
theme as of as_of, or the ledger has open narrative records for its symbols or themes, cfg narrative_snapshot carries
the themes to tag and those records (possibly none), folded at as_of, for research_theme.py to freeze. A run that meets
no theme and no open record gets no key, so its manifest hash is what it was before the ledger existed.

Any failure that leaves no usable evidence raises AssembleError(reason); the runner turns it into needs_review.
Usage (manual): assemble.py --row row.json --run-dir DIR [--as-of 2026-10-21T04:20:00Z]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
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
SNAPSHOT_ALGO = "narrative-snapshot-v1"
SNAPSHOT_CAP = 15  # D11: open records shown to one revision
OPEN_STATUSES = ("unverified", "partially_verified", "stale")


class AssembleError(Exception):
    """Evidence could not be assembled; str(e) is the needs_review reason."""


def iso_z(dt):
    return dt.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def template_for(event_type):
    return TEMPLATE_BY_EVENT.get(event_type, "generic")


def scope_for(symbols, theme_id=None):
    """(scope_key, symbol-or-None): a theme row is a theme scope; a single Taiwan symbol is a spot scope; several form a
    basket scope."""
    if theme_id:
        return f"tw_stock:theme:{theme_id}", None
    names = [s for s in symbols if s != BENCHMARK] or list(symbols)
    if len(names) == 1:
        return f"tw_stock:spot:{names[0]}", names[0]
    return "tw_stock:basket:{}".format("-".join(sorted(names))), None


# ------------------------------------------------------------------------------------------------ theme ledger (read)
def open_ledger(path):
    """Read-only connection to the runner ledger, or None when there is no ledger or it has no theme tables yet."""
    if not path or not Path(path).is_file():
        return None
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'narrative_records'").fetchone():
        db.close()
        return None
    return db


def themes_at(db, as_of):
    """Active themes approved by as_of -> {theme_id: {theme_id, title, kind, approved_at, members [{symbol, role,
    tw_listed, approved_at}]}}."""
    if db is None:
        return {}
    themes = {
        r["theme_id"]: {**dict(r), "members": []}
        for r in db.execute(
            "SELECT theme_id, title, kind, approved_at FROM themes WHERE status = 'active' AND approved_at <= ? "
            "ORDER BY theme_id",
            (as_of,),
        )
    }
    for m in db.execute(
        "SELECT theme_id, symbol, role, tw_listed, approved_at FROM theme_members WHERE approved_at <= ? "
        "ORDER BY theme_id, symbol",
        (as_of,),
    ):
        if m["theme_id"] in themes:
            themes[m["theme_id"]]["members"].append({k: m[k] for k in ("symbol", "role", "tw_listed", "approved_at")})
    return themes


def refuse_bare_foreign(symbols, themes):
    """A bare code is Taiwan-listed (narrative_ledger.symbol_is_tw). When the ledger knows the same ticker as a non-Taiwan
    member ('TSE:3110') and not as a Taiwan member, the bare code is a missing prefix, not a TWSE listing."""
    tw_members = {m["symbol"] for t in themes.values() for m in t["members"] if m["tw_listed"]}
    foreign = {}
    for t in themes.values():
        for m in t["members"]:
            if not m["tw_listed"]:
                foreign.setdefault(m["symbol"].split(":", 1)[1], set()).add(m["symbol"])
    for s in symbols:
        if s in foreign and s not in tw_members:
            raise AssembleError(
                "symbol {} is a bare (Taiwan) code but the theme ledger lists {}; write the "
                "'<EXCHANGE>:<TICKER>' form for a non-Taiwan listing".format(s, ", ".join(sorted(foreign[s])))
            )


def split_symbols(row, themes, as_of):
    """(tw, subject_symbols, theme, notes) for an agenda row. A theme row's Taiwan symbols are the theme's Taiwan-listed
    members; a bare code the row names must be one of them."""
    from narrative_ledger import FOREIGN_SYMBOL

    syms = [str(s).strip().upper() for s in row.get("symbols") or []]
    tw = [s for s in syms if TW_SYMBOL.match(s)]
    subjects = [s for s in syms if FOREIGN_SYMBOL.match(s)]
    notes = [
        f"symbol {s} is neither a Taiwan code nor '<EXCHANGE>:<TICKER>'; ignored"
        for s in syms
        if s not in tw + subjects
    ]
    refuse_bare_foreign(tw, themes)
    theme = None
    if row.get("theme_id"):
        theme = themes.get(row["theme_id"])
        if theme is None:
            raise AssembleError(f"theme {row['theme_id']!r} is not an active theme approved by {as_of}")
        members = [m["symbol"] for m in theme["members"] if m["tw_listed"]]
        if not members:
            raise AssembleError(f"theme {theme['theme_id']} has no Taiwan-listed member to price (design D4)")
        outside = [s for s in tw if s not in members]
        if outside:
            raise AssembleError(
                f"symbols {outside} are not Taiwan-listed members of theme {theme['theme_id']}; approve them as "
                "members or drop them from the row"
            )
        refuse_bare_foreign(members, themes)
        tw = members
    if not tw:
        raise AssembleError(f"no Taiwan-listed symbol in {syms} (a foreign company is researched under a theme_id)")
    return tw, subjects, theme, notes


def narrative_snapshot(db, as_of, themes, run_symbols, theme_id=None):
    """The §4.1 selection, folded at as_of, as a narrative_snapshot evidence item (no recorded_at / hash; research_theme
    adds them), or None when the run meets no theme (T_run is empty) and no open record is selected."""
    import narrative_ledger as nl

    if db is None:
        return None
    s_run = set(run_symbols)
    t_run = {t for t, th in themes.items() if s_run & {m["symbol"] for m in th["members"]}}
    if theme_id:
        t_run.add(theme_id)
    picked = []
    for r in db.execute("SELECT * FROM narrative_records WHERE first_seen_at <= ? ORDER BY narrative_id", (as_of,)):
        events = [e for e in nl.events_of(db, r["narrative_id"]) if e["at"] <= as_of]
        status = nl.fold_status(events)
        subjects, theme_ids = json.loads(r["subject_symbols_json"]), json.loads(r["theme_ids_json"])
        hit = bool(s_run & set(subjects))
        if status in OPEN_STATUSES and (hit or t_run & set(theme_ids)):
            picked.append((hit, r, status, theme_ids, subjects, events[-1]["at"]))
    if not picked and not t_run:
        return None
    picked.sort(key=lambda p: p[1]["first_seen_at"], reverse=True)  # stable: ties stay in narrative_id order
    picked.sort(key=lambda p: not p[0])
    shown, rest = picked[:SNAPSHOT_CAP], picked[SNAPSHOT_CAP:]
    records = [
        {
            "narrative_id": r["narrative_id"],
            "claim": r["claim"],
            "claim_type": r["claim_type"],
            "resolves_on": r["resolves_on"],
            "attributed_to": r["attributed_to"],
            "source_tier": r["source_tier"],
            "status": status,
            "first_seen_at": r["first_seen_at"],
            "theme_ids": theme_ids,
            "subject_symbols": subjects,
            "last_event_at": last_at,
        }
        for _, r, status, theme_ids, subjects, last_at in shown
    ]
    listed = sorted((t_run | {t for rec in records for t in rec["theme_ids"]}) & set(themes))
    # the latest ledger fact shown: a record's first sighting or last event, a listed theme's or member's approval
    shown_at = [t for rec in records for t in (rec["first_seen_at"], rec["last_event_at"])]
    shown_at += [themes[t]["approved_at"] for t in listed]
    shown_at += [m["approved_at"] for t in listed for m in themes[t]["members"]]
    seen = max(shown_at)
    return {
        "id": "narrative_snapshot",
        "kind": "narrative_snapshot",
        "source_uri": f"ledger://narrative/{as_of}",
        "event_time": seen,
        "available_at": seen,
        "payload": {
            "algorithm": SNAPSHOT_ALGO,
            "themes": [
                {
                    "theme_id": t,
                    "title": themes[t]["title"],
                    "kind": themes[t]["kind"],
                    "members": [{"symbol": m["symbol"], "role": m["role"]} for m in themes[t]["members"]],
                }
                for t in listed
            ],
            "records": records,
            "not_shown": [
                {"narrative_id": p[1]["narrative_id"], "reason": f"beyond the cap of {SNAPSHOT_CAP} records (D11)"}
                for p in rest
            ],
        },
    }


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
            raise AssembleError(
                f"Thalassa returned {len(bars)} daily bars for {sym} (need >= 21; a non-Taiwan listing is written "
                "'<EXCHANGE>:<TICKER>')"
            )
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
    ledger_path=None,
):
    """Build <run_dir>/config.json (+ bars.csv, sources/) for one agenda row; returns the config dict. ledger_path is the
    runner ledger, read (never written) for themes and open narrative records."""
    run_dir = Path(run_dir)
    (run_dir / "sources").mkdir(parents=True, exist_ok=True)
    as_of = iso_z(as_of_dt)
    db = open_ledger(ledger_path)
    try:
        themes = themes_at(db, as_of)
        tw, subjects, theme, notes = split_symbols(row, themes, as_of)
        snapshot = narrative_snapshot(db, as_of, themes, tw + subjects, theme and theme["theme_id"])
    finally:
        if db is not None:
            db.close()
    notes += [f"symbol {s} is a subject (not Taiwan-listed); no OHLCV evidence" for s in subjects]
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
    scope_key, symbol = scope_for(tw, theme and theme["theme_id"])
    base = thalassa_base_url or thalassa_env()[0]
    draft = {"model": DEFAULT_DRAFT_MODEL, "reasoning": True, "max_tokens": 32000, **(draft or {})}  # as draft.py
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
        **({"narrative_snapshot": snapshot} if snapshot else {}),
        # runner-only keys (research_theme.py ignores them)
        **({"subject_symbols": subjects} if subjects else {}),
        **({"theme": {k: theme[k] for k in ("theme_id", "title", "kind")}} if theme else {}),
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
