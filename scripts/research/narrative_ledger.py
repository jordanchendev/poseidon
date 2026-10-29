#!/usr/bin/env python3
"""Theme ledger and narrative ledger (design local_dev/design/2026-10-29-research-theme-and-narrative-ledger.md §2-§4).

Nine tables in the runner's SQLite ledger (the file research_theme.py and settle_theme.py use). Every row carries
content_sha256 = Poseidon content_sha256 of its other columns. A narrative's status is the fold of its append-only
events; TRANSITIONS says who may move a record from one status to another. Event `at` never goes back in time: a
caller whose evidence predates the record's last event passes max(evidence available_at, last event at).

  theme-add   --ledger L --id ID --title T --kind K [--keyword W ...] --approval-ref REF [--proposal-id P] [--at T]
  member-add  --ledger L --theme ID --symbol S --role R --approval-ref REF [--at T]
  edge-add    --ledger L --from ID --to ID --relation related|part_of --approval-ref REF [--at T]
  record-add  --ledger L --claim C --claim-type state|forecast [--resolves-on YYYY-MM-DD] --attributed-to A
              --source-tier T --origin-ref R --origin-evidence-id E --quote Q --source-file F --subject S [...]
              --theme ID [...] --first-seen-at T
  set-status  --ledger L --id NR --to STATUS --note TEXT [--at T]      owner override
  sweep-stale --ledger L --as-of T          unverified records past review_by -> stale
  export      --ledger L --as-of T [--out F]  deterministic narrative_export.json

Symbols: Taiwan-listed bare ('2408'), everything else '<EXCHANGE>:<TICKER>' ('NASDAQ:MU', 'TSE:3110').
Approval rows (themes, members, edges) need a non-empty approval_ref; agents never pick theme ids.
Needs the Poseidon worktree on PYTHONPATH (content_sha256).
"""

import argparse
import datetime
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble import TW_SYMBOL
from claim_precheck import check_numbers, lint_atomic

from poseidon.decision_loop.manifest import content_sha256

EXPORT_ALGO = "narrative-export-v1"
THEME_KINDS = ("bottleneck", "tech_transition", "demand_driver", "policy")
MEMBER_ROLES = ("constrained", "supplier", "beneficiary", "customer", "competitor")
EDGE_RELATIONS = ("related", "part_of")
SOURCE_TIERS = ("official", "sell-side", "media", "kol", "our-inference")
CLAIM_TYPES = ("state", "forecast")
STATUSES = ("unverified", "partially_verified", "verified", "refuted", "stale")
# Event kind -> who may write it. 'corroborated' never changes the status.
EVENT_KINDS = {
    "created": {"rule", "owner"},
    "status": {"rule", "sweep"},
    "override": {"owner"},
    "corroborated": {"rule"},
}
STATUS_EVENT_KINDS = ("created", "status", "override")
# (from, to) -> who may make that change (design §2.4). A pair that is absent is refused for everyone.
TRANSITIONS = {
    ("unverified", "partially_verified"): {"rule", "owner"},
    ("unverified", "verified"): {"rule", "owner"},
    ("unverified", "refuted"): {"rule", "owner"},
    ("unverified", "stale"): {"sweep", "owner"},
    ("partially_verified", "verified"): {"rule", "owner"},
    ("partially_verified", "refuted"): {"owner"},
    ("partially_verified", "unverified"): {"owner"},
    ("verified", "refuted"): {"owner"},
    ("verified", "unverified"): {"owner"},
    ("refuted", "verified"): {"owner"},
    ("refuted", "unverified"): {"owner"},
    ("stale", "partially_verified"): {"rule", "owner"},
    ("stale", "verified"): {"rule", "owner"},
    ("stale", "refuted"): {"rule", "owner"},
    ("stale", "unverified"): {"owner"},
}
STATE_REVIEW_DAYS = 180  # D6
FORECAST_REVIEW_DAYS = 90
FORECAST_DEFAULT_RESOLVE_DAYS = 365
EXPORT_QUOTE_CHARS = 120  # D7: the export leaves stormtrooper for the vault

THEME_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FOREIGN_SYMBOL = re.compile(r"^[A-Z]{2,8}:[A-Z0-9][A-Z0-9.\-]*$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

DDL = """
CREATE TABLE IF NOT EXISTS themes (
  theme_id TEXT PRIMARY KEY, title TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('bottleneck','tech_transition','demand_driver','policy')),
  keywords_json TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('active','retired')),
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  proposal_id TEXT, content_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS theme_members (
  theme_id TEXT NOT NULL REFERENCES themes, symbol TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('constrained','supplier','beneficiary','customer','competitor')),
  tw_listed INTEGER NOT NULL CHECK (tw_listed IN (0,1)),
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (theme_id, symbol));
CREATE TABLE IF NOT EXISTS theme_edges (
  from_theme TEXT NOT NULL REFERENCES themes, to_theme TEXT NOT NULL REFERENCES themes,
  relation TEXT NOT NULL CHECK (relation IN ('related','part_of')),
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (from_theme, to_theme, relation), CHECK (from_theme <> to_theme));
CREATE TABLE IF NOT EXISTS theme_proposals (
  proposal_id TEXT PRIMARY KEY, suggested_id TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL, why TEXT NOT NULL,
  source_ref TEXT NOT NULL, proposed_members_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed','accepted','rejected')),
  decided_at TEXT, decided_ref TEXT, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS materials (
  material_id TEXT PRIMARY KEY, received_at TEXT NOT NULL,
  source_tier TEXT NOT NULL CHECK (source_tier IN ('sell-side','media','kol')),
  author TEXT NOT NULL, url TEXT, text_sha256 TEXT NOT NULL, raw_path TEXT NOT NULL, manifest_sha256 TEXT,
  status TEXT NOT NULL CHECK (status IN ('received','intaken','needs_review')), note TEXT,
  content_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS narrative_records (
  narrative_id TEXT PRIMARY KEY, claim TEXT NOT NULL,
  claim_type TEXT NOT NULL CHECK (claim_type IN ('state','forecast')), resolves_on TEXT,
  resolves_on_default INTEGER NOT NULL DEFAULT 0, attributed_to TEXT NOT NULL,
  source_tier TEXT NOT NULL CHECK (source_tier IN ('official','sell-side','media','kol','our-inference')),
  origin_kind TEXT NOT NULL CHECK (origin_kind IN ('material','revision','owner')), origin_ref TEXT NOT NULL,
  origin_evidence_id TEXT NOT NULL, quote TEXT NOT NULL, subject_symbols_json TEXT NOT NULL,
  theme_ids_json TEXT NOT NULL, first_seen_at TEXT NOT NULL, review_by TEXT NOT NULL, created_at TEXT NOT NULL,
  content_sha256 TEXT NOT NULL, CHECK (claim_type = 'state' OR resolves_on IS NOT NULL));
CREATE TABLE IF NOT EXISTS narrative_events (
  narrative_id TEXT NOT NULL REFERENCES narrative_records, seq INTEGER NOT NULL, at TEXT NOT NULL,
  recorded_at TEXT NOT NULL, kind TEXT NOT NULL CHECK (kind IN ('created','status','corroborated','override')),
  from_status TEXT, to_status TEXT NOT NULL, rule TEXT, basis_json TEXT NOT NULL, jev_json TEXT, drafter_verdict TEXT,
  decided_by TEXT NOT NULL CHECK (decided_by IN ('rule','owner','sweep')), note TEXT, content_sha256 TEXT NOT NULL,
  PRIMARY KEY (narrative_id, seq));
CREATE TABLE IF NOT EXISTS narrative_checks (
  revision_id TEXT NOT NULL, narrative_id TEXT NOT NULL REFERENCES narrative_records, status_before TEXT NOT NULL,
  drafter_verdict TEXT NOT NULL CHECK (drafter_verdict IN ('verifies','refutes','partially_verifies','untouched')),
  claim_indices_json TEXT NOT NULL, jev_label TEXT, sweep_label TEXT, sweep_truncated INTEGER NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('status_change','corroborated','no_change','disputed','missed')),
  event_seq INTEGER, checked_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, PRIMARY KEY (revision_id, narrative_id));
CREATE TABLE IF NOT EXISTS claim_theme_links (
  revision_id TEXT NOT NULL, claim_index INTEGER NOT NULL, theme_id TEXT NOT NULL REFERENCES themes,
  about_symbols_json TEXT NOT NULL, claim_kind TEXT NOT NULL CHECK (claim_kind IN ('fact','inference')),
  evidence_tier TEXT NOT NULL, event_time TEXT NOT NULL, source TEXT NOT NULL CHECK (source IN ('drafter','owner')),
  created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, PRIMARY KEY (revision_id, claim_index, theme_id));
"""


class LedgerError(Exception):
    """A write the ledger's rules refuse."""


def iso(value):
    """Timezone-aware ISO text -> 'YYYY-MM-DDTHH:MM:SSZ' (UTC), so stored times compare as strings."""
    try:
        ts = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as e:
        raise LedgerError(f"not an ISO timestamp: {value!r}") from e
    if ts.tzinfo is None:
        raise LedgerError(f"timestamp needs a timezone: {value!r}")
    return ts.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def now():
    return iso(datetime.datetime.now(datetime.UTC))


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(DDL)
    return db


def _insert(db, table, row):
    row = dict(row, content_sha256=content_sha256(row))
    db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    return row


def _one(db, sql, *args):
    r = db.execute(sql, args).fetchone()
    return dict(r) if r else None


def _approval(approval_ref, at):
    if not str(approval_ref or "").strip():
        raise LedgerError("approval_ref is required: themes, members and edges are written only after owner approval")
    return {"approved_by": "owner", "approved_at": iso(at or now()), "approval_ref": approval_ref.strip()}


def _same_or_refuse(existing, wanted, what):
    """Re-running an approval is a no-op; the same key with different content is refused."""
    diff = sorted(k for k, v in wanted.items() if existing[k] != v)
    if diff:
        raise LedgerError(f"{what} already exists with different {', '.join(diff)}")
    return existing


def symbol_is_tw(symbol):
    if TW_SYMBOL.match(symbol):
        return True
    if FOREIGN_SYMBOL.match(symbol):
        return False
    raise LedgerError(f"symbol {symbol!r}: Taiwan-listed is the bare code ('2408'), others '<EXCHANGE>:<TICKER>'")


def _active_theme(db, theme_id):
    t = _one(db, "SELECT * FROM themes WHERE theme_id = ?", theme_id)
    if not t or t["status"] != "active":
        raise LedgerError(f"theme {theme_id!r} is not an active theme")
    return t


# ---------------------------------------------------------------- themes
def add_theme(db, theme_id, title, kind, keywords, approval_ref, proposal_id=None, at=None):
    if not THEME_ID.match(theme_id or ""):
        raise LedgerError(f"theme id {theme_id!r} must be kebab-case")
    if kind not in THEME_KINDS:
        raise LedgerError(f"kind must be one of {'|'.join(THEME_KINDS)}")
    if not str(title or "").strip():
        raise LedgerError("title is required")
    wanted = {
        "theme_id": theme_id,
        "title": title.strip(),
        "kind": kind,
        "keywords_json": json.dumps(list(dict.fromkeys(keywords)), ensure_ascii=False),
        "status": "active",
        "proposal_id": proposal_id,
    }
    approval = _approval(approval_ref, at)
    old = _one(db, "SELECT * FROM themes WHERE theme_id = ?", theme_id)
    if old:
        return _same_or_refuse(old, wanted, f"theme {theme_id}")
    with db:
        return _insert(db, "themes", {**wanted, **approval})


def add_member(db, theme_id, symbol, role, approval_ref, at=None):
    if role not in MEMBER_ROLES:
        raise LedgerError(f"role must be one of {'|'.join(MEMBER_ROLES)}")
    wanted = {"theme_id": theme_id, "symbol": symbol, "role": role, "tw_listed": int(symbol_is_tw(symbol))}
    approval = _approval(approval_ref, at)
    _active_theme(db, theme_id)
    old = _one(db, "SELECT * FROM theme_members WHERE theme_id = ? AND symbol = ?", theme_id, symbol)
    if old:
        return _same_or_refuse(old, wanted, f"member {symbol} of {theme_id}")
    with db:
        return _insert(db, "theme_members", {**wanted, **approval})


def add_edge(db, from_theme, to_theme, relation, approval_ref, at=None):
    if relation not in EDGE_RELATIONS:
        raise LedgerError(f"relation must be one of {'|'.join(EDGE_RELATIONS)}")
    if from_theme == to_theme:
        raise LedgerError("a theme cannot have an edge to itself")
    approval = _approval(approval_ref, at)
    _active_theme(db, from_theme)
    _active_theme(db, to_theme)
    key = (from_theme, to_theme, relation)
    old = _one(db, "SELECT * FROM theme_edges WHERE from_theme = ? AND to_theme = ? AND relation = ?", *key)
    if old:
        return old
    with db:
        return _insert(
            db, "theme_edges", {"from_theme": from_theme, "to_theme": to_theme, "relation": relation, **approval}
        )


# ---------------------------------------------------------------- records
def narrative_id(origin_ref, quote, claim):
    return "nr-" + content_sha256([origin_ref, quote, claim])[:10]


def _date_plus(day, days):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


def review_dates(claim_type, first_seen_at, resolves_on):
    """(resolves_on, resolves_on_default, review_by) per D6. review_by is the end of that UTC day."""
    seen_day = first_seen_at[:10]
    if claim_type == "state":
        if resolves_on:
            raise LedgerError("a state claim has no resolves_on")
        return None, 0, _date_plus(seen_day, STATE_REVIEW_DAYS) + "T23:59:59Z"
    default = 0
    if not resolves_on:
        resolves_on, default = _date_plus(seen_day, FORECAST_DEFAULT_RESOLVE_DAYS), 1
    if not DATE.match(resolves_on):
        raise LedgerError("resolves_on must be YYYY-MM-DD")
    return resolves_on, default, _date_plus(resolves_on, FORECAST_REVIEW_DAYS) + "T23:59:59Z"


def add_record(
    db,
    *,
    claim,
    claim_type,
    attributed_to,
    source_tier,
    origin_kind,
    origin_ref,
    origin_evidence_id,
    quote,
    source_text,
    subject_symbols,
    theme_ids,
    first_seen_at,
    resolves_on=None,
    decided_by="rule",
):
    """Writes the record and its seq-0 'created' event. The quote must be verbatim in source_text, the claim must be
    atomic and its numbers must match the quote (the record is a transcription, so strict)."""
    if claim_type not in CLAIM_TYPES:
        raise LedgerError(f"claim_type must be one of {'|'.join(CLAIM_TYPES)}")
    if source_tier not in SOURCE_TIERS:
        raise LedgerError(f"source_tier must be one of {'|'.join(SOURCE_TIERS)}")
    for name, v in (
        ("claim", claim),
        ("attributed_to", attributed_to),
        ("origin_ref", origin_ref),
        ("origin_evidence_id", origin_evidence_id),
        ("quote", quote),
    ):
        if not str(v or "").strip():
            raise LedgerError(f"{name} is required")
    if quote not in source_text:
        raise LedgerError(f"quote is not a verbatim substring of {origin_evidence_id}")
    atomic = lint_atomic(claim)
    if not atomic["ok"]:
        raise LedgerError(f"claim is not atomic ({atomic['reason']})")
    if check_numbers(claim, quote)["status"] == "mismatch":
        raise LedgerError("a number in the claim is not in the quote")
    if not subject_symbols or not theme_ids:
        raise LedgerError("a record needs at least one subject symbol and one theme")
    for t in theme_ids:
        _active_theme(db, t)
    members = {
        r[0]
        for r in db.execute(
            "SELECT m.symbol FROM theme_members m JOIN themes t USING (theme_id) WHERE t.status = 'active'"
        )
    }
    for s in subject_symbols:
        symbol_is_tw(s)
        if s not in members:
            raise LedgerError(f"subject {s} is not an approved member of an active theme")
    first_seen_at = iso(first_seen_at)
    resolves_on, resolves_default, review_by = review_dates(claim_type, first_seen_at, resolves_on)
    nid = narrative_id(origin_ref, quote, claim)
    wanted = {
        "narrative_id": nid,
        "claim": claim,
        "claim_type": claim_type,
        "resolves_on": resolves_on,
        "resolves_on_default": resolves_default,
        "attributed_to": attributed_to,
        "source_tier": source_tier,
        "origin_kind": origin_kind,
        "origin_ref": origin_ref,
        "origin_evidence_id": origin_evidence_id,
        "quote": quote,
        "subject_symbols_json": json.dumps(sorted(set(subject_symbols))),
        "theme_ids_json": json.dumps(sorted(set(theme_ids))),
        "first_seen_at": first_seen_at,
        "review_by": review_by,
    }
    old = _one(db, "SELECT * FROM narrative_records WHERE narrative_id = ?", nid)
    if old:
        return _same_or_refuse(old, wanted, f"record {nid}")
    with db:
        row = _insert(db, "narrative_records", {**wanted, "created_at": now()})
        append_event(
            db,
            nid,
            kind="created",
            to_status="unverified",
            decided_by=decided_by,
            at=first_seen_at,
            basis={"origin_ref": origin_ref, "origin_evidence_id": origin_evidence_id},
        )
    return row


# ---------------------------------------------------------------- events and the fold
def events_of(db, nid):
    return [dict(r) for r in db.execute("SELECT * FROM narrative_events WHERE narrative_id = ? ORDER BY seq", (nid,))]


def fold_status(events, as_of=None):
    """Status at as_of: to_status of the last created/status/override event with at <= as_of. None before creation."""
    status = None
    for e in events:
        if (as_of is None or e["at"] <= as_of) and e["kind"] in STATUS_EVENT_KINDS:
            status = e["to_status"]
    return status


def check_transition(from_status, to_status, decided_by):
    if decided_by not in TRANSITIONS.get((from_status, to_status), ()):
        raise LedgerError(f"{decided_by} may not move a record from {from_status} to {to_status}")


def append_event(
    db, nid, *, kind, to_status, decided_by, at, basis, rule=None, jev=None, drafter_verdict=None, note=None
):
    """The only way a record's status changes. Raises LedgerError on any move TRANSITIONS does not allow."""
    if decided_by not in EVENT_KINDS.get(kind, ()):
        raise LedgerError(f"{kind} events are not written by {decided_by}")
    if to_status not in STATUSES:
        raise LedgerError(f"status must be one of {'|'.join(STATUSES)}")
    if kind == "override" and not str(note or "").strip():
        raise LedgerError("an owner override needs a non-empty note")
    rec = _one(db, "SELECT claim_type FROM narrative_records WHERE narrative_id = ?", nid)
    if rec is None:
        raise LedgerError(f"no narrative record {nid}")
    events = events_of(db, nid)
    at = iso(at)
    current = fold_status(events)
    if kind == "created":
        if events or to_status != "unverified":
            raise LedgerError("only seq 0 is a created event, and it starts unverified")
    else:
        if at < events[-1]["at"]:
            raise LedgerError(f"event at {at} is before the previous event of {nid} ({events[-1]['at']})")
        if kind == "corroborated":
            if rec["claim_type"] != "forecast" or to_status != current:
                raise LedgerError("corroborated applies to a forecast and keeps its status")
        else:
            check_transition(current, to_status, decided_by)
    with db:
        return _insert(
            db,
            "narrative_events",
            {
                "narrative_id": nid,
                "seq": len(events),
                "at": at,
                "recorded_at": now(),
                "kind": kind,
                "from_status": current,
                "to_status": to_status,
                "rule": rule,
                "basis_json": json.dumps(basis, ensure_ascii=False, sort_keys=True),
                "jev_json": json.dumps(jev, sort_keys=True) if jev is not None else None,
                "drafter_verdict": drafter_verdict,
                "decided_by": decided_by,
                "note": note.strip() if note else None,
            },
        )


def add_check(db, row):
    """One narrative_checks row per (revision_id, narrative_id); a second write of the same key is refused by the key."""
    with db:
        return _insert(db, "narrative_checks", row)


def add_link(db, row):
    """One claim_theme_links row per (revision_id, claim_index, theme_id). A key already written is kept as it is, so a
    rerun writes nothing."""
    key = (row["revision_id"], row["claim_index"], row["theme_id"])
    old = _one(db, "SELECT * FROM claim_theme_links WHERE revision_id = ? AND claim_index = ? AND theme_id = ?", *key)
    if old:
        return old
    with db:
        return _insert(db, "claim_theme_links", row)


def set_status(db, nid, to_status, note, at=None):
    return append_event(
        db,
        nid,
        kind="override",
        to_status=to_status,
        decided_by="owner",
        at=at or now(),
        basis={"override": True},
        note=note,
    )


def sweep_stale(db, as_of):
    """unverified records whose review_by <= as_of -> stale (at = review_by). partially_verified never goes stale.
    Re-running finds nothing left to do."""
    as_of = iso(as_of)
    moved = []
    with db:
        for r in db.execute(
            "SELECT narrative_id, review_by FROM narrative_records WHERE review_by <= ? ORDER BY narrative_id", (as_of,)
        ).fetchall():
            events = events_of(db, r["narrative_id"])
            if fold_status(events) != "unverified":
                continue
            at = max(r["review_by"], events[-1]["at"])
            append_event(
                db,
                r["narrative_id"],
                kind="status",
                to_status="stale",
                decided_by="sweep",
                at=at,
                basis={"review_by": r["review_by"], "as_of": as_of},
            )
            moved.append(r["narrative_id"])
    return moved


# ---------------------------------------------------------------- export
def _clip(text, n=EXPORT_QUOTE_CHARS):
    return text if len(text) <= n else text[: n - 1] + "…"


def export(db, as_of):
    """Point-in-time view at as_of, sorted, without full source texts. Same ledger + same as_of -> same bytes."""
    as_of = iso(as_of)

    def rows(sql, *args):
        return [dict(r) for r in db.execute(sql, args)]

    themes = rows("SELECT * FROM themes WHERE approved_at <= ? ORDER BY theme_id", as_of)
    for t in themes:
        t["keywords"] = json.loads(t.pop("keywords_json"))
        t["members"] = rows(
            "SELECT symbol, role, tw_listed, approved_at, approval_ref, content_sha256 FROM theme_members "
            "WHERE theme_id = ? AND approved_at <= ? ORDER BY symbol",
            t["theme_id"],
            as_of,
        )
    edges = rows("SELECT * FROM theme_edges WHERE approved_at <= ? ORDER BY from_theme, to_theme, relation", as_of)
    records = rows(
        "SELECT * FROM narrative_records WHERE first_seen_at <= ? ORDER BY first_seen_at, narrative_id", as_of
    )
    for r in records:
        r["subject_symbols"] = json.loads(r.pop("subject_symbols_json"))
        r["theme_ids"] = json.loads(r.pop("theme_ids_json"))
        r["quote"] = _clip(r["quote"])
        r["events"] = [e for e in events_of(db, r["narrative_id"]) if e["at"] <= as_of]
        for e in r["events"]:
            e["basis"] = json.loads(e.pop("basis_json"))
            jev = e.pop("jev_json")
            e["jev"] = json.loads(jev) if jev else None
            del e["narrative_id"]
        r["status"] = fold_status(r["events"])
        r["last_event_at"] = r["events"][-1]["at"]
    checks = rows("SELECT * FROM narrative_checks WHERE checked_at <= ? ORDER BY revision_id, narrative_id", as_of)
    links = rows(
        "SELECT * FROM claim_theme_links WHERE created_at <= ? ORDER BY revision_id, claim_index, theme_id", as_of
    )
    doc = {
        "algorithm": EXPORT_ALGO,
        "as_of": as_of,
        "themes": themes,
        "theme_edges": edges,
        "records": records,
        "narrative_checks": checks,
        "claim_theme_links": links,
    }
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


# ---------------------------------------------------------------- CLI
def _cmd(a, db):
    if a.cmd == "theme-add":
        r = add_theme(db, a.id, a.title, a.kind, a.keyword, a.approval_ref, a.proposal_id, a.at)
        return f"theme {r['theme_id']} {r['content_sha256']}"
    if a.cmd == "member-add":
        r = add_member(db, a.theme, a.symbol, a.role, a.approval_ref, a.at)
        return f"member {r['theme_id']} {r['symbol']} {r['role']} tw_listed={r['tw_listed']}"
    if a.cmd == "edge-add":
        r = add_edge(db, a.from_theme, a.to_theme, a.relation, a.approval_ref, a.at)
        return f"edge {r['from_theme']} {r['relation']} {r['to_theme']}"
    if a.cmd == "record-add":
        r = add_record(
            db,
            claim=a.claim,
            claim_type=a.claim_type,
            attributed_to=a.attributed_to,
            source_tier=a.source_tier,
            origin_kind="owner",
            origin_ref=a.origin_ref,
            origin_evidence_id=a.origin_evidence_id,
            quote=a.quote,
            source_text=Path(a.source_file).read_text(encoding="utf-8"),
            subject_symbols=a.subject,
            theme_ids=a.theme,
            first_seen_at=a.first_seen_at,
            resolves_on=a.resolves_on,
            decided_by="owner",
        )
        return f"{r['narrative_id']} review_by={r['review_by']}"
    if a.cmd == "set-status":
        e = set_status(db, a.id, a.to, a.note, a.at)
        return f"{a.id} seq={e['seq']} {e['from_status']} -> {e['to_status']}"
    if a.cmd == "sweep-stale":
        moved = sweep_stale(db, a.as_of)
        return "\n".join([f"stale {n}" for n in moved] + [f"swept {len(moved)}"])
    text = export(db, a.as_of)
    if not a.out:
        return text.rstrip("\n")
    Path(a.out).write_text(text, encoding="utf-8")
    return f"wrote {a.out}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name):
        p = sub.add_parser(name)
        p.add_argument("--ledger", required=True)
        return p

    p = cmd("theme-add")
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--kind", required=True, choices=THEME_KINDS)
    p.add_argument("--keyword", action="append", default=[])
    p.add_argument("--approval-ref", required=True)
    p.add_argument("--proposal-id")
    p.add_argument("--at")
    p = cmd("member-add")
    p.add_argument("--theme", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--role", required=True, choices=MEMBER_ROLES)
    p.add_argument("--approval-ref", required=True)
    p.add_argument("--at")
    p = cmd("edge-add")
    p.add_argument("--from", dest="from_theme", required=True)
    p.add_argument("--to", dest="to_theme", required=True)
    p.add_argument("--relation", required=True, choices=EDGE_RELATIONS)
    p.add_argument("--approval-ref", required=True)
    p.add_argument("--at")
    p = cmd("record-add")
    p.add_argument("--claim", required=True)
    p.add_argument("--claim-type", required=True, choices=CLAIM_TYPES)
    p.add_argument("--resolves-on")
    p.add_argument("--attributed-to", required=True)
    p.add_argument("--source-tier", required=True, choices=SOURCE_TIERS)
    p.add_argument("--origin-ref", required=True)
    p.add_argument("--origin-evidence-id", required=True)
    p.add_argument("--quote", required=True)
    p.add_argument("--source-file", required=True, help="text the quote must appear in verbatim")
    p.add_argument("--subject", action="append", required=True)
    p.add_argument("--theme", action="append", required=True)
    p.add_argument("--first-seen-at", required=True)
    p = cmd("set-status")
    p.add_argument("--id", required=True)
    p.add_argument("--to", required=True, choices=STATUSES)
    p.add_argument("--note", required=True)
    p.add_argument("--at")
    p = cmd("sweep-stale")
    p.add_argument("--as-of", required=True)
    p = cmd("export")
    p.add_argument("--as-of", required=True)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    db = connect(a.ledger)
    try:
        print(_cmd(a, db))
    except LedgerError as e:
        raise SystemExit(f"refused: {e}") from None
    finally:
        db.close()


if __name__ == "__main__":
    main()
