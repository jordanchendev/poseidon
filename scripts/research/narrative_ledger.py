#!/usr/bin/env python3
"""Theme ledger and narrative ledger (design local_dev/design/2026-10-29-research-theme-and-narrative-ledger.md §2-§4).

Nine tables in the runner's SQLite ledger (the file research_theme.py and settle_theme.py use), plus
narrative_suggestions for narrative_apply confirm mode and, for the owner's theme and member approvals (§9 step 9),
member_proposals, theme_proposal_claims (the revision claims a theme proposal cites, linked when it is accepted) and
held_records (intake records that wait for a pending proposal), and claim_link_approvals (the owner's approval of a
claim_theme_links row added with link-add, §5.3). Every row carries content_sha256 = Poseidon
content_sha256 of its other columns; the rows with a lifecycle (materials, theme_proposals, member_proposals,
narrative_suggestions, held_records) are rehashed when it moves.
A narrative's status is the fold of its append-only events; TRANSITIONS says who may move a record from one status to
another. Event `at` never goes back in time: a caller whose evidence predates the record's last event passes
max(evidence available_at, last event at).

  theme-add   --ledger L --id ID --title T --kind K [--keyword W ...] --approval-ref REF [--proposal-id P] [--at T]
  theme-keywords --ledger L --id ID --add W [W ...] --approval-ref REF [--at T]   recall keywords (§5.3) of a theme
  member-add  --ledger L --theme ID --symbol S --role R --approval-ref REF [--at T]
  edge-add    --ledger L --from ID --to ID --relation related|part_of --approval-ref REF [--at T]
  link-add    --ledger L --revision R --claim N --theme ID [--about S ...] --approval-ref REF [--at T]
              tag claims[N] of a completed revision in this ledger with a theme (an untagged_hits claim, §5.3)
  record-add  --ledger L --claim C --claim-type state|forecast [--resolves-on YYYY-MM-DD] --attributed-to A
              --source-tier T --origin-ref R --origin-evidence-id E --quote Q --source-file F --subject S [...]
              --theme ID [...] --first-seen-at T
  set-status  --ledger L --id NR --to STATUS --note TEXT [--at T]      owner override
  sweep-stale --ledger L --as-of T          unverified records past review_by -> stale
  export      --ledger L --as-of T [--out F]  deterministic narrative_export.json (written tmp + rename)

Symbols: Taiwan-listed bare ('2408'), everything else '<EXCHANGE>:<TICKER>' ('NASDAQ:MU', 'TSE:3110').
Approval rows (themes, members, edges, owner links) need a non-empty approval_ref; agents never pick theme ids.
Needs the Poseidon worktree on PYTHONPATH (content_sha256).
"""

import argparse
import datetime
import json
import re
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_rules
from assemble import TW_SYMBOL
from claim_precheck import best_evidence, evidence_tier
from record_rules import CLAIM_TYPES

from poseidon.core.canonical import content_sha256

THEME_KINDS = ("bottleneck", "tech_transition", "demand_driver", "policy")
MEMBER_ROLES = ("constrained", "supplier", "beneficiary", "customer", "competitor")
EDGE_RELATIONS = ("related", "part_of")
SOURCE_TIERS = ("official", "sell-side", "media", "kol", "our-inference")
STATUSES = ("unverified", "partially_verified", "verified", "refuted", "stale")
# Event kind -> who may write it. 'corroborated' never changes the status.
# A rule-table change the owner confirmed in Telegram (narrative_apply confirm) is a status or corroborated event
# decided_by owner, with the rule that suggested it and basis.approval_ref naming the decision.
EVENT_KINDS = {
    "created": {"rule", "owner"},
    "status": {"rule", "sweep", "owner"},
    "override": {"owner"},
    "corroborated": {"rule", "owner"},
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

THEME_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_KEYWORD_CHARS = 40
FOREIGN_SYMBOL = re.compile(r"^[A-Z]{2,8}:[A-Z0-9][A-Z0-9.\-]*$")

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
  decided_at TEXT, decided_ref TEXT, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL,
  keywords_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS theme_keyword_approvals (
  theme_id TEXT NOT NULL REFERENCES themes, keyword TEXT NOT NULL,
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (theme_id, keyword));
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
CREATE TABLE IF NOT EXISTS narrative_suggestions (
  suggestion_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL,
  narrative_id TEXT NOT NULL REFERENCES narrative_records,
  kind TEXT NOT NULL CHECK (kind IN ('status','corroborated')), from_status TEXT NOT NULL, to_status TEXT NOT NULL,
  rule TEXT NOT NULL, at TEXT NOT NULL, basis_json TEXT NOT NULL, jev_json TEXT NOT NULL, drafter_verdict TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('suggested','confirmed','rejected')), created_at TEXT NOT NULL,
  decided_at TEXT, decision_ref TEXT, event_seq INTEGER, content_sha256 TEXT NOT NULL,
  UNIQUE (revision_id, narrative_id));
CREATE TABLE IF NOT EXISTS member_proposals (
  proposal_id TEXT PRIMARY KEY, theme_id TEXT NOT NULL REFERENCES themes, symbol TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('constrained','supplier','beneficiary','customer','competitor')),
  why TEXT NOT NULL, source_ref TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed','accepted','rejected')),
  decided_at TEXT, decided_ref TEXT, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS theme_proposal_claims (
  proposal_id TEXT NOT NULL REFERENCES theme_proposals, revision_id TEXT NOT NULL, claim_index INTEGER NOT NULL,
  about_symbols_json TEXT NOT NULL, claim_kind TEXT NOT NULL CHECK (claim_kind IN ('fact','inference')),
  evidence_tier TEXT NOT NULL, event_time TEXT NOT NULL, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL,
  PRIMARY KEY (proposal_id, revision_id, claim_index));
CREATE TABLE IF NOT EXISTS claim_link_approvals (
  revision_id TEXT NOT NULL, claim_index INTEGER NOT NULL, theme_id TEXT NOT NULL REFERENCES themes,
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (revision_id, claim_index, theme_id));
CREATE TABLE IF NOT EXISTS held_records (
  narrative_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, source_text TEXT NOT NULL, waits_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('held','written','dropped')), reason TEXT, decided_at TEXT,
  created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL);
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
    _migrate(db)
    return db


def _migrate(db):
    """A ledger made before theme proposals carried keywords gets the column, and its proposal rows are rehashed so
    every content_sha256 still covers all other columns."""
    if "keywords_json" in {r[1] for r in db.execute("PRAGMA table_info(theme_proposals)")}:
        return
    with db:
        db.execute("ALTER TABLE theme_proposals ADD COLUMN keywords_json TEXT NOT NULL DEFAULT '[]'")
        for r in db.execute("SELECT * FROM theme_proposals").fetchall():
            row = {k: v for k, v in dict(r).items() if k != "content_sha256"}
            db.execute(
                "UPDATE theme_proposals SET content_sha256 = ? WHERE proposal_id = ?",
                (content_sha256(row), r["proposal_id"]),
            )


def _insert(db, table, row):
    row = dict(row, content_sha256=content_sha256(row))
    db.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    return row


def _update(db, table, key, changes):
    """Set `changes` on the row whose columns match `key` and rehash it; returns the new row."""
    where = " AND ".join(f"{k} = ?" for k in key)
    row = _one(db, f"SELECT * FROM {table} WHERE {where}", *key.values())
    if row is None:
        raise LedgerError(f"no {table} row {key}")
    row = {k: v for k, v in row.items() if k != "content_sha256"} | changes
    row["content_sha256"] = content_sha256(row)
    with db:
        db.execute(
            f"UPDATE {table} SET {', '.join(f'{k} = ?' for k in row)} WHERE {where}",
            [*row.values(), *key.values()],
        )
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
def keyword_list(words):
    """Stripped, non-empty, first occurrence kept. A keyword is a recall hint (§5.3), never a link."""
    out = list(dict.fromkeys(str(w).strip() for w in words if str(w).strip()))
    long = [w for w in out if len(w) > MAX_KEYWORD_CHARS]
    if long:
        raise LedgerError(f"keywords longer than {MAX_KEYWORD_CHARS} characters: {long}")
    return out


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
        "keywords_json": json.dumps(keyword_list(keywords), ensure_ascii=False),
        "status": "active",
        "proposal_id": proposal_id,
    }
    approval = _approval(approval_ref, at)
    old = _one(db, "SELECT * FROM themes WHERE theme_id = ?", theme_id)
    if old:
        return _same_or_refuse(old, wanted, f"theme {theme_id}")
    with db:
        return _insert(db, "themes", {**wanted, **approval})


def add_theme_keywords(db, theme_id, keywords, approval_ref, at=None):
    """The owner adds recall keywords to an existing theme: appended to keywords_json (row rehashed), one
    theme_keyword_approvals row per new keyword with its approval. -> (theme row, keywords added). A keyword the theme
    already has is a no-op."""
    approval = _approval(approval_ref, at)
    theme = _one(db, "SELECT * FROM themes WHERE theme_id = ?", theme_id)
    if theme is None:
        raise LedgerError(f"no theme {theme_id!r}")
    have = json.loads(theme["keywords_json"])
    added = [w for w in keyword_list(keywords) if w not in have]
    if not added:
        return theme, []
    with db:  # one transaction: _update's own commit closes it
        for w in added:
            _insert(db, "theme_keyword_approvals", {"theme_id": theme_id, "keyword": w, **approval})
        theme = _update(
            db, "themes", {"theme_id": theme_id}, {"keywords_json": json.dumps(have + added, ensure_ascii=False)}
        )
    return theme, added


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


def add_owner_link(db, revision_id, claim_index, theme_id, approval_ref, about=None, at=None):
    """The owner tags claims[claim_index] of a completed revision in this ledger with a theme (a §5.3 untagged_hits
    claim): a claim_theme_links row with source owner, its tier and event_time read from the revision's manifest as the
    drafter's are (claim_precheck.best_evidence), and a claim_link_approvals row with the approval, in one
    transaction. about: the companies the claim is about, default its about_symbols. A key already linked is a no-op,
    or refused when about names other companies."""
    approval = _approval(approval_ref, at)
    _active_theme(db, theme_id)
    try:
        rev = str(uuid.UUID(_rev_key(revision_id)))
    except ValueError:
        raise LedgerError(f"revision id {revision_id!r} is not a UUID") from None
    key = (rev, claim_index, theme_id)
    old = _one(db, "SELECT * FROM claim_theme_links WHERE revision_id = ? AND claim_index = ? AND theme_id = ?", *key)
    if old and about:
        wanted = {"about_symbols_json": json.dumps(sorted(set(about)))}
        return _same_or_refuse(old, wanted, f"link {rev} claims[{claim_index}] {theme_id}")
    if old:
        return old
    found = None
    if {"research_revisions", "data_manifests"} <= _tables(db):
        found = _one(
            db,
            "SELECT r.research_json, m.payload_json FROM research_revisions r JOIN data_manifests m "
            "ON m.id = r.manifest_id WHERE r.status = 'completed' AND replace(lower(r.id), '-', '') = ?",
            _rev_key(rev),
        )
    if found is None:
        raise LedgerError(f"no completed revision {rev} with its manifest in this ledger")
    claims = json.loads(found["research_json"]).get("claims") or []
    if not 0 <= claim_index < len(claims):
        raise LedgerError(f"revision {rev} has no claims[{claim_index}]")
    claim = claims[claim_index]
    by_id = {e["id"]: e for e in json.loads(found["payload_json"]).get("evidence") or []}
    if not any(e in by_id for e in claim.get("evidence_ids") or []):
        raise LedgerError(f"claims[{claim_index}] cites no evidence of the revision's manifest")
    about = sorted(set(about or claim.get("about_symbols") or []))
    if not about:
        raise LedgerError(f"claims[{claim_index}] has no about_symbols; name the companies it is about")
    for s in about:
        symbol_is_tw(s)
    best = best_evidence(claim, by_id)
    with db:
        _insert(
            db,
            "claim_link_approvals",
            {"revision_id": rev, "claim_index": claim_index, "theme_id": theme_id, **approval},
        )
        return _insert(
            db,
            "claim_theme_links",
            {
                "revision_id": rev,
                "claim_index": claim_index,
                "theme_id": theme_id,
                "about_symbols_json": json.dumps(about),
                "claim_kind": claim["kind"],
                "evidence_tier": evidence_tier(best),
                "event_time": iso(best["event_time"]),
                "source": "owner",
                "created_at": approval["approved_at"],
            },
        )


# ---------------------------------------------------------------- records
def narrative_id(origin_ref, quote, claim):
    return "nr-" + content_sha256([origin_ref, quote, claim])[:10]


def _date_plus(day, days):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


def review_dates(claim_type, first_seen_at, resolves_on):
    """(resolves_on, resolves_on_default, review_by) per D6 for a record record_rules accepted. review_by is the end
    of that UTC day."""
    seen_day = first_seen_at[:10]
    if claim_type == "state":
        return None, 0, _date_plus(seen_day, STATE_REVIEW_DAYS) + "T23:59:59Z"
    default = 0
    if not resolves_on:
        resolves_on, default = _date_plus(seen_day, FORECAST_DEFAULT_RESOLVE_DAYS), 1
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
    """Writes the record and its seq-0 'created' event. The quote must be verbatim in source_text and the record legal
    (record_rules.record_problems: atomic with no causal link, its numbers in the quote, claim_type / resolves_on).
    Each subject is an approved member of one of the record's themes; a record that needs a pending proposal waits in
    held_records (hold_record)."""
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
    problems = record_rules.record_problems(claim, quote, claim_type, resolves_on, check_period=False)
    if problems:
        raise LedgerError("; ".join(p.message for p in problems))
    if not subject_symbols or not theme_ids:
        raise LedgerError("a record needs at least one subject symbol and one theme")
    for t in theme_ids:
        _active_theme(db, t)
    for s in subject_symbols:
        symbol_is_tw(s)
        if not any(
            _one(db, "SELECT 1 FROM theme_members WHERE theme_id = ? AND symbol = ?", t, s) for t in set(theme_ids)
        ):
            raise LedgerError(f"subject {s} is not an approved member of {', '.join(sorted(set(theme_ids)))}")
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


def append_event(db, nid, **event):
    """The only way a record's status changes. Raises LedgerError on any move TRANSITIONS does not allow."""
    with db:
        return _insert_event(db, nid, **event)


def _insert_event(
    db, nid, *, kind, to_status, decided_by, at, basis, rule=None, jev=None, drafter_verdict=None, note=None
):
    """append_event without the commit, for a caller that writes the event and another row in one transaction."""
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


def set_status(db, nid, to_status, note, at=None, approval_ref=None):
    """Owner override. approval_ref names the Telegram decision when the override came through narrative-set."""
    return append_event(
        db,
        nid,
        kind="override",
        to_status=to_status,
        decided_by="owner",
        at=at or now(),
        basis={"override": True, **({"approval_ref": approval_ref} if approval_ref else {})},
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


# ---------------------------------------------------------------- materials (design §2.2, §6.2)
MATERIAL_TIERS = ("sell-side", "media", "kol")


def add_material(db, *, material_id, received_at, source_tier, author, url, text_sha256, raw_path):
    """The material as received (status received). The same id with the same content is a no-op, so a rerun after a
    crash finds the row it wrote; a different text under the same id is refused."""
    if source_tier not in MATERIAL_TIERS:
        raise LedgerError(f"material tier must be one of {'|'.join(MATERIAL_TIERS)}")
    wanted = {
        "material_id": material_id,
        "received_at": iso(received_at),
        "source_tier": source_tier,
        "author": author,
        "url": url,
        "text_sha256": text_sha256,
        "raw_path": raw_path,
    }
    old = _one(db, "SELECT * FROM materials WHERE material_id = ?", material_id)
    if old:
        return _same_or_refuse(old, wanted, f"material {material_id}")
    with db:
        return _insert(db, "materials", {**wanted, "manifest_sha256": None, "status": "received", "note": None})


def set_material_status(db, material_id, status, *, note=None, manifest_sha256=None):
    changes = {"status": status, "note": note}
    if manifest_sha256:
        changes["manifest_sha256"] = manifest_sha256
    return _update(db, "materials", {"material_id": material_id}, changes)


def _rev_key(revision_id):
    """research_theme.py stores a revision id without dashes (SQLAlchemy Uuid on SQLite); the ledger with them."""
    return str(revision_id).replace("-", "").lower()


def _tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


# ---------------------------------------------------------------- CLI
def _cmd(a, db):
    if a.cmd == "theme-add":
        r = add_theme(db, a.id, a.title, a.kind, a.keyword, a.approval_ref, a.proposal_id, a.at)
        return f"theme {r['theme_id']} {r['content_sha256']}"
    if a.cmd == "theme-keywords":
        r, added = add_theme_keywords(db, a.id, a.add, a.approval_ref, a.at)
        return f"theme {r['theme_id']} keywords {json.loads(r['keywords_json'])} added {added} {r['content_sha256']}"
    if a.cmd == "member-add":
        r = add_member(db, a.theme, a.symbol, a.role, a.approval_ref, a.at)
        return f"member {r['theme_id']} {r['symbol']} {r['role']} tw_listed={r['tw_listed']}"
    if a.cmd == "edge-add":
        r = add_edge(db, a.from_theme, a.to_theme, a.relation, a.approval_ref, a.at)
        return f"edge {r['from_theme']} {r['relation']} {r['to_theme']}"
    if a.cmd == "link-add":
        r = add_owner_link(db, a.revision, a.claim, a.theme, a.approval_ref, a.about, a.at)
        about = json.loads(r["about_symbols_json"])
        return f"link {r['revision_id']} claims[{r['claim_index']}] {r['theme_id']} {about} {r['evidence_tier']} {r['source']}"
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
    import narrative_export  # the export command only; narrative_export builds on this module

    if not a.out:
        return narrative_export.export(db, a.as_of).rstrip("\n")
    return f"wrote {narrative_export.write_export(db, a.as_of, a.out)}"


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
    p = cmd("theme-keywords")
    p.add_argument("--id", required=True)
    p.add_argument("--add", nargs="+", required=True, metavar="KEYWORD")
    p.add_argument("--approval-ref", required=True)
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
    p = cmd("link-add")
    p.add_argument("--revision", required=True, help="revision id (with or without dashes)")
    p.add_argument("--claim", required=True, type=int, metavar="INDEX", help="claims[INDEX] of that revision")
    p.add_argument("--theme", required=True)
    p.add_argument("--about", action="append", metavar="SYMBOL", help="default: the claim's about_symbols")
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
