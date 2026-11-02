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
from claim_precheck import OFFICIAL_TIERS, evidence_tier
from record_rules import CLAIM_TYPES

from poseidon.decision_loop.manifest import content_sha256

EXPORT_ALGO = "narrative-export-v2"
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
EXPORT_QUOTE_CHARS = 120  # D7: the export leaves stormtrooper for the vault
SIGNAL_WINDOW_DAYS = 180  # D11: the cross-company signal window (§5.2)
RELAYED_TIERS = ("sell-side", "media", "kol")

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
    drafter's are (narrative_check.best_evidence), and a claim_link_approvals row with the approval, in one
    transaction. about: the companies the claim is about, default its about_symbols. A key already linked is a no-op,
    or refused when about names other companies."""
    from narrative_check import best_evidence

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


# ---------------------------------------------------------------- proposals (design §3.3, §9 step 9)
# The owner approves a theme or a member in Telegram; agents only propose. A proposal moves proposed -> accepted |
# rejected once; accepting or rejecting it again the same way is a no-op, the other way is refused.
THEME_PROPOSAL_ID = re.compile(r"^T\d{8}-\d{1,4}$")
MEMBER_PROPOSAL_ID = re.compile(r"^M\d{8}-\d{1,4}$")
PROPOSAL_TABLES = {"T": "theme_proposals", "M": "member_proposals"}


def _norm_title(text):
    return re.sub(r"[\s\W_]+", "", str(text)).casefold()


def theme_duplicate(db, suggested_id, title):
    """'theme:<id>' when an active theme has this id or normalised title or one of its keywords is in the title, or an
    accepted proposal had this suggested id or title; 'proposal:<id>' when an open proposal has it; 'rejected:<id>' when
    the owner rejected a proposal with it (never proposed again); else None."""
    norm = _norm_title(title)
    for t in db.execute("SELECT theme_id, title, keywords_json FROM themes WHERE status = 'active' ORDER BY theme_id"):
        words = [_norm_title(k) for k in json.loads(t["keywords_json"])]
        if suggested_id == t["theme_id"] or norm == _norm_title(t["title"]) or any(w and w in norm for w in words):
            return "theme:" + t["theme_id"]
    same = [
        p
        for p in db.execute("SELECT proposal_id, suggested_id, title, status FROM theme_proposals ORDER BY proposal_id")
        if suggested_id == p["suggested_id"] or norm == _norm_title(p["title"])
    ]
    for status in ("proposed", "accepted", "rejected"):
        for p in same:
            if p["status"] != status:
                continue
            if status == "proposed":
                return "proposal:" + p["proposal_id"]
            if status == "rejected":
                return "rejected:" + p["proposal_id"]
            t = _one(db, "SELECT theme_id, status FROM themes WHERE proposal_id = ?", p["proposal_id"])
            if t and t["status"] == "active":
                return "theme:" + t["theme_id"]
    return None


def member_duplicate(db, theme_id, symbol):
    """'member:<theme_id>' when the symbol is already an approved member of that theme, 'proposal:<id>' for an open
    proposal of the same member, 'rejected:<id>' when the owner rejected one (never proposed again), else None."""
    if _one(db, "SELECT 1 FROM theme_members WHERE theme_id = ? AND symbol = ?", theme_id, symbol):
        return "member:" + theme_id
    for status, word in (("proposed", "proposal"), ("rejected", "rejected")):
        p = _one(
            db,
            "SELECT proposal_id FROM member_proposals WHERE theme_id = ? AND symbol = ? AND status = ? "
            "ORDER BY proposal_id LIMIT 1",
            theme_id,
            symbol,
            status,
        )
        if p:
            return f"{word}:{p['proposal_id']}"
    return None


def _new_proposal_id(db, letter, at):
    prefix = letter + at[:10].replace("-", "") + "-"
    table = PROPOSAL_TABLES[letter]
    taken = [r[0] for r in db.execute(f"SELECT proposal_id FROM {table} WHERE proposal_id LIKE ?", (prefix + "%",))]
    return f"{prefix}{1 + max([int(p[len(prefix) :]) for p in taken] or [0])}"


def add_theme_proposal(db, *, suggested_id, title, kind, why, source_ref, proposed_members, at, keywords=()):
    """A theme proposal ('T' + YYYYMMDD + '-' + n, the day of `at`) with the recall keywords it suggests (accepting it
    copies them). The same source_ref and suggested_id again returns the row already written, so a rerun files
    nothing twice. The caller has ruled out duplicates (theme_duplicate)."""
    if kind not in THEME_KINDS:
        raise LedgerError(f"kind must be one of {'|'.join(THEME_KINDS)}")
    old = _one(db, "SELECT * FROM theme_proposals WHERE source_ref = ? AND suggested_id = ?", source_ref, suggested_id)
    if old:
        return old
    at = iso(at)
    with db:
        return _insert(
            db,
            "theme_proposals",
            {
                "proposal_id": _new_proposal_id(db, "T", at),
                "suggested_id": suggested_id,
                "title": title,
                "kind": kind,
                "why": why,
                "source_ref": source_ref,
                "proposed_members_json": json.dumps(proposed_members, ensure_ascii=False, sort_keys=True),
                "keywords_json": json.dumps(keyword_list(keywords), ensure_ascii=False),
                "status": "proposed",
                "decided_at": None,
                "decided_ref": None,
                "created_at": at,
            },
        )


def add_member_proposal(db, *, theme_id, symbol, role, why, source_ref, at):
    """A member proposal ('M' + YYYYMMDD + '-' + n) for an active theme. The same source_ref, theme and symbol again
    returns the row already written. The caller has ruled out duplicates (member_duplicate)."""
    if role not in MEMBER_ROLES:
        raise LedgerError(f"role must be one of {'|'.join(MEMBER_ROLES)}")
    symbol_is_tw(symbol)
    _active_theme(db, theme_id)
    old = _one(
        db,
        "SELECT * FROM member_proposals WHERE source_ref = ? AND theme_id = ? AND symbol = ?",
        source_ref,
        theme_id,
        symbol,
    )
    if old:
        return old
    at = iso(at)
    with db:
        return _insert(
            db,
            "member_proposals",
            {
                "proposal_id": _new_proposal_id(db, "M", at),
                "theme_id": theme_id,
                "symbol": symbol,
                "role": role,
                "why": why,
                "source_ref": source_ref,
                "status": "proposed",
                "decided_at": None,
                "decided_ref": None,
                "created_at": at,
            },
        )


def add_proposal_claim(db, row):
    """The revision claim a theme proposal cites, with the fields its claim_theme_links row will carry once the owner
    accepts the theme. One row per (proposal, revision, claim); a key already written is kept."""
    key = (row["proposal_id"], row["revision_id"], row["claim_index"])
    old = _one(
        db,
        "SELECT * FROM theme_proposal_claims WHERE proposal_id = ? AND revision_id = ? AND claim_index = ?",
        *key,
    )
    if old:
        return old
    with db:
        return _insert(db, "theme_proposal_claims", row)


def _proposal(db, letter, proposal_id):
    p = _one(db, f"SELECT * FROM {PROPOSAL_TABLES[letter]} WHERE proposal_id = ?", proposal_id)
    if p is None:
        raise LedgerError(f"no proposal {proposal_id}")
    return p


def _decide(db, letter, proposal_id, status, decision_ref, decided_at):
    return _update(
        db,
        PROPOSAL_TABLES[letter],
        {"proposal_id": proposal_id},
        {"status": status, "decided_at": iso(decided_at), "decided_ref": decision_ref},
    )


def _member_proposals_of(db, theme_id, members, source_ref, at):
    """Member proposals for a newly approved theme's proposed members, skipping approved, open or rejected ones."""
    filed = []
    for m in members:
        mine = _one(
            db,
            "SELECT * FROM member_proposals WHERE source_ref = ? AND theme_id = ? AND symbol = ?",
            source_ref,
            theme_id,
            m["symbol"],
        )
        if mine or member_duplicate(db, theme_id, m["symbol"]) is None:
            filed.append(
                mine
                or add_member_proposal(
                    db,
                    theme_id=theme_id,
                    symbol=m["symbol"],
                    role=m["role"],
                    why=f"主題提案 {source_ref.split(':', 1)[1]} 提出的成員",
                    source_ref=source_ref,
                    at=at,
                )
            )
    return filed


def accept_theme_proposal(db, proposal_id, theme_id, *, decision_ref, decided_at):
    """The owner accepts a theme proposal under the id the owner typed. Writes the theme (approval_ref
    'telegram:<proposal_id>', the proposal's suggested keywords), claim_theme_links with source owner for the revision
    claims the proposal cites, and a member proposal per proposed member (members need their own approval).
    -> (theme, already, member proposals).
    A theme id outside the grammar, a proposal rejected before or accepted under another id is refused."""
    p = _proposal(db, "T", proposal_id)
    if p["status"] == "rejected":
        raise LedgerError(f"theme proposal {proposal_id} was rejected ({p['decided_ref']})")
    if p["status"] == "accepted":
        t = _one(db, "SELECT * FROM themes WHERE proposal_id = ?", proposal_id)
        if t["theme_id"] != theme_id:
            raise LedgerError(f"theme proposal {proposal_id} was already accepted as {t['theme_id']}")
        return t, True, []
    at = iso(decided_at)
    keywords = json.loads(p["keywords_json"])
    theme = add_theme(db, theme_id, p["title"], p["kind"], keywords, f"telegram:{proposal_id}", proposal_id, at)
    for c in db.execute(
        "SELECT * FROM theme_proposal_claims WHERE proposal_id = ? ORDER BY revision_id, claim_index", (proposal_id,)
    ).fetchall():
        add_link(
            db,
            {
                "revision_id": c["revision_id"],
                "claim_index": c["claim_index"],
                "theme_id": theme_id,
                "about_symbols_json": c["about_symbols_json"],
                "claim_kind": c["claim_kind"],
                "evidence_tier": c["evidence_tier"],
                "event_time": c["event_time"],
                "source": "owner",
                "created_at": at,
            },
        )
    members = _member_proposals_of(
        db, theme_id, json.loads(p["proposed_members_json"]), f"theme-proposal:{proposal_id}", at
    )
    _decide(db, "T", proposal_id, "accepted", decision_ref, at)
    return theme, False, members


def accept_member_proposal(db, proposal_id, *, decision_ref, decided_at):
    """The owner accepts a member proposal: member-add with approval_ref 'telegram:<proposal_id>'.
    -> (member, already). A proposal rejected before is refused."""
    p = _proposal(db, "M", proposal_id)
    if p["status"] == "rejected":
        raise LedgerError(f"member proposal {proposal_id} was rejected ({p['decided_ref']})")
    member = add_member(db, p["theme_id"], p["symbol"], p["role"], f"telegram:{proposal_id}", iso(decided_at))
    if p["status"] == "accepted":
        return member, True
    _decide(db, "M", proposal_id, "accepted", decision_ref, decided_at)
    return member, False


def reject_proposal(db, proposal_id, *, decision_ref, decided_at):
    """The owner rejects a theme (T…) or member (M…) proposal; it is never proposed again. -> (row, already).
    A proposal accepted before is refused."""
    letter = proposal_id[:1]
    if letter not in PROPOSAL_TABLES:
        raise LedgerError(f"no proposal {proposal_id}")
    p = _proposal(db, letter, proposal_id)
    if p["status"] == "rejected":
        return p, True
    if p["status"] == "accepted":
        raise LedgerError(f"proposal {proposal_id} was already accepted ({p['decided_ref']})")
    return _decide(db, letter, proposal_id, "rejected", decision_ref, decided_at), False


# ---------------------------------------------------------------- held records (§9 step 9)
# An intake record whose theme is still a proposal, or whose subject is not yet an approved member of one of its
# themes, waits here until the owner decides every proposal it needs: written then (first_seen_at unchanged), dropped
# with the reason when the only way it had was rejected. Nothing reaches narrative_records without approved themes and
# members of them. A member the owner rejected from theme A is not admitted through a proposal into theme B: the
# record stays under A (owner 2026-10-30; acceptance phase 2a, M20261029-1).
def record_waits(db, theme_refs, subjects):
    """-> (theme ids, open proposal ids the record still needs, reasons it can never be written, needs). A theme ref is
    an active theme id or a theme proposal id. A subject must be an approved member of one of the record's own themes;
    until it is, it waits for an open member proposal of it into one of them, or for an open theme proposal among the
    record's refs that names it (whose accept files that member proposal). Proposals into other themes do not count.
    needs: one sorted list per unmet need, of the open proposals that could meet it. Every need must be met; within a
    need one approval is enough."""
    themes, needs, missing, open_refs = [], [], [], {}
    for ref in theme_refs:
        if THEME_PROPOSAL_ID.match(ref):
            p = _one(db, "SELECT status, proposed_members_json FROM theme_proposals WHERE proposal_id = ?", ref)
            if p is None:
                missing.append(f"沒有主題提案 {ref}")
            elif p["status"] == "proposed":
                needs.append([ref])
                open_refs[ref] = {m["symbol"] for m in json.loads(p["proposed_members_json"])}
            elif p["status"] == "rejected":
                missing.append(f"主題提案 {ref} 已被略過")
            else:
                themes.append(_one(db, "SELECT theme_id FROM themes WHERE proposal_id = ?", ref)["theme_id"])
        elif _one(db, "SELECT 1 FROM themes WHERE theme_id = ? AND status = 'active'", ref):
            themes.append(ref)
        else:
            missing.append(f"{ref} 不是有效主題")
    own = sorted(set(themes))
    for s in subjects if own or open_refs else ():  # no theme of its own left: the theme reasons say why
        member_of = {r[0] for r in db.execute("SELECT theme_id FROM theme_members WHERE symbol = ?", (s,))}
        if member_of & set(own):
            continue
        proposals = db.execute(
            "SELECT proposal_id, theme_id, status FROM member_proposals WHERE symbol = ? ORDER BY proposal_id", (s,)
        ).fetchall()
        cover = [p["proposal_id"] for p in proposals if p["status"] == "proposed" and p["theme_id"] in own]
        cover += [ref for ref, names in open_refs.items() if s in names]
        if cover:
            needs.append(sorted(set(cover)))
            continue
        gone = [p["proposal_id"] for p in proposals if p["status"] == "rejected" and p["theme_id"] in own]
        missing.append(
            f"{s} 不是 {'、'.join(own + sorted(open_refs))} 的核准成員"
            + (f"（成員提案 {'、'.join(gone)} 已被略過）" if gone else "，也沒有待決的提案")
        )
    needs = [list(n) for n in sorted({tuple(n) for n in needs})]
    return own, sorted({w for n in needs for w in n}), missing, needs


def waits_text(needs):
    """zh-TW for a notice: 「M… 或 T…」 within a need, 「，以及」 between needs (record_waits)."""
    return "，以及 ".join(" 或 ".join(n) for n in needs)


def hold_record(db, record, source_text, waits, at):
    """Keep an intake record (add_record keyword arguments, theme_ids may name theme proposals) until the proposals it
    waits for are decided. source_text is the text its quote is verbatim in. The same record again returns its row."""
    nid = narrative_id(record["origin_ref"], record["quote"], record["claim"])
    old = _one(db, "SELECT * FROM held_records WHERE narrative_id = ?", nid)
    if old:
        return old
    with db:
        return _insert(
            db,
            "held_records",
            {
                "narrative_id": nid,
                "record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
                "source_text": source_text,
                "waits_json": json.dumps(sorted(waits)),
                "status": "held",
                "reason": None,
                "decided_at": None,
                "created_at": iso(at),
            },
        )


def release_held(db, at):
    """Re-gate every held record against the ledger as it is now: written when nothing it needs is pending, dropped
    with the reason when something it needs was rejected, otherwise kept with its refreshed waits. -> the records
    whose state changed [{narrative_id, status written|dropped|held, reason, waits, waits_text}]. Re-running changes
    nothing."""
    at = iso(at)
    out = []
    for h in db.execute(
        "SELECT * FROM held_records WHERE status = 'held' ORDER BY created_at, narrative_id"
    ).fetchall():
        nid, rec = h["narrative_id"], json.loads(h["record_json"])
        themes, waits, missing, needs = record_waits(db, rec["theme_ids"], rec["subject_symbols"])
        if not missing and not waits:
            try:
                add_record(db, **{**rec, "theme_ids": themes}, source_text=h["source_text"])
            except LedgerError as e:
                missing = [f"寫入被拒：{e}"]
        if missing:
            _update(
                db,
                "held_records",
                {"narrative_id": nid},
                {"status": "dropped", "reason": "；".join(missing), "decided_at": at},
            )
            out.append({"narrative_id": nid, "status": "dropped", "reason": "；".join(missing), "waits": []})
        elif not waits:
            _update(
                db, "held_records", {"narrative_id": nid}, {"status": "written", "waits_json": "[]", "decided_at": at}
            )
            out.append({"narrative_id": nid, "status": "written", "reason": None, "waits": []})
        elif waits != json.loads(h["waits_json"]):
            _update(db, "held_records", {"narrative_id": nid}, {"waits_json": json.dumps(waits)})
            out.append(
                {"narrative_id": nid, "status": "held", "reason": None, "waits": waits, "waits_text": waits_text(needs)}
            )
    return out


# ---------------------------------------------------------------- narrative_apply confirm mode
def suggestion_id(revision_id, nid):
    return "ns-" + content_sha256([revision_id, nid])[:8]


def add_suggestion(db, *, revision_id, nid, kind, from_status, to_status, rule, at, basis, jev, drafter_verdict):
    """The event a rule would have written, held for the owner. The same (revision, record) again returns the row."""
    sid = suggestion_id(revision_id, nid)
    old = _one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    if old:
        return old
    with db:
        return _insert(
            db,
            "narrative_suggestions",
            {
                "suggestion_id": sid,
                "revision_id": revision_id,
                "narrative_id": nid,
                "kind": kind,
                "from_status": from_status,
                "to_status": to_status,
                "rule": rule,
                "at": iso(at),
                "basis_json": json.dumps(basis, ensure_ascii=False, sort_keys=True),
                "jev_json": json.dumps(jev, sort_keys=True),
                "drafter_verdict": drafter_verdict,
                "status": "suggested",
                "created_at": now(),
                "decided_at": None,
                "decision_ref": None,
                "event_seq": None,
            },
        )


def rejected_suggestion(db, nid, kind, to_status):
    """The owner's earlier rejection of the same change to this record (it is not suggested again), or None."""
    return _one(
        db,
        "SELECT * FROM narrative_suggestions WHERE narrative_id = ? AND kind = ? AND to_status = ? "
        "AND status = 'rejected' ORDER BY decided_at LIMIT 1",
        nid,
        kind,
        to_status,
    )


def corroborations(db, nid, revision_id):
    """The record's corroborations from revisions other than revision_id, oldest first: its corroborated events and its
    pending or confirmed corroborated suggestions -> [(name, basis)]. name is the suggestion id when there is one (a
    confirmed suggestion's event carries it too), else '<narrative id>#<event seq>'."""
    rows = [
        (e["recorded_at"], json.loads(e["basis_json"]), f"{nid}#{e['seq']}")
        for e in events_of(db, nid)
        if e["kind"] == "corroborated"
    ]
    rows += [
        (s["created_at"], json.loads(s["basis_json"]), s["suggestion_id"])
        for s in db.execute(
            "SELECT * FROM narrative_suggestions WHERE narrative_id = ? AND kind = 'corroborated' "
            "AND status IN ('suggested', 'confirmed')",
            (nid,),
        )
    ]
    out = {}
    for _at, basis, name in sorted(rows, key=lambda r: r[0]):
        if basis.get("revision_id") != revision_id:
            out.setdefault(basis.get("suggestion_id", name), basis)
    return list(out.items())


def confirm_suggestion(db, sid, *, decision_ref, decided_at, note):
    """Write the suggested event decided_by owner (basis.approval_ref = decision_ref, note = the owner's sentence) at
    max(decided_at, the record's last event at). -> (row, already): a suggestion confirmed before is a no-op.
    The event and the confirmed row are one transaction. A ledger that crashed between them under the old two-commit
    order has the event (basis.suggestion_id) and a suggested row: the row is finished against that event, and no
    second event is written.
    Raises LedgerError for an unknown or rejected suggestion, or a move TRANSITIONS no longer allows."""
    s = _one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    if s is None:
        raise LedgerError(f"no suggestion {sid}")
    if s["status"] == "confirmed":
        return s, True
    if s["status"] == "rejected":
        raise LedgerError(f"suggestion {sid} was rejected ({s['decision_ref']})")
    events = events_of(db, s["narrative_id"])
    ev = next((e for e in events if json.loads(e["basis_json"]).get("suggestion_id") == sid), None)
    already = ev is not None
    if already:
        decision_ref = json.loads(ev["basis_json"])["approval_ref"]
    with db:  # one transaction: _update's own commit closes it
        if not already:
            ev = _insert_event(
                db,
                s["narrative_id"],
                kind=s["kind"],
                to_status=s["to_status"],
                decided_by="owner",
                at=max(iso(decided_at), events[-1]["at"]),
                basis={**json.loads(s["basis_json"]), "suggestion_id": sid, "approval_ref": decision_ref},
                rule=s["rule"],
                jev=json.loads(s["jev_json"]),
                drafter_verdict=s["drafter_verdict"],
                note=note,
            )
        row = _update(
            db,
            "narrative_suggestions",
            {"suggestion_id": sid},
            {
                "status": "confirmed",
                "decided_at": iso(decided_at),
                "decision_ref": decision_ref,
                "event_seq": ev["seq"],
            },
        )
    return row, already


def reject_suggestion(db, sid, *, decision_ref, decided_at):
    """-> (row, already). Raises LedgerError for an unknown suggestion or one already confirmed (use set-status)."""
    s = _one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    if s is None:
        raise LedgerError(f"no suggestion {sid}")
    if s["status"] == "rejected":
        return s, True
    if s["status"] == "confirmed":
        raise LedgerError(f"suggestion {sid} was already confirmed ({s['decision_ref']}); an owner override changes it")
    row = _update(
        db,
        "narrative_suggestions",
        {"suggestion_id": sid},
        {"status": "rejected", "decided_at": iso(decided_at), "decision_ref": decision_ref},
    )
    return row, False


# ---------------------------------------------------------------- export
def _clip(text, n=EXPORT_QUOTE_CHARS):
    return text if len(text) <= n else text[: n - 1] + "…"


# §5.2, computed here and nowhere else: per theme, the companies with hits in the window (since, as_of]. A theme is a
# signal when its hits cover at least two companies.
#   official: fact claims linked to the theme whose evidence is official, by about_symbols and evidence event_time,
#             from documents of at least two issuing companies (link_issuer, a function cross_company_signals
#             registers): a drafter may name another company in about_symbols, so one company's documents alone are
#             never a cross-company signal (acceptance 2026-10-30: one Micron document listed 2408). The issuer is the
#             company, not the URL host (owner 2026-10-30): two companies' filings on sec.gov are two, one company's
#             documents on two hosts are one. A link whose evidence names no issuer is no hit.
#   relayed:  narrative records of a relayed tier (sell-side / media / kol) not refuted at as_of, by subject_symbols and
#             first_seen_at; a theme that is an official signal is not listed again as relayed.
# A symbol the owner rejected from a theme (REJECTED_SQL) is no hit for that theme in either tier, whenever its link or
# record was written; other themes and non-members are unaffected (acceptance phase 2b: M20261029-1 rejected
# NASDAQ:ASML from cleanroom-capacity, then an ASML revision still linked its official facts there).
# The record tier needs this too: add_record admits a subject that is a member of any one of the record's themes, so a
# record tagged with two themes can carry a subject rejected from the other one.
REJECTED_SQL = """
rejected AS (
  SELECT p.theme_id, p.symbol FROM member_proposals p
  WHERE p.status = 'rejected' AND p.decided_at <= :as_of
    AND NOT EXISTS (SELECT 1 FROM theme_members m WHERE m.theme_id = p.theme_id AND m.symbol = p.symbol
                    AND m.approved_at <= :as_of))"""
SIGNAL_SQL = {
    "official": """
WITH"""
    + REJECTED_SQL
    + """,
hits AS (
  SELECT l.theme_id, s.value AS symbol, l.event_time AS at, link_issuer(l.revision_id, l.claim_index) AS issuer
  FROM claim_theme_links l, json_each(l.about_symbols_json) s
  WHERE l.claim_kind = 'fact' AND l.evidence_tier IN (SELECT value FROM json_each(:official))
    AND l.created_at <= :as_of AND l.event_time > :since AND l.event_time <= :as_of
    AND link_issuer(l.revision_id, l.claim_index) IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM rejected rj WHERE rj.theme_id = l.theme_id AND rj.symbol = s.value))
SELECT theme_id, symbol, MIN(at) AS first_at, MAX(at) AS last_at, COUNT(*) AS n FROM hits
WHERE theme_id IN (SELECT theme_id FROM hits GROUP BY theme_id
                   HAVING COUNT(DISTINCT symbol) >= 2 AND COUNT(DISTINCT issuer) >= 2)
GROUP BY theme_id, symbol ORDER BY theme_id, symbol""",
    "relayed": """
WITH"""
    + REJECTED_SQL
    + """,
status AS (
  SELECT e.narrative_id, e.to_status FROM narrative_events e
  WHERE e.seq = (SELECT MAX(x.seq) FROM narrative_events x WHERE x.narrative_id = e.narrative_id
                 AND x.kind IN ('created', 'status', 'override') AND x.at <= :as_of)),
hits AS (
  SELECT t.value AS theme_id, s.value AS symbol, r.first_seen_at AS at
  FROM narrative_records r JOIN status USING (narrative_id), json_each(r.theme_ids_json) t,
       json_each(r.subject_symbols_json) s
  WHERE r.source_tier IN (SELECT value FROM json_each(:relayed)) AND status.to_status <> 'refuted'
    AND r.first_seen_at > :since AND r.first_seen_at <= :as_of
    AND NOT EXISTS (SELECT 1 FROM rejected rj WHERE rj.theme_id = t.value AND rj.symbol = s.value))
SELECT theme_id, symbol, MIN(at) AS first_at, MAX(at) AS last_at, COUNT(*) AS n FROM hits
WHERE theme_id IN (SELECT theme_id FROM hits GROUP BY theme_id HAVING COUNT(DISTINCT symbol) >= 2)
GROUP BY theme_id, symbol ORDER BY theme_id, symbol""",
}


def cross_company_signals(db, as_of):
    """[{theme_id, level official|relayed, since, companies [{symbol, first_at, last_at, n}]}] at as_of (§5.2)."""
    since = iso(
        datetime.datetime.fromisoformat(as_of.replace("Z", "+00:00")) - datetime.timedelta(days=SIGNAL_WINDOW_DAYS)
    )
    args = {
        "as_of": as_of,
        "since": since,
        "official": json.dumps(OFFICIAL_TIERS),
        "relayed": json.dumps(RELAYED_TIERS),
    }
    issuers = link_issuers(db)
    db.create_function("link_issuer", 2, lambda rev, idx: issuers.get((_rev_key(rev), idx)), deterministic=True)
    out = {}
    for level, sql in SIGNAL_SQL.items():
        for r in db.execute(sql, args):
            if level == "relayed" and (r["theme_id"], "official") in out:
                continue
            sig = out.setdefault((r["theme_id"], level), {"theme_id": r["theme_id"], "level": level, "since": since})
            sig.setdefault("companies", []).append({k: r[k] for k in ("symbol", "first_at", "last_at", "n")})
    return [out[k] for k in sorted(out)]


def rejected_members(db, as_of):
    """{(theme_id, symbol)} the owner rejected at as_of (a rejected member proposal, no approved member row)."""
    rows = db.execute(f"WITH {REJECTED_SQL} SELECT theme_id, symbol FROM rejected", {"as_of": as_of})
    return {(r[0], r[1]) for r in rows}


def _rev_key(revision_id):
    """research_theme.py stores a revision id without dashes (SQLAlchemy Uuid on SQLite); the ledger with them."""
    return str(revision_id).replace("-", "").lower()


def _tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def revision_claims(db):
    """{revision id without dashes: (scope_key, claims)} of the completed revisions research_theme.py wrote into this
    ledger; {} for a ledger without research_revisions."""
    if "research_revisions" not in _tables(db):
        return {}
    return {
        _rev_key(r["id"]): (r["scope_key"], json.loads(r["research_json"]).get("claims") or [])
        for r in db.execute("SELECT id, scope_key, research_json FROM research_revisions WHERE status = 'completed'")
    }


def link_issuers(db):
    """{(revision id without dashes, claim index): issuer} for the claims that cite evidence in the completed revisions
    whose manifest is in this ledger. The issuer is the payload issuer (assemble.issuer_of) of the evidence a
    claim_theme_links row describes (narrative_check.best_evidence); a claim whose evidence names none is left out, so
    a revision frozen before issuers were recorded counts for no one. {} for a ledger without research_revisions or
    data_manifests."""
    from narrative_check import best_evidence

    if not {"research_revisions", "data_manifests"} <= _tables(db):
        return {}
    out = {}
    for r in db.execute(
        "SELECT r.id, r.research_json, m.payload_json FROM research_revisions r "
        "JOIN data_manifests m ON m.id = r.manifest_id WHERE r.status = 'completed'"
    ):
        by_id = {e["id"]: e for e in json.loads(r["payload_json"]).get("evidence") or []}
        for n, c in enumerate(json.loads(r["research_json"]).get("claims") or []):
            if any(e in by_id for e in c.get("evidence_ids") or []):
                issuer = (best_evidence(c, by_id).get("payload") or {}).get("issuer")
                if issuer:
                    out[(_rev_key(r["id"]), n)] = issuer
    return out


def export(db, as_of):
    """Point-in-time view at as_of, sorted, without full source texts. Same ledger + same as_of -> same bytes.
    Each claim_theme_links row carries its revision's scope_key, claim text and quote (clipped) when the revision is
    in this ledger; cross_company_signals is §5.2 over a SIGNAL_WINDOW_DAYS window ending at as_of.
    A symbol rejected from a theme (rejected_members) is not a company of that theme here: it leaves the about_symbols
    of the theme's links (a link left with none is omitted) and a record whose every subject was rejected from a theme
    is not listed under it. The ledger rows stay as they are (a link keeps its row's content_sha256)."""
    as_of = iso(as_of)

    def rows(sql, *args):
        return [dict(r) for r in db.execute(sql, args)]

    rejected = rejected_members(db, as_of)
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
        r["theme_ids"] = [
            t for t in json.loads(r.pop("theme_ids_json")) if any((t, s) not in rejected for s in r["subject_symbols"])
        ]
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
    links = []
    for link in rows(
        "SELECT * FROM claim_theme_links WHERE created_at <= ? ORDER BY revision_id, claim_index, theme_id", as_of
    ):
        about = json.loads(link["about_symbols_json"])
        kept = [s for s in about if (link["theme_id"], s) not in rejected]
        if kept:
            if kept != about:
                link["about_symbols_json"] = json.dumps(kept)
            links.append(link)
    revisions = revision_claims(db)
    for link in links:
        scope_key, claims = revisions.get(_rev_key(link["revision_id"]), (None, []))
        claim = claims[link["claim_index"]] if link["claim_index"] < len(claims) else {}
        link["scope_key"] = scope_key
        link["claim_text"] = _clip(claim["text"]) if claim.get("text") else None
        link["claim_quote"] = _clip(claim["quote"]) if claim.get("quote") else None
    doc = {
        "algorithm": EXPORT_ALGO,
        "as_of": as_of,
        "themes": themes,
        "theme_edges": edges,
        "records": records,
        "narrative_checks": checks,
        "claim_theme_links": links,
        "cross_company_signals": cross_company_signals(db, as_of),
        "signal_window_days": SIGNAL_WINDOW_DAYS,
    }
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def write_export(db, as_of, path):
    """export() to path through a temp file and rename, so a reader never sees half a file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(export(db, as_of), encoding="utf-8")
    tmp.replace(path)
    return path


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
    if not a.out:
        return export(db, a.as_of).rstrip("\n")
    return f"wrote {write_export(db, a.as_of, a.out)}"


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
