"""narrative_export.json (design 2026-10-29 §3.4): the deterministic projection of the narrative ledger as of a time,
which nexus_pull.py on the MacBook turns into vault pages, and the cross-company signals and rejected members it lists.
Moved out of narrative_ledger.py unchanged; `narrative_ledger.py export` still writes it.
"""

import datetime
import json
from pathlib import Path

import narrative_ledger as nl
from claim_precheck import OFFICIAL_TIERS, best_evidence

EXPORT_ALGO = "narrative-export-v2"
EXPORT_QUOTE_CHARS = 120  # D7: the export leaves stormtrooper for the vault
SIGNAL_WINDOW_DAYS = 180  # D11: the cross-company signal window (§5.2)
RELAYED_TIERS = ("sell-side", "media", "kol")


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
    since = nl.iso(
        datetime.datetime.fromisoformat(as_of.replace("Z", "+00:00")) - datetime.timedelta(days=SIGNAL_WINDOW_DAYS)
    )
    args = {
        "as_of": as_of,
        "since": since,
        "official": json.dumps(OFFICIAL_TIERS),
        "relayed": json.dumps(RELAYED_TIERS),
    }
    issuers = link_issuers(db)
    db.create_function("link_issuer", 2, lambda rev, idx: issuers.get((nl._rev_key(rev), idx)), deterministic=True)
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


def revision_claims(db):
    """{revision id without dashes: (scope_key, claims)} of the completed revisions research_theme.py wrote into this
    ledger; {} for a ledger without research_revisions."""
    if "research_revisions" not in nl._tables(db):
        return {}
    return {
        nl._rev_key(r["id"]): (r["scope_key"], json.loads(r["research_json"]).get("claims") or [])
        for r in db.execute("SELECT id, scope_key, research_json FROM research_revisions WHERE status = 'completed'")
    }


def link_issuers(db):
    """{(revision id without dashes, claim index): issuer} for the claims that cite evidence in the completed revisions
    whose manifest is in this ledger. The issuer is the payload issuer (assemble.issuer_of) of the evidence a
    claim_theme_links row describes (claim_precheck.best_evidence); a claim whose evidence names none is left out, so
    a revision frozen before issuers were recorded counts for no one. {} for a ledger without research_revisions or
    data_manifests."""
    if not {"research_revisions", "data_manifests"} <= nl._tables(db):
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
                    out[(nl._rev_key(r["id"]), n)] = issuer
    return out


def export(db, as_of):
    """Point-in-time view at as_of, sorted, without full source texts. Same ledger + same as_of -> same bytes.
    Each claim_theme_links row carries its revision's scope_key, claim text and quote (clipped) when the revision is
    in this ledger; cross_company_signals is §5.2 over a SIGNAL_WINDOW_DAYS window ending at as_of.
    A symbol rejected from a theme (rejected_members) is not a company of that theme here: it leaves the about_symbols
    of the theme's links (a link left with none is omitted) and a record whose every subject was rejected from a theme
    is not listed under it. The ledger rows stay as they are (a link keeps its row's content_sha256)."""
    as_of = nl.iso(as_of)

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
        r["events"] = [e for e in nl.events_of(db, r["narrative_id"]) if e["at"] <= as_of]
        for e in r["events"]:
            e["basis"] = json.loads(e.pop("basis_json"))
            jev = e.pop("jev_json")
            e["jev"] = json.loads(jev) if jev else None
            del e["narrative_id"]
        r["status"] = nl.fold_status(r["events"])
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
        scope_key, claims = revisions.get(nl._rev_key(link["revision_id"]), (None, []))
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
