"""The research ledger's schema: the only place that creates or alters a table of ledger.sqlite.

`PRAGMA user_version` is the schema version. `migrate` applies every migration above the file's version, each in one
transaction together with the version bump, so an interrupted run leaves the file at the previous version. A file
that is already current is not written to.

Migration 1 is the shape of the live ledger on stormtrooper on 2026-11-04 (user_version 0, 17 tables), written to
match it statement for statement: on that file it creates nothing. It adopts a file that has no version yet, so
before it runs every object it would create that already exists must have the same SQL; a different one (a ledger
from an older shape) raises LedgerSchemaError instead of being used silently. Migrations 2 and up run on a file
whose shape migration 1 vouches for.
"""

import sqlite3


class LedgerSchemaError(Exception):
    """The file's schema is not one this code can use."""


# 1: baseline (the live ledger). Runner state, the frozen manifests and revisions research_theme writes through the
# ORM (so they carry the ORM's SQLite rendering), then the narrative ledger.
BASELINE = """
CREATE TABLE IF NOT EXISTS runner_runs (
  agenda_id TEXT PRIMARY KEY, status TEXT NOT NULL, revision_id TEXT, scope_key TEXT, reason TEXT,
  summary TEXT, reported INTEGER NOT NULL DEFAULT 0, run_dir TEXT, started_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS data_manifests (
	id UUID DEFAULT (gen_random_uuid()) NOT NULL,
	content_sha256 VARCHAR(64) NOT NULL,
	market VARCHAR(32) NOT NULL,
	interval VARCHAR(8) NOT NULL,
	as_of DATETIME NOT NULL,
	capability_json JSON DEFAULT '{}' NOT NULL,
	sources_json JSON DEFAULT '[]' NOT NULL,
	payload_json JSON NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT uq_data_manifests_content_sha256 UNIQUE (content_sha256)
);
CREATE TABLE IF NOT EXISTS research_revisions (
	id UUID DEFAULT (gen_random_uuid()) NOT NULL,
	scope_key VARCHAR(256) NOT NULL,
	manifest_id UUID NOT NULL,
	previous_revision_id UUID,
	request_sha256 VARCHAR(64) NOT NULL,
	policy_version VARCHAR(64) NOT NULL,
	provider VARCHAR(64) NOT NULL,
	model VARCHAR(128) NOT NULL,
	runtime_digest VARCHAR(128) NOT NULL,
	content_sha256 VARCHAR(64),
	status VARCHAR(24) DEFAULT 'prepared' NOT NULL,
	research_json JSON DEFAULT '{}' NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT uq_research_revisions_request_sha256 UNIQUE (request_sha256),
	FOREIGN KEY(manifest_id) REFERENCES data_manifests (id),
	FOREIGN KEY(previous_revision_id) REFERENCES research_revisions (id)
);
CREATE INDEX IF NOT EXISTS ix_data_manifests_market_as_of ON data_manifests (market, as_of);
CREATE INDEX IF NOT EXISTS ix_research_revisions_scope_created ON research_revisions (scope_key, created_at);
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

# 2: research_revisions as the ORM model renders it now (migration 048 of the Poseidon database added
# supersedes_revision_id, correction_reason, precheck_json and their CHECKs and unique index). The ledger was created
# before that and create_all never alters a table, so the live file lacks them and the first revision research_theme
# writes would fail with "no such column". SQLite cannot add the CHECKs to a table, so it is rebuilt; the 14 old
# columns are copied unchanged. The old table is renamed, not the new one, so the new one keeps its plain name.
REVISIONS_048 = """
DROP INDEX ix_research_revisions_scope_created;
ALTER TABLE research_revisions RENAME TO research_revisions_old;
CREATE TABLE research_revisions (
    id UUID DEFAULT (gen_random_uuid()) NOT NULL,
    scope_key VARCHAR(256) NOT NULL,
    manifest_id UUID NOT NULL,
    previous_revision_id UUID,
    request_sha256 VARCHAR(64) NOT NULL,
    policy_version VARCHAR(64) NOT NULL,
    provider VARCHAR(64) NOT NULL,
    model VARCHAR(128) NOT NULL,
    runtime_digest VARCHAR(128) NOT NULL,
    content_sha256 VARCHAR(64),
    status VARCHAR(24) DEFAULT 'prepared' NOT NULL,
    research_json JSON DEFAULT '{}' NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    supersedes_revision_id UUID,
    correction_reason VARCHAR(32),
    precheck_json JSON,
    PRIMARY KEY (id),
    CONSTRAINT uq_research_revisions_request_sha256 UNIQUE (request_sha256),
    CONSTRAINT ck_research_revisions_correction_reason CHECK (correction_reason IS NULL OR correction_reason IN ('source_error', 'excerpt_error', 'numeric_error', 'label_policy', 'stale_evidence', 'other')),
    CONSTRAINT ck_research_revisions_correction_link CHECK ((correction_reason IS NULL OR supersedes_revision_id IS NOT NULL) AND (supersedes_revision_id IS NULL OR status <> 'completed' OR correction_reason IS NOT NULL)),
    FOREIGN KEY(manifest_id) REFERENCES data_manifests (id),
    FOREIGN KEY(previous_revision_id) REFERENCES research_revisions (id),
    CONSTRAINT fk_research_revisions_supersedes_revision_id FOREIGN KEY(supersedes_revision_id) REFERENCES research_revisions (id)
);
INSERT INTO research_revisions (id, scope_key, manifest_id, previous_revision_id, request_sha256, policy_version, provider,
    model, runtime_digest, content_sha256, status, research_json, created_at, updated_at)
  SELECT id, scope_key, manifest_id, previous_revision_id, request_sha256, policy_version, provider,
    model, runtime_digest, content_sha256, status, research_json, created_at, updated_at FROM research_revisions_old;
DROP TABLE research_revisions_old;
CREATE INDEX ix_research_revisions_scope_created ON research_revisions (scope_key, created_at);
CREATE UNIQUE INDEX uq_research_revisions_supersedes_completed ON research_revisions (supersedes_revision_id) WHERE supersedes_revision_id IS NOT NULL AND status = 'completed';
"""

# 3: settle-v1 results (SETTLEMENT.md); the live ledger has not run a settlement yet.
SETTLEMENTS = """
CREATE TABLE research_settlements (
  revision_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, symbol TEXT, benchmark TEXT, stance TEXT, direction INTEGER,
  decided_as_of TEXT NOT NULL, horizon_time TEXT, horizon_bar_time TEXT, settle_as_of TEXT, settled_at TEXT,
  settle_algorithm TEXT NOT NULL, bars_sha256 TEXT, abs_return_pct REAL, bench_return_pct REAL, rel_return_pct REAL,
  outcome TEXT NOT NULL CHECK (outcome IN ('hit','miss','flat','unsettled','superseded')), detail TEXT,
  invalidation_flag INTEGER NOT NULL DEFAULT 0, note TEXT, flagged_at TEXT,
  outcome_before_supersede TEXT CHECK (outcome_before_supersede IS NULL OR outcome_before_supersede IN ('hit','miss','flat','unsettled')),
  superseded_by TEXT, content_sha256 TEXT NOT NULL);
"""

MIGRATIONS = ((1, BASELINE), (2, REVISIONS_048), (3, SETTLEMENTS))
LATEST = MIGRATIONS[-1][0]


def _statements(script):
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            yield buf.strip()
            buf = ""
    if buf.strip():
        raise LedgerSchemaError(f"incomplete statement in migration: {buf.strip()[:60]!r}")


def _sql_by_name(db):
    return {
        name: " ".join(sql.split())
        for name, sql in db.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")
    }


def _check_existing_objects(db, version, script):
    scratch = sqlite3.connect(":memory:")
    scratch.executescript(script)
    expected, found = _sql_by_name(scratch), _sql_by_name(db)
    scratch.close()
    differ = sorted(name for name, sql in expected.items() if name in found and found[name] != sql)
    if differ:
        raise LedgerSchemaError(f"migration {version}: {', '.join(differ)} already exist with a different definition")


def migrate(db):
    """Brings `db` (a sqlite3 connection) to LATEST; a no-op on a current file."""
    current = db.execute("PRAGMA user_version").fetchone()[0]
    if current > LATEST:
        raise LedgerSchemaError(f"ledger is at schema version {current}, this code knows up to {LATEST}")
    for version, script in MIGRATIONS:
        if version <= current:
            continue
        db.commit()
        db.execute("BEGIN IMMEDIATE")  # a concurrent opener waits here, then finds the version already bumped
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] >= version:
                db.rollback()
                continue
            if version == 1:
                _check_existing_objects(db, version, script)
            for sql in _statements(script):
                db.execute(sql)
            db.execute(f"PRAGMA user_version = {version}")
            db.commit()
        except BaseException:
            db.rollback()
            raise


def connect(path):
    db = sqlite3.connect(path)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        migrate(db)
    except BaseException:
        db.close()
        raise
    return db
