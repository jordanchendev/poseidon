CREATE TABLE runner_runs (
  agenda_id TEXT PRIMARY KEY, status TEXT NOT NULL, revision_id TEXT, scope_key TEXT, reason TEXT,
  summary TEXT, reported INTEGER NOT NULL DEFAULT 0, run_dir TEXT, started_at TEXT, updated_at TEXT);
CREATE TABLE data_manifests (
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
CREATE INDEX ix_data_manifests_market_as_of ON data_manifests (market, as_of);
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
	PRIMARY KEY (id), 
	CONSTRAINT uq_research_revisions_request_sha256 UNIQUE (request_sha256), 
	FOREIGN KEY(manifest_id) REFERENCES data_manifests (id), 
	FOREIGN KEY(previous_revision_id) REFERENCES research_revisions (id)
);
CREATE INDEX ix_research_revisions_scope_created ON research_revisions (scope_key, created_at);
CREATE TABLE themes (
  theme_id TEXT PRIMARY KEY, title TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('bottleneck','tech_transition','demand_driver','policy')),
  keywords_json TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('active','retired')),
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  proposal_id TEXT, content_sha256 TEXT NOT NULL);
CREATE TABLE theme_members (
  theme_id TEXT NOT NULL REFERENCES themes, symbol TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('constrained','supplier','beneficiary','customer','competitor')),
  tw_listed INTEGER NOT NULL CHECK (tw_listed IN (0,1)),
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (theme_id, symbol));
CREATE TABLE theme_edges (
  from_theme TEXT NOT NULL REFERENCES themes, to_theme TEXT NOT NULL REFERENCES themes,
  relation TEXT NOT NULL CHECK (relation IN ('related','part_of')),
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (from_theme, to_theme, relation), CHECK (from_theme <> to_theme));
CREATE TABLE theme_proposals (
  proposal_id TEXT PRIMARY KEY, suggested_id TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL, why TEXT NOT NULL,
  source_ref TEXT NOT NULL, proposed_members_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed','accepted','rejected')),
  decided_at TEXT, decided_ref TEXT, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL,
  keywords_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE theme_keyword_approvals (
  theme_id TEXT NOT NULL REFERENCES themes, keyword TEXT NOT NULL,
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (theme_id, keyword));
CREATE TABLE materials (
  material_id TEXT PRIMARY KEY, received_at TEXT NOT NULL,
  source_tier TEXT NOT NULL CHECK (source_tier IN ('sell-side','media','kol')),
  author TEXT NOT NULL, url TEXT, text_sha256 TEXT NOT NULL, raw_path TEXT NOT NULL, manifest_sha256 TEXT,
  status TEXT NOT NULL CHECK (status IN ('received','intaken','needs_review')), note TEXT,
  content_sha256 TEXT NOT NULL);
CREATE TABLE narrative_records (
  narrative_id TEXT PRIMARY KEY, claim TEXT NOT NULL,
  claim_type TEXT NOT NULL CHECK (claim_type IN ('state','forecast')), resolves_on TEXT,
  resolves_on_default INTEGER NOT NULL DEFAULT 0, attributed_to TEXT NOT NULL,
  source_tier TEXT NOT NULL CHECK (source_tier IN ('official','sell-side','media','kol','our-inference')),
  origin_kind TEXT NOT NULL CHECK (origin_kind IN ('material','revision','owner')), origin_ref TEXT NOT NULL,
  origin_evidence_id TEXT NOT NULL, quote TEXT NOT NULL, subject_symbols_json TEXT NOT NULL,
  theme_ids_json TEXT NOT NULL, first_seen_at TEXT NOT NULL, review_by TEXT NOT NULL, created_at TEXT NOT NULL,
  content_sha256 TEXT NOT NULL, CHECK (claim_type = 'state' OR resolves_on IS NOT NULL));
CREATE TABLE narrative_events (
  narrative_id TEXT NOT NULL REFERENCES narrative_records, seq INTEGER NOT NULL, at TEXT NOT NULL,
  recorded_at TEXT NOT NULL, kind TEXT NOT NULL CHECK (kind IN ('created','status','corroborated','override')),
  from_status TEXT, to_status TEXT NOT NULL, rule TEXT, basis_json TEXT NOT NULL, jev_json TEXT, drafter_verdict TEXT,
  decided_by TEXT NOT NULL CHECK (decided_by IN ('rule','owner','sweep')), note TEXT, content_sha256 TEXT NOT NULL,
  PRIMARY KEY (narrative_id, seq));
CREATE TABLE narrative_checks (
  revision_id TEXT NOT NULL, narrative_id TEXT NOT NULL REFERENCES narrative_records, status_before TEXT NOT NULL,
  drafter_verdict TEXT NOT NULL CHECK (drafter_verdict IN ('verifies','refutes','partially_verifies','untouched')),
  claim_indices_json TEXT NOT NULL, jev_label TEXT, sweep_label TEXT, sweep_truncated INTEGER NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('status_change','corroborated','no_change','disputed','missed')),
  event_seq INTEGER, checked_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, PRIMARY KEY (revision_id, narrative_id));
CREATE TABLE claim_theme_links (
  revision_id TEXT NOT NULL, claim_index INTEGER NOT NULL, theme_id TEXT NOT NULL REFERENCES themes,
  about_symbols_json TEXT NOT NULL, claim_kind TEXT NOT NULL CHECK (claim_kind IN ('fact','inference')),
  evidence_tier TEXT NOT NULL, event_time TEXT NOT NULL, source TEXT NOT NULL CHECK (source IN ('drafter','owner')),
  created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, PRIMARY KEY (revision_id, claim_index, theme_id));
CREATE TABLE narrative_suggestions (
  suggestion_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL,
  narrative_id TEXT NOT NULL REFERENCES narrative_records,
  kind TEXT NOT NULL CHECK (kind IN ('status','corroborated')), from_status TEXT NOT NULL, to_status TEXT NOT NULL,
  rule TEXT NOT NULL, at TEXT NOT NULL, basis_json TEXT NOT NULL, jev_json TEXT NOT NULL, drafter_verdict TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('suggested','confirmed','rejected')), created_at TEXT NOT NULL,
  decided_at TEXT, decision_ref TEXT, event_seq INTEGER, content_sha256 TEXT NOT NULL,
  UNIQUE (revision_id, narrative_id));
CREATE TABLE member_proposals (
  proposal_id TEXT PRIMARY KEY, theme_id TEXT NOT NULL REFERENCES themes, symbol TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('constrained','supplier','beneficiary','customer','competitor')),
  why TEXT NOT NULL, source_ref TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed','accepted','rejected')),
  decided_at TEXT, decided_ref TEXT, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL);
CREATE TABLE theme_proposal_claims (
  proposal_id TEXT NOT NULL REFERENCES theme_proposals, revision_id TEXT NOT NULL, claim_index INTEGER NOT NULL,
  about_symbols_json TEXT NOT NULL, claim_kind TEXT NOT NULL CHECK (claim_kind IN ('fact','inference')),
  evidence_tier TEXT NOT NULL, event_time TEXT NOT NULL, created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL,
  PRIMARY KEY (proposal_id, revision_id, claim_index));
CREATE TABLE claim_link_approvals (
  revision_id TEXT NOT NULL, claim_index INTEGER NOT NULL, theme_id TEXT NOT NULL REFERENCES themes,
  approved_by TEXT NOT NULL, approved_at TEXT NOT NULL, approval_ref TEXT NOT NULL CHECK (approval_ref <> ''),
  content_sha256 TEXT NOT NULL, PRIMARY KEY (revision_id, claim_index, theme_id));
CREATE TABLE held_records (
  narrative_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, source_text TEXT NOT NULL, waits_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('held','written','dropped')), reason TEXT, decided_at TEXT,
  created_at TEXT NOT NULL, content_sha256 TEXT NOT NULL);
