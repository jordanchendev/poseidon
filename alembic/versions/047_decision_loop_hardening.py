"""Add additive decision-loop hardening constraints and indexes.

Revision ID: 047
Revises: 046
Create Date: 2026-10-15

Phase 101 D-19/D-20/D-27, SOAK-03. 047 only adds CHECK constraints and indexes; it never
modifies 042/043/044/046 objects and its downgrade drops only its own objects (no row is
ever touched). Every CHECK is pre-checked (violators counted per constraint, and any
violator refuses the migration before the first ALTER), then added NOT VALID and VALIDATEd.

``CONSTRAINTS`` and ``INDEXES`` are pure literals: the read-only backfill report
(phase101-backfill-report.py) reads them with ``ast.literal_eval`` so production
pre-checks run the exact expressions applied here.

Vocabularies (exhaustive enumeration of every value written by src/poseidon at
feat/phase101-stage0 36bd7f1; paths relative to src/poseidon):

- decision_records.status: pending_approval (decision_loop/decisions.py:1489),
  approved / rejected (decisions.py:1353-1355), superseded (decisions.py:1523),
  expired (decisions.py:1547), execution_claimed (decisions.py:1917),
  risk_blocked (decision_loop/execution.py:289), executed (decision_loop/reconciliation.py:1055).
- decision_events.event_type: created (decisions.py:1502), approved / rejected
  (decisions.py:1353,1360), superseded (decisions.py:1527), expired (decisions.py:1551),
  execution_claimed (decisions.py:1922), risk_blocked (execution.py:295),
  executed (reconciliation.py:1059); prefixed audit rows ``shadow:<17 hex>``
  (workers/cpu_tasks.py:470) and ``protective_<13 hex>`` (execution.py:1662).
  Deviation from the RESEARCH Q6 candidate: the durable protective-exit event
  ``protective_<digest>`` is admitted by a prefix clause, otherwise every protective exit
  would be refused.
- evaluation_runs.status: pending (decision_loop/evaluation.py:245), complete (evaluation.py:264).
- evaluation_snapshots.status: TERMINAL_STATUSES evaluated / excluded / failed / no_trade
  (evaluation.py:27), enforced before insert (evaluation.py:40,256).
- strategy_versions.status: code writes only draft (decision_loop/admin.py:393;
  models/strategy_version.py:41 default); the release-state vocabulary of engineering
  plan section 4.3 (draft, validated, shadow, paper, eligible, active, rejected, paused,
  retired) is admitted. Release authority lives in strategy_release_events and
  strategies.active_version_id (044/046), never in this column.
- research_revisions.status: prepared (decision_loop/research.py:169), completed
  (research.py:228), needs_review (research.py:238).
- account_reconciliations.status: matched / mismatch / unresolved
  (decision_loop/reconciliation.py:972,988).
- orders.order_origin: Literal allowlist (orders/schemas.py:41); decision
  (execution.py:719); stop_loss / liquidation / manual_emergency durable protective origins
  (orders/schemas.py:11; execution.py:1394,1640); legacy stop_loss
  (workers/cpu_tasks.py:3062), liquidation (cpu_tasks.py:3430), signal default (037).
- Decision-origin orders always carry decision_id, account_scope, account_generation,
  execution_key, client_order_ref, intent_sha256 (execution.py:706-731); protective orders
  carry the anchor decision (execution.py:1641) or NULL (execution.py:1395). Legacy
  writers (orders/manager.py:284-303) never set account_scope, so the signal invariant is
  scoped to account_scope IS NOT NULL and legacy unscoped signal rows stay valid.
- All *_sha256 values come from manifest.content_sha256 (64-char hexdigest).

D-19: the bootstrap release event (strategy_release_events, 046) and Stage 0 exit
decisions (final_action exit inside final_json) are not restricted by any 047 constraint.

Candidates dropped: none. Not added (already present in 040-046): data_manifests
(market, as_of), research_revisions (scope_key, created_at), strategy_versions
(strategy_id, status), evaluation_runs (strategy_version_id, created_at),
evaluation_snapshots (evaluation_run_id, status), decision_records claim selection,
orders decision_id / execution_key / recovery, ck_decision_events_idempotency_pair,
signals (strategy_id, signal_time).
"""

import sqlalchemy as sa

from alembic import op

revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None

CONSTRAINTS = (
    (
        "ck_decision_records_status",
        "decision_records",
        "status IN ('pending_approval', 'approved', 'rejected', 'expired', 'superseded', "
        "'risk_blocked', 'execution_claimed', 'executed')",
    ),
    ("ck_decision_records_revision_positive", "decision_records", "revision >= 1"),
    ("ck_decision_records_validity", "decision_records", "valid_until > decision_as_of"),
    (
        "ck_decision_records_sha256",
        "decision_records",
        "char_length(creation_sha256) = 64 AND char_length(policy_sha256) = 64",
    ),
    ("ck_evaluation_runs_status", "evaluation_runs", "status IN ('pending', 'complete')"),
    (
        "ck_evaluation_snapshots_status",
        "evaluation_snapshots",
        "status IN ('evaluated', 'excluded', 'failed', 'no_trade')",
    ),
    (
        "ck_strategy_versions_status",
        "strategy_versions",
        "status IN ('draft', 'validated', 'shadow', 'paper', 'eligible', 'active', 'rejected', 'paused', 'retired')",
    ),
    (
        "ck_strategy_versions_number_hash",
        "strategy_versions",
        "version_no >= 1 AND char_length(content_sha256) = 64",
    ),
    (
        "ck_research_revisions_status",
        "research_revisions",
        "status IN ('prepared', 'completed', 'needs_review')",
    ),
    (
        "ck_account_reconciliations_status",
        "account_reconciliations",
        "status IN ('matched', 'mismatch', 'unresolved')",
    ),
    (
        "ck_decision_events_type",
        "decision_events",
        "event_type IN ('created', 'approved', 'rejected', 'superseded', 'expired', "
        "'execution_claimed', 'risk_blocked', 'executed') "
        "OR event_type LIKE 'shadow:%' OR left(event_type, 11) = 'protective_'",
    ),
    (
        "ck_orders_order_origin_allowlist",
        "orders",
        "order_origin IN ('signal', 'stop_loss', 'liquidation', 'manual', 'manual_emergency', 'decision')",
    ),
    (
        "ck_orders_decision_origin_identity",
        "orders",
        "order_origin <> 'decision' OR (decision_id IS NOT NULL AND account_scope IS NOT NULL "
        "AND account_generation IS NOT NULL AND execution_key IS NOT NULL "
        "AND client_order_ref IS NOT NULL AND intent_sha256 IS NOT NULL)",
    ),
    (
        "ck_orders_decision_id_origin",
        "orders",
        "decision_id IS NULL OR order_origin IN ('decision', 'stop_loss', 'liquidation', 'manual_emergency')",
    ),
    ("ck_orders_scoped_not_signal", "orders", "account_scope IS NULL OR order_origin <> 'signal'"),
    ("ck_orders_intent_sha256", "orders", "intent_sha256 IS NULL OR char_length(intent_sha256) = 64"),
)

INDEXES = (
    ("ix_evaluation_runs_manifest_id", "evaluation_runs", ("manifest_id",)),
    ("ix_evaluation_runs_status_decision_as_of", "evaluation_runs", ("status", "decision_as_of")),
    ("ix_evaluation_snapshots_market_symbol", "evaluation_snapshots", ("market", "symbol")),
    ("ix_data_manifests_market_interval_as_of", "data_manifests", ("market", "interval", "as_of")),
    ("ix_decision_records_evaluation_run_id", "decision_records", ("evaluation_run_id",)),
    ("ix_decision_records_strategy_version_id", "decision_records", ("strategy_version_id",)),
    ("ix_research_revisions_scope_status", "research_revisions", ("scope_key", "status")),
    (
        "ix_signals_strategy_market_interval_time",
        "signals",
        ("strategy_id", "market", "interval", "signal_time"),
    ),
)


def violator_counts(connection):
    """Count rows violating each 047 CHECK; NULL results count as satisfied, as in PostgreSQL."""
    return [
        (name, connection.execute(sa.text(f"SELECT count(*) FROM {table} WHERE NOT ({expr})")).scalar_one())
        for name, table, expr in CONSTRAINTS
    ]


def upgrade():
    violators = [(name, count) for name, count in violator_counts(op.get_bind()) if count]
    if violators:
        raise RuntimeError("migration 047 refused: " + "; ".join(f"{name}={count}" for name, count in violators))
    for name, table, expr in CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({expr}) NOT VALID")
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")
    for name, table, columns in INDEXES:
        op.create_index(name, table, list(columns))


def downgrade():
    for name, _table, _columns in reversed(INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {name}")
    for name, table, _expr in reversed(CONSTRAINTS):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
