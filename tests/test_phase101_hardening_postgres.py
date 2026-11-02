"""Authentic PostgreSQL migration matrix for additive hardening migration 047."""

import importlib.util
import json
import uuid
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from alembic import command
from tests.test_phase101_migration_postgres import phase101_migration_database as _matrix_database

pytestmark = pytest.mark.postgresql
ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "alembic" / "versions" / "047_decision_loop_hardening.py"
SEEDED_TABLES = (
    "strategies",
    "strategy_versions",
    "strategy_release_events",
    "data_manifests",
    "research_revisions",
    "evaluation_runs",
    "evaluation_snapshots",
    "decision_records",
    "decision_events",
    "account_reconciliations",
    "signals",
    "orders",
)
DECISION_AS_OF = "2026-10-01T06:40:00+00:00"
SHADOW_EVENT = "shadow:0123456789abcdef0"
PROTECTIVE_EVENT = "protective_0123456789abc"


def _load_migration():
    spec = importlib.util.spec_from_file_location("phase101_migration_047", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIGRATION_047 = _load_migration()
CONSTRAINT_NAMES = tuple(name for name, _table, _expr in MIGRATION_047.CONSTRAINTS)
INDEX_NAMES = tuple(name for name, _table, _columns in MIGRATION_047.INDEXES)


@pytest.fixture
def hardening_database(monkeypatch):
    """Reuse the Phase 101 scratch-database fixture (one fresh PostgreSQL database per case)."""
    yield from _matrix_database.__wrapped__(monkeypatch)


def _digest(char):
    return char * 64


def _seed_046(engine):
    """Seed legacy rows plus every Stage 0 loop row shape that 047 must accept."""
    ids = {
        name: uuid.uuid4()
        for name in (
            "strategy",
            "version",
            "manifest",
            "research",
            "run",
            "snapshot",
            "exit_decision",
            "anchor_decision",
            "reconciliation",
            "signal",
            "legacy_order",
            "decision_order",
            "stop_loss_order",
            "manual_order",
            "execution_key",
            "anchor_execution_key",
            "bootstrap",
        )
    }
    exit_final = json.dumps({"final_action": "exit", "selected_evaluation_ids": [str(ids["snapshot"])]})
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO strategies (id,name,strategy_type,config,symbol,market,interval,active) "
                "VALUES (:strategy,'phase101-hardening','sma_cross','{}','2330','tw_stock','1d',true)"
            ),
            ids,
        )
        connection.execute(
            text(
                "INSERT INTO strategy_versions "
                "(id,strategy_id,version_no,status,config_json,policy_json,artifact_json,content_sha256) "
                "VALUES (:version,:strategy,1,'draft','{}','{}','{}',:digest)"
            ),
            {**ids, "digest": _digest("a")},
        )
        connection.execute(
            text(
                "INSERT INTO strategy_release_events "
                "(id,strategy_id,event_type,candidate_version_id,from_version_id,to_version_id,campaign_id,"
                "actor_id,account_scope,expected_revision,resulting_revision,idempotency_key,request_sha256,"
                "reason,gate_json,response_json) "
                "VALUES (:bootstrap,:strategy,'bootstrap',:version,NULL,:version,NULL,'release-manager',"
                "'paper:pilot',0,1,'bootstrap-047',:digest,'phase101 bootstrap','{\"bootstrap\": true}','{}')"
            ),
            {**ids, "digest": _digest("b")},
        )
        connection.execute(
            text(
                "INSERT INTO data_manifests (id,content_sha256,market,interval,as_of,payload_json) "
                "VALUES (:manifest,:digest,'tw_stock','1d',CAST(:as_of AS timestamptz),'{}')"
            ),
            {**ids, "digest": _digest("c"), "as_of": DECISION_AS_OF},
        )
        connection.execute(
            text(
                "INSERT INTO research_revisions "
                "(id,scope_key,manifest_id,request_sha256,policy_version,provider,model,runtime_digest,status) "
                "VALUES (:research,'tw_stock:2330',:manifest,:digest,'v1','bounded','none','runtime','completed')"
            ),
            {**ids, "digest": _digest("d")},
        )
        connection.execute(
            text(
                "INSERT INTO evaluation_runs "
                "(id,strategy_version_id,manifest_id,decision_as_of,universe_json,input_sha256,status) "
                "VALUES (:run,:version,:manifest,CAST(:as_of AS timestamptz),'[]',:digest,'complete')"
            ),
            {**ids, "digest": _digest("e"), "as_of": DECISION_AS_OF},
        )
        connection.execute(
            text(
                "INSERT INTO evaluation_snapshots "
                "(id,evaluation_run_id,symbol,market,instrument,status,content_sha256) "
                "VALUES (:snapshot,:run,'2330','tw_stock','spot','evaluated',:digest)"
            ),
            {**ids, "digest": _digest("f")},
        )
        decision_insert = text(
            "INSERT INTO decision_records "
            "(id,evaluation_run_id,strategy_version_id,account_scope,decision_as_of,valid_until,status,revision,"
            "creation_sha256,policy_sha256,original_json,final_json,portfolio_snapshot_json,risk_snapshot_json,"
            "execution_key,claimed_at) "
            "VALUES (:id,:run,:version,'paper:pilot',CAST(:as_of AS timestamptz),"
            "CAST(:as_of AS timestamptz) + interval '1 day',:status,:revision,:creation,:policy,"
            "CAST(:final AS jsonb),CAST(:final AS jsonb),'{}','{}',:execution_key,"
            "CAST(:as_of AS timestamptz) + interval '1 hour')"
        )
        connection.execute(
            decision_insert,
            {
                **ids,
                "id": ids["exit_decision"],
                "as_of": DECISION_AS_OF,
                "status": "executed",
                "revision": 4,
                "creation": _digest("1"),
                "policy": _digest("2"),
                "final": exit_final,
                "execution_key": ids["execution_key"],
            },
        )
        connection.execute(
            decision_insert,
            {
                **ids,
                "id": ids["anchor_decision"],
                "as_of": DECISION_AS_OF,
                "status": "execution_claimed",
                "revision": 3,
                "creation": _digest("3"),
                "policy": _digest("2"),
                "final": json.dumps({"final_action": "enter"}),
                "execution_key": ids["anchor_execution_key"],
            },
        )
        for event_type, revision, decision in (
            ("created", 0, "exit_decision"),
            ("approved", 1, "exit_decision"),
            ("execution_claimed", 2, "exit_decision"),
            ("executed", 3, "exit_decision"),
            (SHADOW_EVENT, 4, "exit_decision"),
            ("created", 0, "anchor_decision"),
            (PROTECTIVE_EVENT, 3, "anchor_decision"),
        ):
            connection.execute(
                text(
                    "INSERT INTO decision_events (decision_id,event_type,actor_id,expected_revision,payload_json) "
                    "VALUES (:decision,:event_type,'system:test',:revision,'{}')"
                ),
                {"decision": ids[decision], "event_type": event_type, "revision": revision},
            )
        connection.execute(
            text(
                "INSERT INTO account_reconciliations "
                "(id,account_scope,account_generation,as_of,broker_state_watermark,internal_state_watermark,"
                "broker_snapshot_sha256,broker_snapshot_json,internal_snapshot_json,difference_json,policy_sha256,status) "
                "VALUES (:reconciliation,'paper:pilot','g1',CAST(:as_of AS timestamptz),'broker:1','internal:1',"
                ":digest,'{}','{}','{}',:policy,'matched')"
            ),
            {**ids, "digest": _digest("4"), "policy": _digest("2"), "as_of": DECISION_AS_OF},
        )
        connection.execute(
            text(
                "INSERT INTO signals (id,strategy_id,symbol,market,instrument,action,confidence,signal_time,interval) "
                "VALUES (:signal,:strategy,'2330','tw_stock','spot','buy',1,CAST(:as_of AS timestamptz),'1d')"
            ),
            {**ids, "as_of": DECISION_AS_OF},
        )
        order_insert = text(
            "INSERT INTO orders "
            "(id,strategy_name,symbol,market,action,target_weight,quantity,broker_mode,status,signal_id,"
            "order_origin,decision_id,account_scope,account_generation,execution_key,client_order_ref,"
            "instrument,intent_sha256) "
            "VALUES (:id,:strategy_name,'2330','tw_stock',:action,0.1,1,'paper','filled',:signal_id,"
            ":origin,:decision_id,:scope,:generation,:execution_key,:client_ref,:instrument,:intent)"
        )
        unscoped = {"scope": None, "generation": None, "execution_key": None, "client_ref": None, "intent": None}
        for row in (
            {
                "id": ids["legacy_order"],
                "strategy_name": "legacy-signal",
                "action": "buy",
                "signal_id": ids["signal"],
                "origin": "signal",
                "decision_id": None,
                "instrument": None,
                **unscoped,
            },
            {
                "id": ids["manual_order"],
                "strategy_name": "legacy-manual",
                "action": "sell",
                "signal_id": None,
                "origin": "manual",
                "decision_id": None,
                "instrument": None,
                **unscoped,
            },
            {
                "id": ids["decision_order"],
                "strategy_name": "decision-loop",
                "action": "sell",
                "signal_id": None,
                "origin": "decision",
                "decision_id": ids["exit_decision"],
                "scope": "paper:pilot",
                "generation": "g1",
                "execution_key": ids["execution_key"],
                "client_ref": "DL-exit-0000",
                "instrument": "spot",
                "intent": _digest("5"),
            },
            {
                "id": ids["stop_loss_order"],
                "strategy_name": "protective:stop_loss",
                "action": "sell",
                "signal_id": None,
                "origin": "stop_loss",
                "decision_id": ids["anchor_decision"],
                "scope": "paper:pilot",
                "generation": "g1",
                "execution_key": uuid.uuid4(),
                "client_ref": "PX-anchor",
                "instrument": "spot",
                "intent": _digest("6"),
            },
        ):
            connection.execute(order_insert, row)
    return ids


def _version(engine):
    with engine.connect() as connection:
        return connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def _fingerprint(engine):
    result = {}
    with engine.connect() as connection:
        for table in SEEDED_TABLES:
            result[table] = tuple(
                connection.execute(
                    text(f"SELECT count(*), md5(coalesce(string_agg(t::text, '' ORDER BY t.id), '')) FROM {table} AS t")
                ).one()
            )
    return result


def _constraints(engine):
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT conname, convalidated FROM pg_constraint WHERE conname = ANY(:names)"),
            {"names": list(CONSTRAINT_NAMES)},
        ).all()
    return dict(rows)


def _indexes(engine):
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT indexname FROM pg_indexes WHERE indexname = ANY(:names)"),
            {"names": list(INDEX_NAMES)},
        ).all()
    return {row[0] for row in rows}


def _assert_047_applied(engine, version="047"):
    assert _version(engine) == version
    constraints = _constraints(engine)
    assert set(constraints) == set(CONSTRAINT_NAMES)
    assert all(constraints.values()), constraints
    assert _indexes(engine) == set(INDEX_NAMES)


def _assert_047_absent(engine):
    assert _constraints(engine) == {}
    assert _indexes(engine) == set()


def _copy_with(connection, table, source_id, override):
    """Insert a copy of an existing row with column overrides, so each probe changes one rule."""
    connection.execute(
        text(
            f"INSERT INTO {table} SELECT (jsonb_populate_record(NULL::{table}, "
            "to_jsonb(s) || CAST(:override AS jsonb))).* "
            f"FROM {table} AS s WHERE s.id = :source"
        ),
        {"source": source_id, "override": json.dumps({"id": str(uuid.uuid4()), **override})},
    )


def test_047_constraint_and_index_literals_are_well_formed():
    assert MIGRATION_047.revision == "047"
    assert MIGRATION_047.down_revision == "046"
    assert len(CONSTRAINT_NAMES) == len(set(CONSTRAINT_NAMES)) == 16
    assert len(INDEX_NAMES) == len(set(INDEX_NAMES)) == 8
    expressions = {name: expr for name, _table, expr in MIGRATION_047.CONSTRAINTS}
    assert "LIKE 'shadow:%'" in expressions["ck_decision_events_type"]
    assert "left(event_type, 11) = 'protective_'" in expressions["ck_decision_events_type"]
    assert expressions["ck_orders_scoped_not_signal"] == "account_scope IS NULL OR order_origin <> 'signal'"


def test_047_empty_upgrade_head_validates_every_constraint_and_index(hardening_database):
    config, engine = hardening_database
    command.upgrade(config, "head")
    _assert_047_applied(engine, version=ScriptDirectory.from_config(config).get_current_head())


def test_047_seeded_046_upgrade_accepts_legacy_and_loop_rows(hardening_database):
    config, engine = hardening_database
    command.upgrade(config, "046")
    _seed_046(engine)
    before = _fingerprint(engine)
    assert before["orders"][0] == 4
    assert before["decision_events"][0] == 7
    assert before["strategy_release_events"][0] == 1

    command.upgrade(config, "047")
    _assert_047_applied(engine)
    assert _fingerprint(engine) == before


def test_047_round_trip_preserves_rows(hardening_database):
    config, engine = hardening_database
    command.upgrade(config, "046")
    _seed_046(engine)
    baseline = _fingerprint(engine)

    command.upgrade(config, "047")
    _assert_047_applied(engine)
    assert _fingerprint(engine) == baseline

    command.downgrade(config, "046")
    assert _version(engine) == "046"
    _assert_047_absent(engine)
    assert _fingerprint(engine) == baseline
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM pg_constraint WHERE conname = 'ck_strategy_release_events_type'")
            ).scalar_one()
            == 1
        )

    command.upgrade(config, "047")
    _assert_047_applied(engine)
    assert _fingerprint(engine) == baseline


def test_047_refuses_before_any_alter_when_a_violator_exists(hardening_database):
    config, engine = hardening_database
    command.upgrade(config, "046")
    ids = _seed_046(engine)
    with engine.begin() as connection:
        _copy_with(
            connection,
            "decision_records",
            ids["anchor_decision"],
            {"status": "bogus", "creation_sha256": _digest("9"), "execution_key": None},
        )
    before = _fingerprint(engine)

    with pytest.raises(RuntimeError, match=r"migration 047 refused: ck_decision_records_status=1$"):
        command.upgrade(config, "047")

    assert _version(engine) == "046"
    _assert_047_absent(engine)
    assert _fingerprint(engine) == before


VIOLATIONS = (
    ("ck_decision_records_status", "decision_records", "anchor_decision", {"status": "bogus"}),
    ("ck_decision_records_revision_positive", "decision_records", "anchor_decision", {"revision": 0}),
    ("ck_decision_records_validity", "decision_records", "anchor_decision", {"valid_until": DECISION_AS_OF}),
    ("ck_decision_records_sha256", "decision_records", "anchor_decision", {"policy_sha256": "short"}),
    ("ck_evaluation_runs_status", "evaluation_runs", "run", {"status": "running"}),
    ("ck_evaluation_snapshots_status", "evaluation_snapshots", "snapshot", {"status": "insufficient"}),
    ("ck_strategy_versions_status", "strategy_versions", "version", {"status": "bogus"}),
    ("ck_strategy_versions_number_hash", "strategy_versions", "version", {"version_no": 0}),
    ("ck_research_revisions_status", "research_revisions", "research", {"status": "complete"}),
    ("ck_account_reconciliations_status", "account_reconciliations", "reconciliation", {"status": "ok"}),
    ("ck_orders_order_origin_allowlist", "orders", "legacy_order", {"order_origin": "unknown"}),
    ("ck_orders_decision_origin_identity", "orders", "decision_order", {"execution_key": None}),
    (
        "ck_orders_decision_id_origin",
        "orders",
        "manual_order",
        {"decision_id": "anchor_decision"},
    ),
    ("ck_orders_scoped_not_signal", "orders", "legacy_order", {"account_scope": "paper:pilot"}),
    ("ck_orders_intent_sha256", "orders", "decision_order", {"intent_sha256": "abc"}),
)


def test_047_every_constraint_rejects_a_violating_insert_with_23514(hardening_database):
    config, engine = hardening_database
    command.upgrade(config, "046")
    ids = _seed_046(engine)
    command.upgrade(config, "047")
    _assert_047_applied(engine)
    before = _fingerprint(engine)

    covered = set()
    probes = [
        (name, table, ids[source], {key: str(ids[value]) if value in ids else value for key, value in override.items()})
        for name, table, source, override in VIOLATIONS
    ]
    probes.append(("ck_decision_events_type", "decision_events", None, {"event_type": "bogus"}))
    for name, table, source_id, override in probes:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                with pytest.raises(DBAPIError) as error:
                    if table == "decision_events":
                        connection.execute(
                            text(
                                "INSERT INTO decision_events "
                                "(decision_id,event_type,actor_id,expected_revision,payload_json) "
                                "VALUES (:decision,:event_type,'system:test',9,'{}')"
                            ),
                            {"decision": ids["anchor_decision"], **override},
                        )
                    else:
                        _copy_with(connection, table, source_id, override)
                original = error.value.orig
                assert (getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)) == "23514"
                assert original.diag.constraint_name == name
                covered.add(name)
            finally:
                transaction.rollback()

    assert covered == set(CONSTRAINT_NAMES)
    assert _fingerprint(engine) == before

    # Admitted prefixed audit rows still insert after 047.
    with engine.begin() as connection:
        for event_type in ("shadow:fedcba98765432100", "protective_fedcba9876543"):
            connection.execute(
                text(
                    "INSERT INTO decision_events (decision_id,event_type,actor_id,expected_revision,payload_json) "
                    "VALUES (:decision,:event_type,'system:test',8,'{}')"
                ),
                {"decision": ids["anchor_decision"], "event_type": event_type},
            )
