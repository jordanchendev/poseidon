"""ledger_schema: migration 1 is the live ledger exactly, migrating is idempotent and atomic, a stale shape is refused."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from poseidon.research_runner import ledger_schema

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "live_ledger_schema_20261104.sql"


def objects(db):
    return {r[0]: (r[1], r[2]) for r in db.execute("SELECT name, type, sql FROM sqlite_master WHERE sql IS NOT NULL")}


def live_copy(path):
    """A ledger with the live DDL (sqlite_master of stormtrooper's ledger.sqlite, 2026-11-04) and user_version 0."""
    db = sqlite3.connect(path)
    db.executescript(FIXTURE.read_text())
    db.commit()
    assert db.execute("PRAGMA user_version").fetchone()[0] == 0
    return db


REVISION_COLUMNS = (
    "id, scope_key, manifest_id, previous_revision_id, request_sha256, policy_version, provider, model, runtime_digest, "
    "content_sha256, status, research_json, created_at, updated_at"
)


def test_migrating_the_live_shape_rebuilds_only_research_revisions_and_adds_settlements(tmp_path):
    db = live_copy(tmp_path / "live.sqlite")
    db.execute("PRAGMA foreign_keys = ON")
    db.execute(
        "INSERT INTO data_manifests (id, content_sha256, market, interval, as_of, payload_json) "
        "VALUES ('m1', 'c1', 'tw_stock', '1d', '2026-10-26T00:00:00Z', '{}')"
    )
    for rid, prev in (("r1", None), ("r2", "r1")):
        db.execute(
            "INSERT INTO research_revisions (id, scope_key, manifest_id, previous_revision_id, request_sha256, policy_version, "
            "provider, model, runtime_digest, status, research_json) VALUES (?, 's', 'm1', ?, ?, 'p', 'x', 'm', 'script:0', 'completed', '{}')",
            (rid, prev, "q" + rid),
        )
    db.commit()
    before_rows = db.execute(f"SELECT {REVISION_COLUMNS} FROM research_revisions ORDER BY id").fetchall()
    before = objects(db)
    ledger_schema.migrate(db)
    after = objects(db)
    changed = {"research_revisions", "ix_research_revisions_scope_created"}
    assert {k: v for k, v in after.items() if k in before and k not in changed} == {
        k: v for k, v in before.items() if k not in changed
    }  # every other live object byte-identical
    assert set(after) - set(before) == {"research_settlements", "uq_research_revisions_supersedes_completed"}
    assert db.execute(f"SELECT {REVISION_COLUMNS} FROM research_revisions ORDER BY id").fetchall() == before_rows
    assert (
        db.execute("SELECT supersedes_revision_id, correction_reason, precheck_json FROM research_revisions").fetchall()
        == [(None, None, None)] * 2
    )
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.execute("PRAGMA user_version").fetchone()[0] == ledger_schema.LATEST == 3


def test_baseline_alone_is_a_noop_on_the_live_shape(tmp_path):
    db = live_copy(tmp_path / "live.sqlite")
    before = objects(db)
    # migration 1 only, statement by statement as migrate runs it
    for sql in ledger_schema._statements(ledger_schema.BASELINE):
        db.execute(sql)
    assert objects(db) == before


def test_a_fresh_ledger_and_a_migrated_live_ledger_end_in_the_same_schema(tmp_path):
    fresh = ledger_schema.connect(tmp_path / "fresh.sqlite")
    live = live_copy(tmp_path / "live.sqlite")
    ledger_schema.migrate(live)
    norm = lambda objs: {k: (t, " ".join(s.split())) for k, (t, s) in objs.items()}  # noqa: E731
    assert norm(objects(fresh)) == norm(objects(live))


def test_a_current_ledger_is_not_written_to(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger_schema.connect(path).close()
    before = path.read_bytes()
    ledger_schema.connect(path).close()
    assert path.read_bytes() == before


def test_a_stale_object_is_refused_and_nothing_is_applied(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE theme_proposals (proposal_id TEXT PRIMARY KEY, title TEXT NOT NULL)")
    old.commit()
    old.close()
    with pytest.raises(ledger_schema.LedgerSchemaError, match="theme_proposals"):
        ledger_schema.connect(path)
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == 0
    assert list(objects(db)) == ["theme_proposals"]


def test_a_ledger_from_a_newer_code_version_is_refused(tmp_path):
    path = tmp_path / "new.sqlite"
    db = sqlite3.connect(path)
    db.execute(f"PRAGMA user_version = {ledger_schema.LATEST + 1}")
    db.close()
    with pytest.raises(ledger_schema.LedgerSchemaError, match="schema version"):
        ledger_schema.connect(path)


# Rendered in a fresh interpreter that imports what the runner imports: SQLite-backed API test modules register
# process-wide @compiles(PG_UUID, "sqlite") hooks (UUID -> VARCHAR(36)) that the runner never loads, so rendering in
# the pytest process depends on which modules were collected before this one.
RENDER_AS_THE_RUNNER = """
import json, sys
from sqlalchemy.dialects import sqlite
from sqlalchemy.schema import CreateIndex, CreateTable
from poseidon.models.base import Base
from poseidon.research_runner import research_theme  # noqa: F401  (registers JSONB -> JSON for SQLite)
ddl = {}
for name in sys.argv[1:]:
    table = Base.metadata.tables[name]
    ddl[name] = str(CreateTable(table).compile(dialect=sqlite.dialect()))
    ddl.update({i.name: str(CreateIndex(i).compile(dialect=sqlite.dialect())) for i in table.indexes})
print(json.dumps(ddl))
"""


def test_the_orm_tables_are_what_sqlalchemy_renders():
    """research_theme writes data_manifests and research_revisions through the ORM; the ledger's DDL must stay what it renders."""
    tables = ("data_manifests", "research_revisions")
    done = subprocess.run(
        [sys.executable, "-c", RENDER_AS_THE_RUNNER, *tables], capture_output=True, text=True, check=True
    )
    rendered = json.loads(done.stdout.splitlines()[-1])
    ledger = objects(ledger_schema.connect(":memory:"))
    norm = lambda sql: " ".join(str(sql).split())  # noqa: E731
    assert set(tables) <= rendered.keys()
    for name, sql in rendered.items():
        assert norm(sql) == norm(ledger[name][1])
