"""Migration 048 (SEED-002 C3 corrections, C2 precheck persistence) runs on SQLite and matches the ORM."""

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles

from poseidon.models.research_revision import (
    CORRECTION_LINK_CHECK,
    CORRECTION_REASON_CHECK,
    SUPERSEDES_WHERE,
    ResearchRevision,
)

PATH = Path(__file__).resolve().parents[2] / "alembic/versions/048_research_corrections.py"
NEW_COLUMNS = ("supersedes_revision_id", "correction_reason", "precheck_json")


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


def _load():
    spec = importlib.util.spec_from_file_location("migration_048", PATH)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def test_048_literals_match_the_orm():
    migration = _load()
    assert (migration.revision, migration.down_revision) == ("048", "047")
    table = ResearchRevision.__table__
    checks = {c.name: str(c.sqltext) for c in table.constraints if isinstance(c, sa.CheckConstraint)}
    assert (
        dict(migration.CONSTRAINTS)
        == checks
        == {
            "ck_research_revisions_correction_reason": CORRECTION_REASON_CHECK,
            "ck_research_revisions_correction_link": CORRECTION_LINK_CHECK,
        }
    )
    index = next(i for i in table.indexes if i.name == migration.INDEX_NAME)
    assert index.unique and list(index.columns.keys()) == ["supersedes_revision_id"]
    assert migration.INDEX_WHERE == SUPERSEDES_WHERE == str(index.dialect_options["postgresql"]["where"])
    fk = next(iter(table.c.supersedes_revision_id.foreign_keys))
    assert (fk.name, fk.target_fullname) == (migration.FK_NAME, "research_revisions.id")
    assert all(table.c[name].nullable for name in NEW_COLUMNS)


@pytest.fixture
def pre_048():
    """A SQLite research_revisions table in its 047 shape with one legacy completed row."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE research_revisions (id VARCHAR(36) PRIMARY KEY, scope_key VARCHAR(256) NOT NULL, "
            "previous_revision_id VARCHAR(36) REFERENCES research_revisions(id), "
            "status VARCHAR(24) NOT NULL, research_json JSON NOT NULL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO research_revisions VALUES ('legacy', 's', NULL, 'completed', '{\"thesis\": \"x\"}')"
        )
    yield engine
    engine.dispose()


def _run(engine, monkeypatch, step):
    migration = _load()
    with engine.begin() as conn:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(conn)))
        getattr(migration, step)()


def _insert(conn, row_id, status, supersedes=None, reason=None):
    conn.execute(
        sa.text(
            "INSERT INTO research_revisions (id, scope_key, status, research_json, supersedes_revision_id, "
            "correction_reason) VALUES (:id, 's', :status, '{}', :supersedes, :reason)"
        ),
        {"id": row_id, "status": status, "supersedes": supersedes, "reason": reason},
    )


def test_048_upgrade_is_additive_and_enforces_the_correction_contract(pre_048, monkeypatch):
    _run(pre_048, monkeypatch, "upgrade")
    inspector = sa.inspect(pre_048)
    columns = {column["name"]: column for column in inspector.get_columns("research_revisions")}
    assert all(columns[name]["nullable"] for name in NEW_COLUMNS)
    assert {
        (tuple(fk["constrained_columns"]), fk["referred_table"], tuple(fk["referred_columns"]))
        for fk in inspector.get_foreign_keys("research_revisions")
    } >= {(("supersedes_revision_id",), "research_revisions", ("id",))}
    with pre_048.begin() as conn:
        legacy = conn.execute(
            sa.text(
                "SELECT status, research_json, supersedes_revision_id, correction_reason, "
                "precheck_json FROM research_revisions WHERE id = 'legacy'"
            )
        ).one()
        assert tuple(legacy) == ("completed", '{"thesis": "x"}', None, None, None)
        _insert(conn, "prepared-a", "prepared", supersedes="legacy")
        _insert(conn, "prepared-b", "prepared", supersedes="legacy")  # two prepared corrections may coexist
        _insert(conn, "fix-1", "completed", supersedes="legacy", reason="numeric_error")
    for row_id, status, supersedes, reason in (
        ("bad-reason", "prepared", "legacy", "typo"),
        ("reason-without-link", "completed", None, "other"),
        ("completed-without-reason", "completed", "legacy", None),
        ("second-completed", "completed", "legacy", "other"),  # partial unique index: one completed correction
    ):
        with pytest.raises(sa.exc.IntegrityError), pre_048.begin() as conn:
            _insert(conn, row_id, status, supersedes, reason)


def test_048_downgrade_drops_only_its_own_objects(pre_048, monkeypatch):
    _run(pre_048, monkeypatch, "upgrade")
    _run(pre_048, monkeypatch, "downgrade")
    inspector = sa.inspect(pre_048)
    assert [column["name"] for column in inspector.get_columns("research_revisions")] == [
        "id",
        "scope_key",
        "previous_revision_id",
        "status",
        "research_json",
    ]
    assert inspector.get_indexes("research_revisions") == []
    with pre_048.begin() as conn:
        assert conn.execute(sa.text("SELECT status FROM research_revisions WHERE id = 'legacy'")).scalar_one() == (
            "completed"
        )
