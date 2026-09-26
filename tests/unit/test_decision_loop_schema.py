"""Keep migration 040 and ORM schema aligned without a live database."""

import importlib
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateTable

from poseidon.models import DataManifest, EvaluationRun, EvaluationSnapshot, ResearchRevision, StrategyVersion


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


class MigrationOperations:
    def __init__(self):
        self.metadata = sa.MetaData()
        self.dropped = []

    def create_table(self, name, *columns):
        return sa.Table(name, self.metadata, *columns)

    def create_index(self, name, table, columns):
        sa.Index(name, *(self.metadata.tables[table].c[column] for column in columns))

    def drop_table(self, name):
        self.dropped.append(name)


def test_foundation_migration_matches_orm_and_reverses_dependency_order(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "alembic/versions/040_decision_loop_foundation.py"
    spec = importlib.util.spec_from_file_location("foundation_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert (migration.revision, migration.down_revision) == ("040", "039")
    operations = MigrationOperations()
    monkeypatch.setattr(migration, "op", operations)
    migration.upgrade()

    dialect = postgresql.dialect()
    models = (DataManifest, ResearchRevision, StrategyVersion, EvaluationRun, EvaluationSnapshot)
    for model in models:
        table = model.__table__
        migrated = operations.metadata.tables[table.name]
        assert list(table.c.keys()) == list(migrated.c.keys())
        for column in table.c:
            other = migrated.c[column.name]
            assert str(column.type.compile(dialect=dialect)) == str(other.type.compile(dialect=dialect))
            assert (column.nullable, column.primary_key) == (other.nullable, other.primary_key)
            assert {fk.target_fullname for fk in column.foreign_keys} == {
                fk.target_fullname for fk in other.foreign_keys
            }
            assert (str(column.server_default.arg) if column.server_default else None) == (
                str(other.server_default.arg) if other.server_default else None
            )
        assert {(index.name, tuple(index.columns.keys())) for index in table.indexes} == {
            (index.name, tuple(index.columns.keys())) for index in migrated.indexes
        }
        assert {
            (constraint.name, tuple(constraint.columns.keys()))
            for constraint in table.constraints
            if isinstance(constraint, sa.UniqueConstraint)
        } == {
            (constraint.name, tuple(constraint.columns.keys()))
            for constraint in migrated.constraints
            if isinstance(constraint, sa.UniqueConstraint)
        }
        # Existing SQLite tests compile all registered tables via these adapters.
        assert "CREATE TABLE" in str(CreateTable(table).compile(dialect=sqlite.dialect()))

    migration.downgrade()
    assert operations.dropped == [model.__tablename__ for model in reversed(models)]


def test_decision_models_have_exact_phase97_contract():
    record_spec = importlib.util.find_spec("poseidon.models.decision_record")
    event_spec = importlib.util.find_spec("poseidon.models.decision_event")
    assert record_spec is not None, "DecisionRecord model is missing"
    assert event_spec is not None, "DecisionEvent model is missing"

    DecisionRecord = importlib.import_module("poseidon.models.decision_record").DecisionRecord
    DecisionEvent = importlib.import_module("poseidon.models.decision_event").DecisionEvent
    record = DecisionRecord.__table__
    event = DecisionEvent.__table__

    assert list(record.c.keys()) == [
        "id",
        "evaluation_run_id",
        "strategy_version_id",
        "account_scope",
        "decision_as_of",
        "valid_until",
        "status",
        "revision",
        "creation_sha256",
        "policy_sha256",
        "original_json",
        "final_json",
        "portfolio_snapshot_json",
        "risk_snapshot_json",
        "created_at",
        "updated_at",
    ]
    assert {fk.target_fullname for fk in record.c.evaluation_run_id.foreign_keys} == {"evaluation_runs.id"}
    assert {fk.target_fullname for fk in record.c.strategy_version_id.foreign_keys} == {"strategy_versions.id"}
    assert not any(
        name in record.c
        for name in (
            "execution_key",
            "claimed_at",
            "order_id",
            "client_order_ref",
            "lot_id",
            "reconciliation_status",
        )
    )
    assert {(index.name, tuple(index.columns.keys())) for index in record.indexes} == {
        (
            "ix_decision_records_account_status_valid_created",
            ("account_scope", "status", "valid_until", "created_at"),
        )
    }
    assert {
        (constraint.name, tuple(constraint.columns.keys()))
        for constraint in record.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    } == {("uq_decision_records_creation_sha256", ("creation_sha256",))}

    assert list(event.c.keys()) == [
        "id",
        "decision_id",
        "event_type",
        "actor_id",
        "expected_revision",
        "idempotency_key",
        "request_sha256",
        "payload_json",
        "created_at",
    ]
    assert {fk.target_fullname for fk in event.c.decision_id.foreign_keys} == {"decision_records.id"}
    assert event.c.actor_id.nullable is False
    assert not {"credential", "credential_fingerprint", "api_key", "api_key_fingerprint"} & set(event.c.keys())
    assert {
        (constraint.name, tuple(constraint.columns.keys()))
        for constraint in event.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    } == {
        ("uq_decision_events_type_revision", ("decision_id", "event_type", "expected_revision")),
        ("uq_decision_events_idempotency_key", ("decision_id", "idempotency_key")),
    }
    assert {constraint.name for constraint in event.constraints if isinstance(constraint, sa.CheckConstraint)} == {
        "ck_decision_events_idempotency_pair"
    }
    assert {(index.name, tuple(index.columns.keys())) for index in event.indexes} == {
        ("ix_decision_events_decision_created", ("decision_id", "created_at"))
    }
