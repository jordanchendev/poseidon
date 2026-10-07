"""Keep decision-loop migrations and ORM metadata aligned."""

import importlib
import importlib.util
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import pytest
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


class _Result:
    def __init__(self, row=None, scalar_value=False):
        self.row = row
        self.scalar_value = scalar_value

    def first(self):
        return self.row

    def scalar(self):
        return self.scalar_value


class _Connection:
    def __init__(self, row=None, scalar_value=False):
        self.row = row
        self.scalar_value = scalar_value
        self.statements = []

    def execute(self, statement):
        self.statements.append(str(statement))
        return _Result(self.row, self.scalar_value)


class MigrationOperations:
    def __init__(self, *, seed_execution_dependencies=False, duplicate_fill=None, downgrade_history=False):
        self.metadata = sa.MetaData()
        self.dropped = []
        self.dropped_operations = []
        self.added_columns = defaultdict(list)
        self.created_indexes = []
        self.created_unique_constraints = []
        self.created_check_constraints = []
        self.executed = []
        self.mutation_calls = []
        self.connection = _Connection(duplicate_fill, downgrade_history)
        if seed_execution_dependencies:
            sa.Table(
                "decision_records",
                self.metadata,
                sa.Column("id", UUID(as_uuid=True), primary_key=True),
                sa.Column("status", sa.String(24), nullable=False),
                sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
                sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            )
            sa.Table(
                "orders",
                self.metadata,
                sa.Column("id", UUID(as_uuid=True), primary_key=True),
                sa.Column("status", sa.String(32), nullable=False),
            )
            sa.Table(
                "order_fills",
                self.metadata,
                sa.Column("id", UUID(as_uuid=True), primary_key=True),
                sa.Column("order_id", UUID(as_uuid=True), nullable=False),
                sa.Column("broker_fill_id", sa.String(64), nullable=True),
            )

    def get_bind(self):
        return self.connection

    def create_table(self, name, *columns):
        self.mutation_calls.append(("create_table", name))
        return sa.Table(name, self.metadata, *columns)

    def create_index(self, name, table, columns):
        self.mutation_calls.append(("create_index", name))
        self.created_indexes.append((name, table, tuple(columns)))
        if table in self.metadata.tables:
            sa.Index(name, *(self.metadata.tables[table].c[column] for column in columns))

    def create_unique_constraint(self, name, table, columns):
        self.mutation_calls.append(("create_unique_constraint", name))
        self.created_unique_constraints.append((name, table, tuple(columns)))

    def create_check_constraint(self, name, table, condition):
        self.mutation_calls.append(("create_check_constraint", name))
        self.created_check_constraints.append((name, table, str(condition)))

    def add_column(self, table, column):
        self.mutation_calls.append(("add_column", f"{table}.{column.name}"))
        self.added_columns[table].append(column)
        self.metadata.tables[table].append_column(column)

    def execute(self, statement):
        self.mutation_calls.append(("execute", str(statement)))
        self.executed.append(str(statement))

    def drop_index(self, name, table_name=None):
        self.dropped_operations.append(("index", name, table_name))

    def drop_constraint(self, name, table, type_=None):
        self.dropped_operations.append(("constraint", name, table, type_))

    def drop_column(self, table, column):
        self.dropped_operations.append(("column", table, column))

    def drop_table(self, name):
        self.dropped.append(name)
        self.dropped_operations.append(("table", name))


def _assert_migration_matches_models(operations, models):
    dialect = postgresql.dialect()
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
        assert {
            (constraint.name, str(constraint.sqltext))
            for constraint in table.constraints
            if isinstance(constraint, sa.CheckConstraint)
        } == {
            (constraint.name, str(constraint.sqltext))
            for constraint in migrated.constraints
            if isinstance(constraint, sa.CheckConstraint)
        }
        # Existing SQLite tests compile all registered tables via these adapters.
        assert "CREATE TABLE" in str(CreateTable(table).compile(dialect=sqlite.dialect()))


def test_foundation_migration_matches_orm_and_reverses_dependency_order(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "alembic/versions/040_decision_loop_foundation.py"
    spec = importlib.util.spec_from_file_location("foundation_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert (migration.revision, migration.down_revision) == ("040", "039")
    operations = MigrationOperations()
    monkeypatch.setattr(migration, "op", operations)
    migration.upgrade()

    models = (DataManifest, ResearchRevision, StrategyVersion, EvaluationRun, EvaluationSnapshot)
    _assert_migration_matches_models(operations, models)

    migration.downgrade()
    assert operations.dropped == [model.__tablename__ for model in reversed(models)]


def test_decision_migration_041_stays_byte_for_byte_phase97_history(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "alembic/versions/041_decision_execution.py"
    assert path.is_file(), "decision migration 041 is missing"
    assert sha256(path.read_bytes()).hexdigest() == "70037e59f17e52daa95d9cc5532ad92b4f21897964ba0148ea270ca27449325d"

    spec = importlib.util.spec_from_file_location("decision_migration_history", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    operations = MigrationOperations()
    monkeypatch.setattr(migration, "op", operations)
    migration.upgrade()

    record = operations.metadata.tables["decision_records"]
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
    assert "execution_key" not in record.c
    assert "claimed_at" not in record.c


def test_decision_migration_matches_event_orm_and_stays_decision_only(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "alembic/versions/041_decision_execution.py"
    assert path.is_file(), "decision migration 041 is missing"

    spec = importlib.util.spec_from_file_location("decision_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert (migration.revision, migration.down_revision) == ("041", "040")

    import poseidon.models as models

    assert hasattr(models, "DecisionRecord"), "DecisionRecord is not exported"
    assert hasattr(models, "DecisionEvent"), "DecisionEvent is not exported"
    operations = MigrationOperations()
    monkeypatch.setattr(migration, "op", operations)
    migration.upgrade()

    assert list(operations.metadata.tables) == ["decision_records", "decision_events"]
    _assert_migration_matches_models(operations, (models.DecisionEvent,))

    migration.downgrade()
    assert operations.dropped == ["decision_events", "decision_records"]


def _load_execution_migration():
    path = Path(__file__).resolve().parents[2] / "alembic/versions/042_execution_reconciliation.py"
    assert path.is_file(), "execution migration 042 is missing"
    spec = importlib.util.spec_from_file_location("execution_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def _unique_constraints(table):
    return {
        (constraint.name, tuple(constraint.columns.keys()))
        for constraint in table.constraints
        if isinstance(constraint, sa.UniqueConstraint)
    }


def _check_names(table):
    return {constraint.name for constraint in table.constraints if isinstance(constraint, sa.CheckConstraint)}


def _check_sql(table, name):
    return next(
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, sa.CheckConstraint) and constraint.name == name
    )


def _foreign_key_constraints(table):
    return {
        (
            constraint.name,
            tuple(constraint.columns.keys()),
            tuple(element.target_fullname for element in constraint.elements),
        )
        for constraint in table.constraints
        if isinstance(constraint, sa.ForeignKeyConstraint)
    }


def test_execution_migration_aborts_dirty_fill_history_before_mutation(monkeypatch):
    migration = _load_execution_migration()
    assert (migration.revision, migration.down_revision) == ("042", "041")
    operations = MigrationOperations(seed_execution_dependencies=True, duplicate_fill=("order-1", "fill-1", 2))
    monkeypatch.setattr(migration, "op", operations)

    with pytest.raises(RuntimeError, match="duplicate legacy broker fill"):
        migration.upgrade()

    assert operations.mutation_calls == []
    assert "broker_fill_id IS NOT NULL" in operations.connection.statements[0]


def test_execution_migration_refuses_destructive_downgrade_before_any_drop(monkeypatch):
    migration = _load_execution_migration()
    operations = MigrationOperations(downgrade_history=True)
    monkeypatch.setattr(migration, "op", operations)

    with pytest.raises(RuntimeError, match="execution or audit history"):
        migration.downgrade()

    assert operations.dropped_operations == []
    checked_tables = (
        "position_lots",
        "fill_allocations",
        "account_reconciliations",
        "paper_broker_accounts",
        "paper_broker_orders",
        "paper_broker_fills",
        "paper_cash_movements",
        "decision_records",
        "orders",
        "order_fills",
    )
    lock_statement = operations.connection.statements[0]
    assert lock_statement.lstrip().startswith("LOCK TABLE")
    assert "IN SHARE MODE" in lock_statement
    for table_name in checked_tables:
        assert table_name in lock_statement

    statement = operations.connection.statements[1]
    for table_name in checked_tables[:7]:
        assert f"FROM {table_name}" in statement
    for field_name in ("execution_key", "claimed_at"):
        assert f"decision_records.{field_name} IS NOT NULL" in statement
    for field_name in (
        "decision_id",
        "account_scope",
        "account_generation",
        "execution_key",
        "client_order_ref",
        "instrument",
        "intent_json",
        "intent_sha256",
        "reserved_cash_json",
        "reserved_quantity",
        "reservation_status",
        "reconciliation_status",
        "submit_attempted_at",
        "protective_context_json",
    ):
        assert f"orders.{field_name} IS NOT NULL" in statement
    assert "order_fills.projection_status IS NOT NULL" in statement


def test_execution_migration_matches_exact_orm_contract_and_downgrade_order(monkeypatch):
    migration = _load_execution_migration()
    operations = MigrationOperations(seed_execution_dependencies=True)
    monkeypatch.setattr(migration, "op", operations)
    migration.upgrade()

    import poseidon.models as models

    model_names = (
        "PositionLot",
        "FillAllocation",
        "AccountReconciliation",
        "PaperBrokerAccount",
        "PaperBrokerOrder",
        "PaperBrokerFill",
        "PaperCashMovement",
    )
    assert all(hasattr(models, name) for name in model_names)
    execution_models = tuple(getattr(models, name) for name in model_names)
    assert [model.__tablename__ for model in execution_models] == [
        "position_lots",
        "fill_allocations",
        "account_reconciliations",
        "paper_broker_accounts",
        "paper_broker_orders",
        "paper_broker_fills",
        "paper_cash_movements",
    ]
    _assert_migration_matches_models(operations, execution_models)

    expected_added_columns = {
        "decision_records": ["execution_key", "claimed_at"],
        "orders": [
            "decision_id",
            "account_scope",
            "account_generation",
            "execution_key",
            "client_order_ref",
            "instrument",
            "intent_json",
            "intent_sha256",
            "reserved_cash_json",
            "reserved_quantity",
            "reservation_status",
            "reconciliation_status",
            "submit_attempted_at",
            "protective_context_json",
        ],
        "order_fills": ["projection_status"],
    }
    for table_name, expected_columns in expected_added_columns.items():
        assert [column.name for column in operations.added_columns[table_name]] == expected_columns
        model_table = {
            "decision_records": models.DecisionRecord.__table__,
            "orders": models.OrderRecord.__table__,
            "order_fills": models.OrderFillRecord.__table__,
        }[table_name]
        for added in operations.added_columns[table_name]:
            model_column = model_table.c[added.name]
            assert model_column.nullable is True
            assert str(added.type.compile(dialect=postgresql.dialect())) == str(
                model_column.type.compile(dialect=postgresql.dialect())
            )

    assert (
        "uq_decision_records_execution_key",
        "decision_records",
        ("execution_key",),
    ) in operations.created_unique_constraints
    assert (
        "uq_orders_account_generation_client_ref",
        "orders",
        ("account_scope", "account_generation", "client_order_ref"),
    ) in operations.created_unique_constraints
    assert (
        "uq_order_fills_order_broker_fill",
        "order_fills",
        ("order_id", "broker_fill_id"),
    ) in operations.created_unique_constraints
    assert (
        "ck_orders_reserved_quantity_nonnegative_finite",
        "orders",
        "reserved_quantity IS NULL OR (reserved_quantity >= 0 AND reserved_quantity <= 1e308)",
    ) in operations.created_check_constraints
    assert any("fk_orders_decision_id" in statement and "NOT VALID" in statement for statement in operations.executed)

    migration.downgrade()
    assert operations.dropped == [
        "paper_cash_movements",
        "paper_broker_fills",
        "paper_broker_orders",
        "paper_broker_accounts",
        "account_reconciliations",
        "fill_allocations",
        "position_lots",
    ]
    assert operations.dropped_operations[-2:] == [
        ("column", "decision_records", "claimed_at"),
        ("column", "decision_records", "execution_key"),
    ]


def test_execution_models_enforce_replay_quantity_and_independent_broker_truth():
    import poseidon.models as models

    decision = models.DecisionRecord.__table__
    order = models.OrderRecord.__table__
    fill = models.OrderFillRecord.__table__
    lot = models.PositionLot.__table__
    allocation = models.FillAllocation.__table__
    reconciliation = models.AccountReconciliation.__table__
    paper_account = models.PaperBrokerAccount.__table__
    paper_order = models.PaperBrokerOrder.__table__
    paper_fill = models.PaperBrokerFill.__table__
    cash_movement = models.PaperCashMovement.__table__

    assert _unique_constraints(decision) >= {("uq_decision_records_execution_key", ("execution_key",))}
    assert _unique_constraints(order) >= {
        (
            "uq_orders_account_generation_client_ref",
            ("account_scope", "account_generation", "client_order_ref"),
        )
    }
    assert _unique_constraints(fill) >= {("uq_order_fills_order_broker_fill", ("order_id", "broker_fill_id"))}
    assert _unique_constraints(lot) == {("uq_position_lots_opening_fill", ("opening_fill_id",))}
    assert _check_names(lot) == {
        "ck_position_lots_original_quantity_positive",
        "ck_position_lots_open_quantity_range",
        "ck_position_lots_reserved_close_quantity_range",
    }
    assert _unique_constraints(allocation) == {
        ("uq_fill_allocations_closing_fill_lot", ("closing_fill_id", "position_lot_id"))
    }
    assert _check_names(allocation) == {"ck_fill_allocations_quantity_positive"}
    assert _unique_constraints(reconciliation) == {
        (
            "uq_account_reconciliations_replay",
            (
                "account_scope",
                "account_generation",
                "as_of",
                "broker_state_watermark",
                "internal_state_watermark",
                "broker_snapshot_sha256",
                "policy_sha256",
            ),
        )
    }
    assert _unique_constraints(paper_account) == {
        ("uq_paper_broker_accounts_scope_generation", ("account_scope", "account_generation"))
    }
    assert _unique_constraints(paper_order) == {
        (
            "uq_paper_broker_orders_client_ref",
            ("account_scope", "account_generation", "client_order_ref"),
        ),
        (
            "uq_paper_broker_orders_broker_order",
            ("account_scope", "account_generation", "broker_order_id"),
        ),
        (
            "uq_paper_broker_orders_identity",
            (
                "id",
                "account_scope",
                "account_generation",
                "market",
                "symbol",
                "instrument",
                "side",
            ),
        ),
    }
    assert _unique_constraints(paper_fill) == {
        ("uq_paper_broker_fills_order_fill", ("paper_broker_order_id", "broker_fill_id"))
    }
    assert _unique_constraints(cash_movement) == {
        (
            "uq_paper_cash_movements_account_version",
            ("account_scope", "account_generation", "state_version"),
        )
    }

    internal_tables = {"decision_records", "orders", "order_fills", "position_lots", "fill_allocations"}
    for table in (paper_account, paper_order, paper_fill, cash_movement):
        targets = {fk.column.table.name for column in table.c for fk in column.foreign_keys}
        assert not targets & internal_tables

    assert _foreign_key_constraints(paper_fill) == {
        (
            "fk_paper_broker_fills_order_identity",
            (
                "paper_broker_order_id",
                "account_scope",
                "account_generation",
                "market",
                "symbol",
                "instrument",
                "side",
            ),
            (
                "paper_broker_orders.id",
                "paper_broker_orders.account_scope",
                "paper_broker_orders.account_generation",
                "paper_broker_orders.market",
                "paper_broker_orders.symbol",
                "paper_broker_orders.instrument",
                "paper_broker_orders.side",
            ),
        )
    }
    assert _foreign_key_constraints(cash_movement) == {
        (
            "fk_paper_cash_movements_account",
            ("account_scope", "account_generation"),
            (
                "paper_broker_accounts.account_scope",
                "paper_broker_accounts.account_generation",
            ),
        )
    }
    assert "paper_broker_order_id" not in cash_movement.c
    assert "paper_broker_fill_id" not in cash_movement.c

    finite_checks = (
        (order, "ck_orders_reserved_quantity_nonnegative_finite"),
        (lot, "ck_position_lots_original_quantity_positive"),
        (allocation, "ck_fill_allocations_quantity_positive"),
        (paper_account, "ck_paper_broker_accounts_opening_cash_nonnegative"),
        (paper_order, "ck_paper_broker_orders_quantity_positive"),
        (paper_order, "ck_paper_broker_orders_price_nonnegative_finite"),
        (paper_fill, "ck_paper_broker_fills_quantity_positive"),
        (paper_fill, "ck_paper_broker_fills_price_nonnegative"),
        (cash_movement, "ck_paper_cash_movements_amount_nonzero"),
    )
    for table, constraint_name in finite_checks:
        assert "1e308" in _check_sql(table, constraint_name)

    assert [column.name for column in lot.primary_key.columns] == ["id"]
    assert {
        "account_scope",
        "account_generation",
        "market",
        "symbol",
        "instrument",
        "side",
        "original_quantity",
        "open_quantity",
        "reserved_close_quantity",
    } <= set(lot.c.keys())
    assert {"broker_state_watermark", "internal_state_watermark", "policy_sha256", "status"} <= set(
        reconciliation.c.keys()
    )


def _index_contracts(table):
    return {
        (
            index.name,
            tuple(index.columns.keys()),
            index.unique,
            str(index.dialect_options["postgresql"].get("where")),
        )
        for index in table.indexes
    }


def test_phase99_outcome_model_exports_and_revision_contracts():
    import poseidon.models as models

    names = (
        "OutcomeLabelContract",
        "ResearchAssessment",
        "FillCostRevision",
        "FillCostComponent",
        "EconomicReconciliation",
        "OutcomeRecord",
    )
    assert all(hasattr(models, name) for name in names)
    assert models.OutcomeRecord.__tablename__ == "outcome_records"

    outcome = models.OutcomeRecord.__table__
    assert _unique_constraints(outcome) >= {
        ("uq_outcome_records_replay", ("logical_key_sha256", "input_sha256")),
        ("uq_outcome_records_revision", ("logical_key_sha256", "revision_no")),
        ("uq_outcome_records_previous", ("previous_outcome_id",)),
    }
    assert (
        "fk_outcome_records_previous_key_revision",
        ("logical_key_sha256", "previous_revision_no"),
        ("outcome_records.logical_key_sha256", "outcome_records.revision_no"),
    ) in _foreign_key_constraints(outcome)
    assert "revision_no = 1" in _check_sql(outcome, "ck_outcome_records_revision_chain")
    assert "previous_revision_no = revision_no - 1" in _check_sql(
        outcome, "ck_outcome_records_revision_chain"
    )
    identity_check = _check_sql(outcome, "ck_outcome_records_kind_decision")
    assert all(token in identity_check for token in ("signal", "trade", "research", "decision_id"))
    assert all(token in _check_sql(outcome, "ck_outcome_records_status") for token in (
        "available",
        "provisional",
        "unavailable",
    ))


def test_phase99_outcome_label_contract_requires_complete_explicit_json():
    import poseidon.models as models
    from poseidon.models.outcome import outcome_label_contract_sha256

    valid = {
        "calendar": {"name": "XNYS", "version": "2026a"},
        "horizons": {
            "signal": [{"key": "5d", "bars": 5}],
            "trade": [{"key": "5d", "bars": 5}],
            "research": [{"key": "expiry", "rule": "assessment"}],
        },
        "benchmark": "SPY",
        "reporting_currency": "USD",
        "required_cost_components": ["commission", "tax"],
        "cost_model_version": "paper-cost-v1",
        "fx_model_version": "wm-close-v1",
        "pnl_tolerance": "0.000000000000000001",
        "research_assessment": {"types": ["expiry", "invalidation"]},
        "counterfactual_assumption_versions": ["paper-execution-v1"],
    }
    record = models.OutcomeLabelContract(
        version="pilot-v1",
        contract_json=valid,
        contract_sha256=outcome_label_contract_sha256(valid),
    )
    record.verify_contract()

    for missing in valid:
        incomplete = dict(valid)
        incomplete.pop(missing)
        invalid = models.OutcomeLabelContract(
            version="pilot-v1",
            contract_json=incomplete,
            contract_sha256="0" * 64,
        )
        with pytest.raises(ValueError, match=missing):
            invalid.verify_contract()


def test_phase99_fill_cost_metadata_preserves_decimal_fx_and_unbranched_revisions():
    import poseidon.models as models

    revision = models.FillCostRevision.__table__
    component = models.FillCostComponent.__table__
    assert revision.c.fill_key_sha256.nullable is False
    assert revision.c.previous_revision_no.nullable is True
    assert _unique_constraints(revision) >= {
        ("uq_fill_cost_revisions_replay", ("fill_key_sha256", "input_sha256")),
        ("uq_fill_cost_revisions_revision", ("fill_key_sha256", "revision_no")),
        ("uq_fill_cost_revisions_previous", ("previous_fill_cost_revision_id",)),
    }
    assert (
        "fk_fill_cost_revisions_previous_key_revision",
        ("fill_key_sha256", "previous_revision_no"),
        ("fill_cost_revisions.fill_key_sha256", "fill_cost_revisions.revision_no"),
    ) in _foreign_key_constraints(revision)
    chain_check = _check_sql(revision, "ck_fill_cost_revisions_revision_chain")
    assert "revision_no = 1" in chain_check
    assert "previous_revision_no = revision_no - 1" in chain_check
    assert "order_fill_id IS NULL" in _check_sql(revision, "ck_fill_cost_revisions_fill_xor")
    assert _index_contracts(revision) >= {
        (
            "uq_fill_cost_revisions_order_fill_revision",
            ("order_fill_id", "revision_no"),
            True,
            "order_fill_id IS NOT NULL",
        ),
        (
            "uq_fill_cost_revisions_paper_fill_revision",
            ("paper_broker_fill_id", "revision_no"),
            True,
            "paper_broker_fill_id IS NOT NULL",
        ),
    }

    for field in ("native_amount", "reporting_amount", "fx_rate"):
        assert isinstance(component.c[field].type, sa.Numeric)
        assert (component.c[field].type.precision, component.c[field].type.scale) == (38, 18)
    assert all(token in _check_sql(component, "ck_fill_cost_components_type") for token in (
        "commission",
        "tax",
        "funding",
        "borrow",
        "fx",
        "other",
    ))
    assert all(token in _check_sql(component, "ck_fill_cost_components_classification") for token in (
        "actual",
        "estimated",
        "not_applicable",
        "unavailable",
    ))
    assert "fx_source IS NOT NULL" in _check_sql(component, "ck_fill_cost_components_cross_currency_fx")

    forbidden = ("amount", "cost", "fx_rate", "gross", "net")
    for model in (models.FillCostRevision, models.FillCostComponent, models.EconomicReconciliation, models.OutcomeRecord):
        for column in model.__table__.c:
            if any(token in column.name for token in forbidden):
                assert not isinstance(column.type, sa.Float)


def test_phase99_research_assessment_and_economic_reconciliation_are_bounded_facts():
    import poseidon.models as models

    assessment = models.ResearchAssessment.__table__
    targets = {fk.target_fullname for column in assessment.c for fk in column.foreign_keys}
    assert {"evaluation_snapshots.id", "research_revisions.id", "data_manifests.id"} <= targets
    assert _unique_constraints(assessment) == {
        ("uq_research_assessments_replay", ("evaluation_snapshot_id", "input_sha256"))
    }
    assert all(token in _check_sql(assessment, "ck_research_assessments_type") for token in (
        "expiry",
        "invalidation",
    ))
    assert all(token in _check_sql(assessment, "ck_research_assessments_status") for token in (
        "confirmed",
        "not_confirmed",
        "unavailable",
    ))
    assert "jsonb_array_length" in _check_sql(assessment, "ck_research_assessments_citations")

    reconciliation = models.EconomicReconciliation.__table__
    assert {fk.target_fullname for fk in reconciliation.c.account_reconciliation_id.foreign_keys} == {
        "account_reconciliations.id"
    }
    assert _unique_constraints(reconciliation) == {
        (
            "uq_economic_reconciliations_replay",
            ("account_reconciliation_id", "input_sha256"),
        )
    }
    assert all(token in _check_sql(reconciliation, "ck_economic_reconciliations_status") for token in (
        "matched",
        "provisional",
        "unavailable",
    ))
