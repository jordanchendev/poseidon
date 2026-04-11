"""Unit coverage for Phase 47 factor analysis ORM and schemas."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):  # pragma: no cover
    return "VARCHAR(36)"


from poseidon.core.schemas import ShapleyAnalysisRequest  # noqa: E402
from poseidon.models.base import Base  # noqa: E402
from poseidon.models.factor_analysis_run import FactorAnalysisRun  # noqa: E402


ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


def setup_function():
    Base.metadata.create_all(bind=ENGINE)


def teardown_function():
    Base.metadata.drop_all(bind=ENGINE)


def test_shapley_analysis_request_rejects_invalid_uuid():
    with pytest.raises(ValidationError):
        ShapleyAnalysisRequest(model_version_id="not-a-uuid")


def test_shapley_analysis_request_accepts_uuid_string():
    payload = ShapleyAnalysisRequest(model_version_id=str(uuid.uuid4()))

    assert isinstance(payload.model_version_id, str)


def test_factor_analysis_run_declares_expected_constraints_and_indexes():
    table = FactorAnalysisRun.__table__
    constraint_names = {constraint.name for constraint in table.constraints}
    index_names = {index.name for index in table.indexes}

    assert "ck_factor_analysis_runs_run_type" in constraint_names
    assert "ck_factor_analysis_runs_status" in constraint_names
    assert "ix_factor_analysis_runs_market" in index_names
    assert "ix_factor_analysis_runs_created_at" in index_names


def test_factor_analysis_run_table_creates_in_sqlite():
    inspector = inspect(ENGINE)
    columns = {column["name"] for column in inspector.get_columns("factor_analysis_runs")}

    assert "id" in columns
    assert "run_type" in columns
    assert "results_json" in columns
    assert "status" in columns
