"""Tests for ExperimentRecord ORM model (Task 1 TDD RED)."""

from poseidon.models.experiment import ExperimentRecord


def test_experiment_record_tablename():
    assert ExperimentRecord.__tablename__ == "experiments"


def test_experiment_record_columns():
    col_names = [c.name for c in ExperimentRecord.__table__.columns]
    expected = [
        "id",
        "study_name",
        "config_json",
        "metrics_json",
        "composite_score",
        "wfe_score",
        "status",
        "market",
        "interval",
        "optuna_study_name",
        "optuna_trial_number",
        "holdout_boundary",
        "created_at",
        "updated_at",
    ]
    for name in expected:
        assert name in col_names, f"Missing column: {name}"


def test_experiment_record_indexes():
    index_names = [idx.name for idx in ExperimentRecord.__table__.indexes]
    assert "ix_experiments_market_interval" in index_names
    assert "ix_experiments_created_at" in index_names
