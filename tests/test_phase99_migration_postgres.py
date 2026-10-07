"""Authentic PostgreSQL migration matrix for the Phase 99 boundary."""

import os
import re
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url


pytestmark = pytest.mark.postgresql
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def phase99_migration_database(monkeypatch):
    source = os.environ.get("POSEIDON_REAL_DATABASE_URL")
    if not source:
        pytest.fail("POSEIDON_REAL_DATABASE_URL is required for the Phase 99 migration matrix")
    source_url = make_url(source)
    database_name = f"phase99_matrix_{uuid.uuid4().hex[:16]}"
    assert re.fullmatch(r"[a-z0-9_]+", database_name)
    admin_engine = create_engine(source_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{database_name}"'))

    database_url = source_url.set(database=database_name).render_as_string(hide_password=False)
    monkeypatch.setenv("POSEIDON_DATABASE_URL", database_url)
    from poseidon.core.config import settings

    original_database_url = settings.database_url
    settings.database_url = database_url
    engine = create_engine(database_url)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    try:
        yield config, engine
    finally:
        settings.database_url = original_database_url
        engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{database_name}" WITH (FORCE)'))
        admin_engine.dispose()


def _seed_legacy_042(engine):
    ids = {name: uuid.uuid4() for name in ("order", "fill", "experiment")}
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO orders (id,strategy_name,symbol,market,action,target_weight,quantity,broker_mode) "
                "VALUES (:id,'phase99-legacy','TEST','test','buy',1,2,'paper')"
            ),
            {"id": ids["order"]},
        )
        connection.execute(
            text(
                "INSERT INTO order_fills (id,order_id,fill_price,fill_quantity,fill_time,broker_fill_id) "
                "VALUES (:id,:order,12.5,2,now(),'legacy-fill')"
            ),
            {"id": ids["fill"], "order": ids["order"]},
        )
        connection.execute(
            text(
                "INSERT INTO experiments (id,study_name,config_json,status,market,interval) "
                "VALUES (:id,'legacy-study',jsonb_build_object('lookback',20),'running','test','1d')"
            ),
            {"id": ids["experiment"]},
        )
    return ids


def test_empty_database_upgrades_to_the_single_043_head(phase99_migration_database):
    config, engine = phase99_migration_database
    command.upgrade(config, "head")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "043"
    assert set(inspect(engine).get_table_names()) >= {
        "outcome_label_contracts",
        "outcome_records",
        "fill_cost_revisions",
        "experiment_campaigns",
        "campaign_reviews",
    }


def test_seeded_042_legacy_rows_upgrade_unchanged_with_null_extensions(phase99_migration_database):
    config, engine = phase99_migration_database
    command.upgrade(config, "042")
    ids = _seed_legacy_042(engine)
    with engine.connect() as connection:
        before_fill = connection.execute(
            text("SELECT to_jsonb(f)::text FROM order_fills AS f WHERE id = :id"),
            {"id": ids["fill"]},
        ).scalar_one()
        before_experiment = connection.execute(
            text("SELECT to_jsonb(e)::text FROM experiments AS e WHERE id = :id"),
            {"id": ids["experiment"]},
        ).scalar_one()

    command.upgrade(config, "043")
    with engine.connect() as connection:
        after_fill = connection.execute(
            text("SELECT to_jsonb(f)::text FROM order_fills AS f WHERE id = :id"),
            {"id": ids["fill"]},
        ).scalar_one()
        original_experiment = connection.execute(
            text(
                "SELECT (to_jsonb(e) - ARRAY['campaign_id','original_trial_id','trial_role','strategy_version_id',"
                "'ablation_arm','paired_sample_key_sha256','input_sha256','result_sha256','started_at','completed_at',"
                "'terminal_state','terminal_reason_json']::text[])::text FROM experiments AS e WHERE id = :id"
            ),
            {"id": ids["experiment"]},
        ).scalar_one()
        extensions = connection.execute(
            text(
                "SELECT campaign_id,original_trial_id,trial_role,strategy_version_id,ablation_arm,"
                "paired_sample_key_sha256,input_sha256,result_sha256,started_at,completed_at,terminal_state,"
                "terminal_reason_json FROM experiments WHERE id = :id"
            ),
            {"id": ids["experiment"]},
        ).one()
    assert after_fill == before_fill
    assert original_experiment == before_experiment
    assert all(value is None for value in extensions)


def test_empty_043_downgrades_to_042_and_reupgrades(phase99_migration_database):
    config, engine = phase99_migration_database
    command.upgrade(config, "043")
    command.downgrade(config, "042")
    inspector = inspect(engine)
    assert "outcome_records" not in inspector.get_table_names()
    assert "campaign_id" not in {column["name"] for column in inspector.get_columns("experiments")}
    command.upgrade(config, "043")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "043"


def test_populated_043_refuses_downgrade_before_any_schema_mutation(phase99_migration_database):
    config, engine = phase99_migration_database
    command.upgrade(config, "043")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO outcome_label_contracts (version,contract_json,contract_sha256) "
                "VALUES ('matrix-v1','{}',:digest)"
            ),
            {"digest": "a" * 64},
        )

    with pytest.raises(RuntimeError, match="outcome or experiment history exists"):
        command.downgrade(config, "042")

    inspector = inspect(engine)
    expected_tables = {
        "outcome_label_contracts",
        "research_assessments",
        "fill_cost_revisions",
        "fill_cost_components",
        "economic_reconciliations",
        "outcome_records",
        "experiment_campaigns",
        "campaign_events",
        "holdout_uses",
        "campaign_reviews",
    }
    assert expected_tables <= set(inspector.get_table_names())
    assert set(name for name in (
        "campaign_id",
        "original_trial_id",
        "trial_role",
        "strategy_version_id",
        "ablation_arm",
        "paired_sample_key_sha256",
        "input_sha256",
        "result_sha256",
        "started_at",
        "completed_at",
        "terminal_state",
        "terminal_reason_json",
    )) <= {column["name"] for column in inspector.get_columns("experiments")}
