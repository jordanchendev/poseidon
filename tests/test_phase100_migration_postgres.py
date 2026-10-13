"""Authentic PostgreSQL migration matrix for release authority migration 044."""

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
def phase100_migration_database(monkeypatch):
    source = os.environ.get("POSEIDON_REAL_DATABASE_URL")
    if not source:
        pytest.fail("POSEIDON_REAL_DATABASE_URL is required for the Phase 100 migration matrix")
    source_url = make_url(source)
    database_name = f"phase100_matrix_{uuid.uuid4().hex[:16]}"
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


def _seed_legacy_043(engine):
    ids = {name: uuid.uuid4() for name in ("strategy", "incumbent", "candidate", "campaign", "review")}
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO strategies (id,name,strategy_type,config,symbol,market,interval,active) "
                "VALUES (:strategy,'phase100-legacy','test','{}','TEST','test','1d',true)"
            ),
            ids,
        )
        connection.execute(
            text(
                "INSERT INTO strategy_versions "
                "(id,strategy_id,version_no,status,config_json,policy_json,artifact_json,content_sha256) VALUES "
                "(:incumbent,:strategy,1,'draft','{}','{}','{}',:digest_a),"
                "(:candidate,:strategy,2,'draft','{}','{}','{}',:digest_b)"
            ),
            {**ids, "digest_a": "a" * 64, "digest_b": "b" * 64},
        )
        connection.execute(
            text(
                "INSERT INTO experiment_campaigns "
                "(id,incumbent_strategy_version_id,candidate_strategy_version_id,incumbent_content_sha256,"
                "candidate_content_sha256,declared_difference_json,hypothesis,contract_json,contract_sha256,created_by) "
                "VALUES (:campaign,:incumbent,:candidate,:digest_a,:digest_b,'{}','legacy','{}',:digest_c,'phase99')"
            ),
            {**ids, "digest_a": "a" * 64, "digest_b": "b" * 64, "digest_c": "c" * 64},
        )
        connection.execute(
            text(
                "INSERT INTO campaign_reviews (id,campaign_id,input_sha256,result_sha256,status,result_json) "
                "VALUES (:review,:campaign,:digest_a,:digest_b,'passed','{}')"
            ),
            {**ids, "digest_a": "a" * 64, "digest_b": "b" * 64},
        )
    return ids


def test_empty_database_upgrades_to_the_single_044_head(phase100_migration_database):
    config, engine = phase100_migration_database
    command.upgrade(config, "head")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "044"
    inspector = inspect(engine)
    assert "strategy_release_events" in inspector.get_table_names()
    assert {"active_version_id", "release_revision"} <= {
        column["name"] for column in inspector.get_columns("strategies")
    }


def test_seeded_043_upgrades_without_inferred_pointer_or_review_binding(phase100_migration_database):
    config, engine = phase100_migration_database
    command.upgrade(config, "043")
    ids = _seed_legacy_043(engine)
    with engine.connect() as connection:
        strategy_before = connection.execute(
            text("SELECT to_jsonb(s)::text FROM strategies AS s WHERE id = :id"),
            {"id": ids["strategy"]},
        ).scalar_one()
        review_before = connection.execute(
            text("SELECT to_jsonb(r)::text FROM campaign_reviews AS r WHERE id = :id"),
            {"id": ids["review"]},
        ).scalar_one()

    command.upgrade(config, "044")
    with engine.connect() as connection:
        strategy_legacy = connection.execute(
            text(
                "SELECT (to_jsonb(s) - ARRAY['active_version_id','release_revision']::text[])::text "
                "FROM strategies AS s WHERE id = :id"
            ),
            {"id": ids["strategy"]},
        ).scalar_one()
        pointer = connection.execute(
            text("SELECT active_version_id,release_revision,active FROM strategies WHERE id = :id"),
            {"id": ids["strategy"]},
        ).one()
        review_legacy = connection.execute(
            text(
                "SELECT (to_jsonb(r) - ARRAY['holdout_use_id','input_identity_json']::text[])::text "
                "FROM campaign_reviews AS r WHERE id = :id"
            ),
            {"id": ids["review"]},
        ).scalar_one()
        binding = connection.execute(
            text("SELECT holdout_use_id,input_identity_json FROM campaign_reviews WHERE id = :id"),
            {"id": ids["review"]},
        ).one()

    assert strategy_legacy == strategy_before
    assert pointer == (None, 0, True)
    assert review_legacy == review_before
    assert binding == (None, None)


def test_unused_044_downgrades_to_043_and_reupgrades(phase100_migration_database):
    config, engine = phase100_migration_database
    command.upgrade(config, "044")
    command.downgrade(config, "043")
    inspector = inspect(engine)
    assert "strategy_release_events" not in inspector.get_table_names()
    assert "active_version_id" not in {column["name"] for column in inspector.get_columns("strategies")}
    assert "holdout_use_id" not in {column["name"] for column in inspector.get_columns("campaign_reviews")}
    command.upgrade(config, "044")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "044"


def test_populated_044_refuses_downgrade_before_schema_mutation(phase100_migration_database):
    config, engine = phase100_migration_database
    command.upgrade(config, "043")
    ids = _seed_legacy_043(engine)
    command.upgrade(config, "044")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO strategy_release_events "
                "(strategy_id,event_type,candidate_version_id,actor_id,account_scope,expected_revision,"
                "resulting_revision,idempotency_key,request_sha256,reason,gate_json,response_json) "
                "VALUES (:strategy,'reject',:candidate,'release-manager','paper',0,1,'reject-1',:digest,"
                "'not promoted','[]','{}')"
            ),
            {**ids, "digest": "d" * 64},
        )

    with pytest.raises(RuntimeError, match="release authority history exists"):
        command.downgrade(config, "043")

    inspector = inspect(engine)
    assert "strategy_release_events" in inspector.get_table_names()
    assert {"active_version_id", "release_revision"} <= {
        column["name"] for column in inspector.get_columns("strategies")
    }
