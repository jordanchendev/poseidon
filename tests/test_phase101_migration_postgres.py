"""Authentic PostgreSQL migration matrix for bootstrap release migration 046."""

import os
import re
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

from alembic import command

pytestmark = pytest.mark.postgresql
ROOT = Path(__file__).resolve().parents[1]
AUDITED_TABLES = ("strategies", "strategy_versions", "strategy_release_events")


@pytest.fixture
def phase101_migration_database(monkeypatch):
    source = os.environ.get("POSEIDON_REAL_DATABASE_URL")
    if not source:
        pytest.fail("POSEIDON_REAL_DATABASE_URL is required for the Phase 101 migration matrix")
    source_url = make_url(source)
    database_name = f"phase101_matrix_{uuid.uuid4().hex[:16]}"
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


def _seed_044_strategy(engine):
    """Seed a 044-state strategy with two versions and one campaign, zero release events."""
    ids = {name: uuid.uuid4() for name in ("strategy", "incumbent", "candidate", "campaign")}
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO strategies (id,name,strategy_type,config,symbol,market,interval,active) "
                "VALUES (:strategy,'phase101-bootstrap','test','{}','TEST','test','1d',false)"
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
                "VALUES (:campaign,:incumbent,:candidate,:digest_a,:digest_b,'{}','phase101','{}',:digest_c,'phase101')"
            ),
            {**ids, "digest_a": "a" * 64, "digest_b": "b" * 64, "digest_c": "c" * 64},
        )
    return ids


BOOTSTRAP_INSERT = (
    "INSERT INTO strategy_release_events "
    "(strategy_id,event_type,candidate_version_id,from_version_id,to_version_id,campaign_id,"
    "actor_id,account_scope,expected_revision,resulting_revision,idempotency_key,request_sha256,"
    "reason,gate_json,response_json) "
    "VALUES (:strategy,'bootstrap',:candidate,:from_version,:candidate,:campaign_id,"
    "'release-manager','paper',:expected_revision,:expected_revision + 1,:key,:digest,"
    "'phase101 bootstrap','{\"bootstrap\": true}','{}')"
)


def _bootstrap_params(ids, *, key="bootstrap-1", from_version=None, campaign_id=None, expected_revision=0):
    return {
        "strategy": ids["strategy"],
        "candidate": ids["candidate"],
        "from_version": from_version,
        "campaign_id": campaign_id,
        "expected_revision": expected_revision,
        "key": key,
        "digest": "d" * 64,
    }


def _version(engine):
    with engine.connect() as connection:
        return connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()


def _fingerprint(engine):
    result = {}
    with engine.connect() as connection:
        for table in AUDITED_TABLES:
            result[table] = connection.execute(
                text(f"SELECT count(*), md5(coalesce(string_agg(t::text, '' ORDER BY t.id), '')) FROM {table} AS t")
            ).one()
    return result


def _check_definition(engine, name):
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = :name"),
            {"name": name},
        ).scalar_one()


def _assert_check_violation(engine, statement, params):
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with pytest.raises(DBAPIError) as error:
                connection.execute(text(statement), params)
            sqlstate = getattr(error.value.orig, "sqlstate", None) or getattr(error.value.orig, "pgcode", None)
            assert sqlstate == "23514"
        finally:
            transaction.rollback()


def test_046_empty_upgrade_head_and_checks(phase101_migration_database):
    config, engine = phase101_migration_database
    command.upgrade(config, "head")
    assert _version(engine) == "046"
    assert "'bootstrap'" in _check_definition(engine, "ck_strategy_release_events_type")
    shape = _check_definition(engine, "ck_strategy_release_events_shape")
    assert "'bootstrap'" in shape
    assert "'release'" in shape and "'pause'" in shape and "'rollback'" in shape and "'reject'" in shape


def test_046_round_trip_preserves_rows(phase101_migration_database):
    config, engine = phase101_migration_database
    command.upgrade(config, "044")
    _seed_044_strategy(engine)
    baseline = _fingerprint(engine)
    assert baseline["strategies"][0] == 1
    assert baseline["strategy_versions"][0] == 2
    assert baseline["strategy_release_events"][0] == 0

    command.upgrade(config, "046")
    assert _version(engine) == "046"
    assert _fingerprint(engine) == baseline

    command.downgrade(config, "044")
    assert _version(engine) == "044"
    assert _fingerprint(engine) == baseline
    assert "'bootstrap'" not in _check_definition(engine, "ck_strategy_release_events_type")
    assert "'bootstrap'" not in _check_definition(engine, "ck_strategy_release_events_shape")

    command.upgrade(config, "046")
    assert _version(engine) == "046"
    assert _fingerprint(engine) == baseline
    assert "'bootstrap'" in _check_definition(engine, "ck_strategy_release_events_type")


def test_046_downgrade_refuses_with_bootstrap_event(phase101_migration_database):
    config, engine = phase101_migration_database
    command.upgrade(config, "044")
    ids = _seed_044_strategy(engine)
    command.upgrade(config, "046")
    with engine.begin() as connection:
        connection.execute(text(BOOTSTRAP_INSERT), _bootstrap_params(ids))
    before = _fingerprint(engine)
    assert before["strategy_release_events"][0] == 1

    with pytest.raises(RuntimeError, match="migration 046 downgrade refused"):
        command.downgrade(config, "044")

    assert _version(engine) == "046"
    assert _fingerprint(engine) == before
    assert "'bootstrap'" in _check_definition(engine, "ck_strategy_release_events_type")


def test_046_bootstrap_shape_rejects_campaign_evidence(phase101_migration_database):
    config, engine = phase101_migration_database
    command.upgrade(config, "044")
    ids = _seed_044_strategy(engine)
    command.upgrade(config, "046")

    _assert_check_violation(
        engine, BOOTSTRAP_INSERT, _bootstrap_params(ids, key="with-campaign", campaign_id=ids["campaign"])
    )
    _assert_check_violation(
        engine, BOOTSTRAP_INSERT, _bootstrap_params(ids, key="with-from", from_version=ids["incumbent"])
    )
    _assert_check_violation(engine, BOOTSTRAP_INSERT, _bootstrap_params(ids, key="with-revision", expected_revision=1))
    assert _fingerprint(engine)["strategy_release_events"][0] == 0

    # The campaign-free shape itself is admitted.
    with engine.begin() as connection:
        connection.execute(text(BOOTSTRAP_INSERT), _bootstrap_params(ids))
    assert _fingerprint(engine)["strategy_release_events"][0] == 1


def test_046_direct_pointer_write_still_rejected(phase101_migration_database):
    config, engine = phase101_migration_database
    command.upgrade(config, "044")
    ids = _seed_044_strategy(engine)
    command.upgrade(config, "046")

    _assert_check_violation(
        engine,
        "UPDATE strategies SET active_version_id = :candidate WHERE id = :strategy",
        ids,
    )
    # Even with a committed bootstrap event, a later transaction cannot reuse it as a guard.
    with engine.begin() as connection:
        event_id = connection.execute(text(BOOTSTRAP_INSERT + " RETURNING id"), _bootstrap_params(ids)).scalar_one()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(
                text("SELECT set_config('poseidon.release_event_id', :event, true)"),
                {"event": str(event_id)},
            )
            with pytest.raises(DBAPIError) as error:
                connection.execute(
                    text(
                        "UPDATE strategies SET active_version_id = :candidate, release_revision = 1 "
                        "WHERE id = :strategy"
                    ),
                    ids,
                )
            sqlstate = getattr(error.value.orig, "sqlstate", None) or getattr(error.value.orig, "pgcode", None)
            assert sqlstate == "23514"
        finally:
            transaction.rollback()
    with engine.connect() as connection:
        pointer = connection.execute(
            text("SELECT active_version_id, release_revision FROM strategies WHERE id = :strategy"),
            ids,
        ).one()
    assert pointer == (None, 0)
