"""PostgreSQL enforcement for immutable release facts and guarded authority writes."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError


pytestmark = pytest.mark.postgresql


def _assert_sqlstate(session, expected, statement, **params):
    savepoint = session.begin_nested()
    try:
        with pytest.raises(DBAPIError) as error:
            session.execute(text(statement), params)
        sqlstate = getattr(error.value.orig, "sqlstate", None) or getattr(error.value.orig, "pgcode", None)
        assert sqlstate == expected
    finally:
        savepoint.rollback()


def _seed_release_graph(session):
    ids = {name: uuid.uuid4() for name in (
        "strategy", "other_strategy", "version", "other_version", "campaign", "holdout", "review",
        "reconciliation", "event",
    )}
    session.execute(
        text(
            "INSERT INTO strategies (id,name,strategy_type,config,symbol,market,interval,active) VALUES "
            "(:strategy,'phase100-a','test','{}','TEST','test','1d',false),"
            "(:other_strategy,'phase100-b','test','{}','OTHER','test','1d',false)"
        ),
        ids,
    )
    session.execute(
        text(
            "INSERT INTO strategy_versions "
            "(id,strategy_id,version_no,status,config_json,policy_json,artifact_json,content_sha256) VALUES "
            "(:version,:strategy,1,'draft','{}','{}','{}',:digest_a),"
            "(:other_version,:other_strategy,1,'draft','{}','{}','{}',:digest_b)"
        ),
        {**ids, "digest_a": "a" * 64, "digest_b": "b" * 64},
    )
    session.execute(
        text(
            "INSERT INTO experiment_campaigns "
            "(id,incumbent_strategy_version_id,candidate_strategy_version_id,incumbent_content_sha256,"
            "candidate_content_sha256,declared_difference_json,hypothesis,contract_json,contract_sha256,created_by) "
            "VALUES (:campaign,:version,:version,:digest_a,:digest_a,'{}','test','{}',:digest_c,'phase100')"
        ),
        {**ids, "digest_a": "a" * 64, "digest_c": "c" * 64},
    )
    session.execute(
        text(
            "INSERT INTO holdout_uses "
            "(id,holdout_identity_sha256,campaign_id,campaign_contract_sha256,audit_json,consumed_at) "
            "VALUES (:holdout,:digest_d,:campaign,:digest_c,'{}',now())"
        ),
        {**ids, "digest_c": "c" * 64, "digest_d": "d" * 64},
    )
    identity = '{"campaign_contract_sha256":"' + "c" * 64 + '","holdout_identity_sha256":"' + "d" * 64 + '"}'
    session.execute(
        text(
            "INSERT INTO campaign_reviews "
            "(id,campaign_id,input_sha256,result_sha256,status,result_json,holdout_use_id,input_identity_json) "
            "VALUES (:review,:campaign,:digest_e,:digest_f,'passed','{}',:holdout,CAST(:identity AS jsonb))"
        ),
        {**ids, "digest_e": "e" * 64, "digest_f": "f" * 64, "identity": identity},
    )
    session.execute(
        text(
            "INSERT INTO account_reconciliations "
            "(id,account_scope,account_generation,as_of,broker_state_watermark,internal_state_watermark,"
            "broker_snapshot_sha256,broker_snapshot_json,internal_snapshot_json,difference_json,policy_sha256,status) "
            "VALUES (:reconciliation,'paper','g1',now(),'b','i',:digest_a,'{}','{}','{}',:digest_b,'matched')"
        ),
        {**ids, "digest_a": "a" * 64, "digest_b": "b" * 64},
    )
    session.flush()
    return ids


def _insert_release_event(session, ids):
    session.execute(
        text(
            "INSERT INTO strategy_release_events "
            "(id,strategy_id,event_type,candidate_version_id,from_version_id,to_version_id,campaign_id,review_id,"
            "holdout_use_id,reconciliation_id,actor_id,account_scope,expected_revision,resulting_revision,"
            "idempotency_key,request_sha256,reason,gate_json,response_json) "
            "VALUES (:event,:strategy,'release',:version,NULL,:version,:campaign,:review,:holdout,:reconciliation,"
            "'release-manager','paper',0,1,'release-1',:digest,'approved','[]','{}')"
        ),
        {**ids, "digest": "9" * 64},
    )
    session.flush()


def test_release_event_is_append_only_and_guard_allows_exact_same_transaction_update(phase99_session_factory):
    with phase99_session_factory() as session:
        ids = _seed_release_graph(session)
        _insert_release_event(session, ids)
        session.execute(text("SELECT set_config('poseidon.release_event_id', :event, true)"), {"event": str(ids["event"])})
        session.execute(
            text(
                "UPDATE strategies SET active_version_id = :version,release_revision = 1 "
                "WHERE id = :strategy"
            ),
            ids,
        )
        assert session.execute(
            text("SELECT active_version_id,release_revision FROM strategies WHERE id = :strategy"), ids
        ).one() == (ids["version"], 1)
        _assert_sqlstate(
            session,
            "23514",
            "UPDATE strategy_release_events SET reason = 'changed' WHERE id = :event",
            **ids,
        )
        _assert_sqlstate(session, "23514", "DELETE FROM strategy_release_events WHERE id = :event", **ids)
        session.rollback()


@pytest.mark.parametrize(
    "assignment",
    (
        "active = true",
        "release_revision = release_revision + 1",
        "active_version_id = NULL",
    ),
)
def test_raw_strategy_authority_writes_fail_closed(phase99_session_factory, assignment):
    with phase99_session_factory() as session:
        ids = _seed_release_graph(session)
        _assert_sqlstate(
            session,
            "23514",
            f"UPDATE strategies SET {assignment} WHERE id = :strategy",
            **ids,
        )
        session.rollback()


def test_cross_strategy_version_cannot_enter_release_event_or_active_pointer(phase99_session_factory):
    with phase99_session_factory() as session:
        ids = _seed_release_graph(session)
        _assert_sqlstate(
            session,
            "23503",
            "INSERT INTO strategy_release_events "
            "(strategy_id,event_type,candidate_version_id,to_version_id,actor_id,account_scope,expected_revision,"
            "resulting_revision,idempotency_key,request_sha256,reason,gate_json,response_json) "
            "VALUES (:strategy,'release',:other_version,:other_version,'release-manager','paper',0,1,"
            "'cross-strategy',:digest,'invalid','[]','{}')",
            **ids,
            digest="8" * 64,
        )
        session.rollback()


def test_new_review_requires_exact_holdout_and_input_identity_binding(phase99_session_factory):
    with phase99_session_factory() as session:
        ids = _seed_release_graph(session)
        _assert_sqlstate(
            session,
            "23514",
            "INSERT INTO campaign_reviews (campaign_id,input_sha256,result_sha256,status,result_json) "
            "VALUES (:campaign,:digest_a,:digest_b,'passed','{}')",
            **ids,
            digest_a="1" * 64,
            digest_b="2" * 64,
        )
        session.rollback()
