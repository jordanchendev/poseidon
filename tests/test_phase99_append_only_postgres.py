"""PostgreSQL-level append-only enforcement for Phase 99 facts."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError


pytestmark = pytest.mark.postgresql


def _execute(session, statement, **params):
    session.execute(text(statement), params)


def _assert_sqlstate_23514(session, statement, **params):
    _assert_sqlstate(session, "23514", statement, **params)


def _assert_sqlstate(session, expected, statement, **params):
    savepoint = session.begin_nested()
    try:
        with pytest.raises(DBAPIError) as error:
            session.execute(text(statement), params)
        sqlstate = getattr(error.value.orig, "sqlstate", None) or getattr(error.value.orig, "pgcode", None)
        assert sqlstate == expected
    finally:
        savepoint.rollback()


def _seed_phase99_graph(session):
    ids = {name: uuid.uuid4() for name in (
        "strategy", "manifest", "research", "version_a", "version_b", "evaluation_run",
        "evaluation", "order", "fill", "reconciliation", "label", "assessment",
        "cost_revision", "cost_component", "economic", "outcome", "campaign", "event",
        "holdout", "review", "linked_experiment", "legacy_experiment",
    )}
    digest = "a" * 64
    digest_b = "b" * 64

    statements = (
        ("INSERT INTO strategies (id,name,strategy_type,config,symbol,market,interval,active) "
         "VALUES (:strategy,'phase99','test','{}','TEST','test','1d',false)", ids),
        ("INSERT INTO data_manifests (id,content_sha256,market,interval,as_of,capability_json,sources_json,payload_json) "
         "VALUES (:manifest,:digest,'test','1d',now(),'{}','[]','{}')", {**ids, "digest": digest}),
        ("INSERT INTO research_revisions (id,scope_key,manifest_id,request_sha256,policy_version,provider,model,runtime_digest,status,research_json) "
         "VALUES (:research,'phase99',:manifest,:digest,'v1','bounded','none','runtime','complete','{}')",
         {**ids, "digest": digest_b}),
        ("INSERT INTO strategy_versions (id,strategy_id,version_no,status,config_json,policy_json,artifact_json,content_sha256) "
         "VALUES (:version_a,:strategy,1,'draft','{}','{}','{}',:digest),"
         "(:version_b,:strategy,2,'draft','{}','{}','{}',:digest_b)", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO evaluation_runs (id,strategy_version_id,manifest_id,decision_as_of,universe_json,input_sha256,status,coverage_json) "
         "VALUES (:evaluation_run,:version_a,:manifest,now(),'[]',:digest,'complete','{}')", {**ids, "digest": digest}),
        ("INSERT INTO evaluation_snapshots (id,evaluation_run_id,symbol,market,instrument,status,recommendation_json,technical_json,research_revision_ids,reason_codes,content_sha256) "
         "VALUES (:evaluation,:evaluation_run,'TEST','test','spot','complete','{}','{}','[]','[]',:digest)", {**ids, "digest": digest}),
        ("INSERT INTO orders (id,strategy_name,symbol,market,action,target_weight,quantity,broker_mode) "
         "VALUES (:order,'phase99','TEST','test','buy',1,1,'paper')", ids),
        ("INSERT INTO order_fills (id,order_id,fill_price,fill_quantity,fill_time,broker_fill_id) "
         "VALUES (:fill,:order,1,1,now(),'phase99-fill')", ids),
        ("INSERT INTO account_reconciliations (id,account_scope,account_generation,as_of,broker_state_watermark,internal_state_watermark,broker_snapshot_sha256,broker_snapshot_json,internal_snapshot_json,difference_json,policy_sha256,status) "
         "VALUES (:reconciliation,'phase99','g1',now(),'b','i',:digest,'{}','{}','{}',:digest_b,'matched')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO outcome_label_contracts (id,version,contract_json,contract_sha256) "
         "VALUES (:label,'v1','{}',:digest)", {**ids, "digest": digest}),
        ("INSERT INTO research_assessments (id,evaluation_snapshot_id,research_revision_id,manifest_id,assessment_type,assessment_at,status,citation_ids_json,input_sha256,content_sha256,assessment_json) "
         "VALUES (:assessment,:evaluation,:research,:manifest,'expiry',now(),'confirmed','[\"citation\"]',:digest,:digest_b,'{}')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO fill_cost_revisions (id,order_fill_id,fill_key_sha256,reporting_currency,cost_model_version,input_sha256,content_sha256,revision_no) "
         "VALUES (:cost_revision,:fill,:digest,'USD','v1',:digest,:digest_b,1)", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO fill_cost_components (id,fill_cost_revision_id,component_type,native_amount,native_currency,reporting_amount,reporting_currency,classification,source,cost_model_version) "
         "VALUES (:cost_component,:cost_revision,'commission',1,'USD',1,'USD','actual','broker','v1')", ids),
        ("INSERT INTO economic_reconciliations (id,account_reconciliation_id,input_sha256,content_sha256,reporting_currency,cost_revision_ids_json,fx_facts_json,status,reason_codes_json) "
         "VALUES (:economic,:reconciliation,:digest,:digest_b,'USD','[]','[]','matched','[]')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO outcome_records (id,evaluation_snapshot_id,kind,label_contract_id,horizon_key,manifest_id,logical_key_sha256,input_sha256,content_sha256,revision_no,maturity_at,status,reason_code,metrics_json) "
         "VALUES (:outcome,:evaluation,'signal',:label,'1d',:manifest,:digest,:digest,:digest_b,1,now(),'available','mature','{}')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO experiment_campaigns (id,incumbent_strategy_version_id,candidate_strategy_version_id,incumbent_content_sha256,candidate_content_sha256,declared_difference_json,hypothesis,contract_json,contract_sha256,created_by) "
         "VALUES (:campaign,:version_a,:version_b,:digest,:digest_b,jsonb_build_object('change',true),'test','{}',:digest,'phase99')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO campaign_events (id,campaign_id,event_type,payload_json,idempotency_sha256) "
         "VALUES (:event,:campaign,'created','{}',:digest)", {**ids, "digest": digest}),
        ("INSERT INTO holdout_uses (id,holdout_identity_sha256,campaign_id,campaign_contract_sha256,audit_json,consumed_at) "
         "VALUES (:holdout,:digest_b,:campaign,:digest,'{}',now())", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO campaign_reviews (id,campaign_id,input_sha256,result_sha256,status,result_json) "
         "VALUES (:review,:campaign,:digest,:digest_b,'passed','{}')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO experiments (id,study_name,config_json,status,market,interval,campaign_id,original_trial_id,trial_role,strategy_version_id,ablation_arm,paired_sample_key_sha256,input_sha256,result_sha256,started_at,completed_at,terminal_state,terminal_reason_json) "
         "VALUES (:linked_experiment,'phase99','{}','complete','test','1d',:campaign,'trial-1','search',:version_b,'combined',:digest,:digest,:digest_b,now(),now(),'succeeded','{}')", {**ids, "digest": digest, "digest_b": digest_b}),
        ("INSERT INTO experiments (id,study_name,config_json,status,market,interval) "
         "VALUES (:legacy_experiment,'legacy','{}','running','test','1d')", ids),
    )
    for statement, params in statements:
        _execute(session, statement, **params)
    session.flush()
    return ids


@pytest.mark.parametrize(
    ("table", "key", "assignment"),
    (
        ("outcome_label_contracts", "label", "version = 'changed'"),
        ("research_assessments", "assessment", "status = 'unavailable'"),
        ("fill_cost_revisions", "cost_revision", "reporting_currency = 'EUR'"),
        ("fill_cost_components", "cost_component", "source = 'changed'"),
        ("economic_reconciliations", "economic", "status = 'provisional'"),
        ("outcome_records", "outcome", "status = 'provisional'"),
        ("experiment_campaigns", "campaign", "created_by = 'changed'"),
        ("campaign_events", "event", "event_type = 'changed'"),
        ("holdout_uses", "holdout", "audit_json = jsonb_build_object('changed',true)"),
        ("campaign_reviews", "review", "status = 'failed'"),
    ),
)
def test_phase99_authoritative_rows_reject_direct_update_delete_and_remain_stable(
    phase99_session_factory, table, key, assignment
):
    with phase99_session_factory() as session:
        ids = _seed_phase99_graph(session)
        row_id = ids[key]
        before = session.execute(text(f"SELECT to_jsonb(t)::text FROM {table} AS t WHERE id = :id"), {"id": row_id}).scalar_one()
        _assert_sqlstate_23514(session, f"UPDATE {table} SET {assignment} WHERE id = :id", id=row_id)
        _assert_sqlstate_23514(session, f"DELETE FROM {table} WHERE id = :id", id=row_id)
        after = session.execute(text(f"SELECT to_jsonb(t)::text FROM {table} AS t WHERE id = :id"), {"id": row_id}).scalar_one()
        assert after == before
        session.rollback()


def test_campaign_linked_experiment_is_immutable_but_unbound_legacy_remains_mutable(phase99_session_factory):
    with phase99_session_factory() as session:
        ids = _seed_phase99_graph(session)
        linked = ids["linked_experiment"]
        legacy = ids["legacy_experiment"]

        _assert_sqlstate_23514(session, "UPDATE experiments SET status = 'failed' WHERE id = :id", id=linked)
        _assert_sqlstate_23514(session, "DELETE FROM experiments WHERE id = :id", id=linked)
        _assert_sqlstate_23514(
            session,
            "UPDATE experiments SET campaign_id = :campaign WHERE id = :id",
            id=legacy,
            campaign=ids["campaign"],
        )

        assert session.execute(text("SELECT campaign_id FROM experiments WHERE id = :id"), {"id": legacy}).scalar_one() is None
        legacy_extensions = session.execute(
            text(
                "SELECT campaign_id, original_trial_id, trial_role, strategy_version_id, ablation_arm, "
                "paired_sample_key_sha256, input_sha256, result_sha256, started_at, completed_at, "
                "terminal_state, terminal_reason_json FROM experiments WHERE id = :id"
            ),
            {"id": legacy},
        ).one()
        assert all(value is None for value in legacy_extensions)
        _execute(session, "UPDATE experiments SET status = 'complete' WHERE id = :id", id=legacy)
        assert session.execute(text("SELECT status FROM experiments WHERE id = :id"), {"id": legacy}).scalar_one() == "complete"
        _execute(session, "DELETE FROM experiments WHERE id = :id", id=legacy)
        assert session.execute(text("SELECT count(*) FROM experiments WHERE id = :id"), {"id": legacy}).scalar_one() == 0
        session.rollback()


def test_predecessor_uuid_key_and_revision_must_resolve_to_the_same_row(phase99_session_factory):
    with phase99_session_factory() as session:
        ids = _seed_phase99_graph(session)
        second_order = uuid.uuid4()
        second_fill = uuid.uuid4()
        second_cost = uuid.uuid4()
        second_outcome = uuid.uuid4()
        key_a = "a" * 64
        key_b = "c" * 64
        digest = "d" * 64

        _execute(
            session,
            "INSERT INTO orders (id,strategy_name,symbol,market,action,target_weight,quantity,broker_mode) "
            "VALUES (:id,'phase99-2','TEST','test','buy',1,1,'paper')",
            id=second_order,
        )
        _execute(
            session,
            "INSERT INTO order_fills (id,order_id,fill_price,fill_quantity,fill_time,broker_fill_id) "
            "VALUES (:id,:order,1,1,now(),'phase99-fill-2')",
            id=second_fill,
            order=second_order,
        )
        _execute(
            session,
            "INSERT INTO fill_cost_revisions "
            "(id,order_fill_id,fill_key_sha256,reporting_currency,cost_model_version,input_sha256,"
            "content_sha256,revision_no) VALUES (:id,:fill,:key,'USD','v1',:input,:content,1)",
            id=second_cost,
            fill=second_fill,
            key=key_b,
            input=digest,
            content="e" * 64,
        )
        _assert_sqlstate(
            session,
            "23503",
            "INSERT INTO fill_cost_revisions "
            "(id,order_fill_id,fill_key_sha256,reporting_currency,cost_model_version,input_sha256,"
            "content_sha256,revision_no,previous_fill_cost_revision_id,previous_revision_no) "
            "VALUES (:id,:fill,:key,'USD','v1',:input,:content,2,:previous,1)",
            id=uuid.uuid4(),
            fill=ids["fill"],
            key=key_a,
            input="f" * 64,
            content="0" * 64,
            previous=second_cost,
        )

        _execute(
            session,
            "INSERT INTO outcome_records "
            "(id,evaluation_snapshot_id,kind,label_contract_id,horizon_key,manifest_id,logical_key_sha256,"
            "input_sha256,content_sha256,revision_no,maturity_at,status,reason_code,metrics_json) "
            "VALUES (:id,:evaluation,'signal',:label,'2d',:manifest,:key,:input,:content,1,now(),"
            "'available','mature','{}')",
            id=second_outcome,
            evaluation=ids["evaluation"],
            label=ids["label"],
            manifest=ids["manifest"],
            key=key_b,
            input=digest,
            content="e" * 64,
        )
        _assert_sqlstate(
            session,
            "23503",
            "INSERT INTO outcome_records "
            "(id,evaluation_snapshot_id,kind,label_contract_id,horizon_key,manifest_id,logical_key_sha256,"
            "input_sha256,content_sha256,revision_no,previous_outcome_id,previous_revision_no,maturity_at,"
            "status,reason_code,metrics_json) VALUES (:id,:evaluation,'signal',:label,'1d',:manifest,:key,"
            ":input,:content,2,:previous,1,now(),'available','mature','{}')",
            id=uuid.uuid4(),
            evaluation=ids["evaluation"],
            label=ids["label"],
            manifest=ids["manifest"],
            key=key_a,
            input="f" * 64,
            content="0" * 64,
            previous=second_outcome,
        )
        session.rollback()
