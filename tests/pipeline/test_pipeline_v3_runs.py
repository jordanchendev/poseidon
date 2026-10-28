"""Spec v3 daily runs (step 4): frozen CONFIG_VERSION 6 config, stage plan, routing, creation, p003.

Unit tests need nothing; ``*_pg`` tests need the throwaway PostgreSQL (POSEIDON_PIPELINE_TEST_DATABASE_URL).
The synthetic refdata store is small (one index path 1999-2026, two contracts a day).
"""

from __future__ import annotations

import copy
import uuid
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import text

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline.plan import (
    CANONICAL_ORDER,
    DAILY_GATES,
    DAILY_SHADOW_PAPER,
    DAILY_THRESHOLDS,
    STAGE_INTRODUCED_AT,
    V1_STAGE_PLAN,
    V3_DAILY_STAGE_PLAN,
    V3_GATE_STAGES,
    StagePlanError,
    class_settings,
    default_config,
    validate_stage_plan,
)
from poseidon.pipeline.refdata.datasets import TAIEX
from poseidon.pipeline.service import PipelineRequestInvalid
from poseidon.pipeline.shadow_paper_daily import DAILY_PAPER_THRESHOLDS, DEFAULT_DAILY_SHADOW_PAPER
from poseidon.pipeline.stages import STAGE_REGISTRY, EngineRouted
from poseidon.pipeline.v3.config import (
    ENGINE,
    REFDATA_DATASETS,
    daily_data_config,
    daily_run_config,
    is_v3_run,
    series_first_days,
    warmup_record,
)
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.validation import WARMUP_MARGIN_SESSIONS, FeasibilityContext, validate_spec_v3
from tests.pipeline.conftest import Clock, Harness, tx_spec
from tests.pipeline.v3_wiring_support import (
    CLOCK_START,
    Observe,
    build_long_store,
    create_v3,
    planted_edge_spec,
    series_store,
    trace,
    v3_registry,
)

AS_OF = date(2026, 10, 22)
REFDATA = {name: f"{index:064x}" for index, name in enumerate(REFDATA_DATASETS)}


def acceptance_config(**kw) -> dict:
    resolved = validate_spec_v3(tmf_dynamic_leverage_3x()).resolved
    return daily_run_config(resolved, as_of=AS_OF, brief_text="brief", refdata=REFDATA, **kw)


# ----------------------------------------------------------------------------- frozen config


def test_daily_run_config_freezes_the_class_entry_and_the_window():
    config = acceptance_config()
    assert config["version"] == 6 and config["engine"] == ENGINE and is_v3_run(config)
    assert config["strategy_class"] == "daily_equity" and config["spec_validator"] == "spec_v3"
    assert config["thresholds"] == DAILY_THRESHOLDS and config["gates"] == DAILY_GATES
    assert config["shadow_paper"] == DAILY_SHADOW_PAPER and config["shadow_paper"]["mode"] == "daily_target"
    assert config["stage_plan"] == list(V3_DAILY_STAGE_PLAN)
    assert config["data"] == {
        "source": "refdata",
        "frequency": "1d",
        "timezone": "Asia/Taipei",
        "start": "2001-01-02",
        "end": "2026-10-22",
        "end_exclusive": True,
        "holdout_start": daily_data_config(as_of=AS_OF)["holdout_start"],
        "parts": "train = trading days in [start, holdout_start); holdout = [holdout_start, end)",
        "warmup": warmup_record(),
    }
    split = date.fromisoformat(config["data"]["holdout_start"])
    assert date(2021, 8, 1) <= split <= date(2021, 9, 30)  # the last 20 % of 2001-01-02 .. 2026-10-22
    assert config["refdata"]["datasets"] == REFDATA
    assert config["backtest"] == {"truncation_audit_cuts": 5}
    assert config["components"] and all("@" in key and len(sha) == 64 for key, sha in config["components"].items())
    assert config["optimize"]["objective"] == "sharpe_daily" and config["seed"] == 0
    # the DSL default config is untouched (still CONFIG_VERSION 5, no engine)
    assert not is_v3_run(default_config(stage_plan=None, as_of=AS_OF))


SERIES = {
    "taifex_futures_daily/TX": date(1998, 7, 21),
    "twse_taiex_daily": date(1999, 1, 5),
    "twse_taiex_total_return": date(2003, 1, 2),
}


def chosen(spec):
    result = validate_spec_v3(spec, feasibility=FeasibilityContext(series_since=SERIES, choose_start=True))
    assert result.ok
    return result


def test_a_moved_start_is_frozen_with_its_rule_and_an_unmoved_one_freezes_the_same_bytes():
    unmoved = chosen(tmf_dynamic_leverage_3x())  # index:taiex: 2001-01-02 keeps its warm-up
    assert not unmoved.window.moved
    same = daily_run_config(unmoved.resolved, as_of=AS_OF, brief_text="brief", refdata=REFDATA, window=unmoved.window)
    assert content_sha256(same) == content_sha256(acceptance_config())

    moved = chosen(planted_edge_spec("index:taiex_tr"))
    config = daily_run_config(moved.resolved, as_of=AS_OF, brief_text="brief", refdata=REFDATA, window=moved.window)
    data = config["data"]
    assert data["start"] == moved.window.start.isoformat() > "2003-01-02"
    assert data["start_rule"] == moved.window.as_config()
    assert data["start_rule"]["binding_series"] == "twse_taiex_total_return"
    # the holdout is the last 20 % of the shorter window, computed once here
    plain = daily_data_config(as_of=AS_OF, start=data["start"])
    assert data["holdout_start"] == plain["holdout_start"] > acceptance_config()["data"]["holdout_start"]
    assert {key: data[key] for key in plain} == plain


def test_series_first_days_read_the_manifests(tmp_path):
    from poseidon.pipeline.refdata.datasets import FUTURES
    from poseidon.pipeline.refdata.store import DatasetMissingError, RefdataError
    from poseidon.pipeline.v3.instruments import DAILY_PRICE_DATASET

    assert DAILY_PRICE_DATASET == FUTURES
    indices = {"twse_taiex_daily": "1999-01-05", "twse_taiex_total_return": "2003-01-02"}
    # MTX rows first: the TX price source starts with TX's own first row, a month later
    store = series_store(tmp_path / "a", futures={"MTX": "1998-06-01", "TX": "1998-07-21"}, indices=indices)
    assert series_first_days(store) == {
        "taifex_futures_daily/TX": date(1998, 7, 21),
        "twse_taiex_daily": date(1999, 1, 5),
        "twse_taiex_total_return": date(2003, 1, 2),
    }
    late = series_store(
        tmp_path / "b", futures={"TX": "1998-07-21"}, indices={**indices, "twse_taiex_daily": "2004-02-02"}
    )
    assert series_first_days(late)["twse_taiex_daily"] == date(2004, 2, 2)
    with pytest.raises(RefdataError, match="no rows for TX"):
        series_first_days(series_store(tmp_path / "c", futures={"MTX": "2001-04-09"}, indices=indices))
    with pytest.raises(DatasetMissingError):
        series_first_days(series_store(tmp_path / "d", futures={"TX": "1998-07-21"}, indices={}))


def test_daily_class_table_is_the_one_source_of_the_paper_settings():
    assert DEFAULT_DAILY_SHADOW_PAPER is DAILY_SHADOW_PAPER
    assert {k: DAILY_THRESHOLDS[k] for k in DAILY_PAPER_THRESHOLDS} == DAILY_PAPER_THRESHOLDS
    assert class_settings("daily_fixed")["shadow_paper"] == DAILY_SHADOW_PAPER
    assert "shadow_paper" not in class_settings("intraday_fixed")


# ----------------------------------------------------------------------------- plan and routing


def test_v3_plan_validates_at_6_and_never_for_older_versions():
    plan = list(V3_DAILY_STAGE_PLAN)
    assert validate_stage_plan(plan, STAGE_REGISTRY, version=6) == plan
    assert plan[:3] == ["spec_validate", "spec_faithfulness", "backtest"]
    assert validate_stage_plan(plan[:3], STAGE_REGISTRY, version=6) == plan[:3]
    assert validate_stage_plan(["spec_validate", "backtest"], STAGE_REGISTRY, version=6)  # report-only: optional
    assert STAGE_INTRODUCED_AT["execution_timing"] == 6
    with pytest.raises(StagePlanError, match="spec_validate requires spec_generate"):
        validate_stage_plan(plan, STAGE_REGISTRY, version=5)  # read as a DSL plan
    with pytest.raises(StagePlanError, match="execution_timing does not exist at config version 5"):
        validate_stage_plan(
            ["spec_generate", *[n for n in plan if n != "spec_faithfulness"]], STAGE_REGISTRY, version=5
        )
    with pytest.raises(StagePlanError, match="spec_faithfulness does not exist at config version 5"):
        validate_stage_plan(["spec_generate", *plan], STAGE_REGISTRY, version=5)
    with pytest.raises(StagePlanError, match="not part of a spec v3 daily run"):
        validate_stage_plan(["spec_generate", *plan], STAGE_REGISTRY, version=6)
    with pytest.raises(StagePlanError, match="requires execution_timing"):
        validate_stage_plan([name for name in plan if name != "execution_timing"], STAGE_REGISTRY, version=6)
    with pytest.raises(StagePlanError, match="requires backtest"):
        validate_stage_plan(["spec_validate", "optimize"], STAGE_REGISTRY, version=6)
    # the DSL plan is unchanged and still valid at 5
    assert validate_stage_plan(list(V1_STAGE_PLAN), STAGE_REGISTRY) == list(V1_STAGE_PLAN)
    assert list(V3_DAILY_STAGE_PLAN) == [n for n in CANONICAL_ORDER if n in V3_DAILY_STAGE_PLAN]


def test_router_sends_v3_runs_to_the_daily_bodies_and_older_runs_to_dsl():
    v3, v5 = acceptance_config(), default_config(stage_plan=None, as_of=AS_OF)
    for name in ("spec_validate", "backtest", "optimize", "holdout", "shadow_paper"):
        stage = STAGE_REGISTRY[name]
        assert isinstance(stage, EngineRouted)
        assert stage.body(v3) is stage.daily and stage.body(v5) is stage.legacy
    assert STAGE_REGISTRY["spec_validate"].upstream_names(v3) == ()
    assert STAGE_REGISTRY["spec_validate"].upstream_names(v5) == ("spec_generate",)
    assert "execution_timing" in STAGE_REGISTRY["holdout"].upstream_names(v3)
    assert "execution_timing" not in STAGE_REGISTRY["holdout"].upstream_names(v5)
    assert set(STAGE_REGISTRY["shadow_paper"].upstream_names(v3)) == {"backtest", "optimize", *V3_GATE_STAGES}


def test_frozen_config_snapshots_the_owner_decisions():
    # owner decisions 2026-10-24 afternoon #2 / #3, registered before any real result
    thresholds = acceptance_config()["thresholds"]
    assert thresholds["benchmark_alpha_t_min"] == 1.65
    assert thresholds["execution_timing_pnl_positive"] is True
    assert thresholds["execution_timing_sharpe_ratio_min"] == 0.5
    assert acceptance_config()["gates"]["execution_timing"]["scenarios"]["next_open"] == [
        "signal_close",
        "signal_lag_1d",
    ]
    assert class_settings("intraday_fixed")["thresholds"]["benchmark_alpha_t_min"] == 1.0


# ----------------------------------------------------------------------------- creation (PostgreSQL)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_long_store(tmp_path_factory.mktemp("refdata"), kind="ma_band", seed=5)


@pytest.fixture
def v3_harness(pipeline_sessions, tmp_path, store):
    return Harness(pipeline_sessions, tmp_path / "artifacts", Clock(CLOCK_START), registry=v3_registry(store))


@pytest.mark.usefixtures("pipeline_engine")
def test_create_freezes_config_version_6_with_the_instrument(v3_harness, store):
    run_id = create_v3(v3_harness, tmf_dynamic_leverage_3x(), store=store, key="v3-1")
    run = v3_harness.run(run_id)
    assert (run.market, run.symbol, run.status, run.current_stage) == ("tw_futures", "TMF", "queued", "spec_validate")
    config = run.config_json
    assert config["version"] == 6 and config["strategy_class"] == "daily_equity"
    assert config["refdata"]["datasets"] == {name: store.content_sha256(name) for name in REFDATA_DATASETS}
    assert run.config_sha256 == content_sha256(config)
    assert config["data"]["end"] == "2026-10-22"
    # idempotent replay: same key, same body -> the same run, nothing new
    again = create_v3(v3_harness, tmf_dynamic_leverage_3x(), store=store, key="v3-1")
    assert again == run_id


@pytest.mark.usefixtures("pipeline_engine")
def test_create_rejects_invalid_intraday_and_family_mixing(v3_harness, store):
    from poseidon.pipeline.v3.dsl_bridge import v2_to_v3

    broken = tmf_dynamic_leverage_3x()
    broken["sleeves"][0]["sizing"]["component"] = "sizing.nope"
    with pytest.raises(PipelineRequestInvalid, match="does not validate"):
        create_v3(v3_harness, broken, store=store)
    with pytest.raises(PipelineRequestInvalid, match="daily only"):
        create_v3(v3_harness, v2_to_v3(tx_spec()), store=store)  # the ORB template, intraday

    daily = create_v3(v3_harness, planted_edge_spec(), store=store)
    same_class = create_v3(v3_harness, tmf_dynamic_leverage_3x(), store=store, family_id=str(daily))
    assert v3_harness.run(same_class).family_id == daily
    with pytest.raises(PipelineRequestInvalid, match=r"family .* is daily_equity"):
        v3_harness.create(spec=tx_spec(), stage_plan=None, family_id=str(daily))  # a DSL run into a daily family
    dsl = v3_harness.create(spec=tx_spec(), stage_plan=None)
    with pytest.raises(PipelineRequestInvalid, match=r"family .* is intraday_fixed"):
        create_v3(v3_harness, planted_edge_spec(), store=store, family_id=str(dsl))
    # a family joined through a later member still answers with its first run's class
    with pytest.raises(PipelineRequestInvalid, match=r"family .* is daily_equity"):
        v3_harness.create(spec=tx_spec(), stage_plan=None, family_id=str(same_class))


@pytest.mark.usefixtures("pipeline_engine")
def test_create_validates_v3_stage_plans(v3_harness, store):
    with pytest.raises(PipelineRequestInvalid, match="not part of a spec v3 daily run"):
        create_v3(v3_harness, planted_edge_spec(), store=store, stage_plan=["spec_generate", "spec_validate"])
    run_id = create_v3(v3_harness, planted_edge_spec(), store=store, stage_plan=["spec_validate", "backtest"])
    assert v3_harness.run(run_id).config_json["stage_plan"] == ["spec_validate", "backtest"]


@pytest.mark.usefixtures("pipeline_engine")
def test_spec_validate_and_backtest_on_the_engine(v3_harness, store):
    run_id = create_v3(v3_harness, planted_edge_spec(), store=store, stage_plan=["spec_validate", "backtest"])
    v3_harness.drain()
    run = v3_harness.run(run_id)
    attempts = {row.stage: row for row in v3_harness.attempts(run_id)}
    assert run.status == "completed", [(r.stage, r.verdict, r.reason, r.artifact_json) for r in attempts.values()]
    validated = attempts["spec_validate"].artifact_json["report"]
    assert validated["strategy_class"] == "daily_equity" and validated["grid_size"] == 15
    assert validated["feasibility"]["prices"]["TX"][0] > 0  # sized at the synthetic TX prices, train part only
    backtest = attempts["backtest"].artifact_json["report"]
    assert backtest["truncation_audit"]["ok"] and len(backtest["truncation_audit"]["cuts"]) == 5
    assert backtest["liquidation"]["verdict"] == "pass"
    assert backtest["window"]["last_day"] < run.config_json["data"]["holdout_start"]
    assert backtest["metrics"]["last_day"] < run.config_json["data"]["holdout_start"]
    files = {entry["name"] for entry in attempts["backtest"].artifact_json["files"]}
    assert files == {"fills.csv", "positions_daily.csv", "daily_pnl.csv", "events.csv"}
    assert backtest["summary"].startswith("backtest ✅")


@pytest.fixture(scope="module")
def tr_store(tmp_path_factory):
    """The total-return index starts 2003-01-02, as the real one does."""

    return build_long_store(tmp_path_factory.mktemp("refdata_tr"), kind="ma_band", seed=5, tr_from="2003-01-02")


@pytest.mark.usefixtures("pipeline_engine")
def test_a_spec_on_a_late_series_runs_from_a_later_frozen_start(pipeline_sessions, tmp_path, store, tr_store):
    spec = planted_edge_spec("index:taiex_tr")
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(CLOCK_START), registry=v3_registry(tr_store))
    run_id = create_v3(harness, spec, store=tr_store, stage_plan=["spec_validate", "backtest"])
    run = harness.run(run_id)
    data = run.config_json["data"]
    rule = data["start_rule"]
    assert rule["binding_series"] == "twse_taiex_total_return"
    assert rule["series_since"]["twse_taiex_total_return"] == "2003-01-02"  # from the store's manifest
    # the warm-up in sessions of the store's TAIEX days: the start is the first day after warm-up + margin of them
    assert data["warmup"] == warmup_record() and rule["margin_sessions"] == WARMUP_MARGIN_SESSIONS
    assert "warmup_calendar_days" not in rule
    start = pd.Timestamp(data["start"])
    sessions = pd.DatetimeIndex(pd.to_datetime(tr_store.read(TAIEX)["trade_date"]))
    assert start in sessions
    assert ((sessions >= pd.Timestamp("2003-01-02")) & (sessions < start)).sum() == rule[
        "warmup_bars"
    ] + WARMUP_MARGIN_SESSIONS
    assert run.config_sha256 == content_sha256(run.config_json)

    harness.drain()
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    assert harness.run(run_id).status == "completed", [(r.stage, r.verdict, r.reason) for r in attempts.values()]
    validated = attempts["spec_validate"].artifact_json["report"]
    assert validated["feasibility"]["start"] == data["start"]
    assert validated["feasibility"]["series_since"] == rule["series_since"]
    assert (
        f"window from {data['start']} (twse_taiex_total_return data from 2003-01-02 + warm-up)" in validated["summary"]
    )
    backtest = attempts["backtest"].artifact_json["report"]
    assert backtest["window"]["first_day"] == data["start"]
    assert backtest["metrics"]["first_day"] >= data["start"] and backtest["truncation_audit"]["ok"]

    # the same spec on a store whose total-return index has the default window's warm-up: nothing moves
    early = harness.run(create_v3(harness, spec, store=store, stage_plan=["spec_validate"])).config_json["data"]
    assert early["start"] == "2001-01-02" and "start_rule" not in early


@pytest.mark.usefixtures("pipeline_engine")
def test_a_series_too_late_for_the_walk_forward_fails_wfe(pipeline_sessions, tmp_path):
    """Three years of total-return index before the holdout: the run starts, the gates judge the short window."""

    late = build_long_store(tmp_path / "refdata", kind="ma_band", seed=5, tr_from="2022-01-03")
    registry = v3_registry(late)
    for name in V3_GATE_STAGES:  # record each gate's verdict and go on, so every gate runs
        registry[name] = Observe(registry[name])
    harness = Harness(pipeline_sessions, tmp_path / "artifacts", Clock(CLOCK_START), registry=registry)
    plan = list(V3_DAILY_STAGE_PLAN[: V3_DAILY_STAGE_PLAN.index("holdout") + 1])
    run_id = create_v3(harness, planted_edge_spec("index:taiex_tr"), store=late, stage_plan=plan)
    data = harness.run(run_id).config_json["data"]
    assert data["start"] > "2023-01-01" and data["start_rule"]["binding_series"] == "twse_taiex_total_return"
    harness.drain(limit=40)
    rows = {row["stage"]: row for row in trace(harness, run_id)}
    assert list(rows) == plan, rows
    assert all(row["verdict"] == "pass" for row in rows.values()), rows  # Observe: no stage errored
    wfe = rows["wfe"]
    assert wfe["observed_verdict"] == "fail" and "fits none of the frozen layouts" in wfe["reason"], wfe
    # the shorter window's holdout (its last 20 %) has fewer days than the registered minimum: a fail, not a pass
    holdout = next(r for r in harness.attempts(run_id) if r.stage == "holdout").artifact_json["report"]
    min_days = harness.run(run_id).config_json["thresholds"]["holdout_min_days"]
    assert rows["holdout"]["observed_verdict"] == "fail", rows["holdout"]
    assert holdout["inputs"]["holdout_metrics"]["n_days"] < min_days == 250


@pytest.mark.usefixtures("pipeline_engine")
def test_a_changed_component_source_errors_the_stage(v3_harness, store, monkeypatch):
    from poseidon.pipeline.v3 import registry

    run_id = create_v3(v3_harness, planted_edge_spec(), store=store, stage_plan=["spec_validate"])
    real = registry.component_fingerprints
    monkeypatch.setattr(registry, "component_fingerprints", lambda names=None: {k: "0" * 64 for k in real(names)})
    v3_harness.drain()
    run = v3_harness.run(run_id)
    assert run.status == "error" and "registry components changed" in run.reason


# ----------------------------------------------------------------------------- p003


@pytest.mark.usefixtures("pipeline_engine")
def test_p003_accepts_v3_instruments_and_round_trips(pipeline_engine, pipeline_sessions):
    from alembic import command
    from poseidon.core.config import settings
    from tests.pipeline.conftest import pipeline_alembic_config

    url = pipeline_engine.url.render_as_string(hide_password=False)

    def insert(market, symbol):
        with pipeline_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO pipeline_runs (id, family_id, source, idea_text, market, symbol, config_json, "
                    "config_sha256, created_by, status, revision) VALUES (:id, :id, 'api', 'x', :m, :s, '{}', "
                    ":sha, 'human:t', 'queued', 1)"
                ),
                {"id": uuid.uuid4(), "m": market, "s": symbol, "sha": "a" * 64},
            )

    original = settings.database_url
    settings.database_url = url
    try:
        config = pipeline_alembic_config(url)
        command.downgrade(config, "p002")
        insert("tw_futures", "TX")
        with pytest.raises(Exception, match="ck_pipeline_runs_instrument"):
            insert("tw_futures", "TMF")
        command.upgrade(config, "head")
        insert("tw_futures", "TMF")
        insert("tw_stock", "0050")
        insert("multi", "PORTFOLIO")
        for market, symbol in (("multi", "TMF"), ("tw_futures", "PORTFOLIO"), ("crypto", "BTC")):
            with pytest.raises(Exception, match="ck_pipeline_runs_instrument"):
                insert(market, symbol)
        # a database holding v3 runs cannot go back below p003 (runs are never deleted)
        with pytest.raises(Exception, match="ck_pipeline_runs_instrument"):
            command.downgrade(config, "p002")
        with pipeline_engine.connect() as connection:
            assert connection.execute(text("SELECT version_num FROM alembic_version_pipeline")).scalar() == "p004"
    finally:
        command.upgrade(pipeline_alembic_config(url), "head")  # env.py migrates settings.database_url
        settings.database_url = original
    with pipeline_engine.begin() as connection:
        connection.execute(
            text("TRUNCATE pipeline_events, pipeline_trials, stage_runs, pipeline_runs RESTART IDENTITY CASCADE")
        )


def test_daily_run_config_returns_fresh_copies():
    first = acceptance_config()
    first["thresholds"]["holdout_min_days"] = 1
    assert acceptance_config()["thresholds"]["holdout_min_days"] == 250
    assert copy.deepcopy(first) == first
