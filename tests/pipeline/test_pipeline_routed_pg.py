"""Idea-only runs (CONFIG_VERSION 7, step 6) on real PostgreSQL: creation freezes every candidate class,
spec_generate v3 routes the run, the class's frozen plan follows (daily engine or DSL v2), families.

The LLM is a recorded fake (``tests.pipeline.test_spec_llm.FakeClient``); refdata is the synthetic store.
"""

from __future__ import annotations

import pytest

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import openrouter, spec_llm
from poseidon.pipeline.plan import V3_ROUTED_DAILY_STAGE_PLAN
from poseidon.pipeline.service import PipelineRequestInvalid, PipelineService
from poseidon.pipeline.spec_llm_v3 import EXAMPLE_IDEA
from poseidon.pipeline.stages import SpecGenerateStage
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from tests.pipeline.conftest import Clock, Harness, synthetic_registry
from tests.pipeline.jev_support import FakeOpenRouter
from tests.pipeline.test_spec_llm import KEY, LABEL, FakeClient, reply
from tests.pipeline.v3_wiring_support import CLOCK_START, build_long_store, create_v3, planted_edge_spec, v3_registry

pytestmark = pytest.mark.postgresql
PLAN = ["spec_generate", "spec_validate", "spec_faithfulness", "backtest"]
PLANTED_IDEA = "微台做多，槓桿 3 倍，每天收盤後依權益調整口數。台指期收盤跌破 200 日線 2% 就空手，站回 2% 回到 3 倍。"


def planted_reply() -> dict:
    return {
        "status": "ok",
        "spec": planted_edge_spec(),
        "clauses": [
            {"id": "c1", "quote": "微台", "slot": "instruments", "component": None},
            {"id": "c2", "quote": "做多", "slot": "direction", "component": "direction.constant"},
            {"id": "c3", "quote": "槓桿 3 倍", "slot": "leverage", "component": "leverage.base"},
            {"id": "c4", "quote": "每天收盤後依權益調整口數", "slot": "sizing", "component": "sizing.target_leverage"},
            {
                "id": "c5",
                "quote": "台指期收盤跌破 200 日線 2% 就空手，站回 2% 回到 3 倍",
                "slot": "leverage",
                "component": "regime.ma_band",
            },
        ],
        "assumptions": ["換月：結算日前 1 個交易日"],
    }


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_long_store(tmp_path_factory.mktemp("refdata"), kind="ma_band", seed=5)


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch):
    monkeypatch.setenv(openrouter.KEY_ENV, KEY)
    openrouter.reset_identity_cache()
    yield
    openrouter.reset_identity_cache()


def generating(registry, client, store):
    registry["spec_generate"] = SpecGenerateStage(
        client_factory=lambda key, timeout: client,
        pin_provider=lambda: LABEL,
        jev_client_factory=FakeOpenRouter(label=LABEL).factory(),
        refdata=lambda: store,
    )
    return registry


def idea_run(harness, store, idea, **body):
    return create_v3(harness, None, store=store, idea=idea, **body)


def stage_names(sessions, run_id) -> list[str]:
    with sessions() as session:
        return [entry["stage"] for entry in PipelineService(session).stages(run_id)]


@pytest.mark.usefixtures("pipeline_engine")
def test_idea_only_create_freezes_config_7_once(pipeline_sessions, tmp_path, store):
    harness = Harness(pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=v3_registry(store))
    run_id = idea_run(harness, store, EXAMPLE_IDEA, key="idea-1")
    run = harness.run(run_id)
    config = run.config_json
    assert config["version"] == 7 and set(config["by_class"]) == {"intraday_fixed", "daily_equity", "daily_fixed"}
    assert run.config_sha256 == content_sha256(config)
    assert (run.market, run.symbol, run.current_stage) == ("tw_futures", "TX", "spec_generate")
    daily = config["by_class"]["daily_equity"]
    assert daily["refdata"]["datasets"]["taifex_futures_daily"] == store.content_sha256("taifex_futures_daily")
    assert idea_run(harness, store, EXAMPLE_IDEA, key="idea-1") == run_id  # idempotent replay
    stages = stage_names(pipeline_sessions, run_id)
    assert stages == config["stage_plan"]  # not routed yet: every candidate stage


@pytest.mark.usefixtures("pipeline_engine")
def test_a_daily_route_runs_the_generated_spec_on_the_daily_engine(pipeline_sessions, tmp_path, store):
    client = FakeClient([reply(planted_reply())] * 2)  # the fake JEV disagrees once: one repair round
    harness = Harness(
        pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=generating(v3_registry(store), client, store)
    )
    run_id = idea_run(harness, store, PLANTED_IDEA, stage_plan=PLAN)
    frozen = harness.run(run_id)
    sha, config = frozen.config_sha256, frozen.config_json
    assert config["by_class"]["daily_equity"]["stage_plan"] == PLAN
    harness.drain()
    run = harness.run(run_id)
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    assert run.status == "completed", [(r.stage, r.verdict, r.reason) for r in attempts.values()]
    assert (run.config_sha256, run.config_json) == (sha, config)  # the frozen config never changes
    generated = attempts["spec_generate"].artifact_json["report"]
    assert generated["route"]["strategy_class"] == "daily_equity" and len(generated["clauses"]) == 5
    assert generated["summary"].startswith("spec_generate ✅ daily_equity：5 個子句；假設 1 項")
    validated = attempts["spec_validate"].artifact_json["report"]
    assert validated["strategy_class"] == "daily_equity" and validated["spec_sha256"] == content_sha256(
        planted_edge_spec()
    )
    assert attempts["backtest"].artifact_json["report"]["summary"].startswith("backtest ✅")
    stages = stage_names(pipeline_sessions, run_id)
    assert stages == PLAN
    assert len(client.chat_payloads) == 2 and generated["jev_review"]["usage"]["calls"] > 0
    faithfulness = attempts["spec_faithfulness"].artifact_json["report"]
    assert faithfulness["clause_source"] == "spec_generate" and faithfulness["counts"]["clauses"] == 5


TR_IDEA = (
    "微台做多，槓桿 3 倍，每天收盤後依權益調整口數。加權含息報酬指數收盤跌破 200 日線 2% 就空手，站回 2% 回到 3 倍。"
)


@pytest.mark.usefixtures("pipeline_engine")
def test_a_daily_route_on_a_late_series_freezes_its_window_in_the_route(pipeline_sessions, tmp_path):
    """Owner decision 2026-10-25: the total-return index starts 2003-01-02, so the run starts later; spec_generate
    chooses that start once (route ``data``) and every later stage reads it, the frozen config untouched."""

    store = build_long_store(tmp_path / "refdata", kind="ma_band", seed=5, tr_from="2003-01-02")
    body = planted_reply()
    body["spec"] = planted_edge_spec("index:taiex_tr")
    body["clauses"][4]["quote"] = "加權含息報酬指數收盤跌破 200 日線 2% 就空手，站回 2% 回到 3 倍"
    client = FakeClient([reply(body)] * 2)
    harness = Harness(
        pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=generating(v3_registry(store), client, store)
    )
    run_id = idea_run(harness, store, TR_IDEA, stage_plan=PLAN)
    frozen = harness.run(run_id)
    harness.drain()
    run = harness.run(run_id)
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    assert run.status == "completed", [(r.stage, r.verdict, r.reason) for r in attempts.values()]
    assert (run.config_sha256, run.config_json) == (frozen.config_sha256, frozen.config_json)
    assert run.config_json["by_class"]["daily_equity"]["data"]["start"] == "2001-01-02"
    route = attempts["spec_generate"].artifact_json["report"]["route"]
    data = route["data"]
    assert data["start"] > "2003-01-02" and data["start_rule"]["binding_series"] == "twse_taiex_total_return"
    assert data["end"] == run.config_json["by_class"]["daily_equity"]["data"]["end"]
    validated = attempts["spec_validate"].artifact_json["report"]
    assert validated["feasibility"]["start"] == data["start"]
    assert attempts["backtest"].artifact_json["report"]["window"]["first_day"] == data["start"]


@pytest.mark.usefixtures("pipeline_engine")
def test_an_intraday_route_continues_on_the_dsl_path(pipeline_sessions, tmp_path, store):
    dsl_spec = spec_llm._param_example()
    client = FakeClient(
        [reply({"status": "intraday", "reason": "opening range"}), reply({"status": "ok", "spec": dsl_spec})]
    )
    harness = Harness(
        pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=generating(synthetic_registry(), client, store)
    )
    run_id = idea_run(harness, store, "TX 30-minute opening range breakout", stage_plan=PLAN)
    harness.drain()
    run = harness.run(run_id)
    attempts = {row.stage: row for row in harness.attempts(run_id)}
    assert run.status == "completed", [(r.stage, r.verdict, r.reason) for r in attempts.values()]
    generated = attempts["spec_generate"].artifact_json["report"]
    assert generated["route"] == {"strategy_class": "intraday_fixed", "engine": "dsl_v2"}
    assert attempts["spec_validate"].artifact_json["report"]["validator"] == "dsl_v2"
    assert stage_names(pipeline_sessions, run_id) == [
        "spec_generate",
        "spec_validate",
        "backtest",
    ]
    assert "metrics" in attempts["backtest"].artifact_json["report"]  # the legacy runner


@pytest.mark.usefixtures("pipeline_engine")
def test_ambiguous_ends_unsupported_and_can_be_rerun(pipeline_sessions, tmp_path, store):
    ambiguous = {"status": "ambiguous", "reason": "台指 names no series", "missing": ["signal_series"]}
    client = FakeClient([reply(ambiguous)])
    harness = Harness(
        pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=generating(v3_registry(store), client, store)
    )
    run_id = idea_run(harness, store, "微台 3 倍，台指跌破 200 日線降槓桿")
    harness.drain()
    run = harness.run(run_id)
    assert run.status == "unsupported" and run.reason.startswith("spec_generate: ambiguous")
    report = harness.attempts(run_id)[0].artifact_json["report"]
    assert report["missing_capabilities"] == ["signal_series"]


@pytest.mark.usefixtures("pipeline_engine")
def test_family_members_inherit_the_routed_class(pipeline_sessions, tmp_path, store):
    client = FakeClient([reply(planted_reply())] * 2)  # the fake JEV disagrees once: one repair round
    harness = Harness(
        pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=generating(v3_registry(store), client, store)
    )
    first = idea_run(harness, store, PLANTED_IDEA, stage_plan=["spec_generate"])
    with pytest.raises(PipelineRequestInvalid, match="no strategy class yet"):
        idea_run(harness, store, PLANTED_IDEA, family_id=str(first))
    harness.drain()
    member = idea_run(harness, store, PLANTED_IDEA, family_id=str(first))
    assert set(harness.run(member).config_json["by_class"]) == {"daily_equity"}
    hand_written = create_v3(harness, tmf_dynamic_leverage_3x(), store=store, family_id=str(first))
    assert harness.run(hand_written).family_id == first
    with pytest.raises(PipelineRequestInvalid, match=r"family .* is daily_equity"):
        harness.create(stage_plan=None, family_id=str(first))  # a DSL v2 run


@pytest.mark.usefixtures("pipeline_engine")
def test_idea_only_stage_plans_are_validated_per_class(pipeline_sessions, tmp_path, store):
    harness = Harness(pipeline_sessions, tmp_path / "a", Clock(CLOCK_START), registry=v3_registry(store))
    with pytest.raises(PipelineRequestInvalid, match="must start with spec_generate"):
        idea_run(harness, store, "x", stage_plan=["spec_validate"])
    with pytest.raises(PipelineRequestInvalid, match="requires"):
        idea_run(harness, store, "x", stage_plan=["spec_generate", "backtest"])
    requested = ["spec_generate", "spec_validate", "backtest", "optimize", "execution_timing"]
    run_id = idea_run(harness, store, "x", stage_plan=requested)
    by_class = harness.run(run_id).config_json["by_class"]
    assert by_class["intraday_fixed"]["stage_plan"] == requested[:4]  # no execution_timing on the DSL path
    assert by_class["daily_fixed"]["stage_plan"] == [s for s in V3_ROUTED_DAILY_STAGE_PLAN if s in requested]
    assert by_class["daily_fixed"]["stage_plan"] == requested


@pytest.mark.usefixtures("pipeline_engine")
def test_without_refdata_an_idea_only_run_can_still_go_intraday(pipeline_sessions, tmp_path):
    from poseidon.pipeline.refdata.store import RefdataError

    def missing():
        raise RefdataError("no refdata root")

    harness = Harness(pipeline_sessions, tmp_path / "a", Clock(CLOCK_START))
    with harness.sessions() as session:
        response, _ = PipelineService(
            session, registry=harness.dispatcher.registry, clock=harness.clock, refdata=missing
        ).create({"idea_text": "x", "source": "api"}, actor_id="human:researcher", idempotency_key="no-refdata")
        session.commit()
    config = response["config_json"]
    assert set(config["by_class"]) == {"intraday_fixed"} and "no refdata root" in config["unavailable"]["daily_equity"]
