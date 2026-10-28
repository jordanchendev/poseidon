"""Step 6 spec_generate v3: prompt from the registry, clauses (verbatim, coverage), repair, review hook,
ambiguous series, the intraday route to DSL v2, cost cap, routed configs (HTTP mocked, recorded replies)."""

from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import date, timedelta
from pathlib import Path

import pytest

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline import spec_llm
from poseidon.pipeline.plan import (
    DEFAULT_SPEC_GENERATE,
    DEFAULT_SPEC_GENERATE_V3,
    ROUTED_CONFIG_VERSION,
    V1_STAGE_PLAN,
    V3_ROUTED_DAILY_STAGE_PLAN,
    RouteError,
    effective_config,
    validate_stage_plan,
)
from poseidon.pipeline.spec_llm_v3 import (
    EXAMPLE_IDEA,
    PROMPT_SHA256_V3,
    SYSTEM_PROMPT_V3,
    clause_errors,
    example_reply,
    generate_routed,
    generate_spec_v3,
    series_errors,
    split_sentences,
)
from poseidon.pipeline.stages import STAGE_REGISTRY, SpecGenerateStage, StageContext
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.v3.config import REFDATA_DATASETS, idea_run_config, is_v3_run, trading_calendar_record
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.registry import REGISTRY, component_fingerprints
from poseidon.pipeline.v3.validation import validate_spec_v3
from tests.pipeline.jev_support import FakeOpenRouter
from tests.pipeline.test_spec_llm import ENV, KEY, LABEL, FakeClient, assert_no_key, reply

AS_OF = date(2026, 10, 24)
REFDATA = {name: f"{index:064x}" for index, name in enumerate(REFDATA_DATASETS)}
AMBIGUOUS_IDEA = (
    "用微台做多，槓桿 3 倍，每天收盤後依權益調整口數。台指跌破 200 日線 2% 降到 0.5 倍，站回 2% 回到 3 倍。"
)


@pytest.fixture(autouse=True)
def _fresh_identity_cache():
    spec_llm.reset_identity_cache()
    yield
    spec_llm.reset_identity_cache()


def v3_config(**overrides) -> dict:
    cfg = copy.deepcopy(DEFAULT_SPEC_GENERATE_V3)
    cfg.update(overrides)
    return cfg


def ok_reply(**changes) -> dict:
    body = example_reply()
    body.update(changes)
    return body


def run_v3(client, *, idea=EXAMPLE_IDEA, cfg=None, pin=LABEL, env=ENV, **kw):
    created = []

    def factory(key, timeout):
        assert key == KEY
        created.append(timeout)
        return client

    outcome = generate_spec_v3(
        idea, cfg if cfg is not None else v3_config(), pin=pin, environ=env, client_factory=factory, **kw
    )
    return outcome, created


def jev_factory(**kw):
    """A fake JEV behind the same key and pin as the LLM fakes (the identity check is shared)."""

    fake = FakeOpenRouter(**{"label": LABEL, **kw})
    return fake, fake.factory()


def run_routed(client, idea, config=None, **kw):
    if "jev_client_factory" not in kw and "review" not in kw:
        kw["jev_client_factory"] = jev_factory()[1]
    return generate_routed(idea, config or routed(), pin=LABEL, environ=ENV, client_factory=lambda *_: client, **kw)


def routed(classes=("intraday_fixed", "daily_equity", "daily_fixed"), refdata=REFDATA) -> dict:
    return idea_run_config(as_of=AS_OF, brief_text="brief", refdata=refdata, classes=classes, refdata_error="gone")


# --------------------------------------------------------------- prompt material


def test_the_worked_example_passes_every_program_check():
    body = example_reply()
    result = validate_spec_v3(body["spec"])
    assert result.ok and result.strategy_class == "daily_equity"
    used = {ref["component"] for ref in _refs(body["spec"])}
    errors, unimplemented = clause_errors(EXAMPLE_IDEA, body["clauses"], used)
    assert errors == [] and unimplemented == []
    assert series_errors(body["spec"], body["clauses"]) == []


def _refs(spec):
    from poseidon.pipeline.v3.spec import component_slots

    return [ref for _path, _slot, ref in component_slots(spec)]


def test_the_prompt_is_rendered_from_the_registry():
    assert "untrusted_idea" in SYSTEM_PROMPT_V3 and '"status":"intraday"' in SYSTEM_PROMPT_V3
    assert '"missing":["signal_series"]' in SYSTEM_PROMPT_V3
    for name in REGISTRY:  # every registered component, with its catalog entry
        assert f'"name":"{name}"' in SYSTEM_PROMPT_V3
    assert '"hedge.collar"' in SYSTEM_PROMPT_V3  # listed as planned
    assert '"leverage_cap":5.0' in SYSTEM_PROMPT_V3 and '"symbol":"TMF"' in SYSTEM_PROMPT_V3
    for series in ('"self:adjusted"', '"index:taiex"', '"index:taiex_tr"'):
        assert series in SYSTEM_PROMPT_V3
    assert re.fullmatch(r"[0-9a-f]{64}", PROMPT_SHA256_V3) and PROMPT_SHA256_V3 != spec_llm.PROMPT_SHA256


def test_sentences_split_on_cjk_and_ascii_terminators_and_skip_decoration():
    text = "第一句。第二句！\n📈\n——-\nThird one. Fourth?"
    assert [text[a:b] for a, b in split_sentences(text)] == ["第一句", "第二句", "Third one.", "Fourth"]


def test_v3_settings_raise_only_the_cap_and_tokens():
    assert DEFAULT_SPEC_GENERATE_V3["max_cost_usd_per_attempt"] == 0.10
    assert DEFAULT_SPEC_GENERATE["max_cost_usd_per_attempt"] == 0.05  # the DSL settings are untouched
    same = {k: v for k, v in DEFAULT_SPEC_GENERATE_V3.items() if k in DEFAULT_SPEC_GENERATE}
    changed = {k for k, v in same.items() if v != DEFAULT_SPEC_GENERATE[k]}
    assert changed == {"max_cost_usd_per_attempt", "max_tokens", "request_timeout_seconds"}


# --------------------------------------------------------------- ok / clauses / repair


def test_valid_reply_passes_with_route_clauses_and_a_bridge_summary():
    client = FakeClient([reply(ok_reply())])
    outcome, created = run_v3(client)
    assert outcome.verdict == Verdict.PASS and outcome.reason is None, outcome.report
    report = outcome.report
    assert report["route"] == {
        "strategy_class": "daily_equity",
        "engine": "target_v1",
        "components": sorted(
            {
                "allocation.single",
                "direction.constant",
                "execution.next_open",
                "leverage.base",
                "rebalance.daily",
                "regime.ma_band",
                "risk.vol_cap",
                "roll.before_settlement",
                "sizing.target_leverage",
            }
        ),
    }
    assert report["spec"] == tmf_dynamic_leverage_3x() and report["spec_sha256"] == content_sha256(report["spec"])
    assert len(report["clauses"]) == 8 and report["unimplemented"] == []
    assert report["summary"].startswith("spec_generate ✅ daily_equity：8 個子句；假設 3 項：換月")
    assert report["prompt_sha256"] == PROMPT_SHA256_V3 and report["repair_rounds"] == 0
    payload = client.chat_payloads[0]
    assert payload["messages"][0]["content"] == SYSTEM_PROMPT_V3 and payload["max_tokens"] == 16000
    assert created == [240.0] and set(outcome.files) == {"llm_transcript_v3.json"}
    assert_no_key(outcome)


def test_a_paraphrased_quote_is_sent_back_then_fixed():
    bad = ok_reply()
    bad["clauses"][2] = {**bad["clauses"][2], "quote": "槓桿三倍"}
    client = FakeClient([reply(bad), reply(ok_reply())])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.PASS and outcome.report["repair_rounds"] == 1
    first = outcome.report["rounds"][0]
    assert first["outcome"] == "invalid"
    assert {"path": "clauses.2.quote", "msg": first["errors"][0]["msg"]} == first["errors"][0]
    assert "verbatim" in first["errors"][0]["msg"]
    repair = client.chat_payloads[1]["messages"][-1]["content"]
    assert "clauses.2.quote" in repair and "槓桿 3 倍" not in repair  # only the error goes back


def test_an_uncited_sentence_is_sent_back():
    bad = ok_reply()
    bad["clauses"] = [c for c in bad["clauses"] if c["id"] != "c8"]
    client = FakeClient([reply(bad), reply(ok_reply())])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.PASS
    errors = outcome.report["rounds"][0]["errors"]
    assert [e["path"] for e in errors] == ["clauses"] and "我覺得這樣比較穩" in errors[0]["msg"]


def test_clause_component_rules():
    body = example_reply()
    used = {ref["component"] for ref in _refs(body["spec"])}
    clauses = copy.deepcopy(body["clauses"])
    clauses[3]["component"] = "rebalance.daily"  # wrong slot (clause slot sizing)
    clauses[0]["component"] = "leverage.base"  # frame slot takes no component
    clauses[4]["component"] = "rebalance.threshold_band"  # registered, right slot, not in the spec
    clauses[5]["component"] = "regime.nope"
    errors, _ = clause_errors(EXAMPLE_IDEA, clauses, used)
    by_path = {e["path"]: e["msg"] for e in errors}
    assert set(by_path) == {"clauses.0.component", "clauses.3.component", "clauses.4.component", "clauses.5.component"}
    assert "rebalance component" in by_path["clauses.3.component"]
    assert "does not use rebalance.threshold_band" in by_path["clauses.4.component"]


def test_repair_exhaustion_fails_after_two_repair_rounds():
    bad = ok_reply(clauses=[{"id": "c1", "quote": "not in the idea", "slot": "other", "component": None}])
    client = FakeClient([reply(bad), reply(bad), reply(bad)])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.FAIL and outcome.reason == "spec_invalid_after_repair"
    assert len(client.chat_payloads) == 3 and outcome.report["stopped"] == "repair_exhausted"


def test_validator_errors_and_clause_errors_go_back_together():
    bad = ok_reply()
    bad["spec"] = copy.deepcopy(bad["spec"])
    bad["spec"]["sleeves"][0]["leverage"][0]["params"]["value"] = 9.0  # over the 5x cap of leverage.base
    bad["clauses"][1] = {**bad["clauses"][1], "quote": "go long"}
    client = FakeClient([reply(bad), reply(ok_reply())])
    outcome, _ = run_v3(client)
    paths = [e["path"] for e in outcome.report["rounds"][0]["errors"]]
    assert "clauses.1.quote" in paths and any(p.startswith("sleeves.0.leverage.0") for p in paths)


# --------------------------------------------------------------- signal series (owner decision #1)


def test_model_ambiguous_is_unsupported_with_the_question():
    answer = {
        "status": "ambiguous",
        "reason": "ambiguous signal series: 台指 200 日線",
        "missing": ["signal_series"],
        "question": "200 日線要看台指期調整價、加權指數，還是加權含息報酬指數？",
    }
    client = FakeClient([reply(answer)])
    outcome, _ = run_v3(client, idea=AMBIGUOUS_IDEA)
    assert outcome.verdict == Verdict.UNSUPPORTED and outcome.reason.startswith("ambiguous: ambiguous signal series")
    assert outcome.report["missing_capabilities"] == ["signal_series"] and outcome.report["status"] == "ambiguous"
    assert outcome.report["question"].startswith("200 日線要看")
    assert outcome.report["summary"].startswith("spec_generate ⛔ ambiguous")


def test_a_series_the_idea_never_names_is_sent_back_and_the_model_turns_ambiguous():
    spec = tmf_dynamic_leverage_3x()
    spec["sleeves"][0]["leverage"] = spec["sleeves"][0]["leverage"][:2]
    del spec["params"]["vol_k"]
    silent_pick = {
        "status": "ok",
        "spec": spec,
        "clauses": [
            {"id": "c1", "quote": "用微台", "slot": "instruments", "component": None},
            {"id": "c2", "quote": "做多", "slot": "direction", "component": "direction.constant"},
            {"id": "c3", "quote": "槓桿 3 倍", "slot": "leverage", "component": "leverage.base"},
            {"id": "c4", "quote": "每天收盤後依權益調整口數", "slot": "sizing", "component": "sizing.target_leverage"},
            {
                "id": "c5",
                "quote": "台指跌破 200 日線 2% 降到 0.5 倍，站回 2% 回到 3 倍",
                "slot": "leverage",
                "component": "regime.ma_band",
            },
        ],
        "assumptions": [],
    }
    assert validate_spec_v3(spec).ok
    ambiguous = {"status": "ambiguous", "reason": "台指 does not name a series", "missing": ["signal_series"]}
    client = FakeClient([reply(silent_pick), reply(ambiguous)])
    outcome, _ = run_v3(client, idea=AMBIGUOUS_IDEA)
    assert outcome.verdict == Verdict.UNSUPPORTED and outcome.report["missing_capabilities"] == ["signal_series"]
    error = outcome.report["rounds"][0]["errors"][0]
    assert error["path"] == "sleeves.0.leverage.1.params.signal" and "ambiguous" in error["msg"]


@pytest.mark.parametrize(
    ("signal", "quote", "ok"),
    [
        ("index:taiex", "加權指數跌破 200 日線", True),
        ("index:taiex_tr", "加權含息報酬指數跌破 200 日線", True),
        ("self:adjusted", "台指期跌破 200 日線", True),
        ("index:taiex", "台指跌破 200 日線", False),
        ("index:taiex", "加權含息報酬指數跌破 200 日線", False),
        ("index:taiex_tr", "大盤跌破 200 日線", False),
        ("self:close", "台指期跌破 200 日線", False),
    ],
)
def test_series_wording_net(signal, quote, ok):
    spec = tmf_dynamic_leverage_3x()
    spec["sleeves"][0]["leverage"][1]["params"]["signal"] = signal
    spec["sleeves"][0]["leverage"][2]["params"]["signal"] = "index:taiex"
    clauses = [
        {"id": "c1", "quote": quote, "slot": "leverage", "component": "regime.ma_band"},
        {"id": "c2", "quote": "加權股價指數近 20 日波動", "slot": "leverage", "component": "risk.vol_cap"},
    ]
    errors = series_errors(spec, clauses)
    assert (errors == []) is ok, errors


# --------------------------------------------------------------- unsupported / unimplemented


def test_model_unsupported_is_aggregatable():
    client = FakeClient([reply({"status": "unsupported", "reason": "news", "missing": ["News Sentiment"]})])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.UNSUPPORTED and outcome.report["missing_capabilities"] == ["news_sentiment"]


def test_a_planned_component_in_the_spec_is_unsupported_not_a_repair():
    spec = tmf_dynamic_leverage_3x()
    spec["sleeves"][0]["exits"] = [{"component": "exit.stop_loss", "params": {}}]
    client = FakeClient([reply(ok_reply(spec=spec))])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.UNSUPPORTED and len(client.chat_payloads) == 1
    assert outcome.report["missing_capabilities"] == ["exit_stop_loss"]
    assert outcome.report["rejected_spec"] == spec


def test_a_series_that_starts_after_the_frozen_window_is_unsupported_not_a_repair():
    # live sample 8_daily_vol: index:taiex_tr starts 2003-01-02, the daily window 2001-01-02. Without the refdata
    # first days (no series_since) the window is the fixed default, so no repair round could fix it
    sample = json.loads((Path(__file__).parent / "fixtures" / "spec_v3_live_8_daily_vol.json").read_text())
    client = FakeClient([reply(sample["reply"])])
    outcome, _ = run_v3(client, idea=sample["idea"])
    assert outcome.verdict == Verdict.UNSUPPORTED and len(client.chat_payloads) == 1, outcome.report
    assert outcome.report["missing_capabilities"] == ["index_taiex_tr"]
    assert "after the run's data window start 2001-01-02" in outcome.report["unsupported"]["reason"]


SERIES = {
    "taifex_futures_daily/TX": date(1998, 7, 21),
    "twse_taiex_daily": date(1999, 1, 5),
    "twse_taiex_total_return": date(2003, 1, 2),
}


def test_with_the_refdata_first_days_a_late_series_moves_the_window_into_the_route():
    # owner decision 2026-10-25: the same live sample is accepted; its window starts at 2003-01-02 + warm-up
    sample = json.loads((Path(__file__).parent / "fixtures" / "spec_v3_live_8_daily_vol.json").read_text())
    client = FakeClient([reply(sample["reply"])])
    outcome, _ = run_v3(client, idea=sample["idea"], series_since=SERIES, data_end=AS_OF)
    assert outcome.verdict == Verdict.PASS and len(client.chat_payloads) == 1, outcome.report
    rule = outcome.report["window_start"]
    assert (
        rule["binding_series"] == "twse_taiex_total_return"
        and rule["series_since"]["twse_taiex_total_return"] == "2003-01-02"
    )
    data = outcome.report["route"]["data"]
    assert data["start"] == (date(2003, 1, 2) + timedelta(days=rule["warmup_calendar_days"])).isoformat()
    assert data["end"] == AS_OF.isoformat() and data["start_rule"] == rule
    assert f"資料起點 {data['start']}（twse_taiex_total_return 從 2003-01-02 起，加暖機）" in outcome.report["summary"]
    # later stages see that window: the routed class's frozen config with the route's data
    config = routed()
    seen = effective_config(config, outcome.report["route"])
    # the class entry's trading-calendar rule stays with the route's window
    assert seen["data"] == {**data, "trading_calendar": trading_calendar_record()}
    assert config["by_class"]["daily_equity"]["data"]["start"] == "2001-01-02"
    with pytest.raises(RouteError, match="keep the frozen window end"):
        effective_config(config, {**outcome.report["route"], "data": {**data, "end": "2026-10-25"}})
    with pytest.raises(ValueError, match="needs data_end"):
        run_v3(FakeClient([]), series_since=SERIES)


def test_a_spec_whose_series_start_early_routes_without_window_data():
    outcome, _ = run_v3(FakeClient([reply(ok_reply())]), series_since=SERIES, data_end=AS_OF)
    assert outcome.verdict == Verdict.PASS and "data" not in outcome.report["route"]
    assert "window_start" not in outcome.report and "資料起點" not in outcome.report["summary"]


def test_a_not_implemented_add_on_passes_and_is_listed():
    idea = EXAMPLE_IDEA + "再買 put 保險。"
    body = ok_reply()
    body["clauses"].append({"id": "c9", "quote": "再買 put 保險", "slot": "hedge", "component": None})
    client = FakeClient([reply(body)])
    outcome, _ = run_v3(client, idea=idea)
    assert outcome.verdict == Verdict.PASS, outcome.report
    assert [c["quote"] for c in outcome.report["unimplemented"]] == ["再買 put 保險"]
    assert "未實作 1（再買 put 保險）" in outcome.report["summary"]


def test_a_daily_route_without_frozen_daily_config_is_unsupported():
    client = FakeClient([reply(ok_reply())])
    outcome, _ = run_v3(
        client,
        allowed_classes=("intraday_fixed",),
        unavailable={"daily_equity": "refdata was not available at run creation: gone"},
    )
    assert outcome.verdict == Verdict.UNSUPPORTED and "refdata was not available" in outcome.reason
    assert outcome.report["missing_capabilities"] == ["daily_equity"] and "route" not in outcome.report


# --------------------------------------------------------------- review hook (JEV plugs in here)


def test_review_items_go_back_like_validator_errors():
    calls = []

    def review(idea_text, clauses, spec):
        calls.append((idea_text, len(clauses), spec["name"]))
        if len(calls) == 1:
            return [{"path": "clauses.6", "msg": "JEV: c7 maps to sizing, not leverage"}]
        return []

    client = FakeClient([reply(ok_reply()), reply(ok_reply())])
    outcome, _ = run_v3(client, review=review)
    assert outcome.verdict == Verdict.PASS and calls == [(EXAMPLE_IDEA, 8, "tmf_dynamic_leverage_3x")] * 2
    assert "JEV: c7 maps to sizing" in client.chat_payloads[1]["messages"][-1]["content"]
    assert outcome.report["rounds"][0]["outcome"] == "review"
    assert [r["items"] for r in outcome.report["review"]] == [
        [{"path": "clauses.6", "msg": "JEV: c7 maps to sizing, not leverage"}],
        [],
    ]
    assert "review_unresolved" not in outcome.report


def test_review_disagreement_left_after_its_one_repair_is_report_only():
    client = FakeClient([reply(ok_reply())] * 3)
    outcome, _ = run_v3(client, review=lambda *_: ["contradicts: 3 倍 vs 2 倍"])
    assert outcome.verdict == Verdict.PASS and len(client.chat_payloads) == 2
    assert outcome.report["review_unresolved"] == [{"path": "review", "msg": "contradicts: 3 倍 vs 2 倍"}]


def test_a_failing_reviewer_errors_the_attempt():
    def review(*_):
        raise RuntimeError("JEV key label does not match")

    client = FakeClient([reply(ok_reply())])
    with pytest.raises(RuntimeError, match="JEV key label"):
        run_v3(client, review=review)
    assert client.closed


# --------------------------------------------------------------- key and cost


def test_missing_key_or_pin_or_identity_mismatch_never_calls_the_llm():
    client = FakeClient([reply(ok_reply())])
    for env in ({}, {"OPENROUTER_API_KEY": KEY}):
        outcome, created = run_v3(client, env=env)
        assert outcome.verdict == Verdict.ERROR and created == []
    outcome, created = run_v3(client, pin="")
    assert outcome.verdict == Verdict.ERROR and created == []
    mismatch = FakeClient([reply(ok_reply())], label="sk-or-v1-999...bcd")
    outcome, _ = run_v3(mismatch)
    assert outcome.verdict == Verdict.ERROR and mismatch.chat_payloads == []
    assert_no_key(outcome)


def test_worst_case_over_the_cap_refuses_the_call_up_front():
    client = FakeClient([reply(ok_reply())])
    outcome, _ = run_v3(client, cfg=v3_config(max_cost_usd_per_attempt=0.005))
    assert outcome.verdict == Verdict.ERROR and "cost cap" in outcome.reason and client.chat_payloads == []


def test_reported_cost_over_the_cap_stops_with_error():
    client = FakeClient([reply(ok_reply(), cost=0.2)])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.ERROR and "cost cap exceeded" in outcome.reason
    assert outcome.report["total_cost_usd"] == 0.2


def test_the_v3_cap_fits_the_real_prompt_with_two_repairs():
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V3}, {"role": "user", "content": "x" * 4000}]
    first = spec_llm.worst_case_cost(messages, DEFAULT_SPEC_GENERATE_V3)
    assert first < 0.04, first  # three rounds of this size stay under 0.10


# --------------------------------------------------------------- routing: v3 first, intraday -> DSL v2


def test_intraday_route_runs_the_unchanged_dsl_generation():
    dsl_spec = spec_llm._param_example()
    client = FakeClient(
        [
            reply({"status": "intraday", "reason": "5m opening range"}, cost=0.002),
            reply({"status": "ok", "spec": dsl_spec, "assumptions": ["stop at the range"]}, cost=0.001),
        ]
    )
    outcome = run_routed(client, "TX 30-minute opening range breakout")
    assert outcome.verdict == Verdict.PASS, outcome.report
    report = outcome.report
    assert report["route"] == {"strategy_class": "intraday_fixed", "engine": "dsl_v2"}
    assert report["spec"] == dsl_spec and report["prompt_sha256"] == spec_llm.PROMPT_SHA256
    assert report["router"]["status"] == "intraday" and report["total_cost_usd_with_router"] == 0.003
    dsl_payload = client.chat_payloads[1]
    assert dsl_payload["messages"][0]["content"] == spec_llm.SYSTEM_PROMPT and dsl_payload["max_tokens"] == 6000
    assert report["max_cost_usd_per_attempt"] == 0.05  # the intraday config's own DSL cap
    assert set(outcome.files) == {"llm_transcript_v3.json", "llm_transcript.json"}
    assert report["summary"].startswith("spec_generate ✅ intraday_fixed（日內 → DSL v2）")


def test_an_intraday_spec_v3_is_routed_to_dsl_too():
    from poseidon.pipeline.v3.dsl_bridge import v2_to_v3
    from tests.pipeline.conftest import tx_spec

    client = FakeClient(
        [
            reply(ok_reply(spec=v2_to_v3(tx_spec()))),
            reply({"status": "ok", "spec": spec_llm._param_example(), "assumptions": []}),
        ]
    )
    outcome = run_routed(client, "TX ORB")
    assert outcome.verdict == Verdict.PASS and outcome.report["route"]["engine"] == "dsl_v2"


def test_intraday_route_outside_the_family_class_is_unsupported_without_dsl():
    client = FakeClient([reply({"status": "intraday"})])
    outcome = run_routed(client, "TX ORB", routed(classes=("daily_equity",)))
    assert outcome.verdict == Verdict.UNSUPPORTED and len(client.chat_payloads) == 1
    assert "its family's class" in outcome.reason


def test_a_daily_route_is_reviewed_by_jev_with_one_report_only_repair():
    fake, factory = jev_factory()
    client = FakeClient([reply(ok_reply()), reply(ok_reply())])
    outcome = run_routed(client, EXAMPLE_IDEA, jev_client_factory=factory)
    assert outcome.verdict == Verdict.PASS and outcome.report["route"]["strategy_class"] == "daily_equity"
    report = outcome.report
    # the fake JEV puts "做多" in leverage: a slot disagreement goes back once, then stays report-only
    assert len(client.chat_payloads) == 2 and "JEV maps clause c2" in client.chat_payloads[1]["messages"][-1]["content"]
    assert report["rounds"][0]["outcome"] == "review" and report["review_unresolved"]
    jev_review = report["jev_review"]
    assert jev_review["blocking"] is False and len(jev_review["rounds"]) == 2
    assert jev_review["usage"]["calls"] == len(fake.systemone_bodies) > 0
    assert jev_review["usage"]["cost_usd"] == pytest.approx(0.00004 * len(fake.systemone_bodies))
    assert report["total_cost_usd"] == 0.002  # the LLM cost alone
    assert "jev_review_calls.json" in outcome.files and "llm_transcript_v3.json" in outcome.files
    assert report["clauses"][0] == {"id": "c1", "quote": "用微台", "slot": "instruments", "component": None}
    assert_no_key(outcome)


def test_jev_unavailable_errors_before_any_llm_call():
    _fake, factory = jev_factory(label="sk-or-v1-999...bcd")
    client = FakeClient([reply(ok_reply())])
    outcome = run_routed(client, EXAMPLE_IDEA, jev_client_factory=factory)
    assert outcome.verdict == Verdict.ERROR and outcome.reason.startswith("JEV unavailable")
    assert client.chat_payloads == [] and outcome.report["jev_review"]["usage"]["calls"] == 0


def test_a_failing_jev_call_is_recorded_and_never_decides():
    _fake, factory = jev_factory(systemone_status=500, error_message="upstream")
    client = FakeClient([reply(ok_reply())])
    outcome = run_routed(client, EXAMPLE_IDEA, jev_client_factory=factory)
    assert outcome.verdict == Verdict.PASS and len(client.chat_payloads) == 1
    assert "request failed" in outcome.report["jev_review"]["errors"][0]


def test_a_reply_to_review_feedback_that_fails_a_check_keeps_the_reviewed_spec():
    broken = ok_reply(clauses=[{"id": "c1", "quote": "paraphrase", "slot": "other", "component": None}])
    client = FakeClient([reply(ok_reply()), reply(broken)])
    outcome, _ = run_v3(client, review=lambda *_: [{"path": "clauses.1.slot", "msg": "JEV: leverage"}])
    assert outcome.verdict == Verdict.PASS and outcome.report["spec"] == tmf_dynamic_leverage_3x()
    assert outcome.report["review_repair"]["outcome"] == "invalid" and outcome.report["spec_round"] == 0
    assert outcome.report["review_unresolved"] == [{"path": "clauses.1.slot", "msg": "JEV: leverage"}]


def test_a_reply_to_review_feedback_that_turns_unsupported_keeps_the_reviewed_spec():
    client = FakeClient([reply(ok_reply()), reply({"status": "unsupported", "reason": "x", "missing": ["y"]})])
    outcome, _ = run_v3(client, review=lambda *_: ["JEV: contradicts"])
    assert outcome.verdict == Verdict.PASS and outcome.report["review_repair"]["outcome"] == "unsupported"


def test_slot_spellings_are_normalized():
    body = ok_reply()
    body["clauses"][0] = {**body["clauses"][0], "slot": "instrument"}
    client = FakeClient([reply(body)])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.PASS and outcome.report["clauses"][0]["slot"] == "instruments"


# --------------------------------------------------------------- routed config (CONFIG_VERSION 7)


def test_idea_run_config_freezes_every_candidate_class():
    config = routed()
    assert config["version"] == ROUTED_CONFIG_VERSION and config["spec_generate"] == DEFAULT_SPEC_GENERATE_V3
    assert set(config["by_class"]) == {"intraday_fixed", "daily_equity", "daily_fixed"}
    intraday = config["by_class"]["intraday_fixed"]
    assert intraday["version"] == 5 and intraday["stage_plan"] == list(V1_STAGE_PLAN)
    assert intraday["spec_generate"] == DEFAULT_SPEC_GENERATE
    daily = config["by_class"]["daily_equity"]
    assert daily["version"] == 6 and is_v3_run(daily) and daily["spec_source"] == "spec_generate"
    assert daily["stage_plan"] == list(V3_ROUTED_DAILY_STAGE_PLAN)
    assert daily["components"] == component_fingerprints() and daily["refdata"]["datasets"] == REFDATA
    assert daily["thresholds"]["benchmark_alpha_t_min"] == 1.65
    assert config["stage_plan"][0] == "spec_generate" and "execution_timing" in config["stage_plan"]
    assert config["spec_faithfulness"]["blocking"] is False == daily["spec_faithfulness"]["blocking"]
    assert "spec_faithfulness" in daily["stage_plan"] and "spec_faithfulness" not in intraday["stage_plan"]
    assert "unavailable" not in config


def test_without_refdata_the_daily_classes_are_listed_unavailable():
    config = routed(refdata=None)
    assert set(config["by_class"]) == {"intraday_fixed"}
    assert set(config["unavailable"]) == {"daily_equity", "daily_fixed"}
    assert config["stage_plan"] == list(V1_STAGE_PLAN)


def test_effective_config_picks_the_routed_class_without_touching_the_frozen_config():
    config = routed()
    before = content_sha256(config)
    assert effective_config(config, None) is config
    used = ["leverage.base", "regime.ma_band", "direction.constant"]
    daily = effective_config(config, {"strategy_class": "daily_fixed", "engine": "target_v1", "components": used})
    assert daily["strategy_class"] == "daily_fixed"
    assert sorted(key.split("@")[0] for key in daily["components"]) == sorted(used)
    intraday = effective_config(config, {"strategy_class": "intraday_fixed", "engine": "dsl_v2"})
    assert intraday == config["by_class"]["intraday_fixed"]
    assert content_sha256(config) == before
    with pytest.raises(RouteError):
        effective_config(routed(classes=("daily_equity",)), {"strategy_class": "intraday_fixed"})
    with pytest.raises(RouteError, match="not registered"):
        effective_config(config, {"strategy_class": "daily_equity", "components": ["regime.nope"]})
    hand_written = config["by_class"]["daily_equity"]
    assert effective_config(hand_written, {"strategy_class": "x"}) is hand_written  # <= 6: itself


def test_the_routed_daily_plan_validates_only_as_routed():
    plan = list(V3_ROUTED_DAILY_STAGE_PLAN)
    assert validate_stage_plan(plan, STAGE_REGISTRY, version=6, routed=True) == plan
    from poseidon.pipeline.plan import StagePlanError

    with pytest.raises(StagePlanError, match="not part of a spec v3 daily run"):
        validate_stage_plan(plan, STAGE_REGISTRY, version=6)


def test_daily_spec_validate_reads_the_generated_spec_only_for_routed_configs():
    stage = STAGE_REGISTRY["spec_validate"]
    routed_daily = routed()["by_class"]["daily_equity"]
    hand_written = {k: v for k, v in routed_daily.items() if k != "spec_source"}
    assert stage.upstream_names(routed_daily) == ("spec_generate",)
    assert stage.upstream_names(hand_written) == ()


# --------------------------------------------------------------- stage wiring


def _ctx(config) -> StageContext:
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text=EXAMPLE_IDEA,
        input_spec_json=None,
        config=config,
        config_sha256="0" * 64,
        stage="spec_generate",
        attempt=1,
        upstream={},
    )


def first_days_store(root):
    from tests.pipeline.v3_wiring_support import series_store

    return series_store(
        root,
        futures={"TX": "1998-07-21"},
        indices={"twse_taiex_daily": "1999-01-05", "twse_taiex_total_return": "2003-01-02"},
    )


def test_stage_routes_a_routed_run_through_v3_and_hashes_both_prompts(monkeypatch, tmp_path):
    monkeypatch.setenv(spec_llm.KEY_ENV, KEY)
    client = FakeClient([reply(ok_reply())])
    seen = []
    store = first_days_store(tmp_path)
    stage = SpecGenerateStage(
        client_factory=lambda key, timeout: client,
        pin_provider=lambda: LABEL,
        review=lambda idea, clauses, spec: seen.append(len(clauses)) or [],
        refdata=lambda: store,
    )
    ctx = _ctx(routed())
    payload = stage.input_payload(ctx)
    assert payload["prompt_v3_sha256"] == PROMPT_SHA256_V3 and payload["prompt_sha256"] == spec_llm.PROMPT_SHA256
    result = stage.run(ctx)
    assert result.verdict == Verdict.PASS and result.report["route"]["strategy_class"] == "daily_equity"
    assert seen == [8]
    legacy = _ctx(routed()["by_class"]["intraday_fixed"])
    assert "prompt_v3_sha256" not in stage.input_payload(legacy)


def test_stage_errors_before_any_llm_call_when_refdata_is_unreadable(monkeypatch, tmp_path):
    from poseidon.pipeline.refdata.store import RefdataStore

    monkeypatch.setenv(spec_llm.KEY_ENV, KEY)
    stage = SpecGenerateStage(
        client_factory=lambda *_: pytest.fail("no LLM call without the series first days"),
        pin_provider=lambda: LABEL,
        review=lambda *_: [],
        refdata=lambda: RefdataStore(tmp_path / "empty"),
    )
    result = stage.run(_ctx(routed()))
    assert result.verdict == Verdict.ERROR and result.reason.startswith("refdata is not readable")
    # a routed run that froze no daily class never reads refdata
    assert stage.daily_refdata(routed(classes=("intraday_fixed",))) == (None, None)


def test_reply_contract_errors_are_repaired():
    client = FakeClient([reply("not json"), reply({"status": "maybe"}), reply(ok_reply())])
    outcome, _ = run_v3(client)
    assert outcome.verdict == Verdict.PASS and outcome.report["repair_rounds"] == 2
    assert [r["outcome"] for r in outcome.report["rounds"]] == ["contract_error", "contract_error", "valid"]
    assert json.loads(outcome.files["llm_transcript_v3.json"])["prompt_sha256"] == PROMPT_SHA256_V3


def test_configs_of_runs_with_a_spec_are_unchanged_by_spec_generate_v3():
    """Hashes computed on stormtrooper (2026-10-24) with the integration code before this branch: the DSL
    config at 230cb82 and f80ad21 alike; the daily config at f80ad21 (step 6b added spec_faithfulness)."""

    from poseidon.pipeline.plan import default_config
    from poseidon.pipeline.v3.config import daily_run_config

    dsl = default_config(stage_plan=None, as_of=AS_OF)
    # backtest.night_session and thresholds.param_edge_at_bound_pass (configs frozen from 2026-10-28 on): without
    # those two keys it is the 230cb82 config
    assert content_sha256(dsl) == "d057831a7e81ab8dac3da160edaa7b236257f33307a77e51639a247061febad8"
    del dsl["thresholds"]["param_edge_at_bound_pass"]
    assert content_sha256(dsl) == "6087aa7057526c8d6c264d166fd88df1b4377a63e68e4f40933d64b0057571a6"
    del dsl["backtest"]["night_session"]
    assert content_sha256(dsl) == "60648707b1bc20e8544c9ba3baa21cee5c264ab3e34bc91d90c4dc20bc218ae2"
    resolved = validate_spec_v3(tmf_dynamic_leverage_3x()).resolved
    daily = daily_run_config(resolved, as_of=AS_OF, brief_text="brief", refdata=REFDATA)
    # db24f9f4 at REGISTRY_VERSION 2; registry_version moved it (3 adds rebalance.monthly), then data.warmup (the
    # session warm-up rule of configs frozen from 2026-10-28 on), then thresholds.param_edge_at_bound_pass (run #18,
    # 2026-10-28), then data.trading_calendar (2026-10-28): without those keys it is the 681d469 config
    assert content_sha256(daily) == "7ec6ce99749cb5f4bd5c5c84d887e63c9f965d07147a5453537811df35bd228a"
    del daily["data"]["trading_calendar"]
    assert content_sha256(daily) == "0e651e8e66df479e37ae99310c1148f04b95325f227a5a9a668c3c9975170034"
    del daily["thresholds"]["param_edge_at_bound_pass"]
    assert content_sha256(daily) == "a6673bdd55d6de26a08daa56038bbf73e69f4a60e69595c9aa9c95d98769ba00"
    del daily["data"]["warmup"]
    assert content_sha256(daily) == "b3ce5b0dadae872b4fb3908e1fb31ec5e2531e405e2492fca1c6faae951ea14c"
