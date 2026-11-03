"""Step 6b: JEV client, numeric precheck, clause split, the report-only ``spec_faithfulness`` stage and the
spec_generate v3 review hook. HTTP goes through a fake OpenRouter (``httpx.MockTransport``)."""

from __future__ import annotations

import copy
import json
import logging
import uuid
from datetime import date

import pytest

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import faithfulness, jev, openrouter
from poseidon.pipeline.clause_precheck import extract_quantities, precheck
from poseidon.pipeline.faithfulness import (
    FaithfulnessReviewer,
    SpecFaithfulnessStage,
    split_clauses,
    summary_line,
)
from poseidon.pipeline.plan import (
    CANONICAL_ORDER,
    DEFAULT_SPEC_FAITHFULNESS,
    STAGE_INTRODUCED_AT,
    V1_STAGE_PLAN,
    V3_DAILY_STAGE_PLAN,
    StagePlanError,
    validate_stage_plan,
)
from poseidon.pipeline.registry import STAGE_REGISTRY
from poseidon.pipeline.stage_api import IdeaOriginal, StageContext, UpstreamResult
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.v3.config import REFDATA_DATASETS, daily_run_config
from poseidon.pipeline.v3.describe import describe_spec
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.validation import validate_spec_v3
from tests.pipeline.jev_support import COST, ENV, KEY, LABEL, FakeOpenRouter, default_rule, fake_faithfulness_stage

IDEA = "用微台指做多 3 倍槓桿。加權指數跌破 200 日均線 2% 時降到 0.5 倍。買 put 保護。"
AS_OF = date(2026, 10, 22)


@pytest.fixture(autouse=True)
def _fresh_identity_cache():
    openrouter.reset_identity_cache()
    yield
    openrouter.reset_identity_cache()


def acceptance() -> dict:
    return validate_spec_v3(tmf_dynamic_leverage_3x()).template


def run_config(**section) -> dict:
    resolved = validate_spec_v3(tmf_dynamic_leverage_3x()).resolved
    refdata = {name: "0" * 64 for name in REFDATA_DATASETS}
    config = daily_run_config(resolved, as_of=AS_OF, brief_text="brief", refdata=refdata)
    config["spec_faithfulness"].update(section)
    return config


def ctx(config=None, *, idea=IDEA, template=None, generate_report=None, original=None) -> StageContext:
    config = config or run_config()
    upstream = {
        "spec_validate": UpstreamResult(
            "spec_validate", 1, "a" * 64, "b" * 64, {"template": template or acceptance(), "summary": "ok"}
        )
    }
    if generate_report is not None:
        config = {**config, "stage_plan": ["spec_generate", *config["stage_plan"]]}
        upstream["spec_generate"] = UpstreamResult("spec_generate", 1, "c" * 64, "d" * 64, generate_report)
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text=idea,
        input_spec_json=None,
        config=config,
        config_sha256=content_sha256(config),
        stage="spec_faithfulness",
        attempt=1,
        upstream=upstream,
        original=original,
    )


def blob(result) -> str:
    text = json.dumps(result.report, ensure_ascii=False) + (result.reason or "")
    return text + "".join(data.decode("utf-8") for data in result.files.values())


# ----------------------------------------------------------------------------- key identity: fail closed


@pytest.mark.parametrize(
    ("env", "pin", "fake", "reason"),
    [
        ({}, LABEL, FakeOpenRouter(), "STRATEGY_PIPELINE_OPENROUTER_KEY is not set"),
        # the research keys are a different purpose: never a fallback
        ({"JEV_RESEARCH_OPENROUTER_KEY": KEY, "JEV_EVAL_OPENROUTER_KEY": KEY}, LABEL, FakeOpenRouter(), "is not set"),
        (ENV, None, FakeOpenRouter(), "not pinned"),
        (ENV, LABEL, FakeOpenRouter(label="sk-or-v1-fee...566"), "key identity check failed"),
        (ENV, LABEL, FakeOpenRouter(limit_remaining=0.01), "limit_remaining"),
    ],
)
def test_missing_key_pin_or_identity_is_a_stage_error_without_any_question(env, pin, fake, reason):
    result = fake_faithfulness_stage(fake, env=env, pin=pin).run(ctx())
    assert result.verdict == Verdict.ERROR and reason in result.reason
    assert fake.systemone_bodies == []
    assert "JEV unavailable" in result.report["summary"]
    assert KEY not in blob(result)


def test_a_prefix_of_the_label_is_not_enough():
    fake = FakeOpenRouter(label=LABEL + "x")
    assert fake_faithfulness_stage(fake).run(ctx()).verdict == Verdict.ERROR
    assert fake.key_checks == 1 and fake.systemone_bodies == []


def test_identity_is_checked_with_the_spec_generate_pin_before_the_first_question():
    fake = FakeOpenRouter()
    result = fake_faithfulness_stage(fake).run(ctx())
    assert result.verdict == Verdict.PASS
    first = fake.requests[0]
    assert first.url.path == "/api/v1/key" and first.headers["authorization"] == f"Bearer {KEY}"
    assert result.report["jev"]["key_identity"]["checked"] == "live"


# ----------------------------------------------------------------------------- scrubbing


def test_the_key_never_reaches_reports_files_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    idea = f"做多 3 倍槓桿。不小心貼上 {KEY} 這串。跌破 200 日均線降到 0.5 倍"
    result = fake_faithfulness_stage(FakeOpenRouter()).run(ctx(idea=idea))
    assert result.verdict == Verdict.PASS
    assert KEY not in blob(result) and "<key>" in blob(result)
    assert KEY not in caplog.text


def test_a_failed_request_is_an_error_with_the_calls_so_far_and_no_key():
    fake = FakeOpenRouter(systemone_status=502, error_message=f"upstream rejected Bearer {KEY}")
    result = fake_faithfulness_stage(fake).run(ctx())
    assert result.verdict == Verdict.ERROR and "request failed" in result.reason
    assert KEY not in blob(result)
    calls = json.loads(result.files["jev_calls.json"])["calls"]
    assert len(calls) == 1 and "<key>" in calls[0]["error"]


# ----------------------------------------------------------------------------- precheck (strict)


@pytest.mark.parametrize(
    ("clause", "decides", "kinds"),
    [
        ("用微台指做多 3 倍槓桿", False, []),
        ("三倍槓桿", False, []),
        ("3x leverage", False, []),
        ("用 2 倍槓桿", True, ["multiple"]),
        ("跌破 200 日均線 2% 時降到 0.5 倍", False, []),
        ("跌破 200 日均線 2% 時降到 1 倍", True, ["multiple"]),
        ("槓桿不超過 5 倍", False, []),  # the policy cap
        ("做空台指", True, ["sign"]),
        ("不做空", False, []),
        ("左側依技術分析和估值計算做空間規劃", False, []),  # 做空間 is space planning, not a short
        ("空頭時降到 0.5 倍", False, []),  # a bear regime, not a short position
        ("用 300 日均線", False, []),  # days outside the $param range: advisory only
        ("MA200 濾網", False, []),
        ("站回均線上方 5%", False, []),  # percent outside the range: advisory only
        ("槓桿＝3 倍和「75% ÷ 近期波動」取比較小的", False, []),  # 75 % = vol_k 0.75
    ],
)
def test_precheck_strict_decides_only_multiple_and_sign(clause, decides, kinds):
    checked = precheck(clause, acceptance())
    assert checked["decides"] is decides
    assert checked["deciding_kinds"] == kinds
    assert checked["decided_label"] == ("contradicts" if decides else None)


def test_precheck_matches_param_ranges_and_reports_advisory_misses():
    checked = precheck("跌破 300 日均線 2% 時降到 0.5 倍", acceptance())
    by_raw = {item["raw"]: item for item in checked["matched"]}
    assert by_raw["2%"]["spec"][0]["source"] == "$param:band"
    assert by_raw["0.5 倍"]["spec"][0]["path"].endswith("off_leverage")
    assert [(m["kind"], m["value"]) for m in checked["unmatched"]] == [("days", 300.0)]
    assert not checked["decides"]
    assert precheck("用 2 倍槓桿", acceptance(), mode="off")["decides"] is False


def test_a_multiple_is_not_decided_when_the_spec_has_no_leverage():
    template = acceptance()
    template["sleeves"][0]["leverage"] = []
    template["sleeves"][0]["sizing"] = {"component": "sizing.fixed_contracts", "params": {"n": 1}}
    checked = precheck("用 3 倍槓桿", template)
    assert checked["unmatched"][0]["kind"] == "multiple_no_leverage" and not checked["decides"]


def test_quantities_are_read_once_each():
    assert [(q.kind, q.value) for q in extract_quantities("3 倍、200 日均線 ±2%、0.75、二十日、1,000,000")] == [
        ("multiple", 3.0),
        ("days", 200.0),
        ("percent", 2.0),
        ("number", 0.75),
        ("days", 20.0),
        ("number", 1_000_000.0),
    ]


# ----------------------------------------------------------------------------- clauses


def test_sentence_split_is_deterministic_and_verbatim():
    idea = "口數＝權益數 × 3 ÷ 一口價值，帳戶 1,000,000 元。TX\n📈\n收盤跌破 200 日線 2% 以下；隔天降到 0.5 倍！ "
    clauses = split_clauses(idea)
    assert [c.quote for c in clauses] == [
        "口數＝權益數 × 3 ÷ 一口價值，帳戶 1,000,000 元",  # commas stay inside a clause
        "TX",
        "收盤跌破 200 日線 2% 以下",  # the emoji-only line is no clause
        "隔天降到 0.5 倍",
    ]
    assert all(c.quote in idea and c.source == "sentence_split" for c in clauses)
    assert [c.id for c in clauses] == ["s1", "s2", "s3", "s4"]
    assert split_clauses(idea) == clauses
    assert split_clauses("") == [] and split_clauses("。；\n 🛡️ ") == []


# ----------------------------------------------------------------------------- the stage: report-only


def test_hand_written_run_reports_implemented_and_missing_clauses():
    fake = FakeOpenRouter()
    result = fake_faithfulness_stage(fake).run(ctx())
    report = result.report
    assert result.verdict == Verdict.PASS
    assert report["clause_source"] == "sentence_split"
    assert report["summary"] == (
        "spec_faithfulness ⚠️ 已實作 2 句 / 未實作 1 句（「買 put 保護」） / contradicts 0；report-only"
    )
    assert result.reason == report["summary"]
    assert report["counts"] == {
        "clauses": 3,
        "implemented": 2,
        "not_implemented": 1,
        "contradicts": 0,
        "not_strategy": 0,
        "mapping_disagreements": 0,
        "decided_by_precheck": 0,
    }
    put = report["clauses"][2]
    assert put["jev_slot"]["choice"] == "hedge"
    # JEV maps the clause to the registered put; the acceptance spec has no hedge, so it stays not implemented
    assert put["jev_component"]["choice"] == "hedge.protective_put" and put["faithfulness"]["label"] == "says_nothing"
    # the statement JEV judged against is describe_spec of the validated template
    checked = validate_spec_v3(acceptance())
    assert report["statement"] == describe_spec(checked.template, checked.params)
    assert {b["state"].get("spec_statement") for b in fake.systemone_bodies} - {None} == {report["statement"]}


def test_contradicts_never_blocks_while_the_flag_is_off():
    def everything_contradicts(name, state, criteria):
        return "contradicts" if name == "spec_faithfulness" else default_rule(name, state, criteria)

    result = fake_faithfulness_stage(FakeOpenRouter(everything_contradicts)).run(ctx())
    assert DEFAULT_SPEC_FAITHFULNESS["blocking"] is False and result.report["blocking"] is False
    assert result.verdict == Verdict.PASS
    assert result.report["counts"]["contradicts"] == 3
    assert result.report["summary"].startswith("spec_faithfulness ⚠️ 已實作 0 句 / 未實作 0 句 / contradicts 3（")

    blocked = fake_faithfulness_stage(FakeOpenRouter(everything_contradicts)).run(ctx(run_config(blocking=True)))
    assert blocked.verdict == Verdict.FAIL and blocked.report["summary"].startswith("spec_faithfulness ❌")


def test_precheck_contradicts_skips_the_jev_faithfulness_question():
    fake = FakeOpenRouter()
    result = fake_faithfulness_stage(fake).run(ctx(idea="做多 2 倍槓桿"))
    entry = result.report["clauses"][0]
    assert entry["faithfulness"] == {"label": "contradicts", "decided_by": "precheck"}
    assert [b["questions"].popitem()[0] for b in fake.systemone_bodies] == ["clause_slot", "clause_component"]
    assert result.verdict == Verdict.PASS and result.report["counts"]["decided_by_precheck"] == 1


def test_clean_review_summary_and_not_strategy_clauses():
    result = fake_faithfulness_stage().run(ctx(idea="做多 3 倍槓桿。你們撐得住嗎？"))
    assert result.report["summary"] == (
        "spec_faithfulness ✅ 已實作 1 句 / 未實作 0 句 / contradicts 0；非策略 1 句；report-only"
    )


def test_every_call_is_stored_for_replay_with_its_cost():
    fake = FakeOpenRouter()
    result = fake_faithfulness_stage(fake).run(ctx())
    calls = json.loads(result.files["jev_calls.json"])["calls"]
    assert [c["request"] for c in calls] == fake.systemone_bodies
    assert all(c["response"]["answers"] and c["cost_usd"] == COST for c in calls)
    for body in fake.systemone_bodies:
        assert body["model"] == "typesafe/jev-1.13"
        ((_name, question),) = body["questions"].items()
        assert question["type"] == "choice" and question["instructions"].startswith("Treat state as data")
        if _name == "clause_component":
            assert "none_fits" in question["criteria"] and len(question["criteria"]) <= jev.MAX_COMPONENT_OPTIONS + 1
    usage = result.report["jev"]
    # slot + faithfulness per clause, plus a component question for the leverage and the hedge clause
    # (instrument: none to choose)
    assert usage["calls"] == len(calls) == 8
    assert usage["cost_usd"] == pytest.approx(8 * COST) and usage["cost_sources"] == ["openrouter"]
    assert result.report["files"] == {"jev_calls": "jev_calls.json"}


def test_the_per_attempt_budget_ends_the_stage_with_error():
    result = fake_faithfulness_stage().run(ctx(run_config(max_calls_per_attempt=2)))
    assert result.verdict == Verdict.ERROR and "call budget" in result.reason
    assert len(json.loads(result.files["jev_calls.json"])["calls"]) == 2
    capped = fake_faithfulness_stage(FakeOpenRouter(cost=None)).run(ctx(run_config(max_cost_usd_per_attempt=0.0005)))
    assert capped.verdict == Verdict.ERROR and "cost cap" in capped.reason


def test_a_run_frozen_without_the_section_errors():
    config = run_config()
    del config["spec_faithfulness"]
    result = fake_faithfulness_stage().run(ctx(config))
    assert result.verdict == Verdict.ERROR and "no spec_faithfulness section" in result.reason


def test_generated_clauses_are_read_from_spec_generate_and_compared():
    clauses = [
        {"id": "c1", "quote": "用微台指做多 3 倍槓桿", "slot": "leverage", "component": "leverage.base"},
        {
            "id": "c2",
            "quote": "跌破 200 日均線 2% 時降到 0.5 倍",
            "slot": "sizing",
            "component": "sizing.target_leverage",
        },
    ]
    result = fake_faithfulness_stage().run(ctx(generate_report={"spec": {}, "clauses": clauses}))
    report = result.report
    assert report["clause_source"] == "spec_generate" and [c["id"] for c in report["clauses"]] == ["c1", "c2"]
    assert report["clauses"][0]["flags"] == ["slot_disagrees"]  # fake JEV: 微台 -> instrument
    assert report["clauses"][1]["flags"] == ["slot_disagrees", "component_disagrees"]  # leverage / ma_band
    assert "JEV 對應不一致 2 句" in report["summary"] and result.verdict == Verdict.PASS


# ----------------------------------------------------------------------------- spec_generate v3 hook


def test_review_hook_returns_validator_style_items():
    fake = FakeOpenRouter()
    clauses = [
        {"id": "c1", "quote": "做多 3 倍槓桿", "slot": "leverage", "component": "leverage.base"},
        "not a clause",  # skipped; positions still index the reply's list
        {"id": "c3", "quote": "跌破 200 日均線降到 0.5 倍", "slot": "direction", "component": "direction.constant"},
        {"id": "c4", "quote": "用 2 倍槓桿", "slot": "leverage", "component": "leverage.base"},
        {"id": "c5", "quote": "買 put 保護", "slot": "hedge", "component": "hedge.protective_put"},
    ]
    with FaithfulnessReviewer(
        DEFAULT_SPEC_FAITHFULNESS, pin=LABEL, environ=ENV, client_factory=fake.factory()
    ) as review:
        items = review(
            "做多 3 倍槓桿，跌破 200 日均線降到 0.5 倍，用 2 倍槓桿，買 put 保護", clauses, tmf_dynamic_leverage_3x()
        )
    assert all(set(item) == {"path", "msg"} and isinstance(item["msg"], str) for item in items)
    assert all(len(item["msg"]) <= openrouter.MAX_TEXT for item in items)
    assert [item["path"] for item in items] == [
        "clauses.2.slot",  # JEV: leverage
        "clauses.2.component",
        "clauses.3",  # 2x against the spec's 3x: precheck contradicts
    ]  # clauses.4: JEV agrees on hedge.protective_put
    assert "contradicts clause c4" in items[2]["msg"] and "2 倍 (multiple)" in items[2]["msg"]
    assert review.session.cost_usd > 0 and review.session.files()["jev_calls.json"]
    assert len(review.rounds) == 1


def test_review_hook_is_silent_for_an_invalid_spec_and_fails_closed_without_a_key():
    review = FaithfulnessReviewer(
        DEFAULT_SPEC_FAITHFULNESS, pin=LABEL, environ=ENV, client_factory=FakeOpenRouter().factory()
    )
    review.open()
    assert review("做多", [{"id": "c1", "quote": "做多"}], {"spec_version": 3}) == []
    review.close()
    with pytest.raises(jev.JevUnavailable):
        FaithfulnessReviewer(DEFAULT_SPEC_FAITHFULNESS, pin=LABEL, environ={}).open()


# ----------------------------------------------------------------------------- plan and old runs


def test_spec_faithfulness_is_in_new_v3_plans_only():
    assert CANONICAL_ORDER.index("spec_faithfulness") == CANONICAL_ORDER.index("spec_validate") + 1
    assert list(V3_DAILY_STAGE_PLAN[:3]) == ["spec_validate", "spec_faithfulness", "backtest"]
    assert STAGE_INTRODUCED_AT["spec_faithfulness"] == 6 and "spec_faithfulness" not in V1_STAGE_PLAN
    config = run_config()
    assert config["version"] == 6 and config["spec_faithfulness"] == DEFAULT_SPEC_FAITHFULNESS
    assert config["spec_faithfulness"]["blocking"] is False
    with pytest.raises(StagePlanError, match="does not exist at config version 5"):
        validate_stage_plan(["spec_generate", "spec_validate", "spec_faithfulness"], STAGE_REGISTRY, version=5)
    assert isinstance(STAGE_REGISTRY["spec_faithfulness"], SpecFaithfulnessStage)


def test_a_v3_run_frozen_before_it_keeps_its_plan_requires_and_hashes():
    old = copy.deepcopy(run_config())
    del old["spec_faithfulness"]
    old["stage_plan"] = [name for name in old["stage_plan"] if name != "spec_faithfulness"]
    assert validate_stage_plan(old["stage_plan"], STAGE_REGISTRY, version=6) == old["stage_plan"]
    for name in old["stage_plan"]:
        stage = STAGE_REGISTRY[name]
        assert "spec_faithfulness" not in stage.upstream_names(old)
    # nothing downstream of a new run requires it either: report-only can never gate through requires
    new = run_config()
    assert all("spec_faithfulness" not in STAGE_REGISTRY[n].upstream_names(new) for n in new["stage_plan"])


def test_upstream_adds_spec_generate_only_when_planned():
    stage = STAGE_REGISTRY["spec_faithfulness"]
    config = run_config()
    assert stage.upstream_names(config) == ("spec_validate",)
    assert stage.upstream_names({**config, "stage_plan": ["spec_generate", *config["stage_plan"]]}) == (
        "spec_generate",
        "spec_validate",
    )


def test_summary_lists_at_most_five_quotes():
    entries = [
        {"id": f"s{i}", "quote": f"第 {i} 個沒有實作的條件，而且這句話寫得非常非常長", "flags": []} for i in range(7)
    ]
    result = {
        "clauses": entries,
        "not_judged": [],
        "counts": {
            "implemented": 0,
            "not_implemented": 7,
            "contradicts": 0,
            "mapping_disagreements": 0,
            "not_strategy": 0,
        },
        "not_implemented": [e["id"] for e in entries],
        "contradicted": [],
    }
    line, clean = summary_line(result, blocking=False)
    assert not clean and line.count("「") == 5 and "等 7 句" in line and "…」" in line


# ----------------------------------------------------------------------------- direction check (drafted runs)

ORIGINAL_TEXT = "左側分批買入。恐慌時建倉。跌破季線就停損。過去低點在 -30% 附近。末日點位即整個頭部型態壓下來"
ORIGINAL = IdeaOriginal(ORIGINAL_TEXT, "e" * 64, "sisyphus")
# fall-response answers per original sentence: two clear adds (0.99, and 0.9 at the bar), a clear cut, and
# the experiment's two weak answers (an add at 0.49, a spurious cut at 0.79)
FALLS = {
    "左側分批買入": ("add_on_fall", 0.99),
    "恐慌時建倉": ("add_on_fall", 0.9),
    "跌破季線就停損": ("cut_on_fall", 0.99),
    "過去低點在 -30% 附近": ("add_on_fall", 0.49),
    "末日點位即整個頭部型態壓下來": ("cut_on_fall", 0.79),
}


def falls(statement_answer):
    def rule(name, state, criteria):
        if name == "fall_response":
            return FALLS.get(state["text"], statement_answer)
        return default_rule(name, state, criteria)

    return FakeOpenRouter(rule)


def test_a_drafted_run_flags_original_sentences_opposite_to_the_spec_direction():
    fake = falls(("cut_on_fall", 1.0))  # run #11: the spec cuts leverage below the MA band
    result = fake_faithfulness_stage(fake).run(ctx(original=ORIGINAL))
    section = result.report["direction_check"]
    assert result.verdict == Verdict.PASS
    assert section["statement"]["direction"] == "cut_on_fall"
    assert section["flagged"] == ["s1", "s2"]  # the 0.49 add stays under the bar; the cuts agree
    assert section["counts"] == {"clauses": 5, "add_on_fall": 2, "cut_on_fall": 1, "flagged": 2}
    assert (section["threshold"], section["question_version"], section["blocking"]) == (0.9, 1, False)
    assert section["original_sha256"] == "e" * 64 and section["clause_source"] == "sentence_split"
    assert result.report["summary"] == (
        "spec_faithfulness ⚠️ 已實作 2 句 / 未實作 1 句（「買 put 保護」） / contradicts 0；"
        "原文方向相反 2 句（「左側分批買入」、「恐慌時建倉」）；report-only"
    )
    # counted apart: the clause review is the idea's, unchanged by the original
    undrafted = fake_faithfulness_stage().run(ctx())
    assert result.report["counts"] == undrafted.report["counts"]
    assert json.loads(result.files["jev_calls.json"]) == json.loads(undrafted.files["jev_calls.json"])
    direction_calls = json.loads(result.files["jev_direction_calls.json"])["calls"]
    assert [c["request"]["state"]["text"] for c in direction_calls] == [result.report["statement"], *FALLS]
    assert section["jev"]["calls"] == 6


def test_a_faithful_control_is_not_flagged_by_a_weak_opposite_answer():
    result = fake_faithfulness_stage(falls(("add_on_fall", 0.99))).run(ctx(original=ORIGINAL))
    section = result.report["direction_check"]
    assert section["flagged"] == ["s3"]  # the 0.99 cut; the 0.79 cut (s5) is under the bar
    assert "原文方向相反 1 句（「跌破季線就停損」）" in result.report["summary"]
    capped = fake_faithfulness_stage(falls(("add_on_fall", 0.99))).run(
        ctx(run_config(max_clauses=3), original=ORIGINAL)
    )
    assert [c["id"] for c in capped.report["direction_check"]["not_judged"]] == ["s4", "s5"]
    assert "原文方向相反 1 句（「跌破季線就停損」），原文超過上限未判 2 句" in capped.report["summary"]


@pytest.mark.parametrize("statement_answer", [("no_direction", 0.99), ("cut_on_fall", 0.8)])
def test_no_comparison_when_the_spec_direction_is_unclear(statement_answer):
    result = fake_faithfulness_stage(falls(statement_answer)).run(ctx(original=ORIGINAL))
    section = result.report["direction_check"]
    assert section["statement"]["direction"] is None and section["flagged"] == []
    assert "原文方向未比對：spec 下跌時無明確方向；report-only" in result.report["summary"]


def test_the_direction_check_never_fails_the_stage():
    blocking = run_config(blocking=True)
    flagged = fake_faithfulness_stage(falls(("cut_on_fall", 1.0))).run(ctx(blocking, original=ORIGINAL))
    assert flagged.verdict == Verdict.PASS and flagged.report["direction_check"]["counts"]["flagged"] == 2
    assert "原文方向相反 2 句（「左側分批買入」、「恐慌時建倉」）（report-only）" in flagged.report["summary"]

    def broken(name, state, criteria):
        return "bogus" if name == "fall_response" else default_rule(name, state, criteria)

    result = fake_faithfulness_stage(FakeOpenRouter(broken)).run(ctx(original=ORIGINAL))
    assert result.verdict == Verdict.PASS
    assert "no valid choice" in result.report["direction_check"]["error"]
    assert result.report["summary"].startswith("spec_faithfulness ⚠️")
    assert "原文方向檢查失敗" in result.report["summary"]
    assert len(json.loads(result.files["jev_direction_calls.json"])["calls"]) == 1


def test_the_direction_question_enters_the_input_hash_only_with_an_original(monkeypatch):
    stage = STAGE_REGISTRY["spec_faithfulness"]
    undrafted, drafted = ctx(), ctx(original=ORIGINAL)
    before = {
        "undrafted": content_sha256(stage.input_payload(undrafted)),
        "drafted": content_sha256(stage.input_payload(drafted)),
        "questions": jev.questions_sha256(),
    }
    assert set(stage.input_payload(undrafted)) == {
        "stage",
        "config_sha256",
        "upstream",
        "idea_sha256",
        "faithfulness_version",
        "precheck_version",
        "questions_sha256",
    }
    assert before["drafted"] != before["undrafted"]
    other = IdeaOriginal(ORIGINAL_TEXT + "。", "f" * 64, "sisyphus")
    assert content_sha256(stage.input_payload(ctx(original=other))) != before["drafted"]

    changes = [
        ("FALL_RESPONSE_QUESTION", jev.Question("fall_response", "reworded", jev.FALL_RESPONSE_QUESTION.criteria)),
        ("DIRECTION_QUESTION_VERSION", 2),
    ]
    for name, value in changes:
        with monkeypatch.context() as patch:
            patch.setattr(jev, name, value)
            assert content_sha256(stage.input_payload(undrafted)) == before["undrafted"]
            assert jev.questions_sha256() == before["questions"]
            assert content_sha256(stage.input_payload(drafted)) != before["drafted"]
    with monkeypatch.context() as patch:
        patch.setattr(faithfulness, "DIRECTION_MIN_P", 0.8)
        assert content_sha256(stage.input_payload(undrafted)) == before["undrafted"]
        assert content_sha256(stage.input_payload(drafted)) != before["drafted"]


# ----------------------------------------------------------------------------- precheck: ladder rungs, percent units

RUN18_CLAUSE = "加權指數從 N 日高點回檔 10% 以上時加到 2 倍，回到高點 2% 以內再降回 1 倍（跌了加槓桿、回前高退場，對照台股版的 NVDA→NVDL 手法）"


def run18_spec() -> dict:
    with open("tests/pipeline/fixtures/run18_ladder_spec.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_run18_ladder_clause_is_matched_by_the_rungs_not_contradicted():
    checked = precheck(RUN18_CLAUSE, run18_spec())
    assert checked["version"] == 3
    assert checked["decides"] is False and checked["decided_label"] is None
    assert checked["unmatched"] == []
    paths = {item["raw"]: [hit["path"] for hit in item["spec"]] for item in checked["matched"]}
    assert paths["10%"] == ["sleeves.0.leverage.1.params.steps.0.drawdown_pct"]
    assert paths["2 倍"] == ["sleeves.0.leverage.1.params.steps.0.leverage"]
    assert "sleeves.0.leverage.1.params.release_drawdown_pct" in paths["2%"]
    assert paths["1 倍"] == ["sleeves.0.leverage.0.params.value"]


def test_a_leverage_the_ladder_does_not_have_still_contradicts():
    assert precheck(RUN18_CLAUSE.replace("加到 2 倍", "加到 3 倍"), run18_spec())["deciding_kinds"] == ["multiple"]


def _one_component(slot: str, component: str, params: dict) -> dict:
    doc = {
        "instruments": {},
        "sleeves": [
            {
                "direction": {"component": "direction.constant", "params": {"side": "long"}},
                "leverage": [{"component": "leverage.base", "params": {"value": 1.0}}],
                "sizing": {"component": "sizing.target_leverage", "params": {}},
                "rebalance": {"component": "rebalance.daily", "params": {}},
                "exits": [],
                "hedges": [],
            }
        ],
        "allocation": {"component": "allocation.single"},
        "execution": {"component": "execution.next_open"},
    }
    ref = {"component": component, "params": params}
    if slot == "leverage":
        doc["sleeves"][0]["leverage"].append(ref)
    elif slot == "hedge":
        doc["sleeves"][0]["hedges"] = [ref]
    else:
        doc["sleeves"][0][slot] = ref
    return doc


PERCENT_FRACTION_PAIRS = [
    ("leverage", "regime.ma_band", {"ma_days": 200, "band_pct": 0.02, "off_leverage": 0.5}, "跌破 200 日均線 2%", "2%"),
    ("rebalance", "rebalance.threshold_band", {"band_pct": 0.1}, "偏離目標超過 10% 才調整", "10%"),
    ("hedge", "hedge.protective_put", {"moneyness_pct": 0.05}, "買價外 5% 的 put", "5%"),
    (
        "leverage",
        "leverage.drawdown_ladder",
        {"steps": [{"drawdown_pct": 0.15, "leverage": 2.0}], "release_drawdown_pct": 0.03},
        "回檔 15% 時加碼",
        "15%",
    ),
    (
        "leverage",
        "leverage.drawdown_ladder",
        {"steps": [{"drawdown_pct": 0.15, "leverage": 2.0}], "hold": True, "release_drawdown_pct": 0.03},
        "回到高點 3% 以內放開",
        "3%",
    ),
    ("hedge", "hedge.protective_put", {"take_profit_multiple": 3.0}, "漲到買價 3 倍就賣掉", "3 倍"),
]


@pytest.mark.parametrize(("slot", "component", "params", "clause", "raw"), PERCENT_FRACTION_PAIRS)
def test_percent_text_matches_the_fraction_the_registry_stores(slot, component, params, clause, raw):
    template = _one_component(slot, component, params)
    checked = precheck(clause, template)
    assert raw in {item["raw"] for item in checked["matched"]}, checked
    assert not checked["decides"] and checked["unmatched"] == []


def test_a_precheck_version_bump_changes_the_spec_faithfulness_input_hash(monkeypatch):
    stage = STAGE_REGISTRY["spec_faithfulness"]
    before = content_sha256(stage.input_payload(ctx()))
    monkeypatch.setattr(faithfulness, "PRECHECK_VERSION", faithfulness.PRECHECK_VERSION + 1)
    assert content_sha256(stage.input_payload(ctx())) != before
