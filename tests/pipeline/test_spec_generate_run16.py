"""Run #16 replayed: a clause no registered component can do must not be reported as implemented.

The draft says 「每月最後一個交易日收盤判斷」. When run #16 was recorded no registered component could decide
monthly for stocks (``rebalance.monthly`` did not exist yet; ``rebalance.monthly_at_roll`` is futures only),
so the recorded replies cite ``execution.next_open`` for that clause and the replay keeps those recorded
replies as they were, not what a model would answer now. The LLM's three recorded
replies and JEV's 104 recorded answers are replayed through the real ``generate_routed`` (spec checks,
``FaithfulnessReviewer``, repair loop); only the network is faked.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from poseidon.pipeline import spec_llm
from poseidon.pipeline.spec_llm_v3 import generate_routed
from poseidon.pipeline.state import Verdict
from tests.pipeline.jev_support import FakeOpenRouter
from tests.pipeline.test_spec_llm import ENV, LABEL, FakeClient, reply

RUN16 = json.loads((Path(__file__).parent / "fixtures" / "run16_spec_generate.json").read_text())
MONTHLY = "時段：日線，每月最後一個交易日收盤判斷，隔日開盤執行"


@pytest.fixture(autouse=True)
def _fresh_identity_cache():
    spec_llm.reset_identity_cache()
    yield
    spec_llm.reset_identity_cache()


def _replayed_jev() -> tuple[FakeOpenRouter, list[dict]]:
    queue = list(RUN16["jev_calls"])

    def rule(name, state, _criteria):
        call = queue.pop(0)
        sha = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert (name, sha) == (call["question"], call["state_sha256"]), "JEV was asked something run #16 was not"
        return call["choice"], call["confidence"]

    return FakeOpenRouter(rule, label=LABEL), queue


def replay_run16():
    jev, left = _replayed_jev()
    client = FakeClient([reply(content) for content in RUN16["llm_replies"]])
    outcome = generate_routed(
        RUN16["idea_text"],
        RUN16["config"],
        pin=LABEL,
        environ=ENV,
        client_factory=lambda *_: client,
        jev_client_factory=jev.factory(),
        series_since={name: date.fromisoformat(day) for name, day in RUN16["series_since"].items()},
    )
    assert left == [], "the replay asked JEV fewer questions than run #16"
    return outcome, client


def test_the_replay_reproduces_run16():
    outcome, _ = replay_run16()
    report, observed = outcome.report, RUN16["observed"]
    assert outcome.verdict == Verdict.PASS and report["strategy_class"] == "daily_equity"
    assert report["spec_sha256"] == observed["spec_sha256"] and report["spec_round"] == observed["spec_round"]
    assert [c["id"] for c in report["unimplemented"]] == observed["unimplemented"] == ["c8", "c10", "c11", "c19"]
    paths = [item["path"] for item in report["review_unresolved"]]
    assert paths == [item["path"] for item in observed["review_unresolved"]]  # same items; the wording may change
    # the final reply still cites a component for the monthly clause
    assert [(c["id"], c["component"]) for c in report["clauses"] if c["quote"] == MONTHLY] == [
        ("c2", "execution.next_open"),
        ("c3", None),
        ("c4", "execution.next_open"),
    ]


def test_a_clause_the_reviewer_still_contradicts_is_listed_in_the_summary():
    outcome, _ = replay_run16()
    report = outcome.report
    assert outcome.verdict == Verdict.PASS  # JEV is report-only (design §4): listed, never decides
    contradicted = {c["id"] for c in report["review_contradicted"]}
    assert {"c2", "c3", "c4", "c17"} <= contradicted
    assert not contradicted & {c["id"] for c in report["unimplemented"]}
    assert "JEV 判違背" in report["summary"] and MONTHLY[:20] in report["summary"]


def test_every_repair_after_a_contradiction_offers_the_not_implemented_route():
    _, client = replay_run16()
    after_validator, after_review = (p["messages"][-1]["content"] for p in client.chat_payloads[1:])
    # round 1: monthly_at_roll rejected for ETFs; the clauses citing it must not silently keep a component
    assert "rebalance.monthly_at_roll needs a futures instrument" in after_validator
    assert "component null" in after_validator
    # round 2: JEV says the spec contradicts c2; the model is told it may mark it not implemented
    contradicts_c2 = next(line for line in after_review.splitlines() if line.startswith("- clauses.1:"))
    assert "component to null" in contradicts_c2 and "not implemented" in contradicts_c2
