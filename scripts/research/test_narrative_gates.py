"""Design 2026-10-29 §9 step 5: the narrative prompt (schema, rules 11-14), the run_gates gates [evidence] [theme]
[check] [proposal], and the 敘事變化 table in render_theme.py.

The payload is a real frozen manifest from a runner dry run over a seeded ledger; gates run on hand-built narratives.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_gates.py
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import draft  # noqa: E402
import runner  # noqa: E402
import test_narrative_snapshot as snap_t  # noqa: E402
import test_runner as tr  # noqa: E402
from test_runner import AGENDA_ID, DOC_ID, FakeOpenRouter  # noqa: E402
from test_runner import env as runner_env  # noqa: E402,F401  (pytest finds it by this name)

OUTLOOK = "The company expects wafer shipments to increase by high-single digit in the fourth quarter of 2026"


@pytest.fixture()
def env(runner_env):  # noqa: F811
    return runner_env


@pytest.fixture()
def seeded(env):
    """Ledger with two open KOL records about 2303; the snapshot lists the newer one first."""
    _, db = snap_t.ledger(env["tmp"])
    older = snap_t.utilization(db, 1, "2026-10-02T00:00:00Z")
    newer = snap_t.add(
        db,
        2,
        "聯電成熟製程產能利用率偏高",
        "UMC 成熟製程產能利用率偏高",
        ["2303"],
        ["foundry-utilization"],
        "2026-10-03T00:00:00Z",
    )
    db.close()
    run = snap_t.dry_run(env)
    led = runner.Ledger(env["state"] / "dry-run" / "ledger.sqlite")
    payload = led.manifest_payload(run["manifest_sha256"])
    led.close()
    cfg = json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "config.json").read_text())
    prompt = json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "prompt.json").read_text())
    return dict(env=env, payload=payload, cfg=cfg, prompt=prompt, ids=[newer, older])


def ledger_narrative(ids):
    n = tr.good_narrative()
    n["claims"][0].update(theme_ids=["foundry-utilization"], about_symbols=["2303", "NYSE:UMC"])
    n["claims"].append(
        {"kind": "fact", "text": "聯電預期 2026 年第四季晶圓出貨量將增加。", "evidence_ids": [DOC_ID], "quote": OUTLOOK}
    )
    n["narrative_checks"] = [
        {"narrative_id": ids[0], "verdict": "untouched", "claim_indices": [], "why": "本篇未提及。"},
        {
            "narrative_id": ids[1],
            "verdict": "partially_verifies",
            "claim_indices": [1],
            "supported_part": "聯電成熟製程仍有產能在運轉。",
            "open_part": "未說明產能利用率。",
            "why": "毛利率只間接相關。",
        },
    ]
    n["narrative_proposals"] = [
        {
            "claim_index": 3,
            "claim_type": "forecast",
            "resolves_on": "2026-12-31",
            "record_claim": "聯電 2026 年第四季晶圓出貨量將增加",
            "theme_ids": ["foundry-utilization"],
            "subject_symbols": ["2303"],
        }
    ]
    n["theme_proposals"] = [
        {
            "suggested_id": "wafer-shipments",
            "title": "晶圓出貨",
            "kind": "demand_driver",
            "why": "第四季出貨指引。",
            "claim_indices": [3],
            "proposed_members": [{"symbol": "2303", "role": "supplier"}],
        }
    ]
    return n


def gate_reasons(seeded, narrative, payload=None):
    reasons, _ = runner.run_gates(narrative, payload or seeded["payload"], seeded["cfg"])
    return reasons


# ------------------------------------------------------------------------------------------------ prompt
def test_prompt_carries_the_snapshot_schema_and_rules(seeded):
    system, user = seeded["prompt"][0]["content"], seeded["prompt"][1]["content"]
    assert "11. Themes:" in system and "14. Fact claims never cite narrative_snapshot" in system
    assert '"narrative_checks"' in system and '"theme_proposals"' in system and '"about_symbols"' in system
    assert "### evidence_id: narrative_snapshot (kind narrative_snapshot" in user
    assert (
        "- foundry-utilization | 晶圓代工產能利用率 | bottleneck | members 2303(supplier), NYSE:UMC(supplier)" in user
    )
    for nid in seeded["ids"]:
        assert f"- {nid} | state | unverified | kol | 股癌 | 聯電成熟製程產能利用率偏高 | first_seen 2026-10-0" in user


def test_schema_extension_shape():
    ext = draft.NARRATIVE_LEDGER_SCHEMA
    assert set(ext) == {"claims", "narrative_checks", "narrative_proposals", "theme_proposals"}
    assert set(ext["claims"][0]) == {*draft.NARRATIVE_SCHEMA["claims"][0], "theme_ids", "about_symbols"}
    assert set(draft.NARRATIVE_SCHEMA) & {"narrative_checks", "theme_proposals"} == set()


# ------------------------------------------------------------------------------------------------ gates
def test_a_full_ledger_narrative_passes(seeded):
    assert gate_reasons(seeded, ledger_narrative(seeded["ids"])) == []


def _drop_check(n):
    n["narrative_checks"].pop()


def _extra_check(n):
    n["narrative_checks"].append({"narrative_id": "nr-0000000000", "verdict": "untouched", "claim_indices": []})


def _dup_check(n):
    n["narrative_checks"].append(copy.deepcopy(n["narrative_checks"][0]))


def _cite_inference(n):
    n["narrative_checks"][1]["claim_indices"] = [2]


def _cite_exchange_feed(n):
    n["claims"][1]["evidence_ids"] = [DOC_ID, "ohlcv:2303"]


def _no_open_part(n):
    n["narrative_checks"][1]["open_part"] = " "


def _untouched_cites(n):
    n["narrative_checks"][0]["claim_indices"] = [0]


def _verifies_without_claims(n):
    n["narrative_checks"][0].update(verdict="verifies", claim_indices=[])


def _bad_verdict(n):
    n["narrative_checks"][0]["verdict"] = "confirms"


def _fact_cites_snapshot(n):
    n["claims"][0]["evidence_ids"] = [DOC_ID, "narrative_snapshot"]


def _unknown_theme(n):
    n["claims"][0]["theme_ids"] = ["cleanroom-capacity-x"]


def _theme_without_about(n):
    del n["claims"][0]["about_symbols"]


def _about_outside(n):
    n["claims"][0]["about_symbols"] = ["NASDAQ:NVDA"]


def _four_proposals(n):
    n["narrative_proposals"] *= 4


def _non_atomic_record(n):
    n["narrative_proposals"][0]["record_claim"] = "聯電第四季出貨增加 5%，毛利率上升到 35%"


def _record_number_not_in_quote(n):
    n["narrative_proposals"][0]["record_claim"] = "聯電 2026 年第四季晶圓出貨量將增加 9%"


def _resolves_on_without_period(n):
    n["narrative_proposals"][0]["resolves_on"] = "2027-12-31"


def _state_with_resolves_on(n):
    n["narrative_proposals"][0]["claim_type"] = "state"


def _proposal_unknown_theme(n):
    n["narrative_proposals"][0]["theme_ids"] = ["no-such-theme"]


def _proposal_bad_claim_index(n):
    n["narrative_proposals"][0]["claim_index"] = 9


def _theme_proposal_bad_kind(n):
    n["theme_proposals"][0]["kind"] = "rumor"


def _theme_proposal_bad_member(n):
    n["theme_proposals"][0]["proposed_members"] = [{"symbol": "Nittobo", "role": "supplier"}]


@pytest.mark.parametrize(
    "mutate, tag, needle",
    [
        (_drop_check, "[check]", "missing ["),
        (_extra_check, "[check]", "not in the snapshot ['nr-0000000000']"),
        (_dup_check, "[check]", "duplicated ['nr-"),
        (_cite_inference, "[check]", "claims[2] is an inference"),
        (_cite_exchange_feed, "[check]", "evidence is all official"),
        (_no_open_part, "[check]", "must state open_part"),
        (_untouched_cites, "[check]", "untouched check cites no claim"),
        (_verifies_without_claims, "[check]", "verifies needs claim_indices"),
        (_bad_verdict, "[check]", "verdict must be one of"),
        (_fact_cites_snapshot, "[evidence]", "claims[0] is a fact citing ['narrative_snapshot']"),
        (_unknown_theme, "[theme]", "closed list"),
        (_theme_without_about, "[theme]", "no about_symbols"),
        (_about_outside, "[theme]", "about_symbols ['NASDAQ:NVDA']"),
        (_four_proposals, "[proposal]", "at most 3"),
        (_non_atomic_record, "[proposal]", "non-atomic"),
        (_record_number_not_in_quote, "[proposal]", "numbers are not in claims[3]'s quote"),
        (_resolves_on_without_period, "[proposal]", "names no period"),
        (_state_with_resolves_on, "[proposal]", "of a forecast only"),
        (_proposal_unknown_theme, "[proposal]", "theme_ids must be a non-empty subset"),
        (_proposal_bad_claim_index, "[proposal]", "claim_index must point"),
        (_theme_proposal_bad_kind, "[proposal]", "kind must be one of"),
        (_theme_proposal_bad_member, "[proposal]", "needs a valid symbol and role"),
    ],
)
def test_each_gate_refuses(seeded, mutate, tag, needle):
    n = ledger_narrative(seeded["ids"])
    mutate(n)
    reasons = gate_reasons(seeded, n)
    assert any(r.startswith(tag) and needle in r for r in reasons), reasons


def test_partially_verifies_must_state_supported_part(seeded):
    """Owner decision 2026-10-30 (R3): JEV judges the part of the record a partial check says the cited claims
    establish, so the drafter must name it, as it names open_part."""
    for blank in ({}, {"supported_part": " "}):
        n = ledger_narrative(seeded["ids"])
        del n["narrative_checks"][1]["supported_part"]
        n["narrative_checks"][1].update(blank)
        reasons = gate_reasons(seeded, n)
        assert any(r.startswith("[check]") and "partially_verifies must state supported_part" in r for r in reasons), (
            blank,
            reasons,
        )
    assert '"supported_part"' in seeded["prompt"][0]["content"]


def test_proposal_from_an_inference_without_basis_is_refused(seeded):
    n = ledger_narrative(seeded["ids"])
    n["claims"][2]["text"] = "股價反應仍待事件後追蹤。"
    n["narrative_proposals"][0].update(
        claim_index=2, claim_type="state", resolves_on=None, record_claim="聯電股價反應仍待追蹤"
    )
    reasons = gate_reasons(seeded, n)
    assert any(r.startswith("[basis] claims[2]") for r in reasons)
    assert any(r.startswith("[proposal]") and "states no basis" in r for r in reasons)


def test_without_a_snapshot_nothing_may_be_checked_tagged_or_proposed(seeded):
    payload = copy.deepcopy(seeded["payload"])
    payload["evidence"] = [e for e in payload["evidence"] if e["kind"] != "narrative_snapshot"]
    reasons = gate_reasons(seeded, ledger_narrative(seeded["ids"]), payload)
    for tag in ("[check]", "[theme]", "[proposal]"):
        assert any(r.startswith(tag) for r in reasons), (tag, reasons)
    assert gate_reasons(seeded, tr.good_narrative(), payload) == []


# ------------------------------------------------------------------------------------------------ runner + render
def test_refused_checks_are_repaired_then_rendered(seeded):
    env = seeded["env"]
    first = ledger_narrative(seeded["ids"])
    first["narrative_checks"].pop()
    t = FakeOpenRouter([first, ledger_narrative(seeded["ids"])])
    assert env["run"](t) == 0
    run = json.loads((env["run_dir"] / "run.json").read_text())
    assert run["status"] == "completed", run["reasons"]
    assert any(r.startswith("[check]") for r in run["gates"][0]["reasons"]) and run["gates"][1]["reasons"] == []
    assert "[check]" in t.chat_calls[1]["messages"][-1]["content"]
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    research = rev["research"]
    assert [c["verdict"] for c in research["narrative_checks"]] == ["untouched", "partially_verifies"]
    assert research["claims"][0]["theme_ids"] == ["foundry-utilization"]
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(env["run_dir"] / "revision.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    table = md[md.index("## 敘事變化") : md.index("## 可重放身分")]
    assert md.index("## 變更摘要") < md.index("## 敘事變化")
    assert (
        f"| `{seeded['ids'][1]}` | 聯電成熟製程產能利用率偏高 | 股癌（kol） | 傳聞（KOL） | "
        "partially_verifies／jev supports／sweep supports | 部分證實（narrative_apply off，未寫入） |" in table
    )
    assert "claims[1]「Gross margin was 32.5% for the quarter」" in table


def render_with(seeded, tmp_path, results):
    rev = {
        "identity": {
            "revision_id": "0" * 8,
            "status": "completed",
            "manifest_sha256": "f" * 64,
            "stats": {},
            "evidence": [],
            "mutation_refused": "x",
            "provider": "p",
            "model": "m",
            "policy_version": "v",
            **{k: "-" for k in ("manifest_id", "request_sha256", "content_sha256", "runtime_digest")},
            "previous_revision_id": None,
        },
        "research": dict(ledger_narrative(seeded["ids"]), scope_key="tw_stock:spot:2303", as_of="2026-10-21T04:20:00Z"),
        "manifest_payload": seeded["payload"],
        "narrative_results": results,
    }
    path = tmp_path / "rev.json"
    path.write_text(json.dumps(rev, ensure_ascii=False))
    out = subprocess.run([sys.executable, str(HERE / "render_theme.py"), str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_render_shows_outcomes_and_owner_items(seeded, tmp_path):
    a, b = seeded["ids"]
    base = {"claim_indices": [], "jev_label": None, "sweep_truncated": 0, "event_seq": None, "why": ""}
    results = {
        "narrative_apply": "off",
        "jev_model": "typesafe/jev-1.13",
        "records": [
            {
                **base,
                "narrative_id": a,
                "status_before": "unverified",
                "drafter_verdict": "untouched",
                "sweep_label": "supports",
                "rule": "R9",
                "outcome": "missed",
                "target_status": None,
                "why": "sweep found support the drafter missed",
            },
            {
                **base,
                "narrative_id": b,
                "status_before": "unverified",
                "drafter_verdict": "partially_verifies",
                "claim_indices": [1],
                "jev_label": "says_nothing",
                "sweep_label": "says_nothing",
                "rule": "R3",
                "outcome": "status_change",
                "target_status": "partially_verified",
            },
        ],
    }
    md = render_with(seeded, tmp_path, results)
    assert "narrative_apply `off`；JEV `typesafe/jev-1.13`；展示 2 條，未展示 0 條" in md
    assert "partially_verifies／jev says_nothing／sweep says_nothing | 部分證實（narrative_apply off，未寫入） |" in md
    assert "untouched／sweep supports | 傳聞（KOL）（missed） | — |" in md
    owner = md[md.index("### 待 owner 判斷") :]
    assert f"- `{a}` missed（R9）：聯電成熟製程產能利用率偏高；sweep found support the drafter missed" in owner
    results["records"][1]["event_seq"] = 1
    assert "| 部分證實 | claims[1]" in render_with(seeded, tmp_path, results)
