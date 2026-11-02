"""Design 2026-10-29 §9 step 6: narrative_apply (judgement JEV, sweep, rule table R1-R9, writes; off by default).

One runner run over nine seeded records, one per rule, with a fake JEV that answers by (record claim, judgement or
sweep). research_theme.py and the ledger are real. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_apply.py
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

import narrative_check  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import narrative_proposals  # noqa: E402
import runner  # noqa: E402
import testkit as tk  # noqa: E402
from testkit import CASES, JEV_KEY, PART, PART_LABELS, FakeJev, apply_later, ledger_rows, refrozen, run  # noqa: E402

DOC_TIME = "2026-10-06T00:00:00Z"  # available_at of the agenda document (event_date 2026-10-06)
EXPECTED = {  # rule -> (outcome, status after the run with narrative_apply on, event kind written)
    "R1": ("status_change", "verified", "status"),
    "R2": ("status_change", "refuted", "status"),
    "R3": ("status_change", "partially_verified", "status"),
    "R4": ("corroborated", "unverified", "corroborated"),
    "R5": ("status_change", "verified", "status"),
    "R6": ("disputed", "unverified", None),
    "R7": ("disputed", "partially_verified", None),
    "R8": ("no_change", "unverified", None),
    "R9": ("missed", "unverified", None),
}


@pytest.fixture()
def env(runner_env):
    return runner_env


# ------------------------------------------------------------------------------------------------ the rule table
def test_rule_table_rows():
    d = narrative_check.decide
    assert d("state", "verifies", "supports", True, None, "unverified") == ("R1", "status_change", "verified")
    assert d("state", "verifies", "supports", True, None, "stale") == ("R1", "status_change", "verified")
    assert d("state", "verifies", "supports", True, None, "partially_verified") == ("R1", "status_change", "verified")
    assert d("state", "refutes", "contradicts", True, None, "unverified") == ("R2", "status_change", "refuted")
    assert d("state", "partially_verifies", "supports", True, None, "unverified")[1:] == (
        "status_change",
        "partially_verified",
    )
    assert d("state", "partially_verifies", "says_nothing", False, None, "unverified") == (None, "no_change", None)
    assert d("forecast", "partially_verifies", "supports", True, True, "unverified") == ("R4", "corroborated", None)
    assert d("forecast", "verifies", "supports", True, True, "unverified") == ("R4", "corroborated", None)
    assert d("forecast", "refutes", "contradicts", True, False, "unverified") == ("R5", "status_change", "refuted")
    assert d("forecast", "refutes", "contradicts", True, True, "unverified") == (None, "no_change", None)
    assert d("forecast", "refutes", "contradicts", True, False, "partially_verified") == ("R7", "disputed", None)
    assert d("state", "partially_verifies", "contradicts", True, None, "unverified") == ("R6", "disputed", None)
    assert d("state", "refutes", "says_nothing", True, None, "unverified") == ("R6", "disputed", None)
    assert d("forecast", "untouched", "contradicts", False, None, "stale") == ("R9", "missed", None)
    # already there, or a move only the owner may make
    assert d("state", "partially_verifies", "supports", True, None, "partially_verified") == ("R3", "no_change", None)
    assert d("state", "verifies", "supports", True, None, "refuted") == ("R1", "disputed", "verified")
    assert {r[0] for r in narrative_check.RULE_ROWS} == {f"R{n}" for n in range(1, 10)}


def test_a_sweep_that_contradicts_a_verifying_drafter_is_disputed():
    """Acceptance 2026-10-30 §3: nr-392771f0bd and nr-93127dc3ce were forecasts the drafter verified and JEV supported
    while the sweep contradicted them; the table alone made them R4 endorsements."""
    d = narrative_check.decide
    disputed = ("R6", "disputed", None)
    assert d("forecast", "verifies", "supports", True, True, "unverified", "contradicts") == disputed
    assert d("forecast", "partially_verifies", "supports", True, True, "unverified", "contradicts") == disputed
    assert d("state", "verifies", "supports", True, None, "unverified", "contradicts") == disputed
    # a sweep that agrees, or says nothing, leaves the table's answer
    r2 = ("R2", "status_change", "refuted")
    assert d("state", "refutes", "contradicts", True, None, "unverified", "contradicts") == r2
    r4 = ("R4", "corroborated", None)
    assert d("forecast", "verifies", "supports", True, True, "unverified", "says_nothing") == r4
    r9 = ("R9", "missed", None)
    assert d("forecast", "untouched", "contradicts", False, None, "unverified", "contradicts") == r9


def test_sweep_contradiction_is_listed_for_the_owner_not_suggested(env, monkeypatch):
    monkeypatch.setitem(CASES, "R4", (*CASES["R4"][:6], "contradicts"))
    path, ids, _, _, rev = run(env, "confirm")
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}[ids["R4"]]
    assert (res["rule"], res["outcome"], res["sweep_label"]) == ("R6", "disputed", "contradicts")
    assert res["why"] == "drafter verifies and JEV supports, but the sweep contradicts" and "suggestion" not in res
    checks, events = ledger_rows(path, rev["identity"]["revision_id"])
    assert checks[ids["R4"]]["outcome"] == "disputed" and [e["kind"] for e in events[ids["R4"]]] == ["created"]
    db = sqlite3.connect(path)
    n = db.execute("SELECT COUNT(*) FROM narrative_suggestions WHERE narrative_id = ?", (ids["R4"],)).fetchone()[0]
    db.close()
    assert n == 0


def test_a_sweep_that_supports_a_refuting_drafter_is_disputed():
    """The drafter and the independent sweep disagree in direction either way: refutes + sweep supports is disputed,
    as verifies + sweep contradicts is. Same-direction sweeps leave the table's answer."""
    d = narrative_check.decide
    disputed = ("R6", "disputed", None)
    assert d("state", "refutes", "contradicts", True, None, "unverified", "supports") == disputed  # table: R2 refuted
    assert d("forecast", "refutes", "contradicts", True, False, "unverified", "supports") == disputed  # table: R5
    assert d("state", "refutes", "contradicts", True, None, "verified", "supports") == disputed  # table: R7
    # the same direction, or no sweep, leaves the table's answer
    assert d("state", "refutes", "contradicts", True, None, "unverified", "contradicts") == (
        "R2",
        "status_change",
        "refuted",
    )
    assert d("state", "refutes", "contradicts", True, None, "unverified", None) == ("R2", "status_change", "refuted")
    assert d("state", "refutes", "contradicts", True, None, "unverified", "says_nothing") == (
        "R2",
        "status_change",
        "refuted",
    )
    assert d("state", "verifies", "supports", True, None, "unverified", "supports") == (
        "R1",
        "status_change",
        "verified",
    )
    w = narrative_check.why
    assert w("R6", "disputed", "refutes", "contradicts", "supports", "unverified", None) == (
        "drafter refutes and JEV contradicts, but the sweep supports"
    )
    assert w("R6", "disputed", "refutes", "supports", "supports", "unverified", None) == (
        "drafter refutes but JEV supports and the sweep supports"
    )
    assert w("R6", "disputed", "refutes", "supports", "says_nothing", "unverified", None) == (
        "drafter refutes but JEV supports"
    )


def test_sweep_support_of_a_refuting_drafter_is_listed_for_the_owner_not_suggested(env, monkeypatch):
    monkeypatch.setitem(CASES, "R2", (*CASES["R2"][:6], "supports"))
    path, ids, _, _, rev = run(env, "confirm")
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}[ids["R2"]]
    assert (res["rule"], res["outcome"], res["sweep_label"]) == ("R6", "disputed", "supports")
    assert res["why"] == "drafter refutes and JEV contradicts, but the sweep supports" and "suggestion" not in res
    checks, events = ledger_rows(path, rev["identity"]["revision_id"])
    assert checks[ids["R2"]]["outcome"] == "disputed" and [e["kind"] for e in events[ids["R2"]]] == ["created"]
    db = sqlite3.connect(path)
    n = db.execute("SELECT COUNT(*) FROM narrative_suggestions WHERE narrative_id = ?", (ids["R2"],)).fetchone()[0]
    db.close()
    assert n == 0


# ------------------------------------------------------------------------------------------------ R3 supported_part
def test_r3_reads_the_supported_part_label():
    """Owner decision 2026-10-30: a whole-claim says_nothing no longer makes R3; part_label folds the whole-claim label
    and the two supported_part judgements into the label R3's row reads."""
    assert narrative_check.decide("state", "partially_verifies", "says_nothing", True, None, "unverified") == (
        None,
        "no_change",
        None,
    )

    def label(whole, in_record, backed):
        return narrative_check.part_label(whole, {"in_record": {"label": in_record}, "backed": {"label": backed}})

    assert label("says_nothing", "supports", "supports") == "supports"  # N15
    assert label("supports", "supports", "supports") == "supports"
    assert label("says_nothing", "says_nothing", "supports") == "says_nothing"  # ns-712a8221
    assert label("supports", "contradicts", "supports") == "says_nothing"
    assert label("says_nothing", "supports", "says_nothing") == "says_nothing"
    assert label("says_nothing", "supports", "contradicts") == "contradicts"
    assert label("contradicts", "supports", "supports") == "contradicts"


def run_r3(env, monkeypatch, mode, in_record="supports", backed="supports"):
    """The nine-record run with FakeJev's judgements (a) and (b) of the R3 check's supported_part set."""
    monkeypatch.setitem(PART_LABELS, "in_record", in_record)
    monkeypatch.setitem(PART_LABELS, "backed", backed)
    path, ids, t, _, rev = run(env, mode)
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}[ids["R3"]]
    return path, ids, t, rev, res


def suggestion_rows(path, nid):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM narrative_suggestions WHERE narrative_id = ?", (nid,))]
    db.close()
    return rows


def report_md(env):
    return subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(env["run_dir"] / "revision.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_r3_takes_a_supported_part_of_the_record_the_cited_claims_back(env, monkeypatch):
    """Design §7.3 N15: JEV says nothing about the whole record against the cited claim, but the drafter's supported_part
    is part of the record's claim (a) and the cited claim backs it (b): suggested partially_verified, with supported_part
    and both judgements in the suggestion and, once the owner confirms, in the event."""
    path, ids, t, rev, res = run_r3(env, monkeypatch, "confirm")
    assert (res["rule"], res["outcome"], res["jev_label"], res["sweep_label"]) == (
        "R3",
        "status_change",
        "says_nothing",
        "says_nothing",
    )
    assert res["suggestion"]["status"] == "suggested" and res["why"] == ""
    votes = ["supports"] * narrative_check.JEV_VOTES
    assert res["judged"]["in_record"]["votes"] == res["judged"]["backed"]["votes"] == votes
    # (a) reads the record's claim; (b) reads the same cited source as the whole-claim judgement
    record = CASES["R3"][0]
    whole = next(
        c["state"]["source_context"]["text"]
        for c in t.jev_calls
        if c["state"]["claim"] == record and c["state"]["source_context"]["locator"] == "claims[1]"
    )
    asked = {
        (c["state"]["source_context"]["locator"], c["state"]["source_context"]["text"])
        for c in t.jev_calls
        if c["state"]["claim"] == PART
    }
    assert asked == {(narrative_check.IN_RECORD_LOCATOR, record), ("claims[1]", whole)}
    [s] = suggestion_rows(path, ids["R3"])
    basis, jev = json.loads(s["basis_json"]), json.loads(s["jev_json"])
    assert (basis["supported_part"], basis["open_part"]) == (PART, "未說明主動性。")
    assert jev["label"] == "says_nothing"
    assert {k: (v["label"], v["votes"]) for k, v in jev["supported_part"].items()} == {
        "in_record": ("supports", votes),
        "backed": ("supports", votes),
    }
    result, _ = narrative_check.apply_decision(
        path,
        {
            "decision_id": "20261030-d1",
            "action": "confirm",
            "target": s["suggestion_id"],
            "owner_text": "確認敘事 " + s["suggestion_id"],
            "decided_at": "2026-10-30T09:00:00Z",
        },
    )
    assert result == "applied"
    _, events = ledger_rows(path, rev["identity"]["revision_id"])
    ev = events[ids["R3"]][-1]
    assert (ev["to_status"], json.loads(ev["basis_json"])["supported_part"]) == ("partially_verified", PART)
    assert json.loads(ev["jev_json"])["supported_part"]["in_record"]["label"] == "supports"


def test_r3_refuses_a_supported_part_that_is_not_part_of_the_record(env, monkeypatch):
    """ns-712a8221: the cited ASML release claim was true, so it backs what the drafter named (b), but the release never
    mentions cleanrooms: what the drafter named is not part of the record's claim (a). No suggestion; the report says
    which judgement failed."""
    path, ids, _, rev, res = run_r3(env, monkeypatch, "confirm", in_record="says_nothing")
    reason = "supported_part is not part of the record's claim (JEV says_nothing)"
    assert (res["rule"], res["outcome"], res["why"]) == (None, "no_change", reason) and "suggestion" not in res
    assert suggestion_rows(path, ids["R3"]) == []
    checks, _ = ledger_rows(path, rev["identity"]["revision_id"])
    assert checks[ids["R3"]]["outcome"] == "no_change"
    assert f"（不變：{reason}）" in report_md(env)


def test_r3_needs_the_cited_claims_to_back_the_supported_part(env, monkeypatch):
    path, ids, _, rev, res = run_r3(env, monkeypatch, "on", backed="says_nothing")
    reason = "the cited source does not back supported_part (JEV says_nothing)"
    assert (res["rule"], res["outcome"], res["why"]) == (None, "no_change", reason)
    checks, events = ledger_rows(path, rev["identity"]["revision_id"])
    assert checks[ids["R3"]]["event_seq"] is None and [e["kind"] for e in events[ids["R3"]]] == ["created"]


def test_r3_cited_claims_that_contradict_the_supported_part_are_disputed(env, monkeypatch):
    path, ids, _, _, res = run_r3(env, monkeypatch, "confirm", backed="contradicts")
    reason = "the cited source does not back supported_part (JEV contradicts)"
    assert (res["rule"], res["outcome"], res["why"]) == ("R6", "disputed", reason) and "suggestion" not in res
    assert suggestion_rows(path, ids["R3"]) == []
    owner = report_md(env)
    assert f"- `{ids['R3']}` disputed（R6）：{CASES['R3'][0]}；{reason}" in owner[owner.index("### 待 owner 判斷") :]


def test_overlapping_rows_are_refused():
    with pytest.raises(AssertionError, match="overlap"):
        narrative_check._expand(
            (*narrative_check.RULE_ROWS, ("RX", "*", "untouched", "says_nothing", "*", "*", "*", "missed", None))
        )


# ------------------------------------------------------------------------------------------------ runner
def test_on_writes_one_event_per_rule_row(env):
    path, ids, t, run_json, rev = run(env, "on")
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}
    checks, events = ledger_rows(path, rev["identity"]["revision_id"])
    assert rev["narrative_results"]["narrative_apply"] == "on" and run_json["narrative_apply"] == "on"
    for rule, nid in ids.items():
        outcome, status, kind = EXPECTED[rule]
        assert res[nid]["rule"] == rule and res[nid]["outcome"] == checks[nid]["outcome"] == outcome, rule
        assert nl.fold_status(events[nid]) == status, rule
        written = [e for e in events[nid] if e["rule"] == rule]
        assert [e["kind"] for e in written] == ([kind] if kind else []), rule
        assert checks[nid]["event_seq"] == (written[0]["seq"] if kind else None), rule
        assert checks[nid]["jev_label"] == CASES[rule][5] and checks[nid]["sweep_label"] == CASES[rule][6], rule
        sweep = res[nid]["judged"]["sweep"]
        assert sweep["precheck"] == "none" and sweep["votes"] == [CASES[rule][6]] * narrative_check.JEV_VOTES, rule
    assert checks[ids["R7"]]["status_before"] == "partially_verified"
    r1 = next(e for e in events[ids["R1"]] if e["rule"] == "R1")
    basis, jev = json.loads(r1["basis_json"]), json.loads(r1["jev_json"])
    assert r1["at"] == DOC_TIME and r1["decided_by"] == "rule" and r1["drafter_verdict"] == "verifies"
    assert basis == {
        "revision_id": rev["identity"]["revision_id"],
        "claim_indices": [0],
        "evidence_ids": [tk.DOC_ID],
        "quotes": ["Revenue was NT$68.73 billion for the quarter"],
        "tiers": ["company release"],
    }
    assert jev["label"] == "supports" and jev["model"] == runner.JEV_MODEL
    r5 = next(e for e in events[ids["R5"]] if e["rule"] == "R5")
    assert r5["at"] == "2026-10-15T00:00:00Z"  # max(evidence available_at, the record's last event at)
    votes = narrative_check.JEV_VOTES
    # fact claims, then judgements (R1-R7), R3's two supported_part judgements and sweeps (every record)
    assert len(t.jev_calls) == 2 + votes * (7 + 2 + 9)
    assert run_json["costs"]["jev_narrative_usd"] == pytest.approx(votes * 18 * 0.00002)


def test_off_by_default_writes_checks_but_no_events(env):
    path, ids, _, _run_json, rev = run(env)
    assert rev["narrative_results"]["narrative_apply"] == "off"
    checks, events = ledger_rows(path, rev["identity"]["revision_id"])
    assert {checks[ids[r]]["outcome"] for r in ("R1", "R2", "R3", "R5")} == {"status_change"}
    assert checks[ids["R4"]]["outcome"] == "corroborated" and checks[ids["R9"]]["outcome"] == "missed"
    assert all(c["event_seq"] is None for c in checks.values())
    seeded_events = {ids["R7"]: 2, ids["R5"]: 2}
    assert all(len(events[nid]) == seeded_events.get(nid, 1) for nid in ids.values())
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(env["run_dir"] / "revision.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    table = md[md.index("## 敘事變化") : md.index("## 可重放身分")]
    assert "narrative_apply `off`" in table and "已證實（narrative_apply off，未寫入）" in table
    owner = table[table.index("### 待 owner 判斷") :]
    for rule in ("R6", "R7", "R9"):
        assert f"`{ids[rule]}`" in owner and f"（{rule}）" in owner
    done = json.loads(tk.bridge_calls(env, "done")[0]["stdin"])
    assert "## 敘事變化" in done["report_md"]


def test_the_same_document_frozen_again_corroborates_a_record_once(env):
    """Owner decision 2026-10-30, after ns-a9165163 and ns-dd5e60a8 in the trial ledger: one piece of evidence
    corroborates a record once. With narrative_apply on, the same document under a new revision id (new evidence id,
    re-cut excerpts) writes no second corroborated event; its check row is written and names the first event."""
    path, ids, _t, _run_json, rev = run(env, "on")
    (first,) = [e for e in ledger_rows(path, rev["identity"]["revision_id"])[1][ids["R4"]] if e["rule"] == "R4"]
    assert first["kind"] == "corroborated"
    later = refrozen(rev)
    res = apply_later(env, later, "on")[ids["R4"]]
    checks, events = ledger_rows(path, later["identity"]["revision_id"])
    assert [e["kind"] for e in events[ids["R4"]]] == ["created", "corroborated"]
    assert (res["rule"], res["outcome"], res["event_seq"]) == ("R4", "corroborated", None)
    assert res["duplicate_of"] == f"{ids['R4']}#{first['seq']}"
    assert (checks[ids["R4"]]["outcome"], checks[ids["R4"]]["event_seq"]) == ("corroborated", None)


def test_rerun_of_the_same_revision_calls_no_jev_and_writes_nothing(env):
    path, _ids, _, _, rev = run(env, "on")
    payload = rev["manifest_payload"]
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    before = ledger_rows(path, rev["identity"]["revision_id"])
    t = FakeJev([])
    jev = runner.Jev(JEV_KEY, transport=t)
    again = narrative_check.apply(path, rev, payload, cfg, jev, mode="on")
    assert t.jev_calls == [] and ledger_rows(path, rev["identity"]["revision_id"]) == before
    assert [x["outcome"] for x in again["records"]] == [x["outcome"] for x in rev["narrative_results"]["records"]]
    assert all(x["judged"] is None for x in again["records"])
    # a crash after the events but before the check rows: the rerun judges again and reuses the events
    db = sqlite3.connect(path)
    db.execute("DELETE FROM narrative_checks")
    db.commit()
    db.close()
    redo = narrative_check.apply(path, rev, payload, cfg, jev, mode="on")
    checks, events = ledger_rows(path, rev["identity"]["revision_id"])
    assert len(t.jev_calls) == narrative_check.JEV_VOTES * 18 and events == before[1]
    assert {nid: c["event_seq"] for nid, c in checks.items()} == {nid: c["event_seq"] for nid, c in before[0].items()}
    assert [x["outcome"] for x in redo["records"]] == [x["outcome"] for x in again["records"]]


def test_truncated_sweep_is_recorded(env, monkeypatch):
    monkeypatch.setattr(runner.Jev, "SOURCE_CAP", 80)
    path, _, t, _, rev = run(env)
    checks, _ = ledger_rows(path, rev["identity"]["revision_id"])
    assert {c["sweep_truncated"] for c in checks.values()} == {1}
    sweeps = [c for c in t.jev_calls if c["state"]["source_context"]["locator"] == "all official excerpts"]
    assert {len(c["state"]["source_context"]["text"]) for c in sweeps} == {80}


def test_sweep_source_keeps_official_excerpts_in_order():
    def doc(i, tier=None, n=10):
        p = {"excerpts": [{"text": i * n}], **({"source_tier": tier} if tier else {})}
        return {
            "id": i,
            "kind": "secondary" if tier == "kol" else "fundamental",
            "source_uri": "https://x",
            "payload": p,
        }

    text, cut = narrative_check.sweep_source({"evidence": [doc("a"), doc("k", "kol"), doc("b")]}, 100)
    assert text == "[a]\naaaaaaaaaa\n\n[b]\nbbbbbbbbbb" and cut == 0
    text, cut = narrative_check.sweep_source({"evidence": [doc("a", n=200)]}, 100)
    assert len(text) == 100 and cut == 1


def test_run_without_a_snapshot_has_no_narrative_step(env):
    t = tk.FakeOpenRouter([tk.good_narrative()])
    assert env["run"](t) == 0
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert "narrative_results" not in rev and len(t.jev_calls) == 2


# ------------------------------------------------------------------------------------------------ claim_theme_links
def tagged_narrative():
    n = tk.good_narrative()
    n["claims"][0].update(theme_ids=["foundry-utilization"], about_symbols=["NYSE:UMC", "2303"])
    n["claims"][2].update(theme_ids=["foundry-utilization"], about_symbols=["2303"])
    n["narrative_checks"] = []
    return n


def links(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM claim_theme_links ORDER BY claim_index, theme_id")]
    db.close()
    return rows


def run_tagged(env):
    """A theme member with no open record: the snapshot carries the theme list only, the drafter tags two claims."""
    path, db = tk.ledger(env["tmp"])
    db.close()
    t = FakeJev([tagged_narrative()])
    assert env["run"](t) == 0
    run_json = json.loads((env["run_dir"] / "run.json").read_text())
    assert run_json["status"] == "completed", run_json["reasons"]
    return path, t, run_json, json.loads((env["run_dir"] / "revision.json").read_text())


def expected_links(rev):
    by_id = {e["id"]: e for e in rev["manifest_payload"]["evidence"]}
    rev_id = rev["identity"]["revision_id"]
    return [
        (rev_id, 0, "foundry-utilization", '["2303", "NYSE:UMC"]', "fact", "company release", DOC_TIME, "drafter"),
        (
            rev_id,
            2,
            "foundry-utilization",
            '["2303"]',
            "inference",
            "exchange feed（經 Thalassa）",
            nl.iso(by_id["ohlcv:2303"]["event_time"]),
            "drafter",
        ),
    ]


def link_keys(rows):
    cols = ("revision_id", "claim_index", "theme_id", "about_symbols_json", "claim_kind", "evidence_tier", "event_time")
    return [(*(r[c] for c in cols), r["source"]) for r in rows]


def test_tagged_claims_are_linked_once_with_narrative_apply_off(env):
    path, t, run_json, rev = run_tagged(env)
    assert rev["narrative_results"] == {"narrative_apply": "off", "jev_model": runner.JEV_MODEL, "records": []}
    assert "narrative_error" not in run_json and len(t.jev_calls) == 2  # the fact claims only: no record to judge
    rows = links(path)
    assert link_keys(rows) == expected_links(rev)
    assert all(
        r["content_sha256"] == nl.content_sha256({k: v for k, v in r.items() if k != "content_sha256"}) for r in rows
    )


def test_every_invocation_refreshes_the_narrative_export(env):
    """Design §3.4: the runner leaves narrative_export.json next to the ledger for nexus_pull.py, with each linked
    claim's text and the revision's scope (research_revisions as research_theme.py stores it)."""
    path, _, _, rev = run_tagged(env)
    doc = json.loads((path.parent / runner.NARRATIVE_EXPORT).read_text())
    assert doc["algorithm"] == "narrative-export-v2" and doc["as_of"] >= max(x["created_at"] for x in links(path))
    claims, scope = rev["research"]["claims"], json.loads((env["run_dir"] / "config.json").read_text())["scope_key"]
    assert [(x["claim_index"], x["scope_key"], x["claim_text"]) for x in doc["claim_theme_links"]] == [
        (0, scope, claims[0]["text"]),
        (2, scope, claims[2]["text"]),
    ]
    assert not list(path.parent.glob(".narrative_export.json.tmp"))


def test_rerun_writes_no_link_twice(env):
    path, _, _, rev = run_tagged(env)
    before = links(path)
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    t = FakeJev([])
    narrative_check.apply(path, rev, rev["manifest_payload"], cfg, runner.Jev(JEV_KEY, transport=t), mode="on")
    assert links(path) == before and t.jev_calls == []


def test_links_lost_to_a_crash_are_written_by_the_next_apply(env, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("killed before the links")

    monkeypatch.setattr(nl, "add_link", crash)
    path, _, run_json, rev = run_tagged(env)
    assert run_json["narrative_error"].startswith("RuntimeError") and links(path) == []
    assert rev["identity"]["status"] == "completed"
    monkeypatch.undo()
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = runner.Jev(JEV_KEY, transport=FakeJev([]))
    narrative_check.apply(path, rev, rev["manifest_payload"], cfg, jev, mode="off")
    assert link_keys(links(path)) == expected_links(rev)


def test_no_snapshot_writes_no_links(env):
    path, _, _, rev = run_tagged(env)
    db = sqlite3.connect(path)
    db.execute("DELETE FROM claim_theme_links")
    db.commit()
    db.close()
    payload = dict(rev["manifest_payload"])
    payload["evidence"] = [e for e in payload["evidence"] if e["kind"] != "narrative_snapshot"]
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = runner.Jev(JEV_KEY, transport=FakeJev([]))
    assert narrative_check.apply(path, rev, payload, cfg, jev, mode="on") is None
    assert links(path) == []


# ------------------------------------------------------------------------------------------------ narrative_proposals
PROPOSALS = [
    {
        "claim_index": 0,
        "claim_type": "state",
        "resolves_on": None,
        "record_claim": "聯電第三季營收為 NT$68.73 billion",
        "theme_ids": ["foundry-utilization"],
        "subject_symbols": ["2303"],
    },
    {
        "claim_index": 2,
        "claim_type": "forecast",
        "resolves_on": None,
        "record_claim": "聯電股價反應將待事件後確認",
        "theme_ids": ["foundry-utilization"],
        "subject_symbols": ["2303", "NYSE:UMC"],
    },
]


def records(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    rows = {r["narrative_id"]: dict(r) for r in db.execute("SELECT * FROM narrative_records")}
    events = [dict(r) for r in db.execute("SELECT * FROM narrative_events ORDER BY narrative_id, seq")]
    held = [dict(r) for r in db.execute("SELECT * FROM held_records ORDER BY narrative_id")]
    db.close()
    return rows, events, held


def test_revision_narrative_proposals_become_records_in_every_mode(env):
    path, db = tk.ledger(env["tmp"])
    db.close()
    n = tagged_narrative()
    n["narrative_proposals"] = PROPOSALS
    assert env["run"](FakeJev([n])) == 0  # narrative_apply off
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert rev["identity"]["status"] == "completed"
    out = rev["narrative_results"]["narrative_proposals"]
    assert [x["status"] for x in out] == ["written", "written"], out
    rows, events, held = records(path)
    rev_id = rev["identity"]["revision_id"]
    by_id = {e["id"]: e for e in rev["manifest_payload"]["evidence"]}
    official, ours = rows[out[0]["narrative_id"]], rows[out[1]["narrative_id"]]
    assert (official["source_tier"], official["origin_kind"], official["origin_ref"]) == (
        "official",
        "revision",
        f"revision:{rev_id}#claims[0]",
    )
    assert official["attributed_to"] == (by_id[tk.DOC_ID]["payload"].get("publisher") or tk.DOC_ID)
    assert official["origin_evidence_id"] == tk.DOC_ID and official["quote"] == n["claims"][0]["quote"]
    assert (ours["source_tier"], ours["attributed_to"], ours["origin_evidence_id"]) == (
        "our-inference",
        "本系統推論",
        "price_facts:2303",
    )
    assert ours["resolves_on_default"] == 1 and json.loads(ours["subject_symbols_json"]) == ["2303", "NYSE:UMC"]
    assert official["first_seen_at"] == ours["first_seen_at"] == rev["manifest_payload"]["as_of"]
    assert [(e["seq"], e["kind"], e["to_status"]) for e in events] == [(0, "created", "unverified")] * 2
    assert held == []
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(env["run_dir"] / "revision.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    section = md[md.index("## 新增敘事") : md.index("## 可重放身分")]
    assert f"`{official['narrative_id']}`" in section and section.count("已寫入") == 2
    # a rerun of the same revision writes nothing twice
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = runner.Jev(JEV_KEY, transport=FakeJev([]))
    again = narrative_check.apply(path, rev, rev["manifest_payload"], cfg, jev, mode="on")
    assert again["narrative_proposals"] == out and records(path) == (rows, events, held)


def test_a_fact_on_non_official_evidence_is_no_record_and_a_pending_member_holds_one(env):
    """§6.1: only official evidence makes an official record. A subject that is no member of the record's theme waits in
    held_records for a member proposal into that theme and is written once the owner accepts it; a theme proposal
    that names the subject for another theme does not count (acceptance phase 2a)."""
    path, db = tk.ledger(env["tmp"])
    member = narrative_proposals.add_member_proposal(
        db,
        theme_id="foundry-utilization",
        symbol="NASDAQ:INTC",
        role="competitor",
        why="測試",
        source_ref="material:20261019-k9#p1",
        at="2026-10-19T00:00:00Z",
    )["proposal_id"]
    db.close()
    n = tagged_narrative()
    assert env["run"](FakeJev([n])) == 0
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    rev["research"]["claims"][2]["kind"] = "fact"  # a fact on price data: exchange feed / derived, not official
    rev["research"]["narrative_proposals"] = [
        {**PROPOSALS[1], "record_claim": "聯電股價反應將待事件後確認"},
        {**PROPOSALS[0], "subject_symbols": ["NASDAQ:INTC"]},
        {**PROPOSALS[0], "record_claim": "聯電第三季營收創高", "subject_symbols": ["NASDAQ:AMD"]},
    ]
    rev["research"]["theme_proposals"] = [
        {
            "suggested_id": "foundry-pricing",
            "title": "晶圓代工報價",
            "kind": "demand_driver",
            "why": "測試",
            "claim_indices": [0],
            "proposed_members": [
                {"symbol": "NASDAQ:INTC", "role": "competitor"},
                {"symbol": "NASDAQ:AMD", "role": "customer"},
            ],
        }
    ]
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    res = narrative_check.apply(
        path, rev, rev["manifest_payload"], cfg, runner.Jev(JEV_KEY, transport=FakeJev([])), mode="off"
    )
    unfit, waiting, other = res["narrative_proposals"]
    assert unfit["status"] == "unfit" and "不全是官方文件" in unfit["reason"]
    pid = res["theme_proposals"][0]["id"]
    assert waiting["status"] == "held" and waiting["waits"] == [member]
    assert other["status"] == "unfit"
    assert other["reason"] == "NASDAQ:AMD 不是 foundry-utilization 的核准成員，也沒有待決的提案"
    rows, _, held = records(path)
    assert unfit["narrative_id"] not in rows and waiting["narrative_id"] not in rows
    assert [h["narrative_id"] for h in held] == [waiting["narrative_id"]]
    decide = lambda n, action, target, **kw: narrative_check.apply_decision(  # noqa: E731
        path,
        {
            "decision_id": f"20261020-d{n}",
            "action": action,
            "target": target,
            "owner_text": "好",
            "decided_at": "2026-10-20T00:00:00Z",
            **kw,
        },
    )
    result, message = decide(1, "theme-accept", pid, theme_id="foundry-pricing")
    assert result == "applied" and waiting["narrative_id"] not in message
    result, message = decide(2, "member-accept", member)
    assert result == "applied" and f"敘事 {waiting['narrative_id']} 已寫入" in message
    rows, _, _ = records(path)
    assert rows[waiting["narrative_id"]]["source_tier"] == "official"
    assert rows[waiting["narrative_id"]]["theme_ids_json"] == '["foundry-utilization"]'


def test_keyword_hits_on_untagged_claims_are_listed_never_linked(env):
    """§5.3: a theme keyword in a claim the drafter did not tag with that theme is an untagged_hit for the owner."""
    path, db = tk.ledger(env["tmp"])
    nl.add_theme_keywords(db, "foundry-utilization", ["營收", "毛利率"], "cli:owner", at=tk.APPROVED)
    db.close()
    assert env["run"](FakeJev([tagged_narrative()])) == 0  # claims 0 and 2 tagged, claim 1 (毛利率) not
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert rev["narrative_results"]["untagged_hits"] == [
        {"claim_index": 1, "theme_id": "foundry-utilization", "keyword": "毛利率"}
    ]
    assert [r["claim_index"] for r in links(path)] == [0, 2]
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(env["run_dir"] / "revision.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (
        "## 主題召回" in md
        and "claims[1]「聯電第三季毛利率為 32.5%。」命中 `foundry-utilization` 的關鍵字「毛利率」" in md
    )
