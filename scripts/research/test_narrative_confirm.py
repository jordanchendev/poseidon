"""narrative_apply confirm (owner decision 2026-10-29): for the first 4 weeks the rule table's changes are suggestions
the owner confirms or rejects in Telegram (the control set still has two contradicts cases JEV calls supports at
>= 0.95). Same nine seeded records as test_narrative_apply.py; the bridge, draft model and JEV are faked.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_confirm.py
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import narrative_check  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import runner  # noqa: E402
import test_narrative_apply as ta  # noqa: E402
import test_runner as tr  # noqa: E402
from test_runner import JEV_KEY  # noqa: E402
from test_runner import env as runner_env  # noqa: E402,F401  (pytest finds it by this name)

SUGGESTED = ("R1", "R2", "R3", "R4", "R5")  # status_change and corroborated outcomes
DECIDED_AT = "2026-10-22T09:30:00+08:00"


@pytest.fixture()
def env(runner_env):  # noqa: F811
    return runner_env


def suggestions(path):
    db = nl.connect(path)
    try:
        return {r["narrative_id"]: dict(r) for r in db.execute("SELECT * FROM narrative_suggestions")}
    finally:
        db.close()


def decision(n, action, target, text, to_status=None):
    return {
        "decision_id": f"20261022-d{n}",
        "action": action,
        "target": target,
        **({"to_status": to_status} if to_status else {}),
        "owner_text": text,
        "decided_at": DECIDED_AT,
        "status": "pending",
    }


def test_confirm_writes_suggestions_not_events(env):
    path, ids, _t, run_json, rev = ta.run(env, "confirm")
    assert rev["narrative_results"]["narrative_apply"] == "confirm" and run_json["narrative_apply"] == "confirm"
    checks, events = ta.ledger_rows(path, rev["identity"]["revision_id"])
    assert all(c["event_seq"] is None for c in checks.values())
    assert all(len(events[nid]) == {ids["R7"]: 2, ids["R5"]: 2}.get(nid, 1) for nid in ids.values())
    sugg = suggestions(path)
    assert set(sugg) == {ids[r] for r in SUGGESTED}
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}
    for rule in SUGGESTED:
        s, x = sugg[ids[rule]], res[ids[rule]]
        assert (s["rule"], s["status"], s["from_status"]) == (rule, "suggested", "unverified"), rule
        assert (
            x["suggestion"]["id"] == s["suggestion_id"] == nl.suggestion_id(rev["identity"]["revision_id"], ids[rule])
        )
        assert f"回覆「確認敘事 {s['suggestion_id']}」或「駁回敘事 {s['suggestion_id']}」" in x["suggestion"]["line"]
    assert (sugg[ids["R3"]]["kind"], sugg[ids["R3"]]["to_status"]) == ("status", "partially_verified")
    assert (sugg[ids["R4"]]["kind"], sugg[ids["R4"]]["to_status"]) == ("corroborated", "unverified")
    assert "由 未證實 改為 部分證實（R3）" in res[ids["R3"]]["suggestion"]["line"]
    assert "記一筆官方背書（R4，狀態維持 未證實）" in res[ids["R4"]]["suggestion"]["line"]
    assert all("suggestion" not in res[ids[r]] for r in ("R6", "R7", "R8", "R9"))
    # the done notice carries one line per suggestion; the report lists them under 待 owner 確認
    done = json.loads(tr.bridge_calls(env, "done")[0]["stdin"])
    assert sorted((s["id"], s["narrative_id"]) for s in done["suggestions"]) == sorted(
        (res[ids[r]]["suggestion"]["id"], ids[r]) for r in SUGGESTED
    )
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(env["run_dir"] / "revision.json")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    sid = res[ids["R1"]]["suggestion"]["id"]
    assert f"已證實（建議 {sid}，待 owner 確認）" in md
    assert "### 待 owner 確認（narrative_apply confirm）" in md and res[ids["R1"]]["suggestion"]["line"] in md


def test_a_partial_suggestion_shows_supported_part_next_to_open_part(env):
    """Owner decision 2026-10-30: the owner confirms an R3 change against what the cited claims establish and what they
    leave open, so the notice line carries both; a suggestion from another verdict carries neither."""
    _path, ids, _t, _run_json, rev = ta.run(env, "confirm")
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}
    line = res[ids["R3"]]["suggestion"]["line"]
    assert f"；撐住的部分「{ta.PART}」；未撐住的部分「未說明主動性。」；回覆「確認敘事 " in line
    assert "撐住的部分" not in res[ids["R1"]]["suggestion"]["line"]
    done = json.loads(tr.bridge_calls(env, "done")[0]["stdin"])
    assert line in [s["line"] for s in done["suggestions"]]


def test_on_and_off_results_carry_no_suggestion(env):
    _path, _ids, _t, _run_json, rev = ta.run(env, "on")
    assert all("suggestion" not in x for x in rev["narrative_results"]["records"])


def test_owner_decisions_are_applied_once_and_acked(env):
    path, ids, _t, _run_json, rev = ta.run(env, "confirm")
    sid = {r: nl.suggestion_id(rev["identity"]["revision_id"], ids[r]) for r in SUGGESTED}
    pending = [
        decision(1, "confirm", sid["R3"], "確認敘事 " + sid["R3"]),
        decision(2, "reject", sid["R1"], "駁回敘事 " + sid["R1"]),
        decision(3, "confirm", sid["R4"], "確認敘事 " + sid["R4"]),
        decision(4, "set", ids["R8"], "設定敘事 refuted：官方說法相反", "refuted"),
        decision(5, "confirm", "ns-00000000", "確認敘事 ns-00000000"),
        decision(6, "confirm", sid["R1"], "確認敘事 " + sid["R1"]),
    ]
    env["set_bridge"](**{"narrative-pending": pending})
    assert env["run"](tr.FakeOpenRouter([])) == 0
    acks = {c["args"][1]: (c["args"][2], c["stdin"].strip()) for c in tr.bridge_calls(env, "narrative-ack")}
    assert acks["20261022-d1"] == ("applied", f"敘事建議 {sid['R3']} 已確認：{ids['R3']} 改為 部分證實")
    assert acks["20261022-d2"][0] == "applied" and "同一個變化之後不再建議" in acks["20261022-d2"][1]
    assert acks["20261022-d3"] == (
        "applied",
        f"敘事建議 {sid['R4']} 已確認：{ids['R4']} 記一筆官方背書（狀態維持 未證實）",
    )
    assert acks["20261022-d4"] == ("applied", f"敘事 {ids['R8']} 已由 owner 設為 已否證")
    assert acks["20261022-d5"][0] == "refused" and "no suggestion ns-00000000" in acks["20261022-d5"][1]
    assert acks["20261022-d6"][0] == "refused" and "was rejected (telegram:20261022-d2)" in acks["20261022-d6"][1]
    _checks, events = ta.ledger_rows(path, rev["identity"]["revision_id"])
    r3 = events[ids["R3"]][-1]
    basis = json.loads(r3["basis_json"])
    assert (r3["kind"], r3["decided_by"], r3["rule"], r3["to_status"]) == (
        "status",
        "owner",
        "R3",
        "partially_verified",
    )
    assert basis["approval_ref"] == "telegram:20261022-d1" and basis["suggestion_id"] == sid["R3"]
    assert basis["revision_id"] == rev["identity"]["revision_id"] and r3["note"] == "確認敘事 " + sid["R3"]
    assert r3["at"] == "2026-10-22T01:30:00Z"  # the decision time: later than the evidence and the last event
    r4 = events[ids["R4"]][-1]
    assert (r4["kind"], r4["decided_by"], r4["to_status"]) == ("corroborated", "owner", "unverified")
    r8 = events[ids["R8"]][-1]
    assert (r8["kind"], r8["to_status"], json.loads(r8["basis_json"])) == (
        "override",
        "refuted",
        {"override": True, "approval_ref": "telegram:20261022-d4"},
    )
    assert len(events[ids["R1"]]) == 1  # rejected: no event
    sugg = suggestions(path)
    assert (sugg[ids["R3"]]["status"], sugg[ids["R3"]]["event_seq"]) == ("confirmed", r3["seq"])
    assert sugg[ids["R1"]]["status"] == "rejected" and sugg[ids["R2"]]["status"] == "suggested"
    # the same decisions delivered again (the acks were lost): nothing is written twice
    before = ta.ledger_rows(path, rev["identity"]["revision_id"])
    assert env["run"](tr.FakeOpenRouter([])) == 0
    assert ta.ledger_rows(path, rev["identity"]["revision_id"]) == before
    again = [c for c in tr.bridge_calls(env, "narrative-ack") if c["args"][1] == "20261022-d1"]
    assert again[-1]["stdin"].strip().endswith("（先前已套用，未重複寫入）")


def test_a_rejected_change_is_not_suggested_again(env):
    path, ids, _t, _run_json, rev = ta.run(env, "confirm")
    db = nl.connect(path)
    rejected = nl.suggestion_id(rev["identity"]["revision_id"], ids["R1"])
    nl.reject_suggestion(db, rejected, decision_ref="telegram:20261022-d1", decided_at=DECIDED_AT)
    db.close()
    later = json.loads(json.dumps(rev))
    later["identity"]["revision_id"] = str(uuid.uuid4())
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = runner.Jev(JEV_KEY, transport=ta.FakeJev([]))
    out = narrative_check.apply(path, later, later["manifest_payload"], cfg, jev, mode="confirm")
    res = {x["narrative_id"]: x for x in out["records"]}
    assert res[ids["R1"]]["suggestion"] == {"id": None, "status": "suppressed", "suppressed_by": rejected}
    assert res[ids["R2"]]["suggestion"]["status"] == "suggested"
    assert len([s for s in suggestions_all(path) if s["narrative_id"] == ids["R1"]]) == 1
    # rerunning the same revision calls no JEV and reports the stored suggestions
    t = ta.FakeJev([])
    redo = narrative_check.apply(
        path, later, later["manifest_payload"], cfg, runner.Jev(JEV_KEY, transport=t), mode="confirm"
    )
    assert t.jev_calls == []
    assert {x["narrative_id"]: (x.get("suggestion") or {}).get("id") for x in redo["records"]} == {
        x["narrative_id"]: (x.get("suggestion") or {}).get("id") for x in out["records"]
    }


def suggestions_all(path):
    db = nl.connect(path)
    try:
        return [dict(r) for r in db.execute("SELECT * FROM narrative_suggestions")]
    finally:
        db.close()


def test_decision_rows_are_parsed_at_the_boundary(tmp_path):
    path = tmp_path / "l.sqlite"
    nl.connect(path).close()
    for bad, needle in [
        ({**decision(1, "confirm", "nr-0123456789", "x")}, "suggestion id"),
        ({**decision(1, "set", "nr-0123456789", "x", "rumor")}, "status"),
        ({**decision(1, "delete", "ns-01234567", "x")}, "unknown action"),
        ({**decision(1, "confirm", "ns-01234567", "  ")}, "owner_text"),
        ({**decision(1, "confirm", "ns-01234567", "x"), "decision_id": "x; rm"}, "malformed decision id"),
    ]:
        result, message = narrative_check.apply_decision(path, bad)
        assert result == "refused" and needle in message, (needle, message)
