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

import narrative_check  # noqa: E402
import narrative_decisions  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import narrative_suggestions  # noqa: E402
import runner  # noqa: E402
import testkit as tk  # noqa: E402
from testkit import JEV_KEY  # noqa: E402

SUGGESTED = ("R1", "R2", "R3", "R4", "R5")  # status_change and corroborated outcomes
DECIDED_AT = "2026-10-22T09:30:00+08:00"


@pytest.fixture()
def env(runner_env):
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
    path, ids, _t, run_json, rev = tk.run(env, "confirm")
    assert rev["narrative_results"]["narrative_apply"] == "confirm" and run_json["narrative_apply"] == "confirm"
    checks, events = tk.ledger_rows(path, rev["identity"]["revision_id"])
    assert all(c["event_seq"] is None for c in checks.values())
    assert all(len(events[nid]) == {ids["R7"]: 2, ids["R5"]: 2}.get(nid, 1) for nid in ids.values())
    sugg = suggestions(path)
    assert set(sugg) == {ids[r] for r in SUGGESTED}
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}
    for rule in SUGGESTED:
        s, x = sugg[ids[rule]], res[ids[rule]]
        assert (s["rule"], s["status"], s["from_status"]) == (rule, "suggested", "unverified"), rule
        assert (
            x["suggestion"]["id"]
            == s["suggestion_id"]
            == narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids[rule])
        )
        assert f"回覆「確認敘事 {s['suggestion_id']}」或「駁回敘事 {s['suggestion_id']}」" in x["suggestion"]["line"]
    assert (sugg[ids["R3"]]["kind"], sugg[ids["R3"]]["to_status"]) == ("status", "partially_verified")
    assert (sugg[ids["R4"]]["kind"], sugg[ids["R4"]]["to_status"]) == ("corroborated", "unverified")
    assert "由 未證實 改為 部分證實（R3）" in res[ids["R3"]]["suggestion"]["line"]
    assert "記一筆官方背書（R4，狀態維持 未證實）" in res[ids["R4"]]["suggestion"]["line"]
    assert all("suggestion" not in res[ids[r]] for r in ("R6", "R7", "R8", "R9"))
    # the done notice carries one line per suggestion; the report lists them under 待 owner 確認
    done = json.loads(tk.bridge_calls(env, "done")[0]["stdin"])
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
    _path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    res = {x["narrative_id"]: x for x in rev["narrative_results"]["records"]}
    line = res[ids["R3"]]["suggestion"]["line"]
    assert f"；撐住的部分「{tk.PART}」；未撐住的部分「未說明主動性。」；回覆「確認敘事 " in line
    assert "撐住的部分" not in res[ids["R1"]]["suggestion"]["line"]
    done = json.loads(tk.bridge_calls(env, "done")[0]["stdin"])
    assert line in [s["line"] for s in done["suggestions"]]


def test_on_and_off_results_carry_no_suggestion(env):
    _path, _ids, _t, _run_json, rev = tk.run(env, "on")
    assert all("suggestion" not in x for x in rev["narrative_results"]["records"])


def test_owner_decisions_are_applied_once_and_acked(env):
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    sid = {r: narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids[r]) for r in SUGGESTED}
    pending = [
        decision(1, "confirm", sid["R3"], "確認敘事 " + sid["R3"]),
        decision(2, "reject", sid["R1"], "駁回敘事 " + sid["R1"]),
        decision(3, "confirm", sid["R4"], "確認敘事 " + sid["R4"]),
        decision(4, "set", ids["R8"], "設定敘事 refuted：官方說法相反", "refuted"),
        decision(5, "confirm", "ns-00000000", "確認敘事 ns-00000000"),
        decision(6, "confirm", sid["R1"], "確認敘事 " + sid["R1"]),
    ]
    env["set_bridge"](**{"narrative-pending": pending})
    assert env["run"](tk.FakeOpenRouter([])) == 0
    acks = {c["args"][1]: (c["args"][2], c["stdin"].strip()) for c in tk.bridge_calls(env, "narrative-ack")}
    assert acks["20261022-d1"] == ("applied", f"敘事建議 {sid['R3']} 已確認：{ids['R3']} 改為 部分證實")
    assert acks["20261022-d2"][0] == "applied" and "同一個變化之後不再建議" in acks["20261022-d2"][1]
    assert acks["20261022-d3"] == (
        "applied",
        f"敘事建議 {sid['R4']} 已確認：{ids['R4']} 記一筆官方背書（狀態維持 未證實）",
    )
    assert acks["20261022-d4"] == ("applied", f"敘事 {ids['R8']} 已由 owner 設為 已否證")
    assert acks["20261022-d5"][0] == "refused" and "no suggestion ns-00000000" in acks["20261022-d5"][1]
    assert acks["20261022-d6"][0] == "refused" and "was rejected (telegram:20261022-d2)" in acks["20261022-d6"][1]
    _checks, events = tk.ledger_rows(path, rev["identity"]["revision_id"])
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
    before = tk.ledger_rows(path, rev["identity"]["revision_id"])
    assert env["run"](tk.FakeOpenRouter([])) == 0
    assert tk.ledger_rows(path, rev["identity"]["revision_id"]) == before
    again = [c for c in tk.bridge_calls(env, "narrative-ack") if c["args"][1] == "20261022-d1"]
    assert again[-1]["stdin"].strip().endswith("（先前已套用，未重複寫入）")


def confirm_events(path, rev_id, nid, sid):
    _checks, events = tk.ledger_rows(path, rev_id)
    return [e for e in events[nid] if json.loads(e["basis_json"]).get("suggestion_id") == sid]


@pytest.mark.parametrize("rule", ["R3", "R4"])  # a status change and a corroboration
def test_a_confirm_killed_before_its_row_is_written_writes_nothing_and_applies_once_when_resent(env, monkeypatch, rule):
    """P0-3: the event and the confirmed row are one transaction, so a crash between them leaves neither; the bridge's
    re-send of the un-acked decision applies it once, and a re-send after that is a no-op."""
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    rev_id = rev["identity"]["revision_id"]
    nid, sid = ids[rule], narrative_suggestions.suggestion_id(rev_id, ids[rule])
    real, crashed = nl._update, []

    def killed_once(*a, **k):
        if not crashed:
            crashed.append(a)
            raise RuntimeError("killed")
        return real(*a, **k)

    monkeypatch.setattr(nl, "_update", killed_once)
    d = decision(1, "confirm", sid, "確認敘事 " + sid)
    with pytest.raises(RuntimeError, match="killed"):
        narrative_decisions.apply_decision(path, d)
    assert confirm_events(path, rev_id, nid, sid) == [] and suggestions(path)[nid]["status"] == "suggested"
    result, message = narrative_decisions.apply_decision(path, d)
    assert result == "applied" and message.startswith(f"敘事建議 {sid} 已確認") and "先前已套用" not in message
    [ev] = confirm_events(path, rev_id, nid, sid)
    s = suggestions(path)[nid]
    assert (s["status"], s["event_seq"], s["decision_ref"]) == ("confirmed", ev["seq"], "telegram:20261022-d1")
    before = tk.ledger_rows(path, rev_id), suggestions(path)
    result, message = narrative_decisions.apply_decision(path, d)
    assert result == "applied" and message.endswith("（先前已套用，未重複寫入）")
    assert (tk.ledger_rows(path, rev_id), suggestions(path)) == before


@pytest.mark.parametrize("rule", ["R3", "R4"])
def test_a_ledger_left_with_the_event_and_a_suggested_row_is_finished_without_a_second_event(env, rule):
    """P0-3: a ledger that crashed under the old two-commit order holds the owner event (basis.suggestion_id) while the
    suggestion is still suggested. The re-sent decision finishes the row against that event: a status suggestion is
    not refused as a move from its own status, a corroboration is not written twice."""
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    rev_id = rev["identity"]["revision_id"]
    nid, sid = ids[rule], narrative_suggestions.suggestion_id(rev_id, ids[rule])
    db = nl.connect(path)
    s = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    nl.append_event(  # what the old confirm_suggestion had committed when it crashed
        db,
        nid,
        kind=s["kind"],
        to_status=s["to_status"],
        decided_by="owner",
        at=DECIDED_AT,
        basis={**json.loads(s["basis_json"]), "suggestion_id": sid, "approval_ref": "telegram:20261022-d1"},
        rule=s["rule"],
        jev=json.loads(s["jev_json"]),
        drafter_verdict=s["drafter_verdict"],
        note="確認敘事 " + sid,
    )
    db.close()
    [ev] = confirm_events(path, rev_id, nid, sid)
    result, message = narrative_decisions.apply_decision(path, decision(1, "confirm", sid, "確認敘事 " + sid))
    assert result == "applied" and message.endswith("（先前已套用，未重複寫入）"), message
    assert confirm_events(path, rev_id, nid, sid) == [ev]
    s = suggestions(path)[nid]
    assert (s["status"], s["event_seq"], s["decision_ref"]) == ("confirmed", ev["seq"], "telegram:20261022-d1")


def test_a_rejected_change_is_not_suggested_again(env):
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    db = nl.connect(path)
    rejected = narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids["R1"])
    narrative_suggestions.reject_suggestion(db, rejected, decision_ref="telegram:20261022-d1", decided_at=DECIDED_AT)
    db.close()
    later = json.loads(json.dumps(rev))
    later["identity"]["revision_id"] = str(uuid.uuid4())
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = runner.Jev(JEV_KEY, transport=tk.FakeJev([]))
    out = narrative_check.apply(path, later, later["manifest_payload"], cfg, jev, mode="confirm")
    res = {x["narrative_id"]: x for x in out["records"]}
    assert res[ids["R1"]]["suggestion"] == {"id": None, "status": "suppressed", "suppressed_by": rejected}
    assert res[ids["R2"]]["suggestion"]["status"] == "suggested"
    assert len([s for s in suggestions_all(path) if s["narrative_id"] == ids["R1"]]) == 1
    # rerunning the same revision calls no JEV and reports the stored suggestions
    t = tk.FakeJev([])
    redo = narrative_check.apply(
        path, later, later["manifest_payload"], cfg, runner.Jev(JEV_KEY, transport=t), mode="confirm"
    )
    assert t.jev_calls == []
    assert {x["narrative_id"]: (x.get("suggestion") or {}).get("id") for x in redo["records"]} == {
        x["narrative_id"]: (x.get("suggestion") or {}).get("id") for x in out["records"]
    }


def corroborations_of(path, nid):
    return [s for s in suggestions_all(path) if s["narrative_id"] == nid and s["kind"] == "corroborated"]


def test_the_same_document_frozen_again_is_not_suggested_again(env):
    """Owner decision 2026-10-30: trial run 20261030-m99 re-froze the Micron remarks of 20261029-m94 and suggested R4
    again (ns-a9165163, ns-dd5e60a8) for records the owner had confirmed. The same document under a new revision id
    and evidence id is suppressed while the first suggestion is pending and after it is confirmed; the check row is
    written and the report says which suggestion it repeats."""
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    first = narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids["R4"])
    later = tk.refrozen(rev)
    res = tk.apply_later(env, later, "confirm")
    assert res[ids["R4"]]["suggestion"] == {"id": None, "status": "suppressed", "duplicate_of": first}
    assert res[ids["R2"]]["suggestion"]["status"] == "suggested"  # a status change is not an endorsement
    checks, _ = tk.ledger_rows(path, later["identity"]["revision_id"])
    assert (checks[ids["R4"]]["outcome"], checks[ids["R4"]]["event_seq"]) == ("corroborated", None)
    db = nl.connect(path)
    narrative_suggestions.confirm_suggestion(
        db, first, decision_ref="telegram:20261022-d1", decided_at=DECIDED_AT, note="確認敘事"
    )
    db.close()
    again = tk.apply_later(env, tk.refrozen(rev, n=3), "confirm")
    assert again[ids["R4"]]["suggestion"] == {"id": None, "status": "suppressed", "duplicate_of": first}
    assert [s["suggestion_id"] for s in corroborations_of(path, ids["R4"])] == [first]
    # the done report says why no suggestion is listed
    rev_path = env["run_dir"] / "revision.json"
    results = {"narrative_apply": "confirm", "records": [res[ids["R4"]]]}
    rev_path.write_text(json.dumps({**later, "narrative_results": results}))
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(rev_path)], capture_output=True, text=True, check=True
    ).stdout
    assert f"官方背書（同一份證據已背書過 {first}，不再寫入）" in md


def test_a_different_document_corroborates_again(env):
    """Corroboration by other evidence still counts: a different document (other raw files) backing the same record is
    a new suggestion, and that document frozen again repeats it, not the first."""
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    first = narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids["R4"])
    other = tk.refrozen(rev, raw_files={"other.htm": "0" * 64})
    res = tk.apply_later(env, other, "confirm")
    second = narrative_suggestions.suggestion_id(other["identity"]["revision_id"], ids["R4"])
    assert (res[ids["R4"]]["suggestion"]["id"], res[ids["R4"]]["suggestion"]["status"]) == (second, "suggested")
    assert [s["suggestion_id"] for s in corroborations_of(path, ids["R4"])] == [first, second]
    again = tk.refrozen(rev, n=3, raw_files={"other.htm": "0" * 64})
    assert tk.apply_later(env, again, "confirm")[ids["R4"]]["suggestion"] == {
        "id": None,
        "status": "suppressed",
        "duplicate_of": second,
    }


def test_a_corroboration_stored_without_evidence_sha256_is_compared_by_its_quotes(env):
    """A suggestion written before basis carried evidence_sha256 (the trial ledger's ns-48c45f58, ns-3aa4b89b) is the
    same evidence when every quote it stored is in the newly cited document; a document without them is new."""
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    first = narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids["R4"])
    db = nl.connect(path)
    s = nl._one(db, "SELECT basis_json FROM narrative_suggestions WHERE suggestion_id = ?", first)
    basis = json.loads(s["basis_json"])
    del basis["evidence_sha256"]
    nl._update(db, "narrative_suggestions", {"suggestion_id": first}, {"basis_json": json.dumps(basis)})
    db.close()
    same = tk.apply_later(env, tk.refrozen(rev), "confirm")
    assert same[ids["R4"]]["suggestion"] == {"id": None, "status": "suppressed", "duplicate_of": first}
    excerpts = [{"locator": "line 1", "text": "Capacity stays tight next year."}]
    other = tk.refrozen(rev, n=3, raw_files={"other.htm": "0" * 64}, excerpts=excerpts)
    assert tk.apply_later(env, other, "confirm")[ids["R4"]]["suggestion"]["status"] == "suggested"


def test_a_rejected_endorsement_is_still_not_suggested_again(env):
    """The rejection rule is unchanged: after the owner rejects an R4 suggestion the same document frozen again is
    suppressed by that rejection (not reported as a repeat), and nothing is written."""
    path, ids, _t, _run_json, rev = tk.run(env, "confirm")
    rejected = narrative_suggestions.suggestion_id(rev["identity"]["revision_id"], ids["R4"])
    db = nl.connect(path)
    narrative_suggestions.reject_suggestion(db, rejected, decision_ref="telegram:20261022-d1", decided_at=DECIDED_AT)
    db.close()
    res = tk.apply_later(env, tk.refrozen(rev), "confirm")
    assert res[ids["R4"]]["suggestion"] == {"id": None, "status": "suppressed", "suppressed_by": rejected}
    assert [s["status"] for s in corroborations_of(path, ids["R4"])] == ["rejected"]


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
        result, message = narrative_decisions.apply_decision(path, bad)
        assert result == "refused" and needle in message, (needle, message)
