"""Design 2026-10-29 §9 step 9: the owner approves themes and members (runner side).

Theme proposals from a revision (done notice) or a material (material-done notice) and member proposals from a material
carry their program id; the owner's theme-accept / theme-reject / member-accept / member-reject decisions arrive in
the narrative-pending feed. Intake records that need a pending proposal are held and released or dropped by the
decision. The bridge, draft model and JEV are faked; the ledger, gates and research_theme are real. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_theme_decisions.py
"""

import json
import sys
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import intake  # noqa: E402
import narrative_check  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import runner  # noqa: E402
import test_intake as ti  # noqa: E402
import test_narrative_apply as ta  # noqa: E402
import test_runner as tr  # noqa: E402
from test_runner import JEV_KEY  # noqa: E402
from test_runner import env as runner_env  # noqa: E402,F401  (pytest finds it by this name)

DECIDED_AT = "2026-10-22T09:30:00+08:00"
DECIDED_UTC = "2026-10-22T01:30:00Z"
ASML_MEMBER = {
    "theme_id": "cleanroom-capacity",
    "symbol": "NASDAQ:ASML",
    "role": "constrained",
    "why": "ASML 的出貨受客戶無塵室限制",
    "quote": "看來連 asml 都明顯受到 cleanroom 的缺乏限制",
}
MASK = {
    "suggested_id": "mask-12-inch",
    "title": "12 吋光罩轉換",
    "kind": "tech_transition",
    "why": "ASML 與 TSMC 發起 12 吋光罩聯盟",
    "quote": "與 TSMC 發起 12 吋光罩聯盟，2031 試產線、2033 量產",
    "keywords": ["12 吋光罩", "photomask"],
    "proposed_members": [{"symbol": "NASDAQ:ASML", "role": "supplier"}, {"symbol": "2330", "role": "customer"}],
}
MASK_ASML = ti.rec(
    "ASML 發起 12 吋光罩聯盟", "與 TSMC 發起 12 吋光罩聯盟", ["NASDAQ:ASML"], "mask-12-inch", "ASML（經股癌轉述）"
)
MASK_TSMC = ti.rec(
    "TSMC 參與發起 12 吋光罩聯盟", "與 TSMC 發起 12 吋光罩聯盟", ["2330"], "mask-12-inch", "ASML（經股癌轉述）"
)
FOUNDRY_PRICING = {
    "suggested_id": "foundry-pricing",
    "title": "晶圓代工漲價",
    "kind": "demand_driver",
    "why": "聯電第三季毛利率改善",
    "claim_indices": [0],
    "keywords": ["漲價", "price hike"],
    "proposed_members": [{"symbol": "2303", "role": "supplier"}],
}


def out(records=(), proposals=(), members=()):
    return {**ti.intake_out(records, proposals), "member_proposals": list(members)}


def decision(n, action, target, text, **extra):
    return {
        "decision_id": f"20261022-d{n}",
        "action": action,
        "target": target,
        **extra,
        "owner_text": text,
        "decided_at": DECIDED_AT,
        "status": "pending",
    }


def seed(path, approve_asml=False):
    """The seed's verified members only (the candidates, NASDAQ:ASML among them, are not approved)."""
    db = nl.connect(path)
    for t in json.loads((HERE / "themes.seed.json").read_text())["themes"]:
        nl.add_theme(db, t["theme_id"], t["title"], t["kind"], t["keywords"], "cli:test-seed", at=ti.APPROVED)
        for m in t["members"]["verified"]:
            nl.add_member(db, t["theme_id"], m["symbol"], m["role"], "cli:test-seed", at=ti.APPROVED)
    if approve_asml:
        nl.add_member(db, "cleanroom-capacity", "NASDAQ:ASML", "constrained", "cli:test-owner", at=ti.APPROVED)
    db.close()


@pytest.fixture()
def env(runner_env):  # noqa: F811
    runner_env["state"].mkdir(parents=True)
    runner_env["set_bridge"](rows=[])
    runner_env["run_dir"] = runner_env["state"] / "runs" / f"material-{ti.MID}"
    return runner_env


def ledger(env):
    return env["state"] / "ledger.sqlite"


def rows(env, sql, *args):
    return ti.db_rows(env, sql, *args)


def decide(env, *decisions):
    """One runner invocation with these narrative-pending rows and no material; -> {decision id: (result, message)}."""
    env["set_bridge"](**{"narrative-pending": list(decisions), "material-pending": []})
    assert env["run"](tr.FakeOpenRouter([])) == 0
    acks = {}
    for c in tr.bridge_calls(env, "narrative-ack"):
        acks[c["args"][1]] = (c["args"][2], c["stdin"].strip())
    return acks


def take_in(env, out, post="2"):
    _t, rc = ti.run_intake(env, [out], [ti.material_row(post)])
    assert rc == 0 and ti.run_json(env)["status"] == "intaken", ti.run_json(env)["reasons"]
    (done,) = ti.material_done(env)
    return done


# ------------------------------------------------------------------------------------------------ member proposals
def test_member_accept_releases_held_records(env):
    """§7.2 N5-N7 need NASDAQ:ASML in cleanroom-capacity: held until the owner accepts the member proposal. N5 names
    2027 where the post writes 27/28, so the period precheck drops it before (as in step 8)."""
    seed(ledger(env))
    done = take_in(env, out(ti.EXPECTED["2"], members=[ASML_MEMBER]))
    assert rows(env, "SELECT * FROM narrative_records") == []
    held = rows(env, "SELECT * FROM held_records ORDER BY narrative_id")
    assert len(held) == 2 and {(h["status"], h["waits_json"]) for h in held} == {("held", '["M20261021-1"]')}
    (mp,) = rows(env, "SELECT * FROM member_proposals")
    assert (mp["proposal_id"], mp["theme_id"], mp["symbol"], mp["role"], mp["status"], mp["source_ref"]) == (
        "M20261021-1",
        "cleanroom-capacity",
        "NASDAQ:ASML",
        "constrained",
        "proposed",
        f"material:{ti.MID}#p2",
    )
    assert (done["records"], done["held"], done["dropped"], done["member_proposals"]) == (0, 2, 1, ["M20261021-1"])
    assert "等待核准 2 條（主題或成員核准後寫入，略過就不收）：" in done["text"]
    assert "（等 M20261021-1）" in done["text"]
    assert (
        "- 成員提案 M20261021-1：NASDAQ:ASML 加入 cleanroom-capacity（constrained），ASML 的出貨受客戶無塵室限制；"
        "回覆「接受成員 M20261021-1」或「略過成員 M20261021-1」" in done["text"]
    )

    acks = decide(env, decision(1, "member-accept", "M20261021-1", "接受成員 M20261021-1"))
    result, message = acks["20261022-d1"]
    assert result == "applied"
    assert message.startswith("成員提案 M20261021-1 已接受：NASDAQ:ASML 加入 cleanroom-capacity（constrained）")
    written = rows(env, "SELECT * FROM narrative_records ORDER BY narrative_id")
    assert [r["narrative_id"] for r in written] == [h["narrative_id"] for h in held]
    assert all(f"敘事 {h['narrative_id']} 已寫入" in message for h in held)
    assert {(r["source_tier"], r["first_seen_at"], r["origin_ref"]) for r in written} == {
        ("kol", ti.RECEIVED_UTC, f"material:{ti.MID}")
    }
    assert {r["theme_ids_json"] for r in written} == {'["cleanroom-capacity"]'}
    events = rows(env, "SELECT * FROM narrative_events")
    assert len(events) == 2 and {(e["seq"], e["kind"], e["at"]) for e in events} == {(0, "created", ti.RECEIVED_UTC)}
    (member,) = rows(env, "SELECT * FROM theme_members WHERE symbol = 'NASDAQ:ASML'")
    assert (member["theme_id"], member["role"], member["approval_ref"], member["approved_at"]) == (
        "cleanroom-capacity",
        "constrained",
        "telegram:M20261021-1",
        DECIDED_UTC,
    )
    (mp,) = rows(env, "SELECT * FROM member_proposals")
    assert (mp["status"], mp["decided_ref"]) == ("accepted", "telegram:20261022-d1")
    assert {h["status"] for h in rows(env, "SELECT status FROM held_records")} == {"written"}


def test_member_reject_drops_held_records_with_the_reason(env):
    seed(ledger(env))
    take_in(env, out(ti.EXPECTED["2"], members=[ASML_MEMBER]))
    acks = decide(env, decision(1, "member-reject", "M20261021-1", "略過成員 M20261021-1"))
    result, message = acks["20261022-d1"]
    assert result == "applied"
    assert message.startswith("成員提案 M20261021-1 已略過：NASDAQ:ASML 不加入 cleanroom-capacity，之後不再提出")
    assert rows(env, "SELECT * FROM narrative_records") == []
    held = rows(env, "SELECT * FROM held_records")
    reason = "NASDAQ:ASML 不是核准成員（成員提案 M20261021-1 已被略過）"
    assert {(h["status"], h["reason"], h["decided_at"]) for h in held} == {("dropped", reason, DECIDED_UTC)}
    assert all(f"敘事 {h['narrative_id']} 不收：{reason}" in message for h in held)
    # accepting it later is refused; the member is never proposed again
    acks = decide(env, decision(2, "member-accept", "M20261021-1", "接受成員 M20261021-1"))
    assert acks["20261022-d2"][0] == "refused" and "was rejected (telegram:20261022-d1)" in acks["20261022-d2"][1]
    db = nl.connect(ledger(env))
    assert nl.member_duplicate(db, "cleanroom-capacity", "NASDAQ:ASML") == "rejected:M20261021-1"
    db.close()
    m = intake.parse_row(ti.material_row("2"))
    _dup, member_dup = intake._duplicate_fn(ledger(env))
    reasons, plan = intake.gates(
        out([ti.EXPECTED["2"][0]], members=[ASML_MEMBER]),
        m,
        ti.snapshot_payload(ledger(env)),
        lambda s, t: (None, None),
        member_dup,
    )
    assert plan["member_proposals"][0]["duplicate_of"] == "rejected:M20261021-1"
    assert len(reasons) == 1 and reasons[0].startswith("[subject] records[0]") and "NASDAQ:ASML" in reasons[0]


# ------------------------------------------------------------------------------------------------ material themes
def test_theme_accept_releases_records_held_on_it(env):
    """A record tagged with a theme proposal waits for it; accepting it under the owner's id writes the record with that
    id. A subject that only the proposal names then waits for its member proposal, filed by the accept."""
    seed(ledger(env), approve_asml=True)
    done = take_in(env, out([MASK_ASML, MASK_TSMC], [MASK]))
    assert done["theme_proposals"] == ["T20261021-1"] and done["held"] == 2
    assert (
        "- T20261021-1 12 吋光罩轉換（tech_transition）：ASML 與 TSMC 發起 12 吋光罩聯盟；建議 id mask-12-inch；"
        "提案成員 NASDAQ:ASML(supplier)、2330(customer)；回覆「接受主題 T20261021-1 <id>」或「略過主題 T20261021-1」"
    ) in done["text"]
    held = {json.loads(h["record_json"])["claim"]: h for h in rows(env, "SELECT * FROM held_records")}
    assert json.loads(held[MASK_ASML["claim"]]["waits_json"]) == ["T20261021-1"]
    assert json.loads(held[MASK_ASML["claim"]]["record_json"])["theme_ids"] == ["T20261021-1"]

    acks = decide(
        env,
        decision(1, "theme-accept", "T20261021-1", "接受主題 T20261021-1 photomask-12in", theme_id="photomask-12in"),
    )
    result, message = acks["20261022-d1"]
    assert result == "applied" and message.startswith("主題提案 T20261021-1 已接受為 photomask-12in")
    (theme,) = rows(env, "SELECT * FROM themes WHERE proposal_id = 'T20261021-1'")
    assert (theme["theme_id"], theme["title"], theme["kind"], theme["approval_ref"], theme["approved_at"]) == (
        "photomask-12in",
        "12 吋光罩轉換",
        "tech_transition",
        "telegram:T20261021-1",
        DECIDED_UTC,
    )
    assert json.loads(theme["keywords_json"]) == MASK["keywords"]  # the proposal's suggested keywords
    asml_nid = held[MASK_ASML["claim"]]["narrative_id"]
    tsmc_nid = held[MASK_TSMC["claim"]]["narrative_id"]
    (written,) = rows(env, "SELECT * FROM narrative_records")
    assert (written["narrative_id"], written["theme_ids_json"], written["first_seen_at"]) == (
        asml_nid,
        '["photomask-12in"]',
        ti.RECEIVED_UTC,
    )
    # the proposed members are not approved with the theme: each is a member proposal of its own
    mps = {m["symbol"]: m for m in rows(env, "SELECT * FROM member_proposals")}
    assert {s: (m["theme_id"], m["role"], m["status"]) for s, m in mps.items()} == {
        "NASDAQ:ASML": ("photomask-12in", "supplier", "proposed"),
        "2330": ("photomask-12in", "customer", "proposed"),
    }
    assert rows(env, "SELECT * FROM theme_members WHERE theme_id = 'photomask-12in'") == []
    m2330 = mps["2330"]["proposal_id"]
    assert f"回覆「接受成員 {m2330}」或「略過成員 {m2330}」" in message
    assert f"敘事 {asml_nid} 已寫入" in message and f"敘事 {tsmc_nid} 仍等待 {m2330}" in message
    acks = decide(env, decision(2, "member-accept", m2330, f"接受成員 {m2330}"))
    assert acks["20261022-d2"][0] == "applied" and f"敘事 {tsmc_nid} 已寫入" in acks["20261022-d2"][1]
    tsmc = rows(env, "SELECT * FROM narrative_records WHERE narrative_id = ?", tsmc_nid)[0]
    assert (tsmc["subject_symbols_json"], tsmc["theme_ids_json"], tsmc["source_tier"]) == (
        '["2330"]',
        '["photomask-12in"]',
        "kol",
    )


def test_theme_reject_drops_records_held_on_it(env):
    seed(ledger(env), approve_asml=True)
    take_in(env, out([MASK_ASML], [MASK]))
    acks = decide(env, decision(1, "theme-reject", "T20261021-1", "略過主題 T20261021-1"))
    result, message = acks["20261022-d1"]
    assert result == "applied" and message.startswith("主題提案 T20261021-1 已略過，之後不再提出")
    (h,) = rows(env, "SELECT * FROM held_records")
    assert (h["status"], h["reason"]) == ("dropped", "主題提案 T20261021-1 已被略過")
    assert (
        rows(env, "SELECT * FROM narrative_records") == []
        and rows(env, "SELECT * FROM themes WHERE proposal_id IS NOT NULL") == []
    )


# ------------------------------------------------------------------------------------------------ revision themes
def revision_with_proposal(env, mode=None):
    path, ids = ta.seed(env)
    n = ta.narrative(ids)
    n["theme_proposals"] = [FOUNDRY_PRICING]
    t = ta.FakeJev([n])
    assert env["run"](t, *(["--narrative-apply", mode] if mode else [])) == 0
    run_json = json.loads((env["run_dir"] / "run.json").read_text())
    assert run_json["status"] == "completed", run_json["reasons"]
    return path, json.loads((env["run_dir"] / "revision.json").read_text())


@pytest.fixture()
def rev_env(runner_env):  # noqa: F811
    return runner_env


def test_revision_proposal_is_in_the_done_notice_and_accepting_links_its_claims(rev_env):
    env = rev_env
    _path, rev = revision_with_proposal(env)
    rev_id = rev["identity"]["revision_id"]
    (p,) = ti.db_rows(env, "SELECT * FROM theme_proposals")
    assert (p["proposal_id"], p["suggested_id"], p["status"], p["source_ref"]) == (
        "T20261021-1",
        "foundry-pricing",
        "proposed",
        f"revision:{rev_id}#claims[0]",
    )
    (pc,) = ti.db_rows(env, "SELECT * FROM theme_proposal_claims")
    assert (pc["revision_id"], pc["claim_index"], pc["claim_kind"]) == (rev_id, 0, rev["research"]["claims"][0]["kind"])
    done = json.loads(tr.bridge_calls(env, "done")[0]["stdin"])
    line = (
        "主題提案 T20261021-1：晶圓代工漲價（demand_driver），依據 claims[0]；建議 id foundry-pricing；"
        "提案成員 2303(supplier)；回覆「接受主題 T20261021-1 <id>」或「略過主題 T20261021-1」"
    )
    assert done["theme_proposals"] == [{"id": "T20261021-1", "line": line}]
    assert rev["narrative_results"]["theme_proposals"][0]["status"] == "filed"

    first = decision(
        1, "theme-accept", "T20261021-1", "接受主題 T20261021-1 wafer-price-hikes", theme_id="wafer-price-hikes"
    )
    acks = decide(env, first)
    assert acks["20261022-d1"][0] == "applied"
    assert acks["20261022-d1"][1].startswith("主題提案 T20261021-1 已接受為 wafer-price-hikes，補 1 條 claim 主題連結")
    (theme,) = ti.db_rows(env, "SELECT * FROM themes WHERE theme_id = 'wafer-price-hikes'")
    assert (theme["proposal_id"], theme["approval_ref"]) == ("T20261021-1", "telegram:T20261021-1")
    assert json.loads(theme["keywords_json"]) == FOUNDRY_PRICING["keywords"]
    (link,) = ti.db_rows(env, "SELECT * FROM claim_theme_links WHERE theme_id = 'wafer-price-hikes'")
    assert (link["revision_id"], link["claim_index"], link["source"], link["evidence_tier"], link["event_time"]) == (
        rev_id,
        0,
        "owner",
        pc["evidence_tier"],
        pc["event_time"],
    )
    assert json.loads(link["about_symbols_json"]) == ["2303"]
    (p,) = ti.db_rows(env, "SELECT * FROM theme_proposals")
    assert (p["status"], p["decided_ref"], p["decided_at"]) == ("accepted", "telegram:20261022-d1", DECIDED_UTC)

    # a repeated accept is a no-op: the same decision delivered again, or a new one with the same id
    snapshot = [
        ti.db_rows(env, f"SELECT * FROM {t} ORDER BY 1, 2")
        for t in ("themes", "claim_theme_links", "theme_proposals", "member_proposals")
    ]
    acks = decide(
        env,
        first,
        decision(
            2, "theme-accept", "T20261021-1", "接受主題 T20261021-1 wafer-price-hikes", theme_id="wafer-price-hikes"
        ),
    )
    assert acks["20261022-d1"][1].endswith("（先前已套用，未重複寫入）") and acks["20261022-d2"][0] == "applied"
    assert acks["20261022-d2"][1].endswith("（先前已套用，未重複寫入）")
    assert [
        ti.db_rows(env, f"SELECT * FROM {t} ORDER BY 1, 2")
        for t in ("themes", "claim_theme_links", "theme_proposals", "member_proposals")
    ] == snapshot
    acks = decide(env, decision(3, "theme-accept", "T20261021-1", "接受主題 T20261021-1 other-id", theme_id="other-id"))
    assert acks["20261022-d3"][0] == "refused" and "already accepted as wafer-price-hikes" in acks["20261022-d3"][1]


def test_owner_id_outside_the_grammar_is_refused_and_acked(rev_env):
    env = rev_env
    revision_with_proposal(env)
    acks = decide(
        env, decision(1, "theme-accept", "T20261021-1", "接受主題 T20261021-1 Wafer_Pricing", theme_id="Wafer_Pricing")
    )
    result, message = acks["20261022-d1"]
    assert result == "refused" and "Wafer_Pricing" in message and "kebab-case" in message
    assert ti.db_rows(env, "SELECT * FROM themes WHERE proposal_id IS NOT NULL") == []
    assert ti.db_rows(env, "SELECT status FROM theme_proposals") == [{"status": "proposed"}]
    # the owner corrects the id in a new decision
    acks = decide(
        env, decision(2, "theme-accept", "T20261021-1", "接受主題 T20261021-1 wafer-pricing", theme_id="wafer-pricing")
    )
    assert acks["20261022-d2"][0] == "applied"
    assert ti.db_rows(env, "SELECT theme_id FROM themes WHERE proposal_id = 'T20261021-1'") == [
        {"theme_id": "wafer-pricing"}
    ]


def test_rejected_theme_is_not_proposed_again(rev_env):
    env = rev_env
    path, rev = revision_with_proposal(env)
    acks = decide(env, decision(1, "theme-reject", "T20261021-1", "略過主題 T20261021-1"))
    assert acks["20261022-d1"] == ("applied", "主題提案 T20261021-1 已略過，之後不再提出")
    assert ti.db_rows(env, "SELECT status, decided_ref FROM theme_proposals") == [
        {"status": "rejected", "decided_ref": "telegram:20261022-d1"}
    ]
    acks = decide(env, decision(2, "theme-accept", "T20261021-1", "接受主題 T20261021-1 x", theme_id="x"))
    assert acks["20261022-d2"][0] == "refused" and "was rejected (telegram:20261022-d1)" in acks["20261022-d2"][1]
    # a later revision proposing it again (same id or the same title) files nothing and lists no proposal
    later = json.loads(json.dumps(rev))
    later["identity"]["revision_id"] = str(uuid.uuid4())
    later["research"]["theme_proposals"] = [FOUNDRY_PRICING, dict(FOUNDRY_PRICING, suggested_id="wafer-pricing")]
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = runner.Jev(JEV_KEY, transport=ta.FakeJev([]))
    out = narrative_check.apply(path, later, later["manifest_payload"], cfg, jev, mode="off")
    assert [(p["id"], p["status"], p["duplicate_of"]) for p in out["theme_proposals"]] == [
        (None, "duplicate", "rejected:T20261021-1"),
        (None, "duplicate", "rejected:T20261021-1"),
    ]
    assert runner.theme_proposals_of(out) == []
    assert len(ti.db_rows(env, "SELECT * FROM theme_proposals")) == 1
    # a material proposing it again: listed as a duplicate, not filed, and its suggested id admits no record
    db = nl.connect(path)
    assert nl.theme_duplicate(db, "foundry-pricing", "別的標題") == "rejected:T20261021-1"
    assert nl.theme_duplicate(db, "another-id", "晶圓代工 漲價") == "rejected:T20261021-1"
    db.close()


@pytest.mark.parametrize("mode", ["off", "on", "confirm"])
def test_proposals_are_filed_in_every_mode(rev_env, mode):
    revision_with_proposal(rev_env, mode)
    assert [r["proposal_id"] for r in ti.db_rows(rev_env, "SELECT proposal_id FROM theme_proposals")] == ["T20261021-1"]


# ------------------------------------------------------------------------------------------------ faithful check
def test_faithful_check_reads_the_heading_line(env):
    """Post 1's lines do not name 日東紡; the JEV judge gets the heading line with the quote's line."""
    seed(ledger(env))
    t, rc = ti.run_intake(env, [ti.intake_out(ti.EXPECTED["1"][:1])], [ti.material_row("1")])
    assert rc == 0 and ti.run_json(env)["status"] == "intaken"
    heading = ti.POSTS["1"].splitlines()[0]
    (claim_line,) = [ln for ln in ti.POSTS["1"].splitlines() if "份額未來可能略降" in ln]
    for call in t.jev_calls:
        ctx = call["state"]["source_context"]
        assert ctx["text"] == f"{heading}\n{claim_line.strip()}" and ctx["locator"].startswith("p1, p")
    assert len(t.jev_calls) == 3


def test_decision_rows_for_proposals_are_parsed_at_the_boundary(tmp_path):
    path = tmp_path / "l.sqlite"
    nl.connect(path).close()
    for bad, needle in [
        (decision(1, "theme-accept", "T20261021-1", "x"), "theme_id"),
        (decision(1, "theme-accept", "M20261021-1", "x", theme_id="a"), "theme proposal id"),
        (decision(1, "member-reject", "T20261021-1", "x"), "member proposal id"),
        (decision(1, "theme-reject", "T20261021-1", " "), "owner_text"),
        (decision(1, "member-accept", "M20261021-9", "x"), "no proposal M20261021-9"),
    ]:
        result, message = narrative_check.apply_decision(path, bad)
        assert result == "refused" and needle in message, (needle, message)


def test_a_long_proposal_line_keeps_the_reply_within_the_bridge_cap():
    t = dict(
        FOUNDRY_PRICING,
        title="很長的標題" * 40,
        suggested_id="x" * 200,
        proposed_members=[{"symbol": f"NASDAQ:ABC{n}", "role": "supplier"} for n in range(20)],
    )
    line = narrative_check.proposal_line("T20261021-12", t, list(range(30)))
    assert len(line) <= 400 and line.endswith("回覆「接受主題 T20261021-12 <id>」或「略過主題 T20261021-12」")
