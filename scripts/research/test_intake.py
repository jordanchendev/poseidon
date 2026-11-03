"""Design 2026-10-29 §9 step 8: material intake (runner side).

The eight 股癌 posts of §7 (fixtures/gooaye_posts.json, test fixture only) go through runner.py with a fake bridge
(material-pending / material-done / material-needs-review), a fake draft model and a fake JEV. The ledger, the freeze
(ManifestService) and every gate are real. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_intake.py
"""

import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

import intake  # noqa: E402
import testkit as tk  # noqa: E402
from testkit import (  # noqa: E402
    EXPECTED_RECORDS,
    MID,
    POSTS,
    RECEIVED_UTC,
    VERIFY_8,
    VIA,
    db_rows,
    intake_out,
    material_done,
    material_row,
    rec,
    run_intake,
    run_json,
    seed_intake_themes,
    snapshot_payload,
)


@pytest.fixture()
def env(runner_env):
    runner_env["state"].mkdir(parents=True)
    seed_intake_themes(runner_env["state"] / "ledger.sqlite")
    runner_env["set_bridge"](rows=[])
    runner_env["run_dir"] = runner_env["state"] / "runs" / f"material-{MID}"
    return runner_env


@pytest.mark.parametrize("post", sorted(EXPECTED_RECORDS))
def test_section_7_2_records_pass_every_gate(tmp_path, post):
    path = tmp_path / "ledger.sqlite"
    seed_intake_themes(path)
    m = intake.parse_row(material_row(post))
    reasons, plan = intake.gates(
        intake_out(EXPECTED_RECORDS[post]), m, snapshot_payload(path), lambda s, t: (None, None)
    )
    assert reasons == [] and len(plan["records"]) == len(EXPECTED_RECORDS[post])
    for r in plan["records"]:
        assert r["quote"] in r["paragraph"] and r["locator"].startswith("p")


def test_section_7_2_lint_refusals(tmp_path):
    """Post 7's AAOI comparison (two quantities) and post 8's agentic line copied as one record (a 帶動 link)."""
    path = tmp_path / "ledger.sqlite"
    seed_intake_themes(path)
    aaoi = "aaoi 年化約 1200 萬顆，對比 Innolight 4430 萬＋Eoptolink 5670 萬顆"
    agentic = "agentic workload 跑在 CPU 上，帶動 LPDDR、DDR、SSD"
    cases = [
        (
            "7",
            aaoi.replace("aaoi", "AAOI"),
            aaoi,
            ["NASDAQ:AAOI"],
            "optics-export-policy",
            "clauses carry their own quantity",
        ),
        ("8", agentic, agentic, ["NASDAQ:MU"], "memory-supply-tightness", "links a cause to an effect (帶動)"),
    ]
    for post, claim, quote, subjects, theme, why in cases:
        m = intake.parse_row(material_row(post))
        out = intake_out([rec(claim, quote, subjects, theme, "Morgan Stanley" + VIA)])
        reasons, plan = intake.gates(out, m, snapshot_payload(path), lambda s, t: (None, None))
        assert len(reasons) == 1 and reasons[0].startswith("[atomic] records[0]") and why in reasons[0], reasons
        assert plan["records"] == []


def test_subjects_must_be_approved_or_proposed_members(tmp_path):
    path = tmp_path / "ledger.sqlite"
    seed_intake_themes(path)
    m = intake.parse_row(material_row("2"))
    payload = snapshot_payload(path)
    intel = rec(
        "Intel 累計 High-NA 曝光超過 1m 片",
        "Intel 累計 High-NA 曝光超過 1m 片",
        ["NASDAQ:INTC"],
        "cleanroom-capacity",
        "ASML" + VIA,
    )
    proposal = {
        "suggested_id": "high-na-euv",
        "title": "High-NA EUV 導入",
        "kind": "tech_transition",
        "why": "Intel 已大量曝光，Samsung 與 TSMC 排定量產",
        "quote": "Samsung 2028、TSMC 2030 進量產",
        "proposed_members": [{"symbol": "NASDAQ:INTC", "role": "customer"}],
    }
    reasons, _ = intake.gates(intake_out([intel]), m, payload, lambda s, t: (None, None))
    assert len(reasons) == 1 and reasons[0].startswith("[subject] records[0]") and "NASDAQ:INTC" in reasons[0]
    # a theme proposal admits its members into a record tagged with it, never into a record of another theme
    reasons, _ = intake.gates(intake_out([intel], [proposal]), m, payload, lambda s, t: (None, None))
    assert len(reasons) == 1 and reasons[0].startswith("[subject] records[0]") and "NASDAQ:INTC" in reasons[0]
    on_it = dict(intel, theme_ids=["high-na-euv"])
    reasons, plan = intake.gates(intake_out([on_it], [proposal]), m, payload, lambda s, t: (None, None))
    assert reasons == [] and plan["proposals"][0]["duplicate_of"] is None
    # a proposal that repeats an existing theme files nothing, so its members do not admit a subject
    reasons, plan = intake.gates(
        intake_out([on_it], [proposal]), m, payload, lambda s, t: ("theme:cleanroom-capacity", None)
    )
    assert len(reasons) == 1 and reasons[0].startswith("[subject]")
    # one that repeats an open proposal admits that proposal's members
    open_members = [{"symbol": "NASDAQ:INTC", "role": "customer"}]
    reasons, _ = intake.gates(
        intake_out([on_it], [proposal]), m, payload, lambda s, t: ("proposal:T20261019-1", open_members)
    )
    assert reasons == []


def test_gate_refusals_name_the_record(tmp_path):
    path = tmp_path / "ledger.sqlite"
    seed_intake_themes(path)
    m = intake.parse_row(material_row("8"))
    good = EXPECTED_RECORDS["8"][1]
    bad = [
        (dict(good, quote="市場吃緊會一直支撐到 2029"), "[quote] records[0]"),
        (dict(EXPECTED_RECORDS["8"][3], claim="L4 以上自駕車需要 300GB 以上的記憶體"), "[numbers] records[0]"),
        (dict(good, resolves_on="2027-12-31"), "names no period"),
        (dict(good, claim_type="state"), "resolves_on is a YYYY-MM-DD date of a forecast only"),
        (dict(good, theme_ids=["memory-cycle"]), "[theme] records[0]"),
        (dict(good, subject_symbols=["MU"]), "[subject] records[0]"),
    ]
    for r, needle in bad:
        reasons, _ = intake.gates(intake_out([r]), m, snapshot_payload(path), lambda s, t: (None, None))
        assert any(needle in x for x in reasons), (needle, reasons)
    reasons, _ = intake.gates(intake_out([good] * 11), m, snapshot_payload(path), lambda s, t: (None, None))
    assert "[schema] at most 10 records per material (D11); keep the most material ones" in reasons


# ------------------------------------------------------------------------------------------------ runner
def test_prompt_takes_the_claims_the_opening_names_first():
    """Acceptance 2026-10-30 §2: post 8 opens with the rumors it says were confirmed (N15, N17), yet the ten records
    went to the list below it. Rule 2, right after the cap, says what to take first; the rules stay numbered 1-11."""
    rules = intake.INTAKE_PROMPT[intake.INTAKE_PROMPT.index("Rules (") :]
    cap, first, quote = (rules.index(x) for x in ("\n1. records: at most", "\n2. When the material", "\n3. quote:"))
    assert cap < first < quote
    assert "first take the claims its title or opening lines name" in rules and "傳聞獲得驗證" in rules
    assert [int(n) for n in re.findall(r"\n(\d+)\. ", rules)] == list(range(1, 12))


def test_post8_records_are_written_with_the_material_tier_and_first_seen_at(env):
    t, rc = run_intake(env, [intake_out(EXPECTED_RECORDS["8"], verify=VERIFY_8)], [material_row("8")])
    assert rc == 0
    data = run_json(env)
    assert data["status"] == "intaken", data["reasons"]
    rows = db_rows(env, "SELECT * FROM narrative_records ORDER BY narrative_id")
    assert len(rows) == 4
    assert {r["source_tier"] for r in rows} == {"kol"} and {r["first_seen_at"] for r in rows} == {RECEIVED_UTC}
    assert {r["origin_ref"] for r in rows} == {f"material:{MID}"}
    assert {r["origin_evidence_id"] for r in rows} == {f"secondary:material-{MID}"}
    by_claim = {r["claim"]: r for r in rows}
    n16 = by_claim["記憶體供給吃緊會延續到至少 2028 年"]
    assert (n16["claim_type"], n16["resolves_on"], n16["review_by"]) == (
        "forecast",
        "2028-12-31",
        "2029-03-31T23:59:59Z",
    )
    events = db_rows(env, "SELECT * FROM narrative_events")
    assert len(events) == 4 and all(
        (e["seq"], e["kind"], e["to_status"], e["decided_by"], e["at"])
        == (0, "created", "unverified", "rule", RECEIVED_UTC)
        for e in events
    )
    mat = db_rows(env, "SELECT * FROM materials")[0]
    assert (mat["material_id"], mat["status"], mat["source_tier"], mat["author"]) == (MID, "intaken", "kol", "股癌")
    manifest = db_rows(env, "SELECT payload_json FROM data_manifests WHERE content_sha256 = ?", mat["manifest_sha256"])
    payload = json.loads(manifest[0]["payload_json"])
    assert [e["kind"] for e in payload["evidence"]] == ["secondary", "narrative_snapshot"]
    assert payload["evidence"][0]["payload"]["source_tier"] == "kol"
    assert len(payload["evidence"][1]["payload"]["themes"]) == 6 and payload["evidence"][1]["payload"]["records"] == []
    assert db_rows(env, "SELECT id FROM research_revisions") == []  # never a revision
    # prompt: the closed theme list and the material lines; JEV judged each record against its own line, 3 votes
    system, user = (m["content"] for m in t.chat_calls[0]["messages"])
    assert "先參考就好" in system and "- memory-supply-tightness | 記憶體供給吃緊 | bottleneck |" in user
    assert "[p1] mu 26q4" in user
    assert len(t.jev_calls) == 4 * 3
    judged = {j["state"]["claim"]: j["state"]["source_context"] for j in t.jev_calls}
    assert judged["記憶體供給吃緊會延續到至少 2028 年"]["locator"].startswith("p")
    (done,) = material_done(env)
    assert (done["records"], done["dropped"], done["theme_proposals"], done["verify_with"]) == (4, 0, [], VERIFY_8)
    assert done["text"].startswith(f"素材整理完成 {MID}（KOL 股癌）\n新增敘事 4 條：")
    assert "可查證的官方文件：\n- Micron FQ4 FY26 prepared remarks" in done["text"]
    assert data["costs"]["draft_usd"] == 0.0021 and data["costs"]["jev_usd"] == pytest.approx(12 * 0.0001)


@pytest.mark.parametrize(("tier", "word"), [("sell-side", "賣方"), ("media", "媒體")])
def test_record_tier_is_the_material_tier(env, tier, word):
    row = material_row("2", tier=tier, author="產業タイムズ")
    _, rc = run_intake(env, [intake_out(EXPECTED_RECORDS["2"])], [row])
    assert rc == 0
    assert {r["source_tier"] for r in db_rows(env, "SELECT source_tier FROM narrative_records")} == {tier}
    assert material_done(env)[0]["text"].startswith(f"素材整理完成 {MID}（{word} 產業タイムズ）")


def test_post3_yields_no_record_and_says_no_theme(env):
    t, rc = run_intake(env, [intake_out()], [material_row("3")])
    assert rc == 0 and run_json(env)["status"] == "intaken"
    assert db_rows(env, "SELECT * FROM narrative_records") == [] and db_rows(env, "SELECT * FROM theme_proposals") == []
    (done,) = material_done(env)
    assert (done["records"], done["theme_proposals"]) == (0, [])
    assert done["text"] == f"素材整理完成 {MID}（KOL 股癌）\n無對應主題：沒有可追蹤的敘事，也沒有主題提案。"
    assert t.jev_calls == []


def test_quote_not_verbatim_is_refused_then_repaired(env):
    bad = dict(EXPECTED_RECORDS["2"][2], quote="取消成本高於客戶預付款")
    t, rc = run_intake(env, [intake_out([bad]), intake_out([EXPECTED_RECORDS["2"][2]])], [material_row("2")])
    assert rc == 0
    data = run_json(env)
    assert data["status"] == "intaken" and len(t.chat_calls) == 2
    assert data["gates"][0]["reasons"] == [
        "[quote] records[0] quote is not a verbatim substring of the material; copy it exactly"
    ]
    assert "[quote] records[0]" in t.chat_calls[1]["messages"][-1]["content"]
    assert [r["quote"] for r in db_rows(env, "SELECT quote FROM narrative_records")] == ["取消成本高於預付款"]


def test_refused_twice_goes_to_needs_review(env):
    bad = dict(EXPECTED_RECORDS["2"][2], quote="取消成本高於客戶預付款")
    t, rc = run_intake(env, [intake_out([bad]), intake_out([bad])], [material_row("2")])
    assert rc == 0 and len(t.chat_calls) == 2 and t.jev_calls == []
    data = run_json(env)
    assert (data["status"], data["failed_stage"]) == ("needs_review", "gates")
    (call,) = tk.bridge_calls(env, "material-needs-review")
    assert call["args"][1:] == [MID] and call["stdin"].startswith(
        "gates: validator refused after one repair round; [quote]"
    )
    assert db_rows(env, "SELECT status FROM materials") == [{"status": "needs_review"}]
    assert db_rows(env, "SELECT * FROM narrative_records") == [] and material_done(env) == []


def test_record_jev_does_not_support_is_dropped_and_listed(env):
    drop = EXPECTED_RECORDS["8"][0]["claim"]
    _t, rc = run_intake(
        env,
        [intake_out(EXPECTED_RECORDS["8"])],
        [material_row("8")],
        lambda c: "says_nothing" if c == drop else "supports",
    )
    assert rc == 0
    claims = {r["claim"] for r in db_rows(env, "SELECT claim FROM narrative_records")}
    assert len(claims) == 3 and drop not in claims
    (done,) = material_done(env)
    assert (done["records"], done["dropped"]) == (3, 1)
    assert f"未收 1 條（JEV 判定原文段落不支持這個轉述）：\n- {drop}：says_nothing" in done["text"]
    assert run_json(env)["dropped"][0]["judged"]["votes"] == ["says_nothing"] * 3


def test_theme_proposals_are_filed_and_duplicates_listed(env):
    new = {
        "suggested_id": "mask-12-inch",
        "title": "12 吋光罩轉換",
        "kind": "tech_transition",
        "why": "ASML 與 TSMC 發起 12 吋光罩聯盟",
        "quote": "與 TSMC 發起 12 吋光罩聯盟，2031 試產線、2033 量產",
        "proposed_members": [{"symbol": "NASDAQ:ASML", "role": "supplier"}, {"symbol": "2330", "role": "customer"}],
    }
    dup = {
        **new,
        "suggested_id": "cleanroom-bottleneck",
        "title": "設備商的無塵室瓶頸",
        "kind": "bottleneck",
        "quote": "這個環節是許多設備廠商開始刷成績的瓶頸",
    }
    _, rc = run_intake(env, [intake_out(EXPECTED_RECORDS["2"], [new, dup])], [material_row("2")])
    assert rc == 0
    (p,) = db_rows(env, "SELECT * FROM theme_proposals")
    assert (p["proposal_id"], p["suggested_id"], p["status"], p["source_ref"]) == (
        "T20261021-1",
        "mask-12-inch",
        "proposed",
        f"material:{MID}#p12",
    )
    assert json.loads(p["proposed_members_json"])[1] == {"role": "customer", "symbol": "2330"}
    (done,) = material_done(env)
    assert done["theme_proposals"] == ["T20261021-1"]
    assert "- T20261021-1 12 吋光罩轉換（tech_transition）" in done["text"]
    assert "與既有主題或提案重複、未另提：設備商的無塵室瓶頸（theme:cleanroom-capacity）" in done["text"]


def test_bridge_row_problems_go_to_needs_review(env):
    tampered = dict(material_row("8"), text=POSTS["8"] + "（多一句）")
    unreadable = {k: v for k, v in material_row("8", mid="20261020-k2").items() if k != "text"}
    unreadable["text_error"] = "text does not match text_sha256"
    t, rc = run_intake(env, [], [tampered, unreadable])
    assert rc == 0 and t.chat_calls == []
    calls = {c["args"][1]: c["stdin"] for c in tk.bridge_calls(env, "material-needs-review")}
    assert calls == {MID: "intake: material text does not match text_sha256\n"}
    _, rc = run_intake(env, [], [unreadable])
    calls = {c["args"][1]: c["stdin"] for c in tk.bridge_calls(env, "material-needs-review")}
    assert calls["20261020-k2"].startswith("bridge: the bridge could not deliver the text: text does not match")
    assert db_rows(env, "SELECT * FROM materials") == []


def test_listed_again_is_reported_again_without_a_model_call(env):
    _, rc = run_intake(env, [intake_out(EXPECTED_RECORDS["8"])], [material_row("8")])
    assert rc == 0
    first = db_rows(env, "SELECT narrative_id, content_sha256 FROM narrative_records ORDER BY narrative_id")
    t, rc = run_intake(env, [], [material_row("8")])  # the first material-done did not land
    assert rc == 0 and t.chat_calls == [] and t.jev_calls == []
    assert len(material_done(env)) == 2 and material_done(env)[0] == material_done(env)[1]
    # a crash after the checks (run.json never written): intake.json replays the same writes, no model call
    (env["run_dir"] / "run.json").unlink()
    t, rc = run_intake(env, [], [material_row("8")])
    assert rc == 0 and t.chat_calls == [] and run_json(env)["resumed_from"] == "intake.json"
    assert db_rows(env, "SELECT narrative_id, content_sha256 FROM narrative_records ORDER BY narrative_id") == first


def test_host_mode_reads_files_and_queues_reports_in_the_outbox(env):
    import deliver_outbox

    outbox = env["tmp"] / "outbox"
    agenda, materials = env["tmp"] / "agenda.json", env["tmp"] / "materials.json"
    agenda.write_text("[]")
    materials.write_text(json.dumps([material_row("3")]))
    t = tk.FakeOpenRouter([intake_out()])
    rc = env["run"](t, "--agenda-json", str(agenda), "--materials-json", str(materials), "--outbox", str(outbox))
    assert rc == 0 and env["bridge"]().get("calls") is None  # host mode: no bridge call from the container
    entry = json.loads((outbox / f"{MID}.json").read_text())
    assert entry["verb"] == "material-done" and entry["args"] == [MID] and json.loads(entry["stdin"])["records"] == 0
    assert deliver_outbox.main(["--outbox", str(outbox), "--bridge", env["bridge_cmd"]]) == 0
    assert material_done(env)[0]["records"] == 0
    # without --materials-json a host-mode run takes in nothing
    t = tk.FakeOpenRouter([])
    assert env["run"](t, "--agenda-json", str(agenda), "--outbox", str(outbox)) == 0 and t.chat_calls == []
