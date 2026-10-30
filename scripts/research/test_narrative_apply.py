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
sys.path.insert(0, str(HERE))

import narrative_check  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import runner  # noqa: E402
import test_narrative_snapshot as snap_t  # noqa: E402
import test_runner as tr  # noqa: E402
from test_runner import JEV_KEY  # noqa: E402
from test_runner import env as runner_env  # noqa: E402,F401  (pytest finds it by this name)

DOC_TIME = "2026-10-06T00:00:00Z"  # available_at of the agenda document (event_date 2026-10-06)
# rule -> (claim, claim_type, resolves_on, verdict, claim_indices, judgement label, sweep label)
CASES = {
    "R1": ("聯電甲案說法成立", "state", None, "verifies", [0], "supports", "supports"),
    "R2": ("聯電乙案說法成立", "state", None, "refutes", [0], "contradicts", "contradicts"),
    "R3": ("聯電丙案說法成立", "state", None, "partially_verifies", [1], "says_nothing", "says_nothing"),
    "R4": ("聯電丁案預期將成立", "forecast", "2027-06-30", "verifies", [0], "supports", "supports"),
    "R5": ("聯電戊案預期將成立", "forecast", "2026-09-30", "verifies", [0], "supports", "supports"),
    "R6": ("聯電己案說法成立", "state", None, "verifies", [0], "says_nothing", "says_nothing"),
    "R7": ("聯電庚案說法成立", "state", None, "refutes", [0], "contradicts", "contradicts"),
    "R8": ("聯電辛案說法成立", "state", None, "untouched", [], None, "says_nothing"),
    "R9": ("聯電壬案說法成立", "state", None, "untouched", [], None, "supports"),
}
SOURCE = "。".join(c[0] for c in CASES.values()) + "。"
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
def env(runner_env):  # noqa: F811
    return runner_env


class FakeJev(tr.FakeOpenRouter):
    """Drafts as FakeOpenRouter; JEV answers a record claim by locator (judgement vs sweep), any other claim supports."""

    def __init__(self, drafts):
        super().__init__(drafts)
        self.by_claim = {c[0]: (c[5], c[6]) for c in CASES.values()}

    def __call__(self, url, headers, body, timeout):
        if url.endswith("/chat/completions"):
            return super().__call__(url, headers, body, timeout)
        req = json.loads(body)
        self.jev_calls.append(req)
        st = req["state"]
        judged, swept = self.by_claim.get(st["claim"], ("supports", "supports"))
        label = swept if st["source_context"]["locator"] == "all official excerpts" else judged
        return 200, json.dumps(tr.jev_answer(label, 0.9, 0.00002)).encode()


def seed(env):
    path, db = snap_t.ledger(env["tmp"])
    ids = {}
    for n, (rule, (claim, ctype, resolves_on, *_)) in enumerate(CASES.items()):
        ids[rule] = nl.add_record(
            db,
            claim=claim,
            claim_type=ctype,
            resolves_on=resolves_on,
            attributed_to="股癌",
            source_tier="kol",
            origin_kind="material",
            origin_ref=f"material:20261002-k{n}",
            origin_evidence_id="secondary:material",
            quote=claim,
            source_text=SOURCE,
            subject_symbols=["2303"],
            theme_ids=["foundry-utilization"],
            first_seen_at=f"2026-10-{n + 1:02d}T00:00:00Z",
        )["narrative_id"]
    nl.append_event(
        db,
        ids["R7"],
        kind="status",
        to_status="partially_verified",
        decided_by="rule",
        at="2026-10-12T00:00:00Z",
        basis={},
    )
    nl.append_event(
        db,
        ids["R5"],
        kind="corroborated",
        to_status="unverified",
        decided_by="rule",
        at="2026-10-15T00:00:00Z",
        basis={},
    )
    db.close()
    return path, ids


def narrative(ids):
    n = tr.good_narrative()
    n["narrative_checks"] = [
        {
            "narrative_id": ids[rule],
            "verdict": c[3],
            "claim_indices": c[4],
            "why": "測試",
            **({"open_part": "未說明主動性。"} if c[3] == "partially_verifies" else {}),
        }
        for rule, c in CASES.items()
    ]
    return n


def run(env, mode=None):
    path, ids = seed(env)
    t = FakeJev([narrative(ids)])
    assert env["run"](t, *(["--narrative-apply", mode] if mode else [])) == 0
    run_json = json.loads((env["run_dir"] / "run.json").read_text())
    assert run_json["status"] == "completed", run_json["reasons"]
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    return path, ids, t, run_json, rev


def ledger_rows(path, rev_id):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    checks = {
        r["narrative_id"]: dict(r)
        for r in db.execute("SELECT * FROM narrative_checks WHERE revision_id = ?", (rev_id,))
    }
    events = {}
    for r in db.execute("SELECT * FROM narrative_events ORDER BY narrative_id, seq"):
        events.setdefault(r["narrative_id"], []).append(dict(r))
    db.close()
    return checks, events


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
        "evidence_ids": [tr.DOC_ID],
        "quotes": ["Revenue was NT$68.73 billion for the quarter"],
        "tiers": ["company release"],
    }
    assert jev["label"] == "supports" and jev["model"] == runner.JEV_MODEL
    r5 = next(e for e in events[ids["R5"]] if e["rule"] == "R5")
    assert r5["at"] == "2026-10-15T00:00:00Z"  # max(evidence available_at, the record's last event at)
    votes = narrative_check.JEV_VOTES
    assert len(t.jev_calls) == 2 + votes * (7 + 9)  # fact claims, then judgements (R1-R7) and sweeps (every record)
    assert run_json["costs"]["jev_narrative_usd"] == pytest.approx(votes * 16 * 0.00002)


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
    done = json.loads(tr.bridge_calls(env, "done")[0]["stdin"])
    assert "## 敘事變化" in done["report_md"]


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
    assert len(t.jev_calls) == narrative_check.JEV_VOTES * 16 and events == before[1]
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
    t = tr.FakeOpenRouter([tr.good_narrative()])
    assert env["run"](t) == 0
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert "narrative_results" not in rev and len(t.jev_calls) == 2


# ------------------------------------------------------------------------------------------------ claim_theme_links
def tagged_narrative():
    n = tr.good_narrative()
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
    path, db = snap_t.ledger(env["tmp"])
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
    path, db = snap_t.ledger(env["tmp"])
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
    assert official["attributed_to"] == (by_id[tr.DOC_ID]["payload"].get("publisher") or tr.DOC_ID)
    assert official["origin_evidence_id"] == tr.DOC_ID and official["quote"] == n["claims"][0]["quote"]
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
    """§6.1: only official evidence makes an official record. A subject that only an open theme proposal names waits in
    held_records and is written once the owner accepts the theme and the member."""
    path, db = snap_t.ledger(env["tmp"])
    db.close()
    n = tagged_narrative()
    assert env["run"](FakeJev([n])) == 0
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    rev["research"]["claims"][2]["kind"] = "fact"  # a fact on price data: exchange feed / derived, not official
    rev["research"]["narrative_proposals"] = [
        {**PROPOSALS[1], "record_claim": "聯電股價反應將待事件後確認"},
        {**PROPOSALS[0], "subject_symbols": ["NASDAQ:INTC"]},
    ]
    rev["research"]["theme_proposals"] = [
        {
            "suggested_id": "foundry-pricing",
            "title": "晶圓代工報價",
            "kind": "demand_driver",
            "why": "測試",
            "claim_indices": [0],
            "proposed_members": [{"symbol": "NASDAQ:INTC", "role": "competitor"}],
        }
    ]
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    res = narrative_check.apply(
        path, rev, rev["manifest_payload"], cfg, runner.Jev(JEV_KEY, transport=FakeJev([])), mode="off"
    )
    unfit, waiting = res["narrative_proposals"]
    assert unfit["status"] == "unfit" and "不全是官方文件" in unfit["reason"]
    pid = res["theme_proposals"][0]["id"]
    assert waiting["status"] == "held" and waiting["waits"] == [pid]
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
    assert decide(1, "theme-accept", pid, theme_id="foundry-pricing")[0] == "applied"
    db = sqlite3.connect(path)
    (member,) = db.execute("SELECT proposal_id FROM member_proposals WHERE symbol = 'NASDAQ:INTC'").fetchone()
    db.close()
    result, message = decide(2, "member-accept", member)
    assert result == "applied" and f"敘事 {waiting['narrative_id']} 已寫入" in message
    rows, _, _ = records(path)
    assert rows[waiting["narrative_id"]]["source_tier"] == "official"


def test_keyword_hits_on_untagged_claims_are_listed_never_linked(env):
    """§5.3: a theme keyword in a claim the drafter did not tag with that theme is an untagged_hit for the owner."""
    path, db = snap_t.ledger(env["tmp"])
    nl.add_theme_keywords(db, "foundry-utilization", ["營收", "毛利率"], "cli:owner", at=snap_t.APPROVED)
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
