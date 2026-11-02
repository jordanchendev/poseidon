"""Tests for narrative_ledger.py (design 2026-10-29 §2, §9 step 2). SQLite in tmp_path, no network.

PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_ledger.py  (Poseidon repo root)
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
import narrative_export  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import narrative_proposals  # noqa: E402

from poseidon.core.canonical import content_sha256  # noqa: E402

STATUSES = ("unverified", "partially_verified", "verified", "refuted", "stale")
# Design §2.4 state machine: (from, to) -> who may make the change. Every other (from, to, who) must be refused.
DESIGN_TABLE = {
    ("unverified", "partially_verified"): {"rule", "owner"},
    ("unverified", "verified"): {"rule", "owner"},
    ("unverified", "refuted"): {"rule", "owner"},
    ("unverified", "stale"): {"sweep", "owner"},
    ("partially_verified", "verified"): {"rule", "owner"},
    ("partially_verified", "refuted"): {"owner"},
    ("partially_verified", "unverified"): {"owner"},
    ("verified", "refuted"): {"owner"},
    ("verified", "unverified"): {"owner"},
    ("refuted", "verified"): {"owner"},
    ("refuted", "unverified"): {"owner"},
    ("stale", "partially_verified"): {"rule", "owner"},
    ("stale", "verified"): {"rule", "owner"},
    ("stale", "refuted"): {"rule", "owner"},
    ("stale", "unverified"): {"owner"},
}
SOURCE = "Micron: Memory content in Level 4 and higher autonomous vehicles typically exceeds 200GB.\n"
QUOTE = "Memory content in Level 4 and higher autonomous vehicles typically exceeds 200GB"
CLAIM = "L4 以上自駕車需要 200GB 以上的記憶體"
SEEN = "2026-10-27T03:00:00Z"


def cli(ledger, *args):
    env = dict(os.environ, PYTHONPATH=str(HERE.parents[1] / "src"))
    return subprocess.run(
        [sys.executable, str(HERE / "narrative_ledger.py"), args[0], "--ledger", str(ledger), *args[1:]],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture()
def db(tmp_path):
    d = nl.connect(tmp_path / "ledger.sqlite")
    nl.add_theme(d, "physical-ai-demand", "實體 AI 需求時程", "demand_driver", ["自駕"], "cli:test", at=SEEN)
    nl.add_member(d, "physical-ai-demand", "NASDAQ:MU", "supplier", "cli:test", at=SEEN)
    yield d
    d.close()


def record(db, n=0, claim_type="state", resolves_on=None, seen=SEEN):
    return nl.add_record(
        db,
        claim=CLAIM,
        claim_type=claim_type,
        attributed_to="Micron（經股癌轉述）",
        source_tier="kol",
        origin_kind="owner",
        origin_ref=f"owner:test-{n}",
        origin_evidence_id="material:test",
        quote=QUOTE,
        source_text=SOURCE,
        subject_symbols=["NASDAQ:MU"],
        theme_ids=["physical-ai-demand"],
        first_seen_at=seen,
        resolves_on=resolves_on,
        decided_by="owner",
    )["narrative_id"]


def status(db, nid, as_of=None):
    return nl.fold_status(nl.events_of(db, nid), as_of)


def move(db, nid, to, who, at="2026-10-28T00:00:00Z"):
    if who == "owner":
        return nl.set_status(db, nid, to, note="test override", at=at)
    return nl.append_event(db, nid, kind="status", to_status=to, decided_by=who, at=at, basis={"test": True})


# ---------------------------------------------------------------- state machine
@pytest.mark.parametrize("frm", STATUSES)
def test_every_state_machine_cell(db, frm):
    n = 0
    for to in STATUSES:
        for who in ("rule", "owner", "sweep"):
            n += 1
            nid = record(db, n=n)
            if frm != "unverified":
                move(db, nid, frm, "owner", at="2026-10-27T12:00:00Z")
            allowed = who in DESIGN_TABLE.get((frm, to), set())
            if allowed:
                move(db, nid, to, who)
                assert status(db, nid) == to, (frm, to, who)
            else:
                with pytest.raises(nl.LedgerError):
                    move(db, nid, to, who)
                assert status(db, nid) == frm, (frm, to, who)


def test_corroborated_only_for_forecast_and_keeps_status(db):
    f = record(db, n=1, claim_type="forecast", resolves_on="2028-12-31")
    nl.append_event(
        db,
        f,
        kind="corroborated",
        to_status="unverified",
        decided_by="rule",
        at="2026-10-28T00:00:00Z",
        basis={"revision_id": "r1"},
        rule="R4",
    )
    assert status(db, f) == "unverified" and [e["kind"] for e in nl.events_of(db, f)] == ["created", "corroborated"]
    s = record(db, n=2)
    with pytest.raises(nl.LedgerError):
        nl.append_event(
            db,
            s,
            kind="corroborated",
            to_status="unverified",
            decided_by="rule",
            at="2026-10-28T00:00:00Z",
            basis={},
            rule="R4",
        )


def test_fold_is_point_in_time_and_events_cannot_go_back(db):
    nid = record(db)
    move(db, nid, "verified", "rule", at="2026-11-05T00:00:00Z")
    assert status(db, nid, "2026-10-26T00:00:00Z") is None
    assert status(db, nid, "2026-11-01T00:00:00Z") == "unverified"
    assert status(db, nid, "2026-11-05T00:00:00Z") == "verified"
    with pytest.raises(nl.LedgerError):
        nl.set_status(db, nid, "refuted", note="late", at="2026-11-01T00:00:00Z")


# ---------------------------------------------------------------- approvals
def test_theme_add_without_approval_ref_refused(tmp_path):
    ledger = tmp_path / "l.sqlite"
    p = cli(ledger, "theme-add", "--id", "cleanroom-capacity", "--title", "無塵室產能", "--kind", "bottleneck")
    assert p.returncode != 0 and "--approval-ref" in p.stderr
    p = cli(
        ledger,
        "theme-add",
        "--id",
        "cleanroom-capacity",
        "--title",
        "無塵室產能",
        "--kind",
        "bottleneck",
        "--approval-ref",
        "  ",
    )
    assert p.returncode != 0 and "approval_ref is required" in p.stderr
    assert sqlite3.connect(ledger).execute("SELECT count(*) FROM themes").fetchone()[0] == 0
    p = cli(
        ledger,
        "theme-add",
        "--id",
        "cleanroom-capacity",
        "--title",
        "無塵室產能",
        "--kind",
        "bottleneck",
        "--keyword",
        "無塵室",
        "--keyword",
        "cleanroom",
        "--approval-ref",
        "cli:owner-2026-10-29",
    )
    assert p.returncode == 0, p.stderr
    row = sqlite3.connect(ledger).execute("SELECT approved_by, approval_ref, status FROM themes").fetchone()
    assert row == ("owner", "cli:owner-2026-10-29", "active")


def test_approvals_rerun_is_noop_and_conflict_refused(tmp_path):
    ledger = tmp_path / "l.sqlite"
    add = ("theme-add", "--id", "t-glass-supply", "--title", "T-glass", "--kind", "bottleneck", "--approval-ref", "x")
    assert cli(ledger, *add).returncode == 0
    assert cli(ledger, *add).returncode == 0
    p = cli(ledger, *add[:4], "Other", *add[5:])
    assert p.returncode != 0 and "different title" in p.stderr
    for sym, tw in (("1303", 1), ("TSE:3110", 0)):
        p = cli(
            ledger,
            "member-add",
            "--theme",
            "t-glass-supply",
            "--symbol",
            sym,
            "--role",
            "supplier",
            "--approval-ref",
            "x",
        )
        assert p.returncode == 0 and p.stdout.strip().endswith(f"tw_listed={tw}"), p.stdout + p.stderr
    p = cli(
        ledger,
        "member-add",
        "--theme",
        "t-glass-supply",
        "--symbol",
        "Nittobo",
        "--role",
        "supplier",
        "--approval-ref",
        "x",
    )
    assert p.returncode != 0 and "<EXCHANGE>:<TICKER>" in p.stderr
    p = cli(
        ledger,
        "member-add",
        "--theme",
        "no-such-theme",
        "--symbol",
        "1303",
        "--role",
        "supplier",
        "--approval-ref",
        "x",
    )
    assert p.returncode != 0 and "not an active theme" in p.stderr
    assert (
        cli(
            ledger,
            "edge-add",
            "--from",
            "t-glass-supply",
            "--to",
            "t-glass-supply",
            "--relation",
            "related",
            "--approval-ref",
            "x",
        ).returncode
        != 0
    )
    assert sqlite3.connect(ledger).execute("SELECT count(*) FROM theme_members").fetchone()[0] == 2


# ---------------------------------------------------------------- records
def test_record_dates_follow_d6(db):
    s = record(db, n=1)
    f = record(db, n=2, claim_type="forecast", resolves_on="2028-12-31")
    d = record(db, n=3, claim_type="forecast")
    rows = {r["narrative_id"]: r for r in map(dict, db.execute("SELECT * FROM narrative_records"))}
    assert (rows[s]["resolves_on"], rows[s]["review_by"]) == (None, "2027-04-25T23:59:59Z")
    assert (rows[f]["resolves_on"], rows[f]["resolves_on_default"], rows[f]["review_by"]) == (
        "2028-12-31",
        0,
        "2029-03-31T23:59:59Z",
    )
    assert (rows[d]["resolves_on"], rows[d]["resolves_on_default"], rows[d]["review_by"]) == (
        "2027-10-27",
        1,
        "2028-01-25T23:59:59Z",
    )
    assert (
        record(db, n=1) == s
        and db.execute("SELECT count(*) FROM narrative_events WHERE narrative_id = ?", (s,)).fetchone()[0] == 1
    )


@pytest.mark.parametrize(
    "change, why",
    [
        ({"quote": "exceeds 300GB"}, "verbatim"),
        ({"claim": "L4 以上自駕車需要 300GB 以上的記憶體"}, "number"),
        ({"claim": "L4 以上自駕車需要 200GB 以上的記憶體，因為感測器資料量增加 3 倍"}, "atomic"),
        ({"claim": "感測器資料帶動 L4 以上自駕車需要 200GB 以上的記憶體"}, "links a cause to an effect"),
        ({"claim_type": "state", "resolves_on": "2027-06-30"}, "of a forecast only"),
        ({"subject_symbols": ["NYSE:JBL"]}, "not an approved member"),
        ({"theme_ids": ["cleanroom-capacity"]}, "not an active theme"),
    ],
)
def test_record_gates(db, change, why):
    kw = dict(
        claim=CLAIM,
        claim_type="state",
        attributed_to="a",
        source_tier="kol",
        origin_kind="owner",
        origin_ref="owner:g",
        origin_evidence_id="material:test",
        quote=QUOTE,
        source_text=SOURCE,
        subject_symbols=["NASDAQ:MU"],
        theme_ids=["physical-ai-demand"],
        first_seen_at=SEEN,
    )
    with pytest.raises(nl.LedgerError, match=why):
        nl.add_record(db, **{**kw, **change})
    assert db.execute("SELECT count(*) FROM narrative_records").fetchone()[0] == 0


def test_row_hashes_cover_every_other_column(db):
    nid = record(db)
    move(db, nid, "verified", "rule")
    for table in ("themes", "theme_members", "narrative_records", "narrative_events"):
        for r in map(dict, db.execute(f"SELECT * FROM {table}")):
            stored = r.pop("content_sha256")
            assert stored == content_sha256(r), table


# ---------------------------------------------------------------- owner override, sweep, export
def test_override_needs_note(tmp_path, db):
    nid = record(db)
    ledger = tmp_path / "ledger.sqlite"
    p = cli(ledger, "set-status", "--id", nid, "--to", "refuted")
    assert p.returncode != 0 and "--note" in p.stderr
    p = cli(ledger, "set-status", "--id", nid, "--to", "refuted", "--note", " ")
    assert p.returncode != 0 and "non-empty note" in p.stderr
    assert status(db, nid) == "unverified"
    p = cli(ledger, "set-status", "--id", nid, "--to", "refuted", "--note", "ASML 新聞稿否認")
    assert p.returncode == 0, p.stderr
    ev = nl.events_of(db, nid)[-1]
    assert (ev["kind"], ev["decided_by"], ev["to_status"], ev["note"]) == (
        "override",
        "owner",
        "refuted",
        "ASML 新聞稿否認",
    )


def test_sweep_stale_moves_only_unverified(tmp_path, db):
    due, partial, verified, young = (record(db, n=i) for i in range(4))
    db.execute("UPDATE narrative_records SET review_by = '2026-12-01T23:59:59Z' WHERE narrative_id <> ?", (young,))
    db.commit()
    move(db, partial, "partially_verified", "rule")
    move(db, verified, "verified", "rule")
    ledger = tmp_path / "ledger.sqlite"
    p = cli(ledger, "sweep-stale", "--as-of", "2026-12-02T00:00:00Z")
    assert p.returncode == 0 and p.stdout.splitlines() == [f"stale {due}", "swept 1"], p.stdout + p.stderr
    assert [status(db, n) for n in (due, partial, verified, young)] == [
        "stale",
        "partially_verified",
        "verified",
        "unverified",
    ]
    ev = nl.events_of(db, due)[-1]
    assert (ev["decided_by"], ev["at"]) == ("sweep", "2026-12-01T23:59:59Z")
    assert cli(ledger, "sweep-stale", "--as-of", "2026-12-02T00:00:00Z").stdout.splitlines() == ["swept 0"]


def test_export_is_deterministic_and_point_in_time(tmp_path, db):
    early = record(db, n=1)
    record(db, n=2, seen="2026-11-10T00:00:00Z")
    move(db, early, "verified", "rule", at="2026-11-05T00:00:00Z")
    ledger = tmp_path / "ledger.sqlite"
    a = cli(ledger, "export", "--as-of", "2026-11-01T00:00:00Z", "--out", str(tmp_path / "a.json"))
    b = cli(ledger, "export", "--as-of", "2026-11-01T00:00:00Z", "--out", str(tmp_path / "b.json"))
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    doc = json.loads((tmp_path / "a.json").read_text())
    assert [r["narrative_id"] for r in doc["records"]] == [early]
    assert doc["records"][0]["status"] == "unverified" and len(doc["records"][0]["events"]) == 1
    assert [t["theme_id"] for t in doc["themes"]] == ["physical-ai-demand"]
    assert doc["themes"][0]["members"][0]["symbol"] == "NASDAQ:MU"
    later = json.loads(cli(ledger, "export", "--as-of", "2026-11-10T00:00:00Z").stdout)
    assert [r["status"] for r in later["records"]] == ["verified", "unverified"]


def test_export_clips_quotes_to_120_chars(db):
    long_source = "Memory " + "x" * 200 + " ends here."
    nl.add_record(
        db,
        claim="記憶體需求增加",
        claim_type="state",
        attributed_to="a",
        source_tier="kol",
        origin_kind="owner",
        origin_ref="owner:long",
        origin_evidence_id="material:test",
        quote=long_source,
        source_text=long_source,
        subject_symbols=["NASDAQ:MU"],
        theme_ids=["physical-ai-demand"],
        first_seen_at=SEEN,
    )
    quote = json.loads(narrative_export.export(db, "2026-11-01T00:00:00Z"))["records"][0]["quote"]
    assert len(quote) == 120 and quote.endswith("…")


def test_seed_file_loads_into_a_trial_ledger(tmp_path):
    seed = json.loads((HERE / "themes.seed.json").read_text())
    db = nl.connect(tmp_path / "trial.sqlite")
    for t in seed["themes"]:
        nl.add_theme(db, t["theme_id"], t["title"], t["kind"], t["keywords"], "cli:seed-test")
        for m in t["members"]["verified"]:
            nl.add_member(db, t["theme_id"], m["symbol"], m["role"], "cli:seed-test")
        for m in t["members"]["candidate"]:
            nl.symbol_is_tw(m["symbol"])
            assert m["role"] in nl.MEMBER_ROLES and m["why_unconfirmed"]
    for e in seed["edges"]:
        nl.add_edge(db, e["from"], e["to"], e["relation"], "cli:seed-test")
    doc = json.loads(narrative_export.export(db, "2100-01-01T00:00:00Z"))
    assert len(doc["themes"]) == 10
    exported = {(t["theme_id"], m["symbol"]) for t in doc["themes"] for m in t["members"]}
    candidates = {(t["theme_id"], m["symbol"]) for t in seed["themes"] for m in t["members"]["candidate"]}
    assert candidates and not exported & candidates
    # themes-seed-v2: the themes the owner accepted in the trial carry the owner's decision, and so does each member
    approved = [t for t in seed["themes"] if "approval" in t]
    assert [t["theme_id"] for t in approved] == [
        "high-na-euv",
        "custom-asic-capex",
        "xpu-interconnect-content",
        "hbm-base-die-foundry",
    ]
    for item in [*approved, *(m for t in approved for m in t["members"]["verified"])]:
        a = item["approval"]
        assert a["by"] == "owner" and a["decision_ref"].startswith("telegram:20261030-d"), item
        assert a["decided_at"].startswith("2026-10-30T") and a["approval_ref"] == f"telegram:{a['proposal_id']}", item
    # the six §7.1 themes carry the owner's 2026-10-29 seed review ref, and NASDAQ:AMD (M20261029-2) is the one
    # telegram-accepted member that sits in one of them
    assert seed["six_approval_ref"] == "owner-2026-10-29-seed-review"
    assert len(seed["themes"]) - len(approved) == 6
    (optics,) = [t for t in seed["themes"] if t["theme_id"] == "scale-up-optics"]
    (amd,) = [m for m in optics["members"]["verified"] if m["symbol"] == "NASDAQ:AMD"]
    assert amd["role"] == "customer" and amd["proposal_source_ref"] == "material:20261020-k5#p6"
    assert amd["approval"]["approval_ref"] == "telegram:M20261029-2"
    assert amd["approval"]["decided_at"] == "2026-10-30T03:30:01Z"
    assert sum(len(t["members"]["verified"]) for t in seed["themes"]) == 32
    assert ("scale-up-optics", "NASDAQ:AMD") in exported


# ---------------------------------------------------------------- recall keywords (§5.3)
def test_theme_keywords_need_an_approval_and_rehash_the_theme(tmp_path, db):
    path = tmp_path / "ledger.sqlite"
    before = dict(db.execute("SELECT * FROM themes").fetchone())
    assert cli(path, "theme-keywords", "--id", "physical-ai-demand", "--add", "L4").returncode == 2  # no ref
    p = cli(path, "theme-keywords", "--id", "physical-ai-demand", "--add", "L4", "--approval-ref", " ")
    assert p.returncode != 0 and "approval_ref is required" in p.stderr
    p = cli(path, "theme-keywords", "--id", "no-such-theme", "--add", "L4", "--approval-ref", "cli:owner")
    assert p.returncode != 0 and "no theme" in p.stderr
    p = cli(
        path,
        "theme-keywords",
        "--id",
        "physical-ai-demand",
        "--add",
        "自駕",
        "L4",
        "robotaxi",
        "--approval-ref",
        "cli:owner-2026-10-30",
        "--at",
        "2026-10-30T01:00:00Z",
    )
    assert p.returncode == 0, p.stderr
    theme = dict(db.execute("SELECT * FROM themes").fetchone())
    assert json.loads(theme["keywords_json"]) == ["自駕", "L4", "robotaxi"]  # 自駕 was there already
    assert theme["approval_ref"] == before["approval_ref"] and theme["content_sha256"] != before["content_sha256"]
    approvals = [dict(r) for r in db.execute("SELECT * FROM theme_keyword_approvals ORDER BY keyword")]
    assert [(a["keyword"], a["approval_ref"], a["approved_at"]) for a in approvals] == [
        ("L4", "cli:owner-2026-10-30", "2026-10-30T01:00:00Z"),
        ("robotaxi", "cli:owner-2026-10-30", "2026-10-30T01:00:00Z"),
    ]
    for r in [theme, *approvals]:
        stored = r.pop("content_sha256")
        assert stored == content_sha256(r)
    again = cli(path, "theme-keywords", "--id", "physical-ai-demand", "--add", "L4", "--approval-ref", "cli:x")
    assert again.returncode == 0 and "added []" in again.stdout
    assert dict(db.execute("SELECT * FROM themes").fetchone())["content_sha256"] == p.stdout.split()[-1]
    assert db.execute("SELECT count(*) FROM theme_keyword_approvals").fetchone()[0] == 2


def test_a_ledger_from_before_keywords_gets_the_column_and_valid_hashes(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE theme_proposals (proposal_id TEXT PRIMARY KEY, suggested_id TEXT NOT NULL, title TEXT NOT NULL, "
        "kind TEXT NOT NULL, why TEXT NOT NULL, source_ref TEXT NOT NULL, proposed_members_json TEXT NOT NULL, "
        "status TEXT NOT NULL, decided_at TEXT, decided_ref TEXT, created_at TEXT NOT NULL, "
        "content_sha256 TEXT NOT NULL)"
    )
    row = {
        "proposal_id": "T20261020-1",
        "suggested_id": "x",
        "title": "X",
        "kind": "policy",
        "why": "w",
        "source_ref": "material:20261020-k1#p1",
        "proposed_members_json": "[]",
        "status": "proposed",
        "decided_at": None,
        "decided_ref": None,
        "created_at": "2026-10-20T00:00:00Z",
    }
    old.execute("INSERT INTO theme_proposals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [*row.values(), content_sha256(row)])
    old.commit()
    old.close()
    db = nl.connect(path)
    (r,) = [dict(x) for x in db.execute("SELECT * FROM theme_proposals")]
    assert r.pop("keywords_json") == "[]" and r.pop("content_sha256") == content_sha256({**row, "keywords_json": "[]"})
    nl.connect(path).close()  # the migration runs once
    db.close()


# ---------------------------------------------------------------- owner links (§5.3)
REV = "ffc27f0e-f684-4f0e-8b32-4756029c755a"
DOC = {
    "id": "fundamental:agenda-20261029-m94",
    "kind": "fundamental",
    "source_uri": "https://s25.q4cdn.com/621799436/files/Q4-FY26-Prepared-Remarks.pdf",
    "event_time": "2026-09-30T00:00:00Z",
    "payload": {"publisher": "s25.q4cdn.com"},
}


def completed_revision(db, claims):
    """A completed revision and its manifest as research_theme.py stores them (ids without dashes)."""
    db.execute("CREATE TABLE data_manifests (id CHAR(32), payload_json TEXT)")
    db.execute(
        "CREATE TABLE research_revisions (id CHAR(32), scope_key TEXT, status TEXT, research_json TEXT, manifest_id TEXT)"
    )
    db.execute("INSERT INTO data_manifests VALUES ('m94', ?)", (json.dumps({"evidence": [DOC]}),))
    db.execute(
        "INSERT INTO research_revisions VALUES (?, 'tw_stock:basket:2344-2408', 'completed', ?, 'm94')",
        (REV.replace("-", ""), json.dumps({"claims": claims}, ensure_ascii=False)),
    )
    db.commit()


def test_link_add_tags_an_untagged_claim_with_an_approval(tmp_path, db):
    """Acceptance 2026-10-30 §3: claims[19] and [43] hit 「無塵室」 untagged, and the owner had no way to tag them."""
    path = tmp_path / "ledger.sqlite"
    nl.add_theme(db, "cleanroom-capacity", "無塵室產能", "bottleneck", ["無塵室"], "cli:test", at=SEEN)
    tagged = {"kind": "fact", "text": "記憶體供給吃緊", "evidence_ids": [DOC["id"]], "about_symbols": ["NASDAQ:MU"]}
    untagged = {"kind": "fact", "text": "Micron 的新無塵室在 2027 年中出貨", "evidence_ids": [DOC["id"]]}
    completed_revision(db, [tagged, untagged])
    link = ("link-add", "--revision", REV, "--claim", "1", "--theme", "cleanroom-capacity")
    assert cli(path, *link).returncode == 2  # argparse: --approval-ref is required
    p = cli(path, *link, "--approval-ref", " ")
    assert p.returncode != 0 and "approval_ref is required" in p.stderr
    p = cli(path, *link, "--approval-ref", "cli:owner")
    assert p.returncode != 0 and "has no about_symbols" in p.stderr
    for args, why in (
        (("--claim", "2"), "has no claims[2]"),
        (("--theme", "no-such-theme"), "not an active theme"),
        (("--revision", "m94"), "is not a UUID"),
        (("--revision", "0badc0de-0000-4000-8000-000000000000"), "no completed revision"),
        (("--about", "Micron"), "<EXCHANGE>:<TICKER>"),
    ):
        flags = dict(zip(link[1::2], link[2::2], strict=True)) | dict(zip(args[::2], args[1::2], strict=True))
        p = cli(path, "link-add", *[x for kv in flags.items() for x in kv], "--approval-ref", "cli:owner")
        assert p.returncode != 0 and why in p.stderr, (args, p.stderr)
    assert db.execute("SELECT count(*) FROM claim_theme_links").fetchone()[0] == 0
    at = ("--approval-ref", "cli:owner-2026-10-30", "--at", "2026-10-30T02:00:00Z")
    p = cli(path, *link, "--about", "NASDAQ:MU", *at)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == f"link {REV} claims[1] cleanroom-capacity ['NASDAQ:MU'] company release owner"
    (row,) = [dict(r) for r in db.execute("SELECT * FROM claim_theme_links")]
    (approval,) = [dict(r) for r in db.execute("SELECT * FROM claim_link_approvals")]
    assert {k: row[k] for k in ("revision_id", "claim_index", "theme_id", "about_symbols_json", "claim_kind")} == {
        "revision_id": REV,
        "claim_index": 1,
        "theme_id": "cleanroom-capacity",
        "about_symbols_json": '["NASDAQ:MU"]',
        "claim_kind": "fact",
    }
    assert (row["evidence_tier"], row["event_time"], row["source"], row["created_at"]) == (
        "company release",
        "2026-09-30T00:00:00Z",
        "owner",
        "2026-10-30T02:00:00Z",
    )
    assert (approval["approved_by"], approval["approval_ref"], approval["approved_at"]) == (
        "owner",
        "cli:owner-2026-10-30",
        "2026-10-30T02:00:00Z",
    )
    for r in (row, approval):
        stored = r.pop("content_sha256")
        assert stored == content_sha256(r)
    # the same claim again, its id without dashes: a no-op, refused with other companies
    again = cli(path, "link-add", "--revision", REV.replace("-", ""), *link[3:], "--approval-ref", "cli:x")
    assert again.returncode == 0 and again.stdout == p.stdout
    other = cli(path, *link, "--about", "2408", "--approval-ref", "cli:x")
    assert other.returncode != 0 and "different about_symbols_json" in other.stderr
    # a tagged claim takes its own about_symbols
    assert db.execute("SELECT count(*) FROM claim_link_approvals").fetchone()[0] == 1
    p = cli(path, "link-add", "--revision", REV, "--claim", "0", "--theme", "cleanroom-capacity", *at)
    assert p.returncode == 0 and "claims[0] cleanroom-capacity ['NASDAQ:MU']" in p.stdout, p.stderr
    links = json.loads(narrative_export.export(db, "2026-10-31T00:00:00Z"))["claim_theme_links"]
    assert [(x["claim_index"], x["source"]) for x in links] == [(0, "owner"), (1, "owner")]


# ---------------------------------------------------------------- re-validation of filed records (decision 2026-10-31)
def test_revalidate_flags_records_filed_before_a_rule_and_writes_nothing(db, tmp_path, capsys):
    import revalidate_records

    ok = record(db)
    # a record filed before the causal rule applied to it, written as the ledger held it then
    old = dict(nl._one(db, "SELECT * FROM narrative_records WHERE narrative_id = ?", ok))
    old.pop("content_sha256")
    old.update(narrative_id="nr-legacy0001", claim="感測器資料帶動 L4 以上自駕車需要 200GB 以上的記憶體")
    with db:
        nl._insert(db, "narrative_records", old)
    db.commit()
    # an intake record held for a pending proposal, which add_record will gate when it is released
    held = dict(
        claim="感測器資料帶動自駕車記憶體需求",
        claim_type="state",
        attributed_to="a",
        source_tier="kol",
        origin_kind="material",
        origin_ref="material:20261020-k9",
        origin_evidence_id="secondary:material-20261020-k9",
        quote=QUOTE,
        subject_symbols=["NASDAQ:MU"],
        theme_ids=["T20261020-1"],
        first_seen_at=SEEN,
    )
    held_id = narrative_proposals.hold_record(db, held, SOURCE, ["T20261020-1"], SEEN)["narrative_id"]
    path = tmp_path / "ledger.sqlite"
    before = path.read_bytes()
    assert revalidate_records.main(["--ledger", str(path)]) == 0
    out, err = capsys.readouterr()
    line, held_line = out.splitlines()
    nid, origin_kind, origin_ref, status, tags, messages = line.split("\t")
    assert (nid, origin_kind, origin_ref, status, tags) == ("nr-legacy0001", "owner", "owner:test-0", "None", "atomic")
    assert "links a cause to an effect (帶動)" in messages
    assert held_line.split("\t")[:5] == [held_id, "held", "material:20261020-k9", "held", "atomic"]
    assert "1 of 2 records and 1 held records flagged" in err
    assert path.read_bytes() == before
    assert revalidate_records.main(["--ledger", str(path), "--origin", "revision"]) == 0
    assert capsys.readouterr().out == ""
