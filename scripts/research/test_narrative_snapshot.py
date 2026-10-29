"""Design 2026-10-29 §9 step 4: subject symbols, the theme scope, the narrative_snapshot evidence, the secondary kind.

assemble.py runs with fake fetchers against a seeded runner ledger; research_theme.py freezes for real. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_snapshot.py
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import assemble  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import runner  # noqa: E402
import test_runner as tr  # noqa: E402
from claim_precheck import evidence_tier  # noqa: E402
from test_runner import AGENDA_ID, AS_OF, ROW, FakeOpenRouter  # noqa: E402
from test_runner import env as runner_env  # noqa: E402,F401  (pytest finds it by this name)

APPROVED = "2026-10-01T00:00:00Z"
AFTER_AS_OF = "2026-10-25T00:00:00Z"
SOURCE = "UMC 成熟製程產能利用率偏高。ASML 出貨受客戶無塵室限制。Nittobo T glass 產能擴充。記憶體供給吃緊。"
# manifest_sha256 / prompt_sha256 of the test_runner fixture row (2303, dry run), taken at c81d35b before step 4.
RUNNER_MANIFEST_SHA256 = "5eaa71cea89f0625c7a4332940fa76f64912136c888d507bd1c08ed8c51815c3"
RUNNER_PROMPT_SHA256 = "f3f268549dfcb755f383d4d109fec9270d1d4cd4e02ea3ed8699d5468f655473"
# content_sha256 of the completed happy-path revision (good_narrative), same commit.
RUNNER_CONTENT_SHA256 = "dfb75a549dcd0db16c1ecf6223d41724f7b8e4fbb74f7390a6e7b3e7fe757670"


@pytest.fixture()
def env(runner_env):  # noqa: F811
    return runner_env


def seed_themes(db, fixture_at=APPROVED):
    """Four themes. The fixture row's symbol 2303 joins foundry-utilization at fixture_at (None: never)."""
    for tid, title, members in (
        ("foundry-utilization", "晶圓代工產能利用率", [("NYSE:UMC", "supplier")]),
        ("cleanroom-capacity", "無塵室產能", [("NASDAQ:ASML", "constrained"), ("2404", "beneficiary")]),
        ("t-glass-supply", "T-glass 供給", [("TSE:3110", "supplier"), ("1303", "supplier")]),
        (
            "memory-supply-tightness",
            "記憶體供給吃緊",
            [("NASDAQ:MU", "supplier"), ("2408", "supplier"), ("2344", "supplier")],
        ),
    ):
        nl.add_theme(db, tid, title, "bottleneck", [], "test:owner", at=APPROVED)
        for sym, role in members:
            nl.add_member(db, tid, sym, role, "test:owner", at=APPROVED)
    nl.add_member(db, "memory-supply-tightness", "2337", "supplier", "test:owner", at=AFTER_AS_OF)
    if fixture_at:
        nl.add_member(db, "foundry-utilization", "2303", "supplier", "test:owner", at=fixture_at)


def add(db, n, claim, quote, subjects, themes, first_seen, claim_type="state"):
    return nl.add_record(
        db,
        claim=claim,
        claim_type=claim_type,
        attributed_to="股癌",
        source_tier="kol",
        origin_kind="material",
        origin_ref=f"material:20261001-k{n}",
        origin_evidence_id="secondary:material",
        quote=quote,
        source_text=SOURCE,
        subject_symbols=subjects,
        theme_ids=themes,
        first_seen_at=first_seen,
    )["narrative_id"]


def utilization(db, n, first_seen, subjects=("2303",), themes=("foundry-utilization",)):
    return add(db, n, "聯電成熟製程產能利用率偏高", "產能利用率偏高", list(subjects), list(themes), first_seen)


def ledger(tmp_path, fixture_at=APPROVED):
    path = tmp_path / "state" / "ledger.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = nl.connect(path)
    seed_themes(db, fixture_at)
    return path, db


def fake_bars(symbol, start, end):
    return tr.fake_bars("2303" if symbol != "0050" else "0050", start, end)


def assemble_row(env, path, row=ROW, sub="a"):
    kw = dict(env["assemble_kw"], fetch_bars=env.get("fetch_bars", fake_bars))
    return assemble.assemble(row, env["tmp"] / sub, as_of_dt=AS_OF, ledger_path=path, **kw)


def dry_run(env):
    rows = env["tmp"] / "rows.json"
    rows.write_text(json.dumps([ROW]))
    argv = ["--state-dir", str(env["state"]), "--bridge", "/nonexistent", "--dry-run", "--agenda-json", str(rows)]
    assert runner.main(argv, transport=FakeOpenRouter([]), assemble_kw=env["assemble_kw"], as_of=AS_OF) == 0
    return json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "run.json").read_text())


# ------------------------------------------------------------------------------------------------ hash stability
@pytest.mark.parametrize("ledger_state", ["none", "themes", "closed", "future"])
def test_no_theme_member_keeps_manifest_and_prompt(env, ledger_state):
    """A run whose symbol is no theme member as of as_of, with no open record for it, freezes the manifest and builds
    the prompt it did at c81d35b. 2303 joins a theme only after as_of (a record needs a member subject) or never."""
    if ledger_state != "none":
        _, db = ledger(env["tmp"], None if ledger_state == "themes" else AFTER_AS_OF)
        if ledger_state == "closed":
            nid = utilization(db, 1, "2026-10-02T00:00:00Z")
            nl.set_status(db, nid, "refuted", "owner checked", at="2026-10-03T00:00:00Z")
        if ledger_state == "future":
            utilization(db, 1, "2026-10-22T00:00:00Z")
        db.close()
    run = dry_run(env)
    cfg = json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "config.json").read_text())
    assert "narrative_snapshot" not in cfg and "subject_symbols" not in cfg
    assert run["manifest_sha256"] == RUNNER_MANIFEST_SHA256
    assert run["prompt_sha256"] == RUNNER_PROMPT_SHA256


def test_dry_run_reads_the_ledger_without_writing_it(env):
    path, db = ledger(env["tmp"])
    utilization(db, 1, "2026-10-02T00:00:00Z")
    db.close()
    before = path.read_bytes()
    run = dry_run(env)
    assert run["manifest_sha256"] != RUNNER_MANIFEST_SHA256
    assert path.read_bytes() == before


# ------------------------------------------------------------------------------------------------ snapshot trigger
def test_theme_member_without_open_records_gets_the_theme_list(env):
    """The theme list travels only in the snapshot, so a member run with no open record still gets one to tag."""
    _, db = ledger(env["tmp"])
    db.close()
    run = dry_run(env)
    cfg = json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "config.json").read_text())
    snap = cfg["narrative_snapshot"]
    assert snap["payload"]["records"] == [] and snap["payload"]["not_shown"] == []
    assert snap["payload"]["themes"] == [
        {
            "theme_id": "foundry-utilization",
            "title": "晶圓代工產能利用率",
            "kind": "bottleneck",
            "members": [{"symbol": "2303", "role": "supplier"}, {"symbol": "NYSE:UMC", "role": "supplier"}],
        }
    ]
    assert snap["event_time"] == snap["available_at"] == APPROVED
    assert run["manifest_sha256"] != RUNNER_MANIFEST_SHA256
    prompt = json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "prompt.json").read_text())
    text = json.dumps(prompt, ensure_ascii=False)
    assert "foundry-utilization | 晶圓代工產能利用率" in text and "none open for this run" in text
    assert "11. Themes:" in text and "narrative_checks" in text


@pytest.mark.parametrize("case", ["non_member", "member_after_as_of", "retired_theme"])
def test_no_snapshot_without_a_theme_as_of(env, case):
    path, db = ledger(env["tmp"], {"non_member": None, "member_after_as_of": AFTER_AS_OF}.get(case, APPROVED))
    if case == "retired_theme":
        db.execute("UPDATE themes SET status = 'retired' WHERE theme_id = 'foundry-utilization'")
        db.commit()
    db.close()
    assert "narrative_snapshot" not in assemble_row(env, path)


# ------------------------------------------------------------------------------------------------ snapshot
def test_snapshot_is_point_in_time(env):
    path, db = ledger(env["tmp"])
    shown = utilization(db, 1, "2026-10-02T00:00:00Z")
    later_verified = utilization(db, 2, "2026-10-03T00:00:00Z")
    nl.append_event(
        db, later_verified, kind="status", to_status="verified", decided_by="rule", at="2026-10-25T00:00:00Z", basis={}
    )
    already_verified = utilization(db, 3, "2026-10-04T00:00:00Z")
    nl.append_event(
        db,
        already_verified,
        kind="status",
        to_status="verified",
        decided_by="rule",
        at="2026-10-05T00:00:00Z",
        basis={},
    )
    went_stale = utilization(db, 4, "2026-10-05T00:00:00Z")
    nl.append_event(
        db, went_stale, kind="status", to_status="stale", decided_by="sweep", at="2026-10-06T00:00:00Z", basis={}
    )
    first_seen_later = utilization(db, 5, "2026-10-21T04:21:00Z")
    db.close()
    snap = assemble_row(env, path)["narrative_snapshot"]
    recs = {r["narrative_id"]: r for r in snap["payload"]["records"]}
    assert set(recs) == {shown, later_verified, went_stale}
    assert first_seen_later not in recs and already_verified not in recs
    assert (
        recs[later_verified]["status"] == "unverified"
        and recs[later_verified]["last_event_at"] == "2026-10-03T00:00:00Z"
    )
    assert recs[went_stale]["status"] == "stale"
    assert snap["event_time"] == snap["available_at"] == "2026-10-06T00:00:00Z"
    assert snap["source_uri"] == "ledger://narrative/2026-10-21T04:20:00Z"
    assert [t["theme_id"] for t in snap["payload"]["themes"]] == ["foundry-utilization"]
    assert snap["payload"]["not_shown"] == []


def test_snapshot_selection_order_and_cap(env):
    path, db = ledger(env["tmp"])
    by_theme = add(
        db,
        0,
        "ASML 出貨受客戶無塵室限制",
        "出貨受客戶無塵室限制",
        ["NASDAQ:ASML"],
        ["cleanroom-capacity"],
        "2026-10-20T00:00:00Z",
    )
    nl.add_member(db, "cleanroom-capacity", "2303", "customer", "test:owner", at=APPROVED)
    unrelated = add(
        db, 99, "記憶體供給吃緊", "記憶體供給吃緊", ["2408"], ["memory-supply-tightness"], "2026-10-02T00:00:00Z"
    )
    hits = [utilization(db, n, f"2026-10-{n:02d}T00:00:00Z") for n in range(1, 17)]
    db.close()
    snap = assemble_row(env, path)["narrative_snapshot"]["payload"]
    shown = [r["narrative_id"] for r in snap["records"]]
    assert unrelated not in shown + [x["narrative_id"] for x in snap["not_shown"]]
    assert shown == hits[::-1][:15]  # subject hits first, newest first_seen first
    assert [x["narrative_id"] for x in snap["not_shown"]] == [hits[0], by_theme]
    assert all("cap of 15" in x["reason"] for x in snap["not_shown"])


def test_theme_hit_without_subject_hit_is_shown_after_subject_hits(env):
    path, db = ledger(env["tmp"])
    nl.add_member(db, "cleanroom-capacity", "2303", "customer", "test:owner", at=APPROVED)
    by_theme = add(
        db,
        0,
        "ASML 出貨受客戶無塵室限制",
        "出貨受客戶無塵室限制",
        ["NASDAQ:ASML"],
        ["cleanroom-capacity"],
        "2026-10-20T00:00:00Z",
    )
    by_subject = utilization(db, 1, "2026-10-02T00:00:00Z")
    db.close()
    snap = assemble_row(env, path)["narrative_snapshot"]["payload"]
    assert [r["narrative_id"] for r in snap["records"]] == [by_subject, by_theme]
    assert [t["theme_id"] for t in snap["themes"]] == ["cleanroom-capacity", "foundry-utilization"]


def test_no_theme_member_keeps_completed_revision(env):
    _, db = ledger(env["tmp"], None)
    db.close()
    assert env["run"](FakeOpenRouter([tr.good_narrative()])) == 0
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert rev["identity"]["manifest_sha256"] == RUNNER_MANIFEST_SHA256
    assert rev["identity"]["content_sha256"] == RUNNER_CONTENT_SHA256


def test_snapshot_is_frozen_last_and_not_required(env):
    _, db = ledger(env["tmp"])
    nid = utilization(db, 1, "2026-10-02T00:00:00Z")
    db.close()
    run = dry_run(env)
    led = runner.Ledger(env["state"] / "dry-run" / "ledger.sqlite")
    payload = led.manifest_payload(run["manifest_sha256"])
    led.close()
    snap = payload["evidence"][-1]
    assert (
        snap["id"] == snap["kind"] == "narrative_snapshot" and payload["capability_json"]["narrative_snapshot"] is True
    )
    assert snap["recorded_at"] == payload["as_of"] and [r["narrative_id"] for r in snap["payload"]["records"]] == [nid]
    assert "narrative_snapshot" not in payload["required_data"] and evidence_tier(snap) == "derived"


# ------------------------------------------------------------------------------------------------ symbols and theme scope
def test_bare_code_of_a_foreign_member_is_refused(env):
    path, db = ledger(env["tmp"])
    db.close()
    row = dict(ROW, symbols=["3110"])
    with pytest.raises(assemble.AssembleError, match="TSE:3110"):
        assemble_row(env, path, row)


def test_foreign_symbols_are_subjects_without_ohlcv(env):
    path, db = ledger(env["tmp"])
    db.close()
    fetched = []

    def bars(symbol, start, end):
        fetched.append(symbol)
        return fake_bars(symbol, start, end)

    env["fetch_bars"] = bars
    cfg = assemble_row(env, path, dict(ROW, symbols=["2303", "tse:3110", "AAPL"]))
    assert cfg["subject_symbols"] == ["TSE:3110"] and cfg["symbols"] == ["2303", "0050"]
    assert fetched == ["2303", "0050"] and cfg["scope_key"] == "tw_stock:spot:2303"
    assert any("AAPL" in n and "ignored" in n for n in cfg["assemble_notes"])


def test_theme_scope_prices_only_taiwan_members_and_0050(env):
    path, db = ledger(env["tmp"])
    db.close()
    fetched = []

    def bars(symbol, start, end):
        fetched.append(symbol)
        return fake_bars(symbol, start, end)

    env["fetch_bars"] = bars
    row = dict(ROW, symbols=["NASDAQ:MU"], theme_id="memory-supply-tightness")
    cfg = assemble_row(env, path, row)
    assert cfg["scope_key"] == "tw_stock:theme:memory-supply-tightness" and "symbol" not in cfg
    assert cfg["symbols"] == ["2344", "2408", "0050"] and fetched == [
        "2344",
        "2408",
        "0050",
    ]  # 2337 approved after as_of
    assert cfg["subject_symbols"] == ["NASDAQ:MU"]
    assert cfg["theme"] == {"theme_id": "memory-supply-tightness", "title": "記憶體供給吃緊", "kind": "bottleneck"}
    snap = cfg["narrative_snapshot"]["payload"]
    assert snap["records"] == [] and [t["theme_id"] for t in snap["themes"]] == ["memory-supply-tightness"]
    assert [m["symbol"] for m in snap["themes"][0]["members"]] == ["2344", "2408", "NASDAQ:MU"]


@pytest.mark.parametrize(
    "row, needle",
    [
        ({"symbols": ["2303"], "theme_id": "memory-supply-tightness"}, "not Taiwan-listed members"),
        ({"symbols": ["NASDAQ:MU"], "theme_id": "no-such-theme"}, "not an active theme"),
        ({"symbols": ["NASDAQ:MU"]}, "no Taiwan-listed symbol"),
    ],
)
def test_theme_scope_refusals(env, row, needle):
    path, db = ledger(env["tmp"])
    db.close()
    with pytest.raises(assemble.AssembleError, match=needle):
        assemble_row(env, path, dict(ROW, **row))


def test_theme_without_taiwan_member_is_refused(env):
    path, db = ledger(env["tmp"])
    nl.add_theme(db, "optics-export-policy", "光模組對中政策", "policy", [], "test:owner", at=APPROVED)
    nl.add_member(db, "optics-export-policy", "NASDAQ:LITE", "supplier", "test:owner", at=APPROVED)
    db.close()
    with pytest.raises(assemble.AssembleError, match="no Taiwan-listed member"):
        assemble_row(env, path, dict(ROW, symbols=["NASDAQ:LITE"], theme_id="optics-export-policy"))
