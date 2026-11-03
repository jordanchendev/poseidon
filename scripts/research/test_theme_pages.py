"""Design 2026-10-29 §9 step 10: the runner ledger's export projected into the Nexus vault.

A ledger built with narrative_ledger.py (themes, members, an edge, revision claims linked to themes, narrative records
and events) is exported; the vault scripts run as subprocesses against a fixture vault built here through
NEXUS_VAULT, so the real vault is never touched. The scripts are read from $NEXUS_VAULT_SCRIPTS (aquarium
local_dev/scripts/research/nexus, laid out like the vault); without them the vault tests skip, the export tests run.
  PYTHONPATH=src NEXUS_VAULT_SCRIPTS=<dir> python -m pytest -q -p no:cacheprovider scripts/research/test_theme_pages.py
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
import narrative_export  # noqa: E402
import narrative_ledger as nl  # noqa: E402
import narrative_proposals  # noqa: E402
from testkit import insert_revision  # noqa: E402

SCRIPTS = Path(os.environ.get("NEXUS_VAULT_SCRIPTS") or "/nonexistent")
THEME_PAGES = SCRIPTS / "_kb" / "ops" / "compile" / "theme_pages.py"
RESEARCH_LINKS = SCRIPTS / "_kb" / "ops" / "compile" / "research_links.py"
LINT = SCRIPTS / "_kb" / "ops" / "lint" / "lint.py"
NEXUS_PULL = SCRIPTS / "bin" / "nexus_pull.py"
needs_scripts = pytest.mark.skipif(
    not (THEME_PAGES.is_file() and RESEARCH_LINKS.is_file() and LINT.is_file()), reason="vault scripts not present"
)

AS_OF = "2026-10-30T02:30:00Z"
APPROVED = "2026-10-01T00:00:00Z"
REV_MU = "04f5d757-dbf1-4ccc-834b-61baf83cdfa6"
REV_ASML = "1b2c3d4e-0000-4000-8000-00000000a5a1"
REV_OLD = "9a9a9a9a-0000-4000-8000-0000000000aa"
LONG_LINE = "記憶體供給吃緊" + "，客戶持續追加長約與預付款" * 12 + "。"
MATERIAL = (
    "股癌 2026-10-27\n"
    "ASML 的出貨受客戶無塵室不足限制，訂單取消成本很高。\n"
    "Marvell 表示櫃內 scale-up 連接仍使用銅。\n"
    "Astera Labs 表示大型叢集會轉向光學連接。\n"
    "Micron 的擴產受無塵室空間限制。\n"
    "南亞科的 DDR4 報價續漲。\n" + LONG_LINE + "\n"
)
THEMES = (
    (
        "cleanroom-capacity",
        "無塵室產能",
        "bottleneck",
        ["無塵室", "cleanroom"],
        [("NASDAQ:ASML", "constrained"), ("NASDAQ:MU", "constrained"), ("2404", "beneficiary")],
    ),
    ("memory-supply-tightness", "記憶體供給吃緊", "bottleneck", [], [("NASDAQ:MU", "supplier"), ("2408", "supplier")]),
    (
        "scale-up-optics",
        "scale-up 互連由銅轉光",
        "tech_transition",
        [],
        [("NASDAQ:MRVL", "supplier"), ("NASDAQ:ALAB", "supplier")],
    ),
)
CLAIMS = {
    REV_MU: (
        "tw_stock:sector:memory",
        [
            {
                "kind": "fact",
                "text": "Micron 表示需要增建 DRAM 無塵室才能補足供給成長。",
                "quote": "additional cleanroom space",
            },
            {
                "kind": "inference",
                "text": "依 Micron 的說法判斷，2404 的無塵室工程需求可能增加。",
                "quote": "cleanroom",
            },
        ],
    ),
    REV_ASML: (
        "tw_stock:theme:cleanroom-capacity",
        [{"kind": "fact", "text": "ASML 表示客戶的無塵室延後使部分機台延後交貨。", "quote": "cleanroom readiness"}],
    ),
    REV_OLD: (
        "tw_stock:theme:scale-up-optics",
        [
            {"kind": "fact", "text": "Marvell 表示 scale-up 仍以銅為主。"},
            {"kind": "fact", "text": "Astera Labs 表示大型叢集將採用光學連接。"},
        ],
    ),
}
# The official document each revision's claims cite, frozen in its data_manifests payload: (evidence id, url,
# publisher = the url host, event_time, issuer = the company that issued it, None when the row named none). The §5.2
# official tier counts issuers.
DOCS = {
    REV_MU: (
        "fundamental:mu-fq4",
        "https://investors.micron.com/fq4.pdf",
        "investors.micron.com",
        "2026-09-30T20:00:00Z",
        "NASDAQ:MU",
    ),
    REV_ASML: (
        "fundamental:asml-q3",
        "https://www.sec.gov/asml-q3.htm",
        "www.sec.gov",
        "2026-10-15T08:00:00Z",
        "NASDAQ:ASML",
    ),
    REV_OLD: (
        "fundamental:ocp",
        "https://www.opencompute.org/summit.pdf",
        "opencompute.org",
        "2026-10-20T00:00:00Z",
        None,
    ),
}
# (revision, claim index, theme, about_symbols, claim kind, evidence tier, evidence event_time)
LINKS = (
    (REV_MU, 0, "cleanroom-capacity", ["NASDAQ:MU"], "fact", "company release", "2026-09-30T20:00:00Z"),
    (REV_MU, 0, "memory-supply-tightness", ["NASDAQ:MU"], "fact", "company release", "2026-09-30T20:00:00Z"),
    (REV_MU, 1, "cleanroom-capacity", ["2404"], "inference", "company release", "2026-09-30T20:00:00Z"),
    (REV_ASML, 0, "cleanroom-capacity", ["NASDAQ:ASML"], "fact", "primary filing", "2026-10-15T08:00:00Z"),
    (REV_OLD, 0, "scale-up-optics", ["NASDAQ:MRVL"], "fact", "company release", "2026-03-01T00:00:00Z"),
    (REV_OLD, 1, "scale-up-optics", ["NASDAQ:ALAB"], "fact", "company release", "2026-10-20T00:00:00Z"),
)
# (key, claim, quote, subjects, theme, first_seen)
RECORDS = (
    (
        "asml",
        "ASML 的出貨受客戶無塵室不足限制",
        "出貨受客戶無塵室不足限制",
        ["NASDAQ:ASML"],
        "cleanroom-capacity",
        "2026-10-27T03:00:00Z",
    ),
    (
        "mu",
        "Micron 的擴產受無塵室空間限制",
        "擴產受無塵室空間限制",
        ["NASDAQ:MU"],
        "cleanroom-capacity",
        "2026-10-27T03:00:00Z",
    ),
    (
        "mrvl",
        "櫃內 scale-up 連接仍使用銅",
        "櫃內 scale-up 連接仍使用銅",
        ["NASDAQ:MRVL"],
        "scale-up-optics",
        "2026-10-27T03:00:00Z",
    ),
    (
        "alab",
        "大型叢集會轉向光學連接",
        "大型叢集會轉向光學連接",
        ["NASDAQ:ALAB"],
        "scale-up-optics",
        "2026-10-27T03:00:00Z",
    ),
    (
        "nanya",
        "南亞科的 DDR4 報價續漲",
        "南亞科的 DDR4 報價續漲",
        ["2408"],
        "memory-supply-tightness",
        "2026-10-27T03:00:00Z",
    ),
    (
        "long",
        "記憶體客戶持續追加長約",
        LONG_LINE[:-1],
        ["NASDAQ:MU"],
        "memory-supply-tightness",
        "2026-10-27T03:00:00Z",
    ),
)


def add_revision(db, rev, scope, claims, doc):
    """A completed revision as research_theme.py stores it (id without dashes, SQLAlchemy Uuid on SQLite) with its
    manifest; every claim cites the one document."""
    doc_id, uri, publisher, at, issuer = doc
    evidence = {
        "id": doc_id,
        "kind": "fundamental",
        "source_uri": uri,
        "event_time": at,
        "payload": {"publisher": publisher, **({"issuer": issuer} if issuer else {})},
    }
    manifest_id = rev.replace("-", "")[::-1]
    research = {"claims": [{**c, "evidence_ids": [doc_id]} for c in claims]}
    insert_revision(db, rev.replace("-", ""), scope, research, manifest_id, {"evidence": [evidence]})


def link(
    db,
    rev,
    idx,
    theme,
    about,
    kind="fact",
    tier="company release",
    at="2026-09-30T20:00:00Z",
    created="2026-10-28T00:00:00Z",
):
    nl.add_link(
        db,
        {
            "revision_id": rev,
            "claim_index": idx,
            "theme_id": theme,
            "about_symbols_json": json.dumps(about),
            "claim_kind": kind,
            "evidence_tier": tier,
            "event_time": at,
            "source": "drafter",
            "created_at": created,
        },
    )


def build_ledger(path):
    """-> {record key: narrative_id}. The ledger as the runner leaves it, research_revisions included."""
    db = nl.connect(path)
    for tid, title, kind, keywords, members in THEMES:
        nl.add_theme(db, tid, title, kind, keywords, f"telegram:T-{tid}", at=APPROVED)
        for sym, role in members:
            nl.add_member(db, tid, sym, role, f"telegram:M-{tid}-{sym}", at=APPROVED)
    nl.add_edge(db, "cleanroom-capacity", "memory-supply-tightness", "related", "telegram:E-1", at=APPROVED)
    for rev, (scope, claims) in CLAIMS.items():
        add_revision(db, rev, scope, claims, DOCS[rev])
    for rev, idx, theme, about, kind, tier, at in LINKS:
        link(db, rev, idx, theme, about, kind, tier, at)
    ids = {}
    for key, claim, quote, subjects, theme, seen in RECORDS:
        ids[key] = nl.add_record(
            db,
            claim=claim,
            claim_type="state",
            attributed_to="股癌",
            source_tier="kol",
            origin_kind="material",
            origin_ref="material:20261027-k2",
            origin_evidence_id="secondary:material-20261027-k2",
            quote=quote,
            source_text=MATERIAL,
            subject_symbols=subjects,
            theme_ids=[theme],
            first_seen_at=seen,
        )["narrative_id"]
    nl.append_event(
        db,
        ids["asml"],
        kind="status",
        to_status="partially_verified",
        decided_by="rule",
        at="2026-10-28T01:00:00Z",
        rule="R3",
        basis={"revision_id": REV_ASML, "claim_indices": [0]},
    )
    nl.set_status(db, ids["nanya"], "refuted", "owner checked the price list", at="2026-10-29T00:00:00Z")
    db.close()
    return ids


def export(tmp_path):
    ids = build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    path = narrative_export.write_export(db, AS_OF, tmp_path / "narrative_export.json")
    db.close()
    return path, ids


# ------------------------------------------------------------------------------------------------ export (no vault)
def test_cross_company_signals_are_computed_from_the_ledger(tmp_path):
    path, _ = export(tmp_path)
    doc = json.loads(path.read_text())
    assert doc["algorithm"] == "narrative-export-v2" and doc["signal_window_days"] == 180
    signals = {
        (s["theme_id"], s["level"]): [(c["symbol"], c["last_at"][:10], c["n"]) for c in s["companies"]]
        for s in doc["cross_company_signals"]
    }
    assert signals == {
        # two companies with official fact claims in the window; the relayed records are not listed again
        ("cleanroom-capacity", "official"): [("NASDAQ:ASML", "2026-10-15", 1), ("NASDAQ:MU", "2026-09-30", 1)],
        # MRVL's official claim is outside the 180-day window, so only the relayed level stands
        ("scale-up-optics", "relayed"): [("NASDAQ:ALAB", "2026-10-27", 1), ("NASDAQ:MRVL", "2026-10-27", 1)],
        # memory-supply-tightness: one official company, and the 2408 record is refuted -> no signal
    }
    links = {(x["revision_id"], x["claim_index"], x["theme_id"]): x for x in doc["claim_theme_links"]}
    mu = links[(REV_MU, 0, "cleanroom-capacity")]
    assert (mu["scope_key"], mu["claim_text"], mu["claim_quote"]) == (
        "tw_stock:sector:memory",
        CLAIMS[REV_MU][1][0]["text"],
        "additional cleanroom space",
    )
    assert all(len(r["quote"]) <= 120 for r in doc["records"])


def official_signals(db, as_of=AS_OF):
    doc = json.loads(narrative_export.export(db, as_of))
    return {
        s["theme_id"]: [c["symbol"] for c in s["companies"]]
        for s in doc["cross_company_signals"]
        if s["level"] == "official"
    }


def test_one_issuer_is_no_official_signal(tmp_path):
    """Acceptance 2026-10-30 §4: one Micron document whose fact claims also named 2408 made memory-supply-tightness an
    official cross-company signal. The official tier needs two issuing companies; a link whose issuer is unknown (its
    revision is not in the ledger) is no hit."""

    def official():
        return official_signals(db)

    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    rerun = "5e5e5e5e-0000-4000-8000-0000000000b2"  # the same Micron document in another revision
    add_revision(db, rerun, "tw_stock:basket:2344-2408", [{"kind": "fact", "text": "記憶體供需更緊"}], DOCS[REV_MU])
    link(db, rerun, 0, "memory-supply-tightness", ["2408", "NASDAQ:MU"])
    link(db, "0badc0de-0000-4000-8000-0000000000c3", 0, "scale-up-optics", ["NASDAQ:MRVL"], at="2026-10-21T00:00:00Z")
    assert official() == {"cleanroom-capacity": ["NASDAQ:ASML", "NASDAQ:MU"]}
    link(db, REV_ASML, 0, "memory-supply-tightness", ["2408"], tier="primary filing", at="2026-10-15T08:00:00Z")
    assert official()["memory-supply-tightness"] == ["2408", "NASDAQ:MU"]  # a second issuer
    db.close()


def test_the_official_tier_counts_issuing_companies_not_url_hosts(tmp_path):
    """Owner 2026-10-30: the official tier counts the companies that published the cited documents. Two companies'
    filings on sec.gov are two issuers; one company's documents on two hosts are one."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    nl.add_theme(db, "high-na-euv", "High-NA EUV", "tech_transition", [], "telegram:T-euv", at=APPROVED)
    for rev, doc_id, at, issuer in (
        ("a5a1a5a1-0000-4000-8000-0000000000d1", "fundamental:asml-6k", "2026-10-15T08:00:00Z", "NASDAQ:ASML"),
        ("1e7e1e7e-0000-4000-8000-0000000000d2", "fundamental:intc-8k", "2026-10-23T20:00:00Z", "NASDAQ:INTC"),
    ):
        doc = (doc_id, f"https://www.sec.gov/{doc_id[12:]}.htm", "www.sec.gov", at, issuer)
        add_revision(db, rev, "tw_stock:theme:high-na-euv", [{"kind": "fact", "text": "High-NA 導入"}], doc)
        link(db, rev, 0, "high-na-euv", [issuer], tier="primary filing", at=at)
    assert official_signals(db)["high-na-euv"] == ["NASDAQ:ASML", "NASDAQ:INTC"]  # one host, two issuers

    mu_8k = "8c8c8c8c-0000-4000-8000-0000000000d3"  # Micron's 8-K on sec.gov, a host other than its fq4 PDF's
    doc = ("fundamental:mu-8k", "https://www.sec.gov/mu-8k.htm", "www.sec.gov", "2026-09-30T20:05:00Z", "NASDAQ:MU")
    add_revision(db, mu_8k, "tw_stock:basket:2344-2408", [{"kind": "fact", "text": "DRAM 供給吃緊"}], doc)
    link(db, mu_8k, 0, "memory-supply-tightness", ["2408"], tier="primary filing", at=doc[3])
    assert "memory-supply-tightness" not in official_signals(db)  # two hosts, one issuer
    db.close()


def test_a_link_whose_evidence_names_no_issuer_is_no_official_hit(tmp_path):
    """A revision frozen before issuers were recorded counts for no one: an old ledger loses the official tier rather
    than gains a false one."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    nanya = "2408a0a0-0000-4000-8000-0000000000d4"
    doc = ("fundamental:nanya", "https://mops.twse.com.tw/nanya.pdf", "mops.twse.com.tw", "2026-10-10T08:00:00Z", None)
    add_revision(db, nanya, "tw_stock:spot:2408", [{"kind": "fact", "text": "DDR4 合約價上漲"}], doc)
    link(db, nanya, 0, "memory-supply-tightness", ["2408"], tier="company release", at=doc[3])
    assert "memory-supply-tightness" not in official_signals(db)
    rerun = "2408b0b0-0000-4000-8000-0000000000d5"  # the same document frozen with its issuer
    add_revision(db, rerun, "tw_stock:spot:2408", [{"kind": "fact", "text": "DDR4 合約價上漲"}], (*doc[:4], "2408"))
    link(db, rerun, 0, "memory-supply-tightness", ["2408"], tier="company release", at=doc[3])
    assert official_signals(db)["memory-supply-tightness"] == ["2408", "NASDAQ:MU"]
    db.close()


def test_signals_are_point_in_time(tmp_path):
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    before_asml = json.loads(narrative_export.export(db, "2026-10-14T00:00:00Z"))["cross_company_signals"]
    much_later = json.loads(narrative_export.export(db, "2027-06-01T00:00:00Z"))["cross_company_signals"]
    db.close()
    assert [s["theme_id"] for s in before_asml if s["level"] == "official"] == []  # ASML's claim is not out yet
    assert much_later == []  # everything slid out of the window


# A theme NASDAQ:ASML is not a member of, and the owner's rejection of the proposal to add it. Acceptance phase 2b: the
# owner rejected M20261029-1 (NASDAQ:ASML into cleanroom-capacity: ASML's official documents never mention cleanrooms),
# then the ASML theme revision 20261030-m96 still linked two ASML official facts to that theme, because the [theme]
# gate admits the run's own subject.
FAB = "fab-buildout"
REJECTED_AT = "2026-10-29T03:00:00Z"
AFTER_REJECTION = "2026-10-30T01:00:00Z"


def fab_theme(db):
    """-> the open member proposal of NASDAQ:ASML into FAB."""
    nl.add_theme(db, FAB, "晶圓廠擴建", "demand_driver", [], "telegram:T-fab", at=APPROVED)
    nl.add_member(db, FAB, "NASDAQ:MU", "customer", "telegram:M-fab-mu", at=APPROVED)
    nl.add_member(db, FAB, "2404", "beneficiary", "telegram:M-fab-2404", at=APPROVED)
    return narrative_proposals.add_member_proposal(
        db,
        theme_id=FAB,
        symbol="NASDAQ:ASML",
        role="supplier",
        why="ASML 交機受客戶建廠進度影響",
        source_ref=f"revision:{REV_ASML}",
        at="2026-10-28T12:00:00Z",
    )["proposal_id"]


def reject(db, proposal_id):
    narrative_proposals.reject_proposal(db, proposal_id, decision_ref="telegram:20261029-r1", decided_at=REJECTED_AT)


def fab_links(db, created="2026-10-28T00:00:00Z"):
    """Micron's official fact first, then ASML's: ASML is the second company and the second issuer."""
    link(db, REV_MU, 0, FAB, ["NASDAQ:MU"], created=created)
    link(db, REV_ASML, 0, FAB, ["NASDAQ:ASML"], tier="primary filing", at="2026-10-15T08:00:00Z", created=created)


def raw_links(db, theme):
    return sorted(
        (r[0], r[1], json.loads(r[2]))
        for r in db.execute(
            "SELECT revision_id, claim_index, about_symbols_json FROM claim_theme_links WHERE theme_id = ?", (theme,)
        )
    )


def test_a_member_the_owner_rejected_is_no_official_hit_for_that_theme(tmp_path):
    """ASML's official facts linked to a theme the owner rejected it from do not make ASML the second company: the
    theme has no official signal. The links stay in the ledger as the audit record."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    reject(db, fab_theme(db))
    fab_links(db, created=AFTER_REJECTION)
    assert FAB not in official_signals(db)
    assert raw_links(db, FAB) == [(REV_MU, 0, ["NASDAQ:MU"]), (REV_ASML, 0, ["NASDAQ:ASML"])]
    db.close()


def test_a_link_written_before_the_rejection_is_no_official_hit_after_it(tmp_path):
    """A write-time gate cannot reach links already in the ledger; the signal reads the rejection at as_of."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    proposal = fab_theme(db)
    fab_links(db)
    reject(db, proposal)
    assert official_signals(db, "2026-10-29T00:00:00Z")[FAB] == ["NASDAQ:ASML", "NASDAQ:MU"]  # not rejected yet
    assert FAB not in official_signals(db)
    db.close()


def test_a_member_rejected_from_one_theme_still_counts_in_the_others(tmp_path):
    """The rejection is of one (symbol, theme) pair: ASML still counts where it is a member (cleanroom-capacity in this
    fixture) and where it is merely not a member (memory-supply-tightness), and counts again in the rejected theme once
    the owner adds it as a member after all."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    reject(db, fab_theme(db))
    fab_links(db, created=AFTER_REJECTION)
    link(db, REV_ASML, 0, "memory-supply-tightness", ["NASDAQ:ASML"], tier="primary filing", at="2026-10-15T08:00:00Z")
    assert official_signals(db) == {
        "cleanroom-capacity": ["NASDAQ:ASML", "NASDAQ:MU"],
        "memory-supply-tightness": ["NASDAQ:ASML", "NASDAQ:MU"],
    }
    nl.add_member(db, FAB, "NASDAQ:ASML", "supplier", "telegram:M-fab-asml", at="2026-10-30T00:00:00Z")
    assert official_signals(db)[FAB] == ["NASDAQ:ASML", "NASDAQ:MU"]
    db.close()


def test_the_export_leaves_a_rejected_member_out_of_that_theme(tmp_path):
    """Through the export the theme page sees no rejected company: it leaves the about_symbols of the theme's links, a
    link left with none is omitted, and the same symbol's links to other themes are kept."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    proposal = fab_theme(db)
    fab_links(db)
    link(db, REV_MU, 1, FAB, ["2404", "NASDAQ:ASML"], kind="inference")
    reject(db, proposal)
    doc = json.loads(narrative_export.export(db, AS_OF))
    links = {(x["theme_id"], x["revision_id"], x["claim_index"]): x for x in doc["claim_theme_links"]}
    assert {k[1:]: json.loads(x["about_symbols_json"]) for k, x in links.items() if k[0] == FAB} == {
        (REV_MU, 0): ["NASDAQ:MU"],
        (REV_MU, 1): ["2404"],
    }
    assert json.loads(links[("cleanroom-capacity", REV_ASML, 0)]["about_symbols_json"]) == ["NASDAQ:ASML"]
    assert raw_links(db, FAB) == [
        (REV_MU, 0, ["NASDAQ:MU"]),
        (REV_MU, 1, ["2404", "NASDAQ:ASML"]),
        (REV_ASML, 0, ["NASDAQ:ASML"]),
    ]
    db.close()


def two_theme_records(db):
    """-> narrative ids of an ASML record under cleanroom-capacity (ASML is a member) and FAB (ASML was rejected), which
    add_record admits, and a Micron record under FAB."""
    ids = []
    for claim, quote, subject, themes in (
        ("ASML 的出貨受客戶無塵室不足限制", "出貨受客戶無塵室不足限制", "NASDAQ:ASML", ["cleanroom-capacity", FAB]),
        ("Micron 的擴產受無塵室空間限制", "擴產受無塵室空間限制", "NASDAQ:MU", [FAB]),
    ):
        ids.append(
            nl.add_record(
                db,
                claim=claim,
                claim_type="state",
                attributed_to="股癌",
                source_tier="kol",
                origin_kind="material",
                origin_ref="material:20261029-k3",
                origin_evidence_id="secondary:material-20261029-k3",
                quote=quote,
                source_text=MATERIAL,
                subject_symbols=[subject],
                theme_ids=themes,
                first_seen_at="2026-10-29T05:00:00Z",
            )["narrative_id"]
        )
    return ids


def test_a_record_under_two_themes_is_no_relayed_hit_where_its_subject_was_rejected(tmp_path):
    """add_record admits a subject that is a member of any one of the record's themes, so the relayed tier can carry a
    rejected pair too. The record stays under the theme ASML is a member of."""
    build_ledger(tmp_path / "ledger.sqlite")
    db = nl.connect(tmp_path / "ledger.sqlite")
    reject(db, fab_theme(db))
    asml, mu = two_theme_records(db)
    doc = json.loads(narrative_export.export(db, AS_OF))
    assert [s for s in doc["cross_company_signals"] if s["theme_id"] == FAB] == []
    records = {r["narrative_id"]: r for r in doc["records"]}
    assert (records[asml]["theme_ids"], records[mu]["theme_ids"]) == (["cleanroom-capacity"], [FAB])
    stored = db.execute("SELECT theme_ids_json FROM narrative_records WHERE narrative_id = ?", (asml,)).fetchone()[0]
    assert json.loads(stored) == ["cleanroom-capacity", FAB]
    db.close()


# ------------------------------------------------------------------------------------------------ the vault
HUMAN_TOP = """---
type: theme
title: "無塵室產能"
status: draft
created: 2026-10-02
updated: 2026-10-20
tags: [theme, bottleneck]
origin: human
theme_id: cleanroom-capacity
---

# 無塵室產能

人寫的上半段：無塵室是記憶體與設備商共同的瓶頸。

## 主題時間線

"""
HUMAN_BOTTOM = "\n\n人寫的結尾。\n"
COMPANY = """---
type: company
title: "{title}"
status: draft
created: 2026-10-21
updated: 2026-10-21
tags: [company]
origin: agent
symbol: "{symbol}"
market: "{market}"
sector: "[[sectors/memory]]"
sources: []
related: []
---

# {title}

## 研究時間線

<!-- research-timeline:start (generated by _kb/ops/compile/research_links.py; do not edit) -->
_尚未生成。_
<!-- research-timeline:end -->
"""
TAXONOMY = """---
type: reference
title: Taxonomy
status: draft
created: 2026-10-21
updated: 2026-10-21
tags: [taxonomy, schema]
origin: agent
sources: []
---

```yaml
sectors:
  memory: {title: "記憶體"}
```
"""


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture()
def vault(tmp_path):
    v = tmp_path / "Nexus"
    write(v, "AGENTS.md", "# fake vault\n")
    write(v, "_kb/wiki/_taxonomy.md", TAXONOMY)
    write(
        v,
        "_kb/wiki/sectors/memory.md",
        "---\ntype: sector\ntitle: m\nstatus: stub\ncreated: 2026-10-21\n"
        "updated: 2026-10-21\ntags: [sector]\norigin: agent\n---\n",
    )
    write(v, "_kb/wiki/companies/mu-micron.md", COMPANY.format(title="美光（mu）", symbol="mu", market="NASDAQ"))
    write(v, "_kb/wiki/companies/2408-nanya.md", COMPANY.format(title="南亞科（2408）", symbol="2408", market="TWSE"))
    write(
        v,
        "_kb/wiki/themes/cleanroom-capacity.md",
        HUMAN_TOP + "<!-- theme-timeline:start (generated by _kb/ops/compile/theme_pages.py; do not edit) -->\n"
        "舊的機器內容\n<!-- theme-timeline:end -->" + HUMAN_BOTTOM,
    )
    for script in (THEME_PAGES, RESEARCH_LINKS):
        write(v, f"_kb/ops/compile/{script.name}", script.read_text(encoding="utf-8"))
    write(v, "_kb/ops/lint/lint.py", LINT.read_text(encoding="utf-8"))
    path, ids = export(tmp_path)
    return {"v": v, "export": path, "ids": ids}


def vault_run(k, script, *args):
    return subprocess.run(
        [sys.executable, str(k["v"] / script), *args],
        capture_output=True,
        text=True,
        env=dict(os.environ, NEXUS_VAULT=str(k["v"])),
    )


def theme_pages(k):
    p = vault_run(k, "_kb/ops/compile/theme_pages.py", "--export", str(k["export"]), "--apply")
    assert p.returncode == 0, p.stderr + p.stdout
    return p.stdout


def snapshot(v):
    return {str(p.relative_to(v)): p.read_bytes() for p in sorted(v.rglob("*")) if p.is_file()}


def block_of(text):
    return re.search(r"<!-- theme-timeline:start[^\n]*-->\n(.*?)<!-- theme-timeline:end -->", text, re.S).group(1)


@needs_scripts
def test_bytes_outside_the_markers_are_kept_and_a_rerun_writes_nothing(vault):
    out = theme_pages(vault)
    themes = vault["v"] / "_kb" / "wiki" / "themes"
    page = (themes / "cleanroom-capacity.md").read_text()
    head, rest = page.split("<!-- theme-timeline:start", 1)
    assert head == HUMAN_TOP and rest.endswith("<!-- theme-timeline:end -->" + HUMAN_BOTTOM)
    assert "舊的機器內容" not in page
    assert sorted(p.name for p in themes.iterdir()) == [
        "_registry.md",
        "cleanroom-capacity.md",
        "memory-supply-tightness.md",
        "scale-up-optics.md",
    ]
    assert "applied: 4 file(s) changed; 3 theme(s)" in out
    before = snapshot(vault["v"])
    again = theme_pages(vault)
    assert "applied: 0 file(s) changed" in again and snapshot(vault["v"]) == before  # log.md included


@needs_scripts
def test_theme_page_block_has_signals_timeline_and_tracking(vault):
    theme_pages(vault)
    ids = vault["ids"]
    block = block_of((vault["v"] / "_kb/wiki/themes/cleanroom-capacity.md").read_text())
    assert (
        "- 官方級（180 天內至少兩家公司各有官方文件的 fact claim）：NASDAQ:ASML（2026-10-15，1 條）、NASDAQ:MU（2026-09-30，1 條）"
        in block
    )
    assert "- 轉述級（僅轉述來源）：已達官方級，不另列" in block
    timeline = block[block.index("### 時間線") : block.index("### 敘事追蹤")]
    rows = [r for r in timeline.splitlines() if r.startswith("| 20")]
    assert [(r.split(" | ")[0][2:], r.split(" | ")[2]) for r in rows] == [
        ("2026-09-30", "官方事實"),
        ("2026-09-30", "本系統推論"),
        ("2026-10-15", "官方事實"),
        ("2026-10-27", "傳聞（KOL）"),
        ("2026-10-27", "傳聞（KOL）"),
        ("2026-10-28", "狀態變化"),
    ]
    assert f"[[Research/tw_stock-sector-memory/{REV_MU}\\|{REV_MU[:8]}]] claims[0]" in rows[0]
    assert f"`{ids['asml']}` 未證實 → 部分證實" in rows[5] and "claims[0]（R3）" in rows[5]
    tracking = block[block.index("### 敘事追蹤") :]
    assert f"| `{ids['asml']}` | ASML 的出貨受客戶無塵室不足限制 | 股癌 | 2026-10-27 | 部分證實 |" in tracking
    assert (
        f"| `{ids['mu']}` | Micron 的擴產受無塵室空間限制 | 股癌 | 2026-10-27 | 傳聞（KOL） | 建立 2026-10-27 |"
        in tracking
    )
    scale = block_of((vault["v"] / "_kb/wiki/themes/scale-up-optics.md").read_text())
    assert "官方級（180 天內至少兩家公司各有官方文件的 fact claim）：無" in scale
    assert "轉述級（僅轉述來源）：NASDAQ:ALAB（2026-10-27，1 條）、NASDAQ:MRVL（2026-10-27，1 條）" in scale


@needs_scripts
def test_an_official_endorsement_is_not_a_status_change_on_the_timeline(vault, tmp_path):
    """An owner-confirmed R4 endorsement (a corroborated event, status kept) is 官方背書 on the timeline; 狀態變化 is kept
    for events that move the status (acceptance phase 2a, d9 / d10)."""
    db = nl.connect(tmp_path / "ledger.sqlite")  # the ledger the vault fixture exported
    nid = nl.add_record(
        db,
        claim="ASML 的訂單取消成本將維持很高",
        claim_type="forecast",
        attributed_to="股癌",
        source_tier="kol",
        origin_kind="material",
        origin_ref="material:20261027-k2",
        origin_evidence_id="secondary:material-20261027-k2",
        quote="訂單取消成本很高",
        source_text=MATERIAL,
        subject_symbols=["NASDAQ:ASML"],
        theme_ids=["cleanroom-capacity"],
        first_seen_at="2026-10-27T03:00:00Z",
    )["narrative_id"]
    nl.append_event(
        db,
        nid,
        kind="corroborated",
        to_status="unverified",
        decided_by="owner",
        at="2026-10-29T01:00:00Z",
        rule="R4",
        basis={"revision_id": REV_ASML, "claim_indices": [0], "approval_ref": "telegram:20261030-d9"},
    )
    narrative_export.write_export(db, AS_OF, vault["export"])
    db.close()
    theme_pages(vault)
    block = block_of((vault["v"] / "_kb/wiki/themes/cleanroom-capacity.md").read_text())
    timeline = block[block.index("### 時間線") : block.index("### 敘事追蹤")]
    rows = [r for r in timeline.splitlines() if r.startswith("| 20")]
    (endorsed,) = [r for r in rows if f"`{nid}` 官方背書，狀態維持未證實" in r]
    assert endorsed.split(" | ")[0] == "| 2026-10-29" and endorsed.split(" | ")[2] == "官方背書"
    (moved,) = [r for r in rows if f"`{vault['ids']['asml']}` 未證實 → 部分證實" in r]
    assert moved.split(" | ")[2] == "狀態變化"


@needs_scripts
def test_a_theme_page_omits_a_company_the_owner_rejected_from_the_theme(vault, tmp_path):
    """Acceptance phase 2b: ASML showed up on the page of a theme the owner had rejected it from. It is gone from that
    page (signals, timeline, tracking, registry linked_symbols) and stays on the page of the theme it is a member of."""
    db = nl.connect(tmp_path / "ledger.sqlite")  # the ledger the vault fixture exported
    reject(db, fab_theme(db))
    fab_links(db, created=AFTER_REJECTION)
    asml, mu = two_theme_records(db)
    narrative_export.write_export(db, AS_OF, vault["export"])
    db.close()
    theme_pages(vault)
    themes = vault["v"] / "_kb/wiki/themes"
    fab = block_of((themes / f"{FAB}.md").read_text())
    assert "NASDAQ:ASML" not in fab and f"`{asml}`" not in fab
    assert "NASDAQ:MU" in fab and f"`{mu}`" in fab
    assert "- 官方級（180 天內至少兩家公司各有官方文件的 fact claim）：無" in fab
    assert f"`{asml}`" in block_of((themes / "cleanroom-capacity.md").read_text())
    registry = (themes / "_registry.md").read_text()
    data = __import__("yaml").safe_load(re.search(r"```yaml\n(.*?)```", registry, re.S).group(1))["themes"]
    assert data[FAB]["linked_symbols"] == ["NASDAQ:MU"]
    assert "NASDAQ:ASML" in data["cleanroom-capacity"]["linked_symbols"]


@needs_scripts
def test_pages_carry_no_quote_over_120_chars_and_no_material_text(vault):
    theme_pages(vault)
    for p in (vault["v"] / "_kb/wiki/themes").glob("*.md"):
        text = p.read_text()
        assert LONG_LINE[:121] not in text and MATERIAL not in text, p.name
        assert all(len(q) <= 120 for q in re.findall(r"「([^」]*)」", text)), p.name
    memory = (vault["v"] / "_kb/wiki/themes/memory-supply-tightness.md").read_text()
    assert LONG_LINE[:119] + "…" in memory


@needs_scripts
def test_registry_lists_themes_and_new_pages_are_stubs(vault):
    theme_pages(vault)
    registry = (vault["v"] / "_kb/wiki/themes/_registry.md").read_text()
    data = __import__("yaml").safe_load(re.search(r"```yaml\n(.*?)```", registry, re.S).group(1))["themes"]
    assert sorted(data) == ["cleanroom-capacity", "memory-supply-tightness", "scale-up-optics"]
    assert data["cleanroom-capacity"]["members"][0] == {"symbol": "2404", "role": "beneficiary", "tw_listed": True}
    assert data["cleanroom-capacity"]["linked_symbols"] == ["2404", "NASDAQ:ASML", "NASDAQ:MU"]
    assert data["cleanroom-capacity"]["keywords"] == ["無塵室", "cleanroom"]
    assert "- [[themes/cleanroom-capacity]] related [[themes/memory-supply-tightness]]（telegram:E-1）" in registry
    stub = (vault["v"] / "_kb/wiki/themes/scale-up-optics.md").read_text()
    assert stub.startswith("---\ntype: theme\n") and "\nstatus: stub\n" in stub and "\ncreated: 2026-10-01\n" in stub


def lint_findings(k):
    p = vault_run(k, "_kb/ops/lint/lint.py")
    report = p.stdout
    found = {}
    for section in re.split(r"\n(?=## )", report):
        m = re.match(r"## (ERROR|WARN|INFO) `([a-z-]+)`", section)
        if m:
            found[m.group(2)] = re.findall(r"^- `([^`]+)`(?: — (.*))?$", section, re.M)
    return p.returncode, found


@needs_scripts
def test_an_unregistered_theme_link_is_a_lint_error(vault):
    theme_pages(vault)
    rc, found = lint_findings(vault)
    assert "theme-id" not in found, found.get("theme-id")
    assert not [f for f in found.get("type-folder", []) if "themes/" in f[0]]
    write(
        vault["v"],
        f"Research/tw_stock-sector-memory/{REV_MU}.md",
        "---\ntype: research\ntitle: m\nstatus: active\ncreated: 2026-10-28\nupdated: 2026-10-28\ntags: [research]\n"
        f'origin: agent\nrevision_id: "{REV_MU}"\nscope_key: "tw_stock:sector:memory"\ncompany: "[[companies/mu-micron]]"\n'
        'themes: ["[[themes/cleanroom-capacity]]", "[[themes/hbm-bottleneck]]", "[[companies/mu-micron]]"]\n---\n\n'
        "見 [[themes/memory-supply-tightness|記憶體]] 與 [[_kb/wiki/themes/not-approved]]。\n",
    )
    rc, found = lint_findings(vault)
    assert rc == 1
    assert sorted(found["theme-id"]) == sorted(
        [
            (
                f"Research/tw_stock-sector-memory/{REV_MU}.md",
                "`themes/hbm-bottleneck` is not in _kb/wiki/themes/_registry.md",
            ),
            (
                f"Research/tw_stock-sector-memory/{REV_MU}.md",
                "`themes/not-approved` is not in _kb/wiki/themes/_registry.md",
            ),
            (
                f"Research/tw_stock-sector-memory/{REV_MU}.md",
                "themes: `companies/mu-micron` does not point into themes/",
            ),
        ]
    )


@needs_scripts
def test_theme_links_without_a_registry_are_lint_errors(vault):
    write(
        vault["v"],
        "_kb/wiki/companies/mu-micron.md",
        COMPANY.format(title="mu", symbol="mu", market="NASDAQ") + "\n見 [[themes/cleanroom-capacity]]\n",
    )
    rc, found = lint_findings(vault)
    assert rc == 1 and found["theme-id"] == [
        ("_kb/wiki/companies/mu-micron.md", "`themes/cleanroom-capacity` but _kb/wiki/themes/_registry.md is missing")
    ]


@needs_scripts
def test_company_block_gets_one_related_themes_line(vault):
    theme_pages(vault)
    p = vault_run(vault, "_kb/ops/compile/research_links.py", "--apply")
    assert p.returncode == 0, p.stderr + p.stdout
    mu = (vault["v"] / "_kb/wiki/companies/mu-micron.md").read_text()
    nanya = (vault["v"] / "_kb/wiki/companies/2408-nanya.md").read_text()
    assert (
        "-->\n相關主題：[[themes/cleanroom-capacity|無塵室產能]]、[[themes/memory-supply-tightness|記憶體供給吃緊]]\n\n"
        "_尚無研究修訂。_\n<!-- research-timeline:end -->" in mu
    )
    assert "-->\n相關主題：[[themes/memory-supply-tightness|記憶體供給吃緊]]\n\n_尚無研究修訂。_" in nanya
    assert mu.count("相關主題") == 1
    before = snapshot(vault["v"])
    again = vault_run(vault, "_kb/ops/compile/research_links.py", "--apply")
    assert "0 company block(s) changed" in again.stdout and snapshot(vault["v"]) == before


# ------------------------------------------------------------------------------------------------ nexus_pull.py
FAKE_SSH = "import subprocess, sys\nsys.exit(subprocess.run(['sh', '-c', sys.argv[2]]).returncode)\n"


@pytest.fixture()
def pull(vault, tmp_path, monkeypatch):
    """nexus_pull.py loaded against the fixture vault, its ssh replaced by a local shell, the export path absolute."""
    if not NEXUS_PULL.is_file():
        pytest.skip("nexus_pull.py not present")
    monkeypatch.setenv("NEXUS_VAULT", str(vault["v"]))
    monkeypatch.setenv("NEXUS_PULL_STATE", str(tmp_path / "pull-state"))
    monkeypatch.setenv("NEXUS_PULL_EXPORT", str(vault["export"]))
    spec = importlib.util.spec_from_file_location("nexus_pull_under_test", NEXUS_PULL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SSH = [sys.executable, str(write(tmp_path, "fake_ssh.py", FAKE_SSH))]
    mod.PY = sys.executable
    mod.STATE.mkdir(parents=True)
    return mod


@needs_scripts
def test_nexus_pull_runs_theme_pages_once_per_export(vault, pull):
    st = pull.load_state()
    assert pull.pull_themes(st, False) == 1
    assert st["themes"]["status"] == "projected" and st["themes"]["result"].startswith("applied: 4 file(s) changed")
    assert (vault["v"] / "_kb/wiki/themes/_registry.md").is_file()
    assert (pull.STATE / "cache" / "narrative_export.json").read_bytes() == vault["export"].read_bytes()
    before = snapshot(vault["v"])
    assert pull.pull_themes(st, False) == 0 and snapshot(vault["v"]) == before  # same bytes: theme_pages not run
    pull.RESEARCH_EXPORT = str(vault["export"].with_name("missing.json"))
    assert pull.pull_themes(st, False) == 0  # the runner has written no export yet


@needs_scripts
def test_nexus_pull_quarantines_an_export_with_a_secret(vault, pull):
    doc = json.loads(vault["export"].read_text())
    doc["records"][0]["attributed_to"] = "api_key=abcdef0123456789"
    vault["export"].write_text(json.dumps(doc, ensure_ascii=False))
    before = snapshot(vault["v"])
    st = pull.load_state()
    assert pull.pull_themes(st, False) == 1 and st["themes"]["status"] == "quarantined"
    assert snapshot(vault["v"]) == before
    (q,) = (pull.STATE / "quarantine").iterdir()
    assert q.stat().st_mode & 0o777 == 0o600
