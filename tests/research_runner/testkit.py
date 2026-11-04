"""Shared helpers of the research runner tests: the fixture row and document, fakes for OpenRouter / JEV / Thalassa /
TWSE, the theme ledger seeds, the rule-table run of test_narrative_apply and the §7 material-intake cases (股癌 posts,
expected records). Fixtures live in conftest.py."""

import copy
import datetime
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

from poseidon import research_runner
from poseidon.research_runner import intake, jev_client, narrative_check, runner
from poseidon.research_runner import narrative_ledger as nl

HERE = Path(__file__).resolve().parent
PKG = Path(research_runner.__file__).resolve().parent  # the package: its data files and modules
WORKTREE = HERE.parents[1]  # Poseidon repo root (src/ on PYTHONPATH)
PY = sys.executable

AGENDA_ID = "20261021-3"
DOC_ID = f"fundamental:agenda-{AGENDA_ID}"
AS_OF = datetime.datetime(2026, 10, 21, 4, 20, tzinfo=datetime.UTC)
DRAFT_KEY = "sk-or-test-draft-0000"
JEV_KEY = "sk-or-test-jev-1111"
RETURNED_MODEL = "deepseek/deepseek-v4.1-flash-1017"
HTML = """<html><body>
<p>UMC Reports Third Quarter 2026 Results. Revenue was NT$68.73 billion for the quarter, up 12.6% quarter over quarter.</p>

<p>Gross margin was 32.5% for the quarter and capacity utilization rate was 85% across the fabs this quarter.</p>

<p>The company expects wafer shipments to increase by high-single digit in the fourth quarter of 2026, per its outlook.</p>
</body></html>
"""


ROW = {
    "id": AGENDA_ID,
    "topic": "UMC 3Q26 results",
    "symbols": ["2303"],
    "event_date": "2026-10-06",
    "event_type": "earnings",
    "primary_source_url": "https://ir.example.test/umc-3q26.htm",
    "mirror_url": None,
    "why_now": "results published",
    "status": "accepted",
}


def good_narrative():
    return {
        "tl_dr": "聯電第三季營收與毛利率穩健，股價走勢仍待法說確認。",
        "thesis": "聯電第三季營收 NT$68.73 billion，毛利率 32.5%。",
        "stance": "watch",
        "change_summary": "首版。",
        "next_review_at": "2026-10-28T09:00:00Z 法說會後補版",
        "catalysts": [
            {"date": "2026-10-28", "type": "earnings_call", "expected_signal": "第四季指引", "source": DOC_ID}
        ],
        "risks": ["證據只有新聞稿。"],
        "invalidation_conditions": ["第四季出貨衰退。"],
        "unknowns": ["第四季毛利率。"],
        "claims": [
            {
                "kind": "fact",
                "text": "聯電第三季營收為 NT$68.73 billion。",
                "evidence_ids": [DOC_ID],
                "quote": "Revenue was NT$68.73 billion for the quarter",
            },
            {
                "kind": "fact",
                "text": "聯電第三季毛利率為 32.5%。",
                "evidence_ids": [DOC_ID],
                "quote": "Gross margin was 32.5% for the quarter",
            },
            {
                "kind": "inference",
                "text": "依 2303 的價格事實判斷，股價反應仍待事件後追蹤。",
                "evidence_ids": ["ohlcv:2303", "price_facts:2303"],
                "quote": "2303 price facts (algorithm price-facts-v2",
            },
        ],
    }


def fake_bars(symbol, start, end):
    base = {"2303": 150.0, "0050": 110.0}[symbol]
    out, d, i = [], datetime.date(2026, 7, 1), 0
    while d <= end.date():
        if d.weekday() < 5:
            ts = datetime.datetime(d.year, d.month, d.day, 5, 30, tzinfo=datetime.UTC)
            if ts <= end:
                c = base + (i % 7) - 3 + i * 0.1
                out.append((ts.isoformat(), c - 0.5, c + 1, c - 1, c, 1000.0 + i))
                i += 1
        d += datetime.timedelta(days=1)
    return out


def fake_twse():
    rows = [
        {
            "出表日期": "1151012",
            "資料年月": "11509",
            "公司代號": "2303",
            "公司名稱": "聯電",
            "營業收入-當月營收": "25350431",
            "營業收入-上月營收": "23180000",
            "營業收入-去年當月營收": "19925000",
            "營業收入-上月比較增減(%)": "9.36",
            "營業收入-去年同月增減(%)": "27.22",
        },
        {"出表日期": "1151012", "資料年月": "11509", "公司代號": "2330", "營業收入-當月營收": "1"},
    ]
    return rows, "https://openapi.twse.com.tw/v1/opendata/t187ap05_L", "2026-10-21T04:00:00Z"


class FakeOpenRouter:
    """Routes chat completions to a queue of draft responses and JEV calls to a label function."""

    def __init__(self, drafts, jev_label=lambda claim: "supports"):
        self.drafts = list(drafts)
        self.jev_label = jev_label
        self.chat_calls, self.jev_calls, self.headers_seen = [], [], []

    def __call__(self, url, headers, body, timeout):
        self.headers_seen.append(headers)
        req = json.loads(body)
        if url.endswith("/chat/completions"):
            self.chat_calls.append(req)
            item = self.drafts.pop(0)
            if isinstance(item, tuple):  # (status, raw body)
                return item
            if isinstance(item, dict) and "choices" in item:
                return 200, json.dumps(item).encode()
            return 200, json.dumps(chat_response(json.dumps(item, ensure_ascii=False))).encode()
        self.jev_calls.append(req)
        label = self.jev_label(req["state"]["claim"])
        return 200, json.dumps(jev_answer(label, 0.97, 0.0001)).encode()


def jev_answer(label, p, cost):
    """A JEV response body: `label` at probability p, the rest split over the other two labels."""
    probs = {k: p if k == label else round((1 - p) / 2, 4) for k in ("supports", "contradicts", "says_nothing")}
    return {
        "answers": {"citation_relation": {"choice": label, "confidence": p, "probabilities": probs}},
        "usage": {"cost": cost},
    }


def chat_response(content, finish="stop"):
    return {
        "id": "gen-test",
        "model": RETURNED_MODEL,
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 800, "cost": 0.0021},
    }


BRIDGE_SCRIPT = r"""
import json, sys
state_path, args = sys.argv[1], sys.argv[2:]
st = json.load(open(state_path))
st.setdefault("calls", []).append({"args": args, "stdin": sys.stdin.read() if args[0] != "pending" else ""})
json.dump(st, open(state_path, "w"), ensure_ascii=False)
if st.get("refuse", {}).get(args[0]):
    sys.stderr.write("refused: " + st["refuse"][args[0]])
    sys.exit(2)
if args[0] == "pending":
    print(json.dumps(st["rows"], ensure_ascii=False))
if args[0] in ("material-pending", "narrative-pending"):
    print(json.dumps(st.get(args[0], []), ensure_ascii=False))
"""


def insert_revision(db, rev_id, scope_key, research, manifest_id, manifest_payload, status="completed"):
    """A manifest and a revision as research_theme.py stores them (ids without dashes), filled for the ledger's real tables."""
    key = hashlib.sha256(f"{rev_id}{manifest_id}".encode()).hexdigest()
    db.execute(
        "INSERT INTO data_manifests (id, content_sha256, market, interval, as_of, payload_json) VALUES (?, ?, 'tw_stock', '1d', ?, ?)",
        (manifest_id, key, "2026-10-21T00:00:00Z", json.dumps(manifest_payload)),
    )
    db.execute(
        "INSERT INTO research_revisions (id, scope_key, manifest_id, request_sha256, policy_version, provider, model, "
        "runtime_digest, status, research_json) VALUES (?, ?, ?, ?, 'p', 'x', 'm', 'script:0', ?, ?)",
        (rev_id, scope_key, manifest_id, key, status, json.dumps(research, ensure_ascii=False)),
    )
    db.commit()


def ledger_row(env):
    db = sqlite3.connect(env["state"] / "ledger.sqlite")
    db.row_factory = sqlite3.Row
    r = db.execute("SELECT * FROM runner_runs WHERE agenda_id = ?", (AGENDA_ID,)).fetchone()
    revs = [dict(x) for x in db.execute("SELECT id, status FROM research_revisions")]
    db.close()
    return (dict(r) if r else None), revs


def bridge_calls(env, verb):
    return [c for c in env["bridge"]().get("calls", []) if c["args"][0] == verb]


def no_key_leak(env):
    for p in env["state"].rglob("*"):
        if p.is_file() and p.suffix != ".sqlite":
            text = p.read_text(encoding="utf-8", errors="replace")
            assert DRAFT_KEY not in text and JEV_KEY not in text, p


APPROVED = "2026-10-01T00:00:00Z"
AFTER_AS_OF = "2026-10-25T00:00:00Z"
THEME_SOURCE = "UMC 成熟製程產能利用率偏高。ASML 出貨受客戶無塵室限制。Nittobo T glass 產能擴充。記憶體供給吃緊。"


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
        source_text=THEME_SOURCE,
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


def dry_run(env):
    rows = env["tmp"] / "rows.json"
    rows.write_text(json.dumps([ROW]))
    argv = ["--state-dir", str(env["state"]), "--bridge", "/nonexistent", "--dry-run", "--agenda-json", str(rows)]
    assert runner.main(argv, transport=FakeOpenRouter([]), assemble_kw=env["assemble_kw"], as_of=AS_OF) == 0
    return json.loads((env["state"] / "dry-run" / "runs" / AGENDA_ID / "run.json").read_text())


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


RULE_SOURCE = "。".join(c[0] for c in CASES.values()) + "。"
PART = "聯電丙案有一部分成立"  # the R3 check's supported_part
PART_LABELS = {"in_record": "supports", "backed": "supports"}  # FakeJev's answers for PART, judgements (a) and (b)


class FakeJev(FakeOpenRouter):
    """Drafts as FakeOpenRouter; JEV answers a record claim by locator (judgement vs sweep), PART by PART_LABELS (against
    the record's claim vs the cited claims), any other claim supports."""

    def __init__(self, drafts):
        super().__init__(drafts)
        self.by_claim = {c[0]: (c[5], c[6]) for c in CASES.values()}

    def __call__(self, url, headers, body, timeout):
        if url.endswith("/chat/completions"):
            return super().__call__(url, headers, body, timeout)
        req = json.loads(body)
        self.jev_calls.append(req)
        st = req["state"]
        locator = st["source_context"]["locator"]
        if st["claim"] == PART:
            label = PART_LABELS["in_record" if locator == narrative_check.IN_RECORD_LOCATOR else "backed"]
        else:
            judged, swept = self.by_claim.get(st["claim"], ("supports", "supports"))
            label = swept if locator == "all official excerpts" else judged
        return 200, json.dumps(jev_answer(label, 0.9, 0.00002)).encode()


def seed(env):
    path, db = ledger(env["tmp"])
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
            source_text=RULE_SOURCE,
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
    n = good_narrative()
    n["narrative_checks"] = [
        {
            "narrative_id": ids[rule],
            "verdict": c[3],
            "claim_indices": c[4],
            "why": "測試",
            **({"supported_part": PART, "open_part": "未說明主動性。"} if c[3] == "partially_verifies" else {}),
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


def refrozen(rev, n=2, raw_files=None, excerpts=None):
    """rev as a later run under a new revision id with its agenda document frozen again, as run 20261030-m99 re-froze
    the Micron remarks of 20261029-m94: a new evidence id, an issuer and one excerpt more (so the item's content_sha256
    changes), the same raw files. raw_files (and excerpts) make it a different document."""
    later = json.loads(json.dumps(rev))
    later["identity"]["revision_id"] = str(uuid.uuid4())
    new_id = f"{DOC_ID}-r{n}"
    for e in later["manifest_payload"]["evidence"]:
        if e["id"] == DOC_ID:
            p = e["payload"]
            p["excerpts"] = excerpts or [*p["excerpts"], {"locator": "line 999", "text": "Guidance follows."}]
            p["issuer"] = "2303"
            p["raw_files"] = raw_files or p["raw_files"]
            e.update(id=new_id, content_sha256=nl.content_sha256(p))
    for c in later["research"]["claims"]:
        c["evidence_ids"] = [new_id if x == DOC_ID else x for x in c["evidence_ids"]]
    return later


def apply_later(env, later, mode):
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    jev = jev_client.Jev(JEV_KEY, transport=FakeJev([]))
    out = narrative_check.apply(env["state"] / "ledger.sqlite", later, later["manifest_payload"], cfg, jev, mode=mode)
    return {x["narrative_id"]: x for x in out["records"]}


def rec(claim, quote, subjects, theme, attributed, claim_type="state", resolves_on=None):
    return {
        "claim": claim,
        "claim_type": claim_type,
        "resolves_on": resolves_on,
        "attributed_to": attributed,
        "quote": quote,
        "subject_symbols": subjects,
        "theme_ids": [theme],
    }


# ------------------------------------------------------------------------------------------------ material intake (§7)
POSTS = {k: v["text"] for k, v in json.loads((HERE / "fixtures" / "gooaye_posts.json").read_text())["posts"].items()}
MID = "20261020-k1"
RECEIVED = "2026-10-20T18:00:00+08:00"
RECEIVED_UTC = "2026-10-20T10:00:00Z"
# §7.1 lists NASDAQ:ASML under cleanroom-capacity but it is not in the seed's verified members, so these tests approve
# it the way the owner would (member-add). NASDAQ:AMD, the subject N11 needs, is a verified member of scale-up-optics
# in the seed since the owner accepted M20261029-2, so seed_intake_themes gets it from the seed.
EXTRA_MEMBERS = (("cleanroom-capacity", "NASDAQ:ASML", "constrained"),)
VIA = "（經股癌轉述）"


NITTOBO = "日東紡（經產業タイムズ、股癌轉述）"
# §7.2 N1-N19 as the gates accept them. Where the design table differs: N1 drops 「隨競爭者進入而」 (a second
# statement); N9 and N19 keep resolves_on null, because 「明年」 and 「兩年以上」 name no year the period gate can match
# (§2.3), so the program sets first_seen_at + 365 days; N10 keeps the post's 144/288/576 (check_numbers refuses a bare
# 144); N17 says 帶出 as the post does (帶動 is a causal link the intake gate refuses).
EXPECTED_RECORDS = {
    "1": [
        rec(
            "日東紡的 T glass 市佔率未來可能略降",
            "份額未來可能略降",
            ["TSE:3110"],
            "t-glass-supply",
            NITTOBO,
            "forecast",
        ),
        rec(
            "日東紡 T glass 的金額規模會持續向上",
            "但金額規模會持續向上",
            ["TSE:3110"],
            "t-glass-supply",
            NITTOBO,
            "forecast",
        ),
        rec(
            "日東紡把到 2027 年度的投資上調 400 億日圓至 1,200 億日圓",
            "把到 2027 年度的投資上調 400 億日圓至 1,200 億日圓",
            ["TSE:3110"],
            "t-glass-supply",
            NITTOBO,
        ),
        rec(
            "日東紡的 T glass 產能目標為 2025 年度的 3 倍",
            "T glass 產能目標為 2025 年度的 3 倍",
            ["TSE:3110"],
            "t-glass-supply",
            NITTOBO,
        ),
    ],
    "2": [
        rec(
            "ASML 目前規劃的 2027 與 2028 產出無法滿足客戶需求",
            "目前規劃的 27/28 產出無法滿足客戶需求",
            ["NASDAQ:ASML"],
            "cleanroom-capacity",
            "ASML" + VIA,
        ),
        rec(
            "ASML 的出貨受客戶無塵室不足限制",
            "看來連 asml 都明顯受到 cleanroom 的缺乏限制",
            ["NASDAQ:ASML"],
            "cleanroom-capacity",
            "股癌推論",
        ),
        rec(
            "ASML 的訂單取消成本高於客戶預付款",
            "取消成本高於預付款",
            ["NASDAQ:ASML"],
            "cleanroom-capacity",
            "ASML" + VIA,
        ),
    ],
    "3": [],
    "4": [
        rec(
            "櫃內 scale-up 連接 100% 使用銅",
            "櫃內 scale up 100% 走銅",
            ["NASDAQ:MRVL"],
            "scale-up-optics",
            "Marvell" + VIA,
        ),
        rec(
            "Marvell 的 scale-up optics 營收明年會從接近零成長到數億美元",
            "scale up optics 在明年會從幾乎為零的營收往數億 USD 跑",
            ["NASDAQ:MRVL"],
            "scale-up-optics",
            "Marvell" + VIA,
            "forecast",
        ),
    ],
    "5": [
        rec(
            "GPU 叢集擴大到 144/288/576 顆時銅連接不可行",
            "cluster 從 72 顆擴大到 144/288/576 顆 GPU 時，物理限制使銅不可行",
            ["NASDAQ:ALAB"],
            "scale-up-optics",
            "Astera Labs" + VIA,
        ),
        rec(
            "AMD MI450 系列採用 UALink over Ethernet",
            "MI450 系列採用 UALink over Ethernet",
            ["NASDAQ:AMD"],
            "scale-up-optics",
            "AMD（經 Astera CEO、股癌轉述）",
        ),
    ],
    "6": [
        rec(
            "實體 AI 的實際商業化還要兩年以上",
            "JBL 意見是實體 AI 還要兩年以上",
            ["NYSE:JBL"],
            "physical-ai-demand",
            "Jabil" + VIA,
            "forecast",
        ),
    ],
    "7": [
        rec(
            "3.2T 會成為美國對中政策的施力點",
            "3.2T 會是對中政策施力點",
            ["SZSE:300308", "SZSE:300502"],
            "optics-export-policy",
            "Morgan Stanley" + VIA,
            "forecast",
        ),
        rec(
            "中國模組廠可透過採購 Marvell DSP 與 LITE／COHR 雷射湊足 65% 美國 BOM 門檻",
            "innolight 這類公司只要去採購 marvell DSP + LITE/COHR 的雷射湊足 65% bom content 還是可以輸美",
            ["SZSE:300308", "NASDAQ:MRVL", "NASDAQ:LITE", "NYSE:COHR"],
            "optics-export-policy",
            "Morgan Stanley" + VIA,
        ),
        rec(
            "中國可能以 InP 基板出口管制反制",
            "中國可能控制 inp 基板出口反制",
            ["SZSE:300308", "SZSE:300502"],
            "optics-export-policy",
            "Morgan Stanley" + VIA,
            "forecast",
        ),
    ],
    "8": [
        rec(
            "客戶為保出貨量主動壓低每台伺服器的記憶體用量",
            "因為客戶為了保出貨量，主動壓低每台記憶體用量",
            ["NASDAQ:MU"],
            "memory-supply-tightness",
            "市場傳聞（股癌稱已獲 Micron 證實）",
        ),
        rec(
            "記憶體供給吃緊會延續到至少 2028 年",
            "市場吃緊會一直支撐到 2028",
            ["NASDAQ:MU"],
            "memory-supply-tightness",
            "Micron" + VIA,
            "forecast",
            "2028-12-31",
        ),
        rec(
            "agentic AI 帶出伺服器 CPU 的新需求",
            "agentic AI 帶出 CPU 新需求",
            ["NASDAQ:MU"],
            "memory-supply-tightness",
            "Micron" + VIA,
        ),
        rec(
            "L4 以上自駕車需要 200GB 以上的記憶體",
            "L4 以上自駕車用 200GB 以上記憶體",
            ["NASDAQ:MU"],
            "physical-ai-demand",
            "Micron" + VIA,
        ),
    ],
}
VERIFY_8 = ["Micron FQ4 FY26 prepared remarks", "Micron FQ4 FY26 earnings call transcript (Q&A)"]


def intake_out(records=(), proposals=(), verify=()):
    return {"records": list(records), "theme_proposals": list(proposals), "verify_with": list(verify)}


# The ledger the §7.2 cases were written against: the seed's six §7.1 themes. The four themes the owner accepted in the
# trial (themes-seed-v2: high-na-euv and the rest) came from these same posts, so the cases propose them anew.
SECTION_7_1 = (
    "cleanroom-capacity",
    "memory-supply-tightness",
    "t-glass-supply",
    "scale-up-optics",
    "optics-export-policy",
    "physical-ai-demand",
)


def section_7_1_themes():
    return [t for t in json.loads((PKG / "themes.seed.json").read_text())["themes"] if t["theme_id"] in SECTION_7_1]


def seed_intake_themes(path):
    db = nl.connect(path)
    for t in section_7_1_themes():
        nl.add_theme(db, t["theme_id"], t["title"], t["kind"], t["keywords"], "cli:test-seed", at=APPROVED)
        for m in t["members"]["verified"]:
            nl.add_member(db, t["theme_id"], m["symbol"], m["role"], "cli:test-seed", at=APPROVED)
    for theme_id, symbol, role in EXTRA_MEMBERS:
        nl.add_member(db, theme_id, symbol, role, "cli:test-owner", at=APPROVED)
    db.close()


def material_row(post, mid=MID, tier="kol", author="股癌", url=None):
    text = POSTS[post]
    return {
        "id": mid,
        "tier": tier,
        "author": author,
        "url": url,
        "received_at": RECEIVED,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "text": text,
        "status": "accepted",
    }


def run_intake(env, drafts, rows, jev_label=lambda claim: "supports", *extra):
    env["set_bridge"](**{"material-pending": rows})
    t = FakeOpenRouter(drafts, jev_label)
    rc = env["run"](t, *extra)
    return t, rc


def db_rows(env, sql, *args):
    db = sqlite3.connect(env["state"] / "ledger.sqlite")
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args)]
    finally:
        db.close()


def run_json(env):
    return json.loads((env["run_dir"] / "run.json").read_text())


def material_done(env):
    return [json.loads(c["stdin"]) for c in bridge_calls(env, "material-done")]


# ------------------------------------------------------------------------------------------------ the §7.2 records
def snapshot_payload(path, as_of="2026-10-21T04:20:00Z"):
    snap = intake.intake_snapshot(path, as_of)
    return {"as_of": as_of, "evidence": [snap]}


# ------------------------------------------------------------------------------------------------ offline tooling: research_theme.py on synthetic bars
def _sessions_dir():
    if os.environ.get("RESEARCH_SESSIONS_DIR"):
        return Path(os.environ["RESEARCH_SESSIONS_DIR"])
    for parent in HERE.parents:
        if (parent / "local_dev" / "sessions").is_dir():
            return parent / "local_dev" / "sessions"
    return HERE / "_no_sessions_"


SESSIONS = _sessions_dir()
N, EVENT_IDX = 30, 19  # base = close of bar 19; T+1 = bar 20; T+5 = bar 24
START = date(2026, 9, 1)


def day(i):
    return (START + timedelta(days=i)).isoformat()


def aaa(i):
    return 100.0 + 2.0 * i


def bbb(i):
    return 50.0 + 0.5 * i


def ccc(i):
    return 20.0 + 0.1 * i


def run_script(args, **kw):
    env = dict(os.environ, PYTHONPATH=str(WORKTREE / "src"))
    return subprocess.run([PY, *args], cwd=WORKTREE, env=env, capture_output=True, text=True, **kw)


def pct(x):
    return round(x * 100, 2)


def go(env, name, cfg=None, narrative=None, ledger=None, extra=()):
    d = env["tmp"] / name
    d.mkdir(exist_ok=True)
    (d / "cfg.json").write_text(json.dumps(cfg or env["cfg"]))
    args = [
        "-m",
        "poseidon.research_runner.research_theme",
        "--config",
        str(d / "cfg.json"),
        "--csv",
        str(env["csv"]),
        "--ledger",
        str(ledger or d / "ledger.sqlite"),
        *extra,
    ]
    if narrative is not None:
        (d / "narr.json").write_text(json.dumps(narrative))
        args += ["--narrative", str(d / "narr.json"), "--out", str(d / "rev.json")]
    return run_script(args), d


ACTUAL_Q4 = "Fourth quarter revenue was US$53.1 billion and gross margin was 61.2%. For the first quarter, revenue is expected to be between US$55 billion and US$57 billion."


def render(path):
    return subprocess.run(
        [sys.executable, "-m", "poseidon.research_runner.render_theme", str(path)], capture_output=True, text=True
    )


def doc_with(env, name, extra_text, **doc_keys):
    """Copy of fundamental:doc1 whose raw file also carries `extra_text` as a second excerpt ('line 2')."""
    base = env["cfg"]["documents"][0]
    raw = env["tmp"] / f"{name}.txt"
    raw.write_text(base["excerpts"][0]["text"] + "\n" + extra_text + "\n")
    return dict(
        copy.deepcopy(base),
        raw_text_path=str(raw),
        raw_files={raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()},
        excerpts=[base["excerpts"][0], {"locator": "line 2", "text": extra_text}],
        **doc_keys,
    )
