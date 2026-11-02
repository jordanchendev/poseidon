"""Offline tests for research_theme.py / render_theme.py (price-facts-v2, catalysts, tl_dr layering, guidance bridge,
kpi_series, old-revision read-across, source tiers, disclosure).

SQLite ledger, synthetic bars, no network. Run from the Poseidon repo root:
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_offline_tooling.py
Reference revisions from aquarium sessions are read from $RESEARCH_SESSIONS_DIR (default: the aquarium
local_dev/sessions next to this checkout); those tests skip when the files are absent.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]  # Poseidon repo root (src/ on PYTHONPATH)
PY = sys.executable


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


def run(args, **kw):
    env = dict(os.environ, PYTHONPATH=str(WORKTREE / "src"))
    return subprocess.run([PY, *args], cwd=WORKTREE, env=env, capture_output=True, text=True, **kw)


def pct(x):
    return round(x * 100, 2)


@pytest.fixture()
def env(tmp_path):
    csv_path = tmp_path / "bars.csv"
    rows = []
    for sym, f in (("AAA", aaa), ("BBB", bbb), ("CCC", ccc)):
        for i in range(N):
            c = f(i)
            rows.append(f"{day(i)}T00:00:00Z,{sym},{c - 0.5},{c + 1},{c - 1},{c},{1000 + i}")
    csv_path.write_text("\n".join(rows) + "\n")
    raw = tmp_path / "raw.txt"
    text = "Company announced record revenue of 100 units for the quarter, up 20% year over year."
    raw.write_text(text + "\n")
    cfg = {
        "scope_key": "tw_stock:spot:AAA",
        "symbol": "AAA",
        "as_of": f"{day(N - 1)}T09:00:00Z",
        "event_time": f"{day(EVENT_IDX)}T10:00:00Z",
        "account_scope": "paper:test",
        "universe_id": "u-test",
        "thalassa_base_url": "http://127.0.0.1:1",
        "symbols": ["AAA", "BBB"],
        "benchmark": "BBB",
        "required_data": {"fundamental": {"max_age_seconds": 2592000}, "ohlcv": {"max_age_seconds": 172800}},
        "policy_version": "test/v0",
        "provider": "test",
        "model": "test",
        "documents": [
            {
                "id": "fundamental:doc1",
                "kind": "fundamental",
                "title": "Doc",
                "publisher": "Pub",
                "source_uri": "https://example.com/doc1",
                "event_time": f"{day(EVENT_IDX)}T10:00:00Z",
                "available_at": f"{day(EVENT_IDX)}T10:00:00Z",
                "raw_text_path": str(raw),
                "raw_files": {"raw.txt": hashlib.sha256(raw.read_bytes()).hexdigest()},
                "excerpts": [{"locator": "line 1", "text": text}],
            }
        ],
    }
    narrative = {
        "thesis": "Test thesis.",
        "stance": "watch",
        "change_summary": "First version.",
        "next_review_at": "2026-10-05T01:00:00Z review the October revenue",
        "tl_dr": "AAA beat and outperformed the benchmark.",
        "risks": ["r"],
        "invalidation_conditions": ["i"],
        "unknowns": ["u"],
        "claims": [
            {"kind": "fact", "text": "Revenue was 100 units.", "evidence_ids": ["fundamental:doc1"]},
            {
                "kind": "fact",
                "text": "AAA outperformed BBB after the event.",
                "evidence_ids": ["price_facts:AAA", "ohlcv:AAA"],
            },
            {
                "kind": "inference",
                "text": "The move is probably not fully priced in.",
                "evidence_ids": ["price_facts:AAA"],
            },
        ],
        "catalysts": [
            {"date": "2026-09-10", "type": "other", "expected_signal": "past, ignored", "source": "s"},
            {
                "date": "2026-10-05",
                "type": "monthly_revenue",
                "expected_signal": "revenue stays above 100",
                "source": "MOPS",
            },
            {"date": "2026-11-12", "type": "earnings_call", "expected_signal": "guidance", "source": "IR"},
        ],
    }
    return dict(tmp=tmp_path, csv=csv_path, cfg=cfg, narrative=narrative)


def go(env, name, cfg=None, narrative=None, ledger=None, extra=()):
    d = env["tmp"] / name
    d.mkdir(exist_ok=True)
    (d / "cfg.json").write_text(json.dumps(cfg or env["cfg"]))
    args = [
        str(HERE / "research_theme.py"),
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
    return run(args), d


def test_price_facts_v2_math(env):
    p, d = go(env, "facts")
    assert p.returncode == 0, p.stderr
    info = json.loads(p.stdout)
    assert [e["algorithm"] for e in info["evidence"] if e["kind"] == "price_facts"] == ["price-facts-v2"] * 2
    a, b = info["stats"]["AAA"], info["stats"]["BBB"]
    base_a, base_b = aaa(EVENT_IDX), bbb(EVENT_IDX)
    t1a, t5a = aaa(EVENT_IDX + 1) / base_a - 1, aaa(EVENT_IDX + 5) / base_a - 1
    t1b, t5b = bbb(EVENT_IDX + 1) / base_b - 1, bbb(EVENT_IDX + 5) / base_b - 1
    sa, sb = aaa(N - 1) / base_a - 1, bbb(N - 1) / base_b - 1
    assert a["close_at_event"] == base_a
    assert (a["ret_t1_pct"], a["ret_t5_pct"]) == (pct(t1a), pct(t5a)) == (1.45, 7.25)
    assert (b["ret_t1_pct"], b["ret_t5_pct"]) == (pct(t1b), pct(t5b))
    assert a["rel_ret_t1_pct"] == pct(t1a - t1b) == 0.61
    assert a["rel_ret_t5_pct"] == pct(t5a - t5b) == 3.04
    assert a["rel_ret_since_event_pct"] == pct(sa - sb)
    assert a["ret_since_event_pct"] == pct(sa)  # v1 field unchanged
    assert b["rel_ret_t1_pct"] == 0 and b["rel_ret_since_event_pct"] == 0  # benchmark vs itself


def test_not_enough_bars_is_null(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["event_time"] = f"{day(N - 3)}T10:00:00Z"  # only 2 bars after the event
    p, _ = go(env, "late", cfg)
    assert p.returncode == 0, p.stderr
    a = json.loads(p.stdout)["stats"]["AAA"]
    assert a["ret_t1_pct"] == pct(aaa(N - 2) / aaa(N - 3) - 1)
    assert a["ret_t5_pct"] is None and a["rel_ret_t5_pct"] is None and a["rel_ret_t1_pct"] is not None


def test_benchmark_missing_gives_null_relative(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["benchmark"] = "NOPE"
    p, _ = go(env, "nobench", cfg)
    a = json.loads(p.stdout)["stats"]["AAA"]
    assert a["ret_t1_pct"] is not None and a["rel_ret_t1_pct"] is None and a["rel_ret_since_event_pct"] is None


def test_complete_and_render(env):
    p, d = go(env, "ok", narrative=env["narrative"])
    assert p.returncode == 0, p.stderr + p.stdout
    rev = json.loads((d / "rev.json").read_text())
    assert rev["research"]["catalysts"][1]["type"] == "monthly_revenue"
    r = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(d / "rev.json")], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    md = r.stdout
    assert md.index("> **推論｜一句話結論：** AAA beat") < md.index("## 論點")
    assert "## 催化劑日曆" in md and "| 2026-10-05 | monthly_revenue | revenue stays above 100 | MOPS |" in md
    assert (
        md.index("## 事實（引用凍結證據）")
        < md.index("Revenue was 100 units.")
        < md.index("## 推論\n")
        < md.index("probably not fully priced")
    )
    assert "- Revenue was 100 units." in md and "_引用：price_facts:AAA, ohlcv:AAA_" in md
    assert "T+1 / T+5 報酬（絕對）" in md and "+1.45% / +7.25%" in md
    assert "T+1 / T+5 報酬（相對 BBB）" in md and "+0.61% / +3.04%" in md


def test_catalyst_next_review_mismatch_rejected(env):
    n = copy.deepcopy(env["narrative"])
    n["next_review_at"] = "2026-10-09T01:00:00Z later"  # earliest future catalyst is 2026-10-05
    p, d = go(env, "mismatch", narrative=n)
    assert p.returncode != 0 and "earliest future catalyst date 2026-10-05" in p.stderr
    assert not (d / "rev.json").exists()


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda c: c[0].update(type="bogus"), "catalysts[0].type"),
        (lambda c: c[0].update(date="10/05/2026"), "catalysts[0].date"),
        (lambda c: c[0].pop("expected_signal"), "catalysts[0].expected_signal"),
    ],
)
def test_catalyst_shape_rejected(env, mutate, msg):
    n = copy.deepcopy(env["narrative"])
    mutate(n["catalysts"])
    p, _ = go(env, "shape", narrative=n)
    assert p.returncode != 0 and msg in p.stderr


def test_legacy_narrative_without_new_fields_renders(env):
    n = copy.deepcopy(env["narrative"])
    n.pop("catalysts")
    n.pop("tl_dr")
    p, d = go(env, "legacy", narrative=n)
    assert p.returncode == 0, p.stderr
    md = subprocess.run(
        [sys.executable, str(HERE / "render_theme.py"), str(d / "rev.json")], capture_output=True, text=True, check=True
    ).stdout
    assert "催化劑日曆" not in md and "一句話結論" not in md and "## 事實（引用凍結證據）" in md


# ------------------------------------------------------------------ claim_precheck (stdlib, no Poseidon needed)
from claim_precheck import check_numbers, extract_quantities, lint_atomic  # noqa: E402


def canon(text):
    return [(round(q.value, 6), q.unit_class, q.currency) for q in extract_quantities(text)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("美光 2026 會計年度營收為 1,331.9 億美元", [(133190000000.0, "money", "USD")]),
        ("營收 1.3319 兆美元", [(1331900000000.0, "money", "USD")]),
        ("revenue of $133.19 billion", [(133190000000.0, "money", "USD")]),
        ("新台幣 1 兆 4,942.4 億元", [(1494240000000.0, "money", "TWD")]),
        ("consolidated revenue of NT$1,494.24 billion", [(1494240000000.0, "money", "TWD")]),
        ("2025年合併營收新臺幣8.099兆元", [(8099000000000.0, "money", "TWD")]),
        ("鴻海 2025 年合併營收為新台幣 8,099 億元", [(809900000000.0, "money", "TWD")]),
        ("稀釋每股盈餘為新台幣 29.43 元", [(29.43, "money", "TWD")]),
        ("每單位 ADR 4.06 美元", [(4.06, "money", "USD")]),
        ("交易金額預計為1,150億日圓", [(115000000000.0, "money", "JPY")]),
        ("Gross margin for the quarter was 66.4%", [(66.4, "percent", None)]),
        ("up 210 basis points sequentially", [(2.1, "pp", None)]),
        ("較上季上升 290 個基點", [(2.9, "pp", None)]),
        ("提升 2.1 個百分點", [(2.1, "pp", None)]),
        ("up 210 percentage points", [(210.0, "pp", None)]),
    ],
)
def test_normaliser_values(text, expected):
    assert canon(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "TSMC (TWSE: 2330, NYSE: TSM) on Oct. 15, 2026",  # ticker, month-day, year
        "3Q26 vs 2Q26 and FQ1-27 guidance, 1H26",  # glued period identifiers
        "LPDDR6, HBM4E and DDR5 in fiscal Q4 2026",  # product names
        "2026年10月16日，預計於2027年10月1日完成",  # zh dates
        "2330 財報後與 0050 同幅下跌",  # tickers incl. leading zero
        "grow in the low-20s percentage range in calendar 2027",
        "on 10-15 the board met; net(2) and GAAP(1) footnotes",
    ],
)
def test_normaliser_skips_dates_periods_ids(text):
    assert extract_quantities(text) == []


def test_normaliser_ranges_signs_periods():
    (q,) = extract_quantities("Revenue is expected to be between US$52.3 billion and US$54.3 billion")
    assert q.range == (52.3e9, 54.3e9) and q.currency == "USD"
    (q,) = extract_quantities("毛利率 65% 到 67%")
    assert q.range == (65.0, 67.0) and q.unit_class == "percent"
    (q,) = extract_quantities("資本支出 600 億至 640 億美元")
    assert q.range == (6e10, 6.4e10) and q.currency == "USD"
    (q,) = extract_quantities("guidance 52.3–54.3 billion")
    assert q.range == (52.3e9, 54.3e9)
    (q,) = extract_quantities("revenue decreased 17.6% from the second quarter")
    assert q.sign == -1 and q.period == "qoq"
    (q,) = extract_quantities("third quarter revenue increased 50.9% year-over-year")
    assert q.sign == 1 and q.period == "yoy"
    (q,) = extract_quantities("伺服器產品組合營收在第四季年增超過一倍")
    assert (q.value, q.unit_class, q.qualifier, q.period) == (2.0, "multiple", ">", "yoy")
    (q,) = extract_quantities("$61.5 billion ± $1.5 billion")
    assert q.value == 61.5e9 and q.pm == 1.5e9


@pytest.mark.parametrize(
    "claim,excerpt,status,kind",
    [
        # the four JEV gold-set misses
        (
            "美光 2026 會計年度營收為 133.19 億美元。",
            "Revenue of $133.19 billion versus $37.38 billion for the prior year",
            "mismatch",
            "scale",
        ),  # micron-pr-04
        (
            "鴻海 2025 年合併營收為新台幣 8,099 億元。",
            "2025年合併營收新臺幣8.099兆元。",
            "mismatch",
            "scale",
        ),  # honhai-07
        (
            "Micron expects server unit growth in the high-teens percentage range in calendar 2026 and 2027, driven mainly by AI training clusters.",
            "We expect server unit growth in the high-teens percentage range in both calendar 2026 and 2027.",
            "no_numbers",
            None,
        ),  # micron-rm-28 (lint's job)
        (
            "美光伺服器 LPDDR SOCAMM 產品組合營收在第四季年增超過一倍。",
            "Our industry leading server LPDDR SOCAMM portfolio revenue more than doubled sequentially in the fourth quarter of 2026.",
            "ok",
            None,
        ),  # micron-pr-12
        # derivable quantities must not be flagged
        (
            "Micron's fiscal Q4 2026 operating cash flow was more than seven times the level of the same quarter a year earlier.",
            "Operating cash flow of $43.97 billion versus $25.39 billion for the prior quarter and $5.73 billion for the same period last year",
            "ok",
            None,
        ),
        (
            "台積電第三季毛利為新台幣 9,923.79 億元，年增 68.6%。",
            "(Unit: NT$ million, except for EPS) 3Q26 Amounta 3Q25 Amount YoY Inc. (Dec.) % Net sales 1,494,243 989,918 50.9 Gross profit 992,379 588,543 68.6",
            "ok",
            None,
        ),
        (
            "美光 2026 會計年度營收為 1,331.9 億美元，前一年度為 373.8 億美元。",
            "Revenue of $133.19 billion versus $37.38 billion for the prior year",
            "ok",
            None,
        ),
        (
            "Q4 revenue guidance midpoint is US$53.3 billion.",
            "Revenue is expected to be between US$52.3 billion and US$54.3 billion",
            "ok",
            None,
        ),
        # contradictions
        (
            "Third-quarter revenue decreased 17.6% from the second quarter.",
            "third quarter results represented a 17.6% increase in revenue",
            "mismatch",
            "sign",
        ),
        (
            "gross margin was 87.0%, up 210 percentage points sequentially.",
            "gross margin for fiscal Q4 was 87.0%, up 210 basis points sequentially.",
            "mismatch",
            "scale",
        ),
        (
            "第三季毛利率為 56.4%。",
            "Gross margin for the quarter was 66.4%, operating margin was 58.7%",
            "mismatch",
            "value",
        ),
    ],
)
def test_check_numbers(claim, excerpt, status, kind):
    r = check_numbers(claim, excerpt)
    assert r["status"] == status, r
    if kind:
        assert r["unmatched"][0]["kind"] == kind


def test_check_numbers_derived_flag_and_period_mismatch():
    r = check_numbers(
        "Operating cash flow was more than seven times the year-ago level.",
        "Operating cash flow of $43.97 billion versus $5.73 billion for the same period last year",
    )
    assert r["matched"][0]["match"] == "derived" and r["matched"][0]["via"][0]["how"] == "A/B"
    r = check_numbers(
        "美光第四季 NAND 營收年增 42%。", "Sequentially, NAND revenue increased 42%. Up 526% year over year."
    )
    assert r["status"] == "ok" and r["period_mismatch"][0]["claim_period"] == "yoy"


@pytest.mark.parametrize(
    "claim,ok",
    [
        (
            "Micron expects server unit growth in the high-teens percentage range in calendar 2026 and 2027, driven mainly by AI training clusters.",
            False,
        ),
        ("第三季毛利率 66.4%、營業利益率 58.7%。", False),
        ("營收 523 億到 543 億美元。", True),
        ("Gross margin is expected to be between 65% and 67%.", True),
        (
            "Hon Hai and Mitsubishi Electric plan to jointly build a high-quality EV platform originating in Japan.",
            True,
        ),
        ("Compared with the second quarter, TSMC's third-quarter net income rose 17.6%.", True),
        ("第四季毛利率指引低於第三季是因為 2 奈米爬坡的稀釋效應。", False),
    ],
)
def test_lint_atomic(claim, ok):
    assert lint_atomic(claim)["ok"] is ok


def test_numeric_mismatch_refuses_completion(env):
    n = copy.deepcopy(env["narrative"])
    n["claims"][0]["text"] = "Revenue was 1,000 units."  # excerpt says 100 units
    p, d = go(env, "numref", narrative=n)
    assert (
        p.returncode != 0
        and "claim precheck refused completion" in p.stderr
        and "claims[0] numeric mismatch" in p.stderr
    )
    assert "1,000 (scale)" in p.stderr and not (d / "rev.json").exists()


def test_non_atomic_claim_refused(env):
    n = copy.deepcopy(env["narrative"])
    n["claims"][0]["text"] = "Revenue was 100 units, up 20% year over year."
    p, d = go(env, "atomref", narrative=n)
    assert p.returncode != 0 and "claims[0] is non-atomic" in p.stderr and not (d / "rev.json").exists()


@pytest.mark.parametrize(
    "claim,excerpts,expected",
    [
        (
            "Diluted EPS was $33.42.",
            [
                "GAAP net income of $37.70 billion, or $32.87 per diluted share",
                "Non-GAAP net income of $38.40 billion, or $33.42 per diluted share",
            ],
            "missing",
        ),
        (
            "Non-GAAP diluted EPS was $33.42.",
            [
                "GAAP net income of $37.70 billion, or $32.87 per diluted share",
                "Non-GAAP net income of $38.40 billion, or $33.42 per diluted share",
            ],
            "ok",
        ),
        ("每股盈餘為新台幣 29.43 元。", ["diluted earnings per share of NT$29.43"], "n/a"),
        ("Revenue was $54.23 billion.", ["GAAP net income", "Non-GAAP net income"], "n/a"),
    ],
)
def test_check_basis(claim, excerpts, expected):
    from research_theme import check_basis

    assert check_basis(claim, excerpts) == expected


# ------------------------------------------------------------------------------------------------ guidance bridge (B2)
GUIDE_Q4 = "For the fourth quarter, revenue is expected to be between US$50 billion and US$52 billion; gross margin is expected to be between 60% and 62%, based on the exchange rate assumption of 1 US dollar to 31.8 NT dollars."
ACTUAL_Q4 = "Fourth quarter revenue was US$53.1 billion and gross margin was 61.2%. For the first quarter, revenue is expected to be between US$55 billion and US$57 billion."


def render(path):
    return subprocess.run([sys.executable, str(HERE / "render_theme.py"), str(path)], capture_output=True, text=True)


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


def q4_guidance(**over):
    g = [
        {
            "metric": "revenue",
            "period": "Q4",
            "low": 50,
            "high": 52,
            "unit": "USD billion",
            "basis": "IFRS",
            "fx_assumption": "1 US dollar to 31.8 NT dollars",
            "excerpt_locator": "line 2",
        },
        {
            "metric": "gross margin",
            "period": "Q4",
            "low": 60,
            "high": 62,
            "unit": "%",
            "basis": "IFRS",
            "excerpt_locator": "line 2",
        },
    ]
    g[0].update(over)
    return g


def test_guidance_frozen_hashed_and_rendered_without_predecessor(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"] = [doc_with(env, "g1", GUIDE_Q4, guidance=q4_guidance())]
    p, d = go(env, "guid", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    doc = next(e for e in rev["manifest_payload"]["evidence"] if e["id"] == "fundamental:doc1")
    assert doc["payload"]["guidance"][0]["low"] == 50 and rev["identity"]["guidance_bridge"] is None
    from research_theme import content_sha256 as _h  # noqa: F401  (import check only)

    md = render(d / "rev.json").stdout
    assert (
        "## 本期指引（Guidance）" in md
        and "| revenue | Q4 | 50–52 | USD billion | IFRS | 1 US dollar to 31.8 NT dollars |" in md
    )
    assert "Guidance bridge" not in md


def test_guidance_changes_manifest_hash(env):
    p0, _ = go(env, "nog", env["cfg"])
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"] = [doc_with(env, "g2", GUIDE_Q4, guidance=q4_guidance())]
    p1, _ = go(env, "withg", cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2["documents"][0]["guidance"][1]["high"] = 63  # warns (value) but must change the hash
    p2, _ = go(env, "withg2", cfg2)
    assert p0.returncode == p1.returncode == p2.returncode == 0, p0.stderr + p1.stderr + p2.stderr
    shas = [json.loads(x.stdout)["manifest_sha256"] for x in (p0, p1, p2)]
    assert len(set(shas)) == 3 and "warning:" in p2.stderr and "'63%'" in p2.stderr


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"low": 5.0, "high": 5.2}, "refusing to freeze"),  # scale: US$5.0bn vs US$50bn
        ({"low": -50, "high": 52}, "refusing to freeze"),  # sign
        ({"excerpt_locator": "line 9"}, "excerpt_locator 'line 9'"),
        ({"basis": "adjusted"}, "basis must be one of"),
        ({"high": None}, "low/high/value must be numbers"),
        ({"value": 51}, "needs either low and high, or value"),
    ],
)
def test_guidance_refusals(env, over, msg):
    g = q4_guidance(**over)
    if over.get("high", 0) is None:
        g[0]["high"] = None
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"] = [doc_with(env, "gr", GUIDE_Q4, guidance=g)]
    p, d = go(env, "gref", cfg, narrative=env["narrative"])
    assert p.returncode != 0 and msg in p.stderr, p.stderr
    assert not (d / "rev.json").exists()


def test_guidance_value_mismatch_only_warns(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"] = [doc_with(env, "gw", GUIDE_Q4, guidance=q4_guidance(low=49))]
    p, _ = go(env, "gwarn", cfg)
    assert p.returncode == 0 and "warning: fundamental:doc1 guidance[0]: 'US$49 billion' not found" in p.stderr, (
        p.stderr
    )


def bridge_pair(env, actuals=None, narrative2=None):
    """Revision 1 (as_of day 25) freezes Q4 guidance; revision 2 (as_of day 29, same ledger) reports Q4 actuals."""
    ledger = env["tmp"] / "bridge.sqlite"
    c1 = copy.deepcopy(env["cfg"])
    c1["as_of"] = f"{day(25)}T09:00:00Z"
    c1["documents"] = [doc_with(env, "b1", GUIDE_Q4, guidance=q4_guidance())]
    p1, d1 = go(env, "b1", c1, narrative=env["narrative"], ledger=ledger)
    assert p1.returncode == 0, p1.stderr
    q1 = [
        {
            "metric": "Revenue",
            "period": "Q1",
            "low": 55,
            "high": 57,
            "unit": "USD billion",
            "basis": "IFRS",
            "excerpt_locator": "line 2",
        }
    ]
    c2 = copy.deepcopy(env["cfg"])
    c2["documents"] = [doc_with(env, "b2", ACTUAL_Q4, guidance=q1)]
    c2["actuals"] = (
        actuals
        if actuals is not None
        else [
            {
                "metric": "revenue",
                "period": "Q4",
                "value": 53.1,
                "unit": "USD billion",
                "evidence_id": "fundamental:doc1",
                "excerpt_locator": "line 2",
            }
        ]
    )
    n2 = narrative2 or copy.deepcopy(env["narrative"])
    p2, d2 = go(env, "b2", c2, narrative=n2, ledger=ledger)
    return p1, d1, p2, d2


def test_guidance_bridge_from_predecessor(env):
    n2 = copy.deepcopy(env["narrative"])
    n2["claims"].append(
        {
            "kind": "fact",
            "text": "Q4 gross margin was 61.2%.",
            "evidence_ids": ["fundamental:doc1"],
            "actual": {"metric": "gross margin", "period": "Q4", "value": 61.2, "unit": "%"},
        }
    )
    p1, d1, p2, d2 = bridge_pair(env, narrative2=n2)
    assert p2.returncode == 0, p2.stderr
    rev1, rev2 = (json.loads((x / "rev.json").read_text()) for x in (d1, d2))
    br = rev2["identity"]["guidance_bridge"]
    assert rev2["identity"]["previous_revision_id"] == rev1["identity"]["revision_id"] == br["predecessor_revision_id"]
    rows = {r["metric"]: r for r in br["rows"]}
    assert (
        rows["revenue"]["prior"] == {"low": 50, "high": 52}
        and rows["revenue"]["actual"]["value"] == 53.1
        and rows["revenue"]["vs_prior"] == "above"
    )
    assert rows["revenue"]["new"] == [{"period": "Q1", "low": 55, "high": 57, "unit": "USD billion", "basis": "IFRS"}]
    assert (
        rows["gross margin"]["actual"]["source"] == "claims[3]"
        and rows["gross margin"]["vs_prior"] == "within"
        and rows["gross margin"]["new"] == []
    )
    md = render(d2 / "rev.json").stdout
    assert "## Guidance bridge（前版 `%s` 的 Q4 指引 → 實際 → 新指引）" % rev1["identity"]["revision_id"][:8] in md
    assert (
        "| revenue（Q4，IFRS） | 50–52 USD billion | 53.1 USD billion（fundamental:doc1） | above | Q1: 55–57 USD billion |"
        in md
    )
    assert "| gross margin（Q4，IFRS） | 60–62 % | 61.2 %（claims[3]） | within | n/a |" in md
    assert md.index("## Guidance bridge") < md.index("## 本期指引（Guidance）")


def test_bridge_without_actual_shows_na(env):
    _, _, p2, d2 = bridge_pair(env, actuals=[])
    assert p2.returncode == 0, p2.stderr
    assert (
        json.loads((d2 / "rev.json").read_text())["identity"]["guidance_bridge"] is None
    )  # no reported period -> no bridge
    c = json.loads((d2 / "cfg.json").read_text())
    c["reported_period"] = "Q4"
    c["as_of"] = f"{day(N - 1)}T09:30:00Z"
    p3, d3 = go(env, "b3", c, narrative=env["narrative"], ledger=env["tmp"] / "bridge.sqlite")
    assert p3.returncode == 0, p3.stderr
    md = render(d3 / "rev.json").stdout
    assert "| revenue（Q4，IFRS） | 50–52 USD billion | n/a（無引用實際值） | n/a | Q1: 55–57 USD billion |" in md


@pytest.mark.parametrize(
    "actual,msg",
    [
        ({"value": 5.31}, "refusing to freeze"),  # scale
        ({"value": 53.5}, "refusing to freeze"),  # value mismatch is strict for actuals
        ({"evidence_id": "fundamental:nope"}, "is not a config document"),
        ({"excerpt_locator": "line 7"}, "excerpt_locator 'line 7'"),
    ],
)
def test_actuals_refusals(env, actual, msg):
    a = dict(
        {
            "metric": "revenue",
            "period": "Q4",
            "value": 53.1,
            "unit": "USD billion",
            "evidence_id": "fundamental:doc1",
            "excerpt_locator": "line 2",
        },
        **actual,
    )
    _, _, p2, d2 = bridge_pair(env, actuals=[a])
    assert p2.returncode != 0 and msg in p2.stderr, p2.stderr
    assert not (d2 / "rev.json").exists()


def test_narrative_actual_must_be_in_claim_text(env):
    n2 = copy.deepcopy(env["narrative"])
    n2["claims"].append(
        {
            "kind": "fact",
            "text": "Q4 gross margin was 61.2%.",
            "evidence_ids": ["fundamental:doc1"],
            "actual": {"metric": "gross margin", "period": "Q4", "value": 62.1, "unit": "%"},
        }
    )
    _, _, p2, d2 = bridge_pair(env, narrative2=n2)
    assert p2.returncode != 0 and "claims[3].actual vs its claim text" in p2.stderr and not (d2 / "rev.json").exists()


# ------------------------------------------------------------------------------------------------ kpi_series (B3)
def roc(ym_iso):
    y, m = ym_iso.split("-")
    return "%d%s" % (int(y) - 1911, m)


def kpi_rows():
    """AAA monthly revenue 2025-08..2026-08 (13 months) published on the 10th of the next month, plus a 2026-09 row
    published 2026-10-10 (after as_of, must be invisible). Values in NTD thousand."""
    rows, months = [], ["2025-%02d" % m for m in range(8, 13)] + ["2026-%02d" % m for m in range(1, 10)]
    for k, ym in enumerate(months):
        y, m = int(ym[:4]), int(ym[5:])
        py, pm = (y, m + 1) if m < 12 else (y + 1, 1)
        v, prior = 1_000_000 + 100_000 * k, 800_000 + 50_000 * k
        rows.append(
            {
                "出表日期": "%d%02d10" % (py - 1911, pm),
                "資料年月": roc(ym),
                "公司代號": "AAA",
                "公司名稱": "A",
                "營業收入-當月營收": str(v),
                "營業收入-上月營收": str(v - 100_000),
                "營業收入-去年當月營收": str(prior),
                "營業收入-上月比較增減(%)": str((v / (v - 100_000) - 1) * 100),
                "營業收入-去年同月增減(%)": str((v / prior - 1) * 100),
            }
        )
    rows.append({"出表日期": "1150910", "資料年月": "11508", "公司代號": "ZZZ", "營業收入-當月營收": "5"})
    return rows


def kpi_env(env, rows=None):
    path = env["tmp"] / "kpi.json"
    path.write_text(
        json.dumps(
            {
                "rows": rows or kpi_rows(),
                "source_uri": "https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
                "fetched_at": "2026-09-30T08:00:00Z",
            },
            ensure_ascii=False,
        )
    )
    return ["--kpi-json", str(path)]


def test_kpi_series_payload_point_in_time(env):
    p, d = go(env, "kpi", extra=kpi_env(env))
    assert p.returncode == 0, p.stderr
    assert "no t187ap05_L row visible at as_of for BBB" in p.stderr
    info = json.loads(p.stdout)
    k = next(e for e in info["evidence"] if e["kind"] == "kpi_series")
    assert k["id"] == "kpi_series:AAA" and k["points"] == 12
    # payload check via a completed revision
    n = copy.deepcopy(env["narrative"])
    n["claims"].append(
        {"kind": "fact", "text": "AAA revenue in August 2026 was NT$2.2 billion.", "evidence_ids": ["kpi_series:AAA"]}
    )
    p, d = go(env, "kpi2", narrative=n, extra=kpi_env(env))
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    pay = next(e for e in rev["manifest_payload"]["evidence"] if e["id"] == "kpi_series:AAA")["payload"]
    assert (
        pay["unit"] == "NTD thousand"
        and pay["metric"] == "monthly_revenue"
        and pay["fetched_at"] == "2026-09-30T08:00:00Z"
    )
    assert [x["period"] for x in pay["points"]][[0, -1][0]] == "2025-09" and pay["points"][-1][
        "period"
    ] == "2026-08"  # 2026-09 published after as_of
    assert pay["points"][-1]["value"] == 2_200_000 and pay["points"][-1]["yoy_pct"] == round(
        (2_200_000 / 1_400_000 - 1) * 100, 2
    )
    t3 = (2_000_000 + 2_100_000 + 2_200_000) / (1_300_000 + 1_350_000 + 1_400_000) - 1
    assert pay["trailing_3m_yoy_pct"] == round(t3 * 100, 2) and len(pay["content_sha256"]) == 64
    assert rev["identity"]["claim_precheck"][3]["numbers"] == "ok"
    md = render(d / "rev.json").stdout
    sec = md.split("## 月營收")[1].split("\n## ")[0]
    assert (
        "## 月營收（`kpi_series:AAA`，單位：NTD thousand）" in md
        and sec.count("| 2026-0") == 6
        and "| 2025-" not in sec
    )
    assert "| 2026-08 | 2,200,000 | +57.14% | +4.76% |" in md and "近三個月合計 YoY：%+.2f%%" % round(t3 * 100, 2) in md


def test_kpi_claim_mismatch_refused(env):
    n = copy.deepcopy(env["narrative"])
    n["claims"].append(
        {"kind": "fact", "text": "AAA revenue in August 2026 was NT$22 billion.", "evidence_ids": ["kpi_series:AAA"]}
    )
    p, d = go(env, "kpibad", narrative=n, extra=kpi_env(env))
    assert (
        p.returncode != 0
        and "claims[3] numeric mismatch" in p.stderr
        and "(scale)" in p.stderr
        and not (d / "rev.json").exists()
    )


def test_kpi_gap_gives_null_trailing_and_no_rows_refused(env):
    rows = [r for r in kpi_rows() if r["資料年月"] != roc("2026-07")]
    p, d = go(env, "kpigap", narrative=env["narrative"], extra=kpi_env(env, rows))
    assert p.returncode == 0, p.stderr
    pay = next(
        e for e in json.loads((d / "rev.json").read_text())["manifest_payload"]["evidence"] if e["kind"] == "kpi_series"
    )["payload"]
    assert pay["trailing_3m_yoy_pct"] is None
    assert "需連續三個月資料" in render(d / "rev.json").stdout
    p, _ = go(env, "kpinone", extra=kpi_env(env, [r for r in kpi_rows() if r["公司代號"] == "ZZZ"]))
    assert p.returncode != 0 and "no symbol has a monthly revenue row visible at as_of" in p.stderr


NUM_CASES = [  # P0-9: from 1000 up, ,.4g gave 1.235e+04 / 9.877e+05 / 1,234
    (12345.6, "12,345.6"),
    (987654.3, "987,654.3"),
    (1234.5, "1,234.5"),
    (-12345.6, "-12,345.6"),
    (1234.567, "1,234.57"),
    (1234.999, "1,235"),
    (1000.004, "1,000"),
    (2_200_000, "2,200,000"),
    (2_200_000.0, "2,200,000"),
    (999.5, "999.5"),
    (1.2345, "1.2345"),
    (None, "n/a"),
]


def test_kpi_values_render_with_thousands_separators_and_up_to_two_decimals(env):
    p, d = go(env, "kpinum", narrative=env["narrative"], extra=kpi_env(env))
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    kpi = next(e for e in rev["manifest_payload"]["evidence"] if e["kind"] == "kpi_series")["payload"]
    for start in range(0, len(NUM_CASES), 6):  # the table shows the last 6 points
        cases = NUM_CASES[start : start + 6]
        kpi["points"] = [{"period": f"2026-{k + 1:02d}", "value": v} for k, (v, _) in enumerate(cases)]
        (d / "rev.json").write_text(json.dumps(rev))
        md = render(d / "rev.json").stdout
        for k, (v, want) in enumerate(cases):
            assert f"| 2026-{k + 1:02d} | {want} | n/a | n/a |" in md, (v, want)


# ------------------------------------------------------------------------------------------------ read-across + disclosure
def test_config_read_across_refused_toward_about_symbols(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["read_across"] = [{"symbol": "CCC", "relation": "supplier", "why": "sells parts to AAA"}]
    p, _ = go(env, "rabad", cfg, narrative=env["narrative"])
    assert p.returncode != 0 and "about_symbols" in p.stderr
    p, d = go(env, "ranone", narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    assert "read_across" not in json.loads((d / "rev.json").read_text())["identity"]


def test_old_revision_identity_read_across_still_renders(env):
    p, d = go(env, "ra", narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    rev["identity"]["read_across"] = [
        {"symbol": "AAA", "relation": "self", "why": "frozen symbol"},
        {"symbol": "DDD", "relation": "customer", "why": "not in symbols"},
    ]
    (d / "old.json").write_text(json.dumps(rev))
    md = render(d / "old.json").stdout
    b = EVENT_IDX
    t1 = pct(aaa(b + 1) / aaa(b) - 1 - (bbb(b + 1) / bbb(b) - 1))
    t5 = pct(aaa(b + 5) / aaa(b) - 1 - (bbb(b + 5) / bbb(b) - 1))
    assert "## Read-across（推論；相對 BBB 的事件後報酬）" in md
    assert f"| AAA | self | frozen symbol | {t1:+.2f}% | {t5:+.2f}% |" in md
    assert "| DDD | customer | not in symbols | n/a（未凍結價格） | n/a（未凍結價格） |" in md


# sha256 of render_theme.py's stdout for the UMC v2 revision (3de55971, the one old revision with identity.read_across),
# taken at 6620350 before config read_across was removed.
UMC_V2_RENDER_SHA256 = "91cc26960273986d39e6f50e5f9c9d8a4547c14179d9fc536b3a40172e569490"


def test_umc_v2_revision_renders_byte_identical():
    src = SESSIONS / "2026-10-21-research-umc-2303" / "revision_umc_v2.json"
    if not src.exists():
        pytest.skip("UMC v2 reference revision not present")
    out = subprocess.run([sys.executable, str(HERE / "render_theme.py"), str(src)], capture_output=True)
    assert out.returncode == 0, out.stderr
    assert "## Read-across" in out.stdout.decode()
    assert hashlib.sha256(out.stdout).hexdigest() == UMC_V2_RENDER_SHA256


# manifest_sha256 of the env fixture config (no documents[].source_tier), taken at 6620350 before SOURCE_TIERS changed.
ENV_MANIFEST_SHA256 = "e1d137d7199a143ee645cb33e3b7e6ca7800ad46662ec9c1ce1fdc554871f706"


def test_config_without_tier_keeps_manifest_hash(env):
    p, _ = go(env, "notier")
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["manifest_sha256"] == ENV_MANIFEST_SHA256


@pytest.mark.parametrize("tier", ["sell-side", "media", "kol"])
def test_secondary_tiers_split(env, tier):
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"][0]["source_tier"] = tier
    p, d = go(env, "tier", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    assert f"`fundamental:doc1`：{tier}" in render(d / "rev.json").stdout
    cfg["documents"][0]["source_tier"] = "secondary"
    p, _ = go(env, "tierold", cfg)
    assert p.returncode != 0 and "source_tier must be one of" in p.stderr


def secondary_cfg(env, tier):
    cfg = copy.deepcopy(env["cfg"])
    doc = copy.deepcopy(cfg["documents"][0])
    doc.update(id="secondary:post", kind="secondary", source_uri="https://www.facebook.com/post/1")
    if tier:
        doc["source_tier"] = tier
    cfg["documents"].append(doc)
    return cfg


def test_secondary_kind_tier_comes_from_the_payload(env):
    sys.path.insert(0, str(HERE))
    from claim_precheck import evidence_tier

    p, d = go(env, "sec", secondary_cfg(env, "kol"), narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    item = next(e for e in rev["manifest_payload"]["evidence"] if e["id"] == "secondary:post")
    assert rev["manifest_payload"]["capability_json"]["secondary"] is True
    assert evidence_tier(item) == "kol" and evidence_tier(dict(item, payload={})) != "company release"
    md = render(d / "rev.json").stdout
    assert "`secondary:post`：kol；" in md and "二手素材" in md


@pytest.mark.parametrize("tier", [None, "primary filing"])
def test_secondary_kind_without_a_secondary_tier_is_refused(env, tier):
    p, _ = go(env, f"sec-{tier}", secondary_cfg(env, tier))
    assert p.returncode != 0 and "needs source_tier sell-side|media|kol" in p.stderr


def test_disclosure_block(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"][0]["source_tier"] = "primary filing"
    p, d = go(env, "disc", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    md = render(d / "rev.json").stdout
    tail = md[md.index("## 揭露（Disclosure）") :]
    assert "## " not in tail[3:]  # last section
    assert "origin: agent" in tail and "provider `test`，model `test`，policy `test/v0`" in tail
    assert (
        "`fundamental:doc1`：primary filing" in tail
        and "`ohlcv:AAA`：exchange feed（經 Thalassa）" in tail
        and "`price_facts:AAA`：derived" in tail
    )
    assert (
        "節錄逐字驗證：1/1" in tail
        and "數字 precheck：ok 1，mismatch 0，推論數字未對上（僅記錄） 0，無數字 0，略過 2；原子化 ok 3/3" in tail
    ), tail
    assert "JEV：未執行（not run）" in tail and tail.rstrip().endswith("not investment advice）。")
    rev = json.loads((d / "rev.json").read_text())
    rev["jev_results"] = {"model": "jev-x", "n": 3, "supports": 2, "says_nothing": 1}
    (d / "rev2.json").write_text(json.dumps(rev))
    assert "JEV：model jev-x，n 3，supports 2，says_nothing 1" in render(d / "rev2.json").stdout
    bad = copy.deepcopy(env["cfg"])
    bad["documents"][0]["source_tier"] = "blog"
    p, _ = go(env, "discbad", bad)
    assert p.returncode != 0 and "source_tier must be one of" in p.stderr


def test_legacy_revision_file_still_renders():
    legacy = SESSIONS / "2026-10-20-research-tsmc-3q26-2330" / "revision_tsmc.json"
    if not legacy.exists():
        pytest.skip("TSMC reference revision not present")
    r = render(legacy)  # read-only
    assert r.returncode == 0, r.stderr
    assert (
        "節錄逐字驗證：未記錄" in r.stdout and "not investment advice" in r.stdout and "Guidance bridge" not in r.stdout
    )


def test_inference_numeric_mismatch_recorded_fact_refused(env):
    base = copy.deepcopy(env["narrative"])
    base["claims"][2] = {
        "kind": "inference",
        "text": "Revenue is probably about 120 units next quarter.",
        "evidence_ids": ["fundamental:doc1"],
    }
    p, d = go(env, "infok", narrative=base)
    assert p.returncode == 0, p.stderr
    pc = json.loads((d / "rev.json").read_text())["identity"]["claim_precheck"][2]
    assert pc["numbers"] == "mismatch-recorded" and pc["unmatched"] == ["120 (value)"]
    assert "推論數字未對上（僅記錄） 1" in render(d / "rev.json").stdout
    fact = copy.deepcopy(base)
    fact["claims"][2]["kind"] = "fact"
    p, d = go(env, "factref", narrative=fact)
    assert p.returncode != 0 and "claims[2] numeric mismatch" in p.stderr and not (d / "rev.json").exists()
    scale = copy.deepcopy(base)
    scale["claims"][2]["text"] = "Revenue is probably about 1,000 units next quarter."
    p, d = go(env, "infscale", narrative=scale)
    assert p.returncode != 0 and "claims[2] numeric mismatch" in p.stderr and "(scale)" in p.stderr


def test_price_labels_from_payload_and_plain_unknowns(env):
    p, d = go(env, "labels", narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    md = render(d / "rev.json").stdout
    ev, last = day(EVENT_IDX)[5:10], day(N - 1)[5:10]
    assert f"| 收盤（{last}） |" in md and f"| {ev} 收盤 → {last} |" in md
    assert "09-30 收盤 → 10-20" not in md and "## 未知\n" in md and "股癌" not in md


def test_umc_reference_revision_renders_dollar_text_unchanged(tmp_path):
    src = SESSIONS / "2026-10-21-research-umc-2303" / "revision_umc.json"
    if not src.exists():
        pytest.skip("UMC reference revision not present")
    rev = json.loads(src.read_text())
    r = render(src)  # read-only
    assert r.returncode == 0, r.stderr
    md = r.stdout
    assert " ".join(rev["research"]["thesis"].split()) in " ".join(md.split())
    for c in rev["research"]["claims"]:
        assert c["text"] in md
    pf = next(e["payload"] for e in rev["manifest_payload"]["evidence"] if e["kind"] == "price_facts")
    last = rev["identity"]["stats"][pf["symbol"]]["last_bar"][5:10]
    assert f"| {pf['event_time'][5:10]} 收盤 → {last} |" in md and "## 未知\n" in md


# ------------------------------------------------------------------------------------------------ consensus / surprise (B1), variant perception (B5)
# SYNTHETIC consensus numbers (test fixtures only; not real analyst estimates).
def consensus_file(env, as_of_day=15, items=None, sym="AAA", name="cons", tamper=False):
    raw = env["tmp"] / f"{name}-raw.html"
    raw.write_text("<html>synthetic consensus capture for tests</html>")
    data = {
        "symbol": sym,
        "provider": "synthetic-test",
        "as_of": f"{day(as_of_day)}T00:00:00Z",
        "source_uri": "https://example.com/consensus/AAA",
        "raw_path": str(raw),
        "raw_sha256": "0" * 64 if tamper else hashlib.sha256(raw.read_bytes()).hexdigest(),
        "fetched_at": f"{day(as_of_day)}T00:00:00Z",
        "items": items
        or [
            {
                "metric": "revenue",
                "period": "Q4 2026",
                "mean": 52e9,
                "low": 51e9,
                "high": 53e9,
                "n_analysts": 20,
                "unit": "USD",
                "currency": "USD",
                "basis": "IFRS",
            },
            {
                "metric": "gross_margin",
                "period": "4Q26",
                "mean": 61.5,
                "median": 61.4,
                "low": 60.5,
                "high": 62.0,
                "n_analysts": 18,
                "unit": "%",
                "basis": "IFRS",
            },
            {
                "metric": "eps",
                "period": "FY2026",
                "mean": 10.0,
                "low": 9.0,
                "high": 11.0,
                "n_analysts": 12,
                "unit": "USD",
                "currency": "USD",
                "basis": "IFRS",
            },
        ],
    }
    path = env["tmp"] / f"{name}.json"
    path.write_text(json.dumps(data))
    return ["--consensus-json", str(path)]


def surprise_case(env, name="sur", cons_day=15, narrative_extra=None, cfg_extra=None, items=None):
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"] = [doc_with(env, name, ACTUAL_Q4)]
    cfg["actuals"] = [
        {
            "metric": "revenue",
            "period": "4Q26",
            "value": 53.1,
            "unit": "USD billion",
            "evidence_id": "fundamental:doc1",
            "excerpt_locator": "line 2",
        }
    ]
    cfg.update(cfg_extra or {})
    n = copy.deepcopy(env["narrative"])
    n["claims"].append(
        {
            "kind": "fact",
            "text": "Q4 gross margin was 61.2%.",
            "evidence_ids": ["fundamental:doc1"],
            "actual": {"metric": "gross margin", "period": "Q4'26", "value": 61.2, "unit": "%"},
        }
    )
    n.update(narrative_extra or {})
    return go(env, name, cfg, narrative=n, extra=consensus_file(env, cons_day, items=items, name=name + "-c"))


def test_consensus_frozen_and_surprise_rendered(env):
    p, d = surprise_case(env)
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    ev = {e["id"]: e for e in rev["manifest_payload"]["evidence"]}
    c = ev["consensus:AAA"]
    assert (
        c["kind"] == "consensus"
        and c["payload"]["source_tier"] == "third-party aggregate"
        and c["payload"]["items"][0]["period"] == "4Q26"
    )
    assert (
        c["payload"]["items"][2]["period"] == "FY26"
        and len(c["payload"]["raw_sha256"]) == 64
        and c["event_time"] == f"{day(15)}T00:00:00Z"
    )
    rows = {r["metric"]: r for r in ev["surprise:AAA"]["payload"]["rows"]}
    assert ev["surprise:AAA"]["payload"]["algorithm"] == "surprise-v1" and set(rows) == {"revenue", "gross_margin"}
    rv, gm = rows["revenue"], rows["gross_margin"]
    assert (
        rv["status"] == "ok"
        and rv["surprise_abs"] == 1.1e9
        and rv["surprise_pct"] == round(1.1 / 52 * 100, 2)
        and rv["vs_range"] == "above"
    )
    assert (
        gm["status"] == "ok"
        and gm["surprise_abs"] == -0.3
        and gm["surprise_pct"] == round(-0.3 / 61.5 * 100, 2)
        and gm["vs_range"] == "within"
    )
    assert gm["actual"]["source"] == "claims[3]" and gm["actual"]["published_at"] == f"{day(EVENT_IDX)}T10:00:00Z"
    from consensus import replay_surprise

    assert replay_surprise(rev["manifest_payload"]) == {"surprise:AAA": True}
    tampered = copy.deepcopy(rev["manifest_payload"])
    next(e for e in tampered["evidence"] if e["id"] == "surprise:AAA")["payload"]["rows"][0]["surprise_abs"] = 0
    assert replay_surprise(tampered) == {"surprise:AAA": False}
    md = render(d / "rev.json").stdout
    assert "## Consensus & surprise" in md
    assert (
        "| revenue | 4Q26 | 52.00 billion USD（51.00 billion USD–53.00 billion USD，n=20） | 53.1 USD billion（fundamental:doc1） | +1.10 billion USD（+2.12%） | above |"
        in md
    ), md
    assert "| gross_margin | 4Q26 | 61.5%（60.5%–62%，n=18） | 61.2 %（claims[3]） | -0.3 pp（-0.49%） | within |" in md
    assert "| eps | FY26 | 10 USD（9 USD–11 USD，n=12） | n/a | n/a | n/a |" in md
    tail = md[md.index("## 揭露（Disclosure）") :]
    assert "`consensus:AAA`：third-party aggregate" in tail and "`surprise:AAA`：derived" in tail


def test_consensus_changes_manifest_hash_and_claim_precheck(env):
    p0, _ = go(env, "c0")
    p1, _ = go(env, "c1", extra=consensus_file(env, name="c1c"))
    assert p0.returncode == p1.returncode == 0, p1.stderr
    assert json.loads(p0.stdout)["manifest_sha256"] != json.loads(p1.stdout)["manifest_sha256"]
    n = copy.deepcopy(env["narrative"])
    n["claims"].append(
        {"kind": "fact", "text": "Analysts expected 4Q26 revenue of US$52 billion.", "evidence_ids": ["consensus:AAA"]}
    )
    p, d = go(env, "c2", narrative=n, extra=consensus_file(env, name="c2c"))
    assert p.returncode == 0, p.stderr
    assert json.loads((d / "rev.json").read_text())["identity"]["claim_precheck"][3]["numbers"] == "ok"
    n["claims"][3]["text"] = "Analysts expected 4Q26 revenue of US$5.2 billion."
    p, d = go(env, "c3", narrative=n, extra=consensus_file(env, name="c3c"))
    assert p.returncode != 0 and "claims[3] numeric mismatch" in p.stderr


def test_post_event_consensus_gives_no_surprise_number(env):
    p, d = surprise_case(env, "post", cons_day=25)  # observed after the actual was published (day 19)
    assert p.returncode == 0, p.stderr
    rows = next(
        e for e in json.loads((d / "rev.json").read_text())["manifest_payload"]["evidence"] if e["kind"] == "surprise"
    )["payload"]["rows"]
    assert all(r["status"].startswith("post-event") and r["surprise_abs"] is None for r in rows)
    assert "post-event (consensus observed" in render(d / "rev.json").stdout


def test_unit_mismatch_and_late_snapshot(env):
    items = [
        {
            "metric": "revenue",
            "period": "4Q26",
            "mean": 1600.0,
            "low": 1500.0,
            "high": 1700.0,
            "n_analysts": 5,
            "unit": "TWD billion",
            "currency": "TWD",
            "basis": "IFRS",
        }
    ]
    p, d = surprise_case(env, "fx", items=items)
    assert p.returncode == 0, p.stderr
    row = next(
        e for e in json.loads((d / "rev.json").read_text())["manifest_payload"]["evidence"] if e["kind"] == "surprise"
    )["payload"]["rows"][0]
    assert row["status"].startswith("n/a (unit differs") and row["surprise_abs"] is None  # never FX-converts
    p, _ = go(
        env, "late", extra=consensus_file(env, as_of_day=N + 3, name="latec")
    )  # observed after the manifest as_of
    assert p.returncode == 0 and "not point-in-time, skipped" in p.stderr and "consensus:AAA" not in p.stdout


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda d: d.update(raw_sha256="0" * 64), "refusing to freeze"),
        (lambda d: d["items"][0].update(low=53e9, high=54e9), "low <= mean <= high"),
        (lambda d: d["items"][0].update(period="next quarter"), "does not normalise"),
        (lambda d: d["items"][0].update(n_analysts=0), "n_analysts"),
        (lambda d: d["items"][0].update(currency="TWD"), "does not match unit"),
        (lambda d: d.pop("source_uri"), "source_uri must be a non-empty string"),
    ],
)
def test_consensus_json_refusals(env, mutate, msg):
    args = consensus_file(env, name="bad")
    path = Path(args[1])
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    p, _ = go(env, "cbad", extra=args)
    assert p.returncode != 0 and msg in p.stderr, p.stderr


VP = {
    "metric": "revenue",
    "period": "4Q26",
    "consensus_ref": "consensus:AAA",
    "our_view": 54e9,
    "direction": "above",
    "why": "backlog in the doc",
}


def test_variant_perception_and_pilot_position_rendered(env):
    vp = [
        VP,
        dict(
            VP, metric="gross_margin", our_view="margins hold despite mix", direction="in_line", why="no new fab costs"
        ),
    ]
    p, d = surprise_case(
        env,
        "vp",
        narrative_extra={"variant_perception": vp},
        cfg_extra={"pilot_position": {"position": "long", "size_note": "1 lot paper"}},
    )
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    assert (
        rev["research"]["variant_perception"][0]["direction"] == "above"
        and rev["research"]["disclosure"]["pilot_position"] == "long"
    )
    md = render(d / "rev.json").stdout
    assert "## Variant perception（推論；我們 vs 共識）" in md
    assert (
        "| revenue | 4Q26 | 52.00 billion USD（`consensus:AAA`） | 54.00 billion USD | above | backlog in the doc |"
        in md
    )
    assert (
        "| gross_margin | 4Q26 | 61.5%（`consensus:AAA`） | margins hold despite mix | in_line | no new fab costs |"
        in md
    )
    assert (
        "- 部位揭露：pilot_position long（1 lot paper），as_of " in md
        and "未記錄" not in md.split("部位揭露")[1].split("\n")[0]
    )


@pytest.mark.parametrize(
    "vp,msg",
    [
        (dict(VP, our_view=50e9), "inconsistent with direction 'above'"),
        (dict(VP, direction="in_line", our_view=55e9), "inconsistent with direction 'in_line'"),
        (dict(VP, why=""), "variant_perception[0].why"),
        (dict(VP, consensus_ref="fundamental:doc1"), "is not a frozen consensus evidence"),
        (dict(VP, period="1Q27"), "has no revenue 1Q27 item"),
        (dict(VP, direction="higher"), "direction must be one of"),
    ],
)
def test_variant_perception_refusals(env, vp, msg):
    p, d = surprise_case(env, "vpbad", narrative_extra={"variant_perception": [vp]})
    assert p.returncode != 0 and msg in p.stderr, p.stderr
    assert not (d / "rev.json").exists()


def test_pilot_position_invalid_and_undisclosed(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["pilot_position"] = "maybe"
    p, _ = go(env, "ppbad", cfg, narrative=env["narrative"])
    assert p.returncode != 0 and "pilot_position must be one of" in p.stderr
    cfg["pilot_position"] = "undisclosed"
    p, d = go(env, "ppund", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    assert "- 部位揭露：pilot_position undisclosed，as_of" in render(d / "rev.json").stdout


class FakeResp:
    def __init__(self, status, body):
        self.status, self.body = status, body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """Maps URL prefixes to (status, body); records every request. No network."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def open(self, req, timeout=None):
        import io
        import urllib.error

        url = req.full_url
        self.seen.append(url)
        st, body = next((v for k, v in self.routes.items() if url.startswith(k)), (404, b"not found"))
        if st != 200:
            raise urllib.error.HTTPError(url, st, "x", {}, io.BytesIO(body))
        return FakeResp(st, body)


def yahoo_page(trend):
    body = json.dumps({"quoteSummary": {"result": [{"earningsTrend": {"trend": trend}}], "error": None}})
    blob = json.dumps({"status": 200, "body": body}).replace("<", "\\u003c")
    return (
        '<html><script type="application/json" data-sveltekit-fetched data-url="https://query1.finance.yahoo.com/v10/finance/quoteSummary/TSM?'
        'formatted=true&amp;modules=earningsTrend%2CcalendarEvents">' + blob + "</script></html>"
    ).encode()


TREND = [
    {
        "period": "0q",
        "endDate": "2026-12-31",
        "earningsEstimate": {
            "avg": {"raw": 5.16},
            "low": {"raw": 5.1},
            "high": {"raw": 5.28},
            "numberOfAnalysts": {"raw": 8},
            "earningsCurrency": "USD",
        },
        "revenueEstimate": {
            "avg": {"raw": 1.705e12},
            "low": {"raw": 1.626e12},
            "high": {"raw": 1.753e12},
            "numberOfAnalysts": {"raw": 26},
            "revenueCurrency": "TWD",
        },
    },
    {
        "period": "0y",
        "endDate": "2026-12-31",
        "earningsEstimate": {
            "avg": 17.5,
            "low": 17.18,
            "high": 17.95,
            "numberOfAnalysts": 11,
            "earningsCurrency": "USD",
        },
        "revenueEstimate": {},
    },
    {"period": "+5y", "endDate": None, "growth": {"raw": 0.2}},
]


def test_yahoo_probe_parses_embedded_quote_summary(tmp_path):
    from consensus import fetch_yahoo_consensus

    op = FakeOpener({"https://finance.yahoo.com/quote/TSM/analysis/": (200, yahoo_page(TREND))})
    data, st = fetch_yahoo_consensus("2330", "TSM", tmp_path / "raw", adr=True, opener=op)
    assert st["status"] == "ok" and st["attempts"][0]["status"] == 200 and len(st["attempts"][0]["sha256"]) == 64
    assert [(i["metric"], i["period"], i["unit"], i["n_analysts"]) for i in data["items"]] == [
        ("eps_per_adr", "4Q26", "USD", 8),
        ("revenue", "4Q26", "TWD", 26),
        ("eps_per_adr", "FY26", "USD", 11),
    ]
    assert hashlib.sha256(Path(data["raw_path"]).read_bytes()).hexdigest() == data["raw_sha256"] and data[
        "source_uri"
    ].endswith("/TSM/analysis/")


def test_yahoo_probe_failures_are_clean(tmp_path):
    from consensus import fetch_yahoo_consensus

    cf = FakeOpener({"https://finance.yahoo.com/": (200, b"<html><title>Just a moment...</title>cf-chl</html>")})
    data, st = fetch_yahoo_consensus("2303", "UMC", tmp_path, opener=cf)
    assert data is None and st["status"] == "unavailable" and "Cloudflare" in st["reason"] and len(cf.seen) == 1
    limited = FakeOpener(
        {
            "https://finance.yahoo.com/": (200, b"<html>no embedded data</html>"),
            "https://fc.yahoo.com/": (404, b""),
            "https://query2.finance.yahoo.com/v1/test/getcrumb": (429, b"Too Many Requests"),
        }
    )
    data, st = fetch_yahoo_consensus("2303", "UMC", tmp_path, opener=limited)
    assert (
        data is None and "getcrumb HTTP 429" in st["reason"] and all("crumb=" not in a["url"] for a in st["attempts"])
    )

    class Down:
        def open(self, req, timeout=None):
            raise OSError("network unreachable")

    data, st = fetch_yahoo_consensus("2303", "UMC", tmp_path, opener=Down())
    assert data is None and st["status"] == "unavailable"


def test_consensus_live_without_mapping_does_not_fail(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["consensus_live"] = True
    cfg["adr_ticker"] = {}  # no mapping -> no network call
    p, d = go(env, "live", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    assert rev["identity"]["consensus_status"] == [
        {"symbol": "AAA", "status": "unavailable", "reason": "no adr_ticker mapping for a TW symbol"}
    ]
    assert "共識不可用：AAA：unavailable（no adr_ticker mapping for a TW symbol）" in render(d / "rev.json").stdout


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3Q26", "3Q26"),
        ("Q3 2026", "3Q26"),
        ("2026Q3", "3Q26"),
        ("3Q2026", "3Q26"),
        ("Q3'26", "3Q26"),
        ("FY2026", "FY26"),
        ("fy26", "FY26"),
        ("1H26", "1H26"),
        ("Q4", "Q4"),
    ],
)
def test_norm_period(raw, expected):
    from consensus import norm_period

    assert norm_period(raw) == expected


# ---------------------------------------------------------------------------------------------------------------------
# project_vault.py knowledge-graph edges + _kb/ops/compile/research_links.py (design 2026-10-21-vault-knowledge-graph-linking)

REAL_VAULT = Path(
    os.environ.get("NEXUS_VAULT") or os.path.expanduser("~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Nexus")
)
RESEARCH_LINKS = REAL_VAULT / "_kb" / "ops" / "compile" / "research_links.py"
COMPANY_HUMAN = """---
type: company
title: "Alpha（AAA）"
status: draft
created: 2026-10-21
updated: 2026-10-21
tags: [company, "AAA"]
origin: human
symbol: "AAA"
sector: "[[sectors/testsector]]"
sources: []
related: []
confidence: medium
---

# Alpha（AAA）

Human-written half.

## 研究時間線

<!-- research-timeline:start (generated by _kb/ops/compile/research_links.py; do not edit) -->
_尚未生成。_
<!-- research-timeline:end -->

Human footer.
"""
TAXONOMY = """---
type: reference
title: "Taxonomy"
---

```yaml
sectors:
  testsector: {title: "Test sector", keywords: ["widgets"]}
```

```yaml
concepts:
  revenue-test: {title: "Revenue concept", keywords: ["revenue"]}
  never-matched: {title: "Unmatched", keywords: ["zzzqqq"]}
  benchmark-concept: {title: "Benchmark", keywords: ["outperformed"]}
```
"""


@pytest.fixture()
def kvault(env):
    if not RESEARCH_LINKS.exists():
        pytest.skip("research_links.py not found in the Nexus vault")
    p, d = go(env, "kg", narrative=env["narrative"])
    assert p.returncode == 0, p.stderr + p.stdout
    v = env["tmp"] / "vault"
    (v / "_kb" / "wiki" / "companies").mkdir(parents=True)
    (v / "_kb" / "ops" / "compile").mkdir(parents=True)
    (v / "_kb" / "wiki" / "concepts").mkdir()
    (v / "AGENTS.md").write_text("# fake vault\n")
    (v / "_kb" / "wiki" / "_taxonomy.md").write_text(TAXONOMY)
    (v / "_kb" / "wiki" / "companies" / "aaa-alpha.md").write_text(COMPANY_HUMAN)
    (v / "_kb" / "wiki" / "concepts" / "benchmark-concept.md").write_text(
        "human concept page\n"
    )  # must never be overwritten
    (v / "_kb" / "ops" / "compile" / "research_links.py").write_bytes(RESEARCH_LINKS.read_bytes())
    rev = json.loads((d / "rev.json").read_text())
    rev["identity"]["read_across"] = [{"symbol": "CCC", "relation": "peer", "why": "test"}]  # pre-2026-10-29 revision
    (d / "rev.json").write_text(json.dumps(rev))
    return dict(v=v, rev=d / "rev.json", rid=rev["identity"]["revision_id"])


def project(k, *extra):
    return run(
        [
            str(HERE / "project_vault.py"),
            str(k["rev"]),
            "--title",
            "Test note",
            "--evidence-dir",
            "ev",
            "--out-dir",
            str(k["v"]),
            "--company-slug",
            "CCC=gamma",
            "--benchmark",
            "BBB",
            *extra,
        ]
    )


def note_path(k):
    return k["v"] / "Research" / "tw_stock-spot-AAA" / ("%s.md" % k["rid"])


def split_note(text):
    fm, body = text[4:].split("\n---\n", 1)
    return __import__("yaml").safe_load(fm), body


def test_projection_edges_links_section_and_stubs(kvault):
    k = kvault
    p = project(k)
    assert p.returncode == 0, p.stderr + p.stdout
    fm, body = split_note(note_path(k).read_text())
    assert fm["company"] == "[[companies/aaa-alpha]]" and fm["sector"] == "[[sectors/testsector]]"
    assert fm["concepts"] == [
        "[[concepts/revenue-test]]",
        "[[concepts/benchmark-concept]]",
    ]  # taxonomy ids only, in taxonomy order
    assert fm["peers"] == ["[[companies/ccc-gamma]]"] and "timeline" not in fm  # timelines retired 2026-10-21
    assert fm["summary"] == "AAA beat and outperformed the benchmark." and str(fm["next_review_at"]).startswith(
        "2026-10-05"
    )
    assert [c["date"] for c in fm["catalysts"]] == ["2026-09-10", "2026-10-05", "2026-11-12"]
    sec = body[body.index("\n## 知識連結\n") :]
    for t in (
        "companies/aaa-alpha",
        "sectors/testsector",
        "concepts/revenue-test",
        "companies/ccc-gamma",
    ):
        assert "[[%s]]" % t in sec
    wiki = k["v"] / "_kb" / "wiki"
    for t in ("sectors/testsector", "concepts/revenue-test", "companies/ccc-gamma"):
        sfm, _ = split_note((wiki / (t + ".md")).read_text())
        assert sfm["status"] == "stub" and sfm["origin"] == "agent", t
    assert not (wiki / "concepts" / "never-matched.md").exists()
    assert (wiki / "concepts" / "benchmark-concept.md").read_text() == "human concept page\n"  # existing page untouched
    company = (wiki / "companies" / "aaa-alpha.md").read_text()
    assert company.split("<!-- research-timeline:start")[0] == COMPANY_HUMAN.split("<!-- research-timeline:start")[0]
    assert company.endswith("<!-- research-timeline:end -->\n\nHuman footer.\n")
    assert "\\|%s]] | watch | AAA beat and outperformed the benchmark. | 2026-10-05 |" % k["rid"][:8] in company
    assert not (wiki / "timelines").exists()
    assert "\n### 催化劑\n" in company and (
        "| 2026-10-05 | monthly_revenue | revenue stays above 100 | [[Research/tw_stock-spot-AAA/%s\\|%s]] |"
        % (k["rid"], k["rid"][:8])
        in company
    )
    log = (wiki / "log.md").read_text()
    assert (
        "append-only" in log.lower()
        and "## [%s] research | AAA %s watch\n" % (date.today().isoformat(), k["rid"][:8]) in log
    )


def test_projection_theme_edges_come_from_claim_theme_ids(kvault):
    """Design 2026-10-29 §3.4: `themes` frontmatter and a 主題 line from the claims' theme_ids; no theme stubs."""
    k = kvault
    rev = json.loads(k["rev"].read_text())
    assert project(k).returncode == 0
    fm, _ = split_note(note_path(k).read_text())
    assert "themes" not in fm  # an untagged revision keeps its old shape
    note_path(k).unlink()
    rev["research"]["claims"][0]["theme_ids"] = ["cleanroom-capacity"]
    rev["research"]["claims"][1]["theme_ids"] = ["memory-supply-tightness", "cleanroom-capacity"]
    k["rev"].write_text(json.dumps(rev))
    p = project(k)
    assert p.returncode == 0, p.stderr + p.stdout
    fm, body = split_note(note_path(k).read_text())
    assert fm["themes"] == ["[[themes/cleanroom-capacity]]", "[[themes/memory-supply-tightness]]"]
    sec = body[body.rindex("## 知識連結") :]
    assert "- 主題：[[themes/cleanroom-capacity]]\n- 主題：[[themes/memory-supply-tightness]]" in sec
    assert not (k["v"] / "_kb" / "wiki" / "themes").exists()


def test_projection_never_overwrites_without_replace(kvault):
    k = kvault
    assert project(k).returncode == 0
    before = note_path(k).read_bytes()
    p = project(k)
    assert p.returncode != 0 and "refusing to overwrite" in p.stderr
    assert note_path(k).read_bytes() == before


def test_replace_keeps_body_and_refuses_on_hash_mismatch(kvault):
    k = kvault
    assert project(k).returncode == 0
    path = note_path(k)
    original = path.read_text()
    fm0, body0 = split_note(original)
    core0 = body0[: body0.rfind("\n## 知識連結\n")]
    p = project(k, "--replace")
    assert p.returncode == 0, p.stderr
    fm1, body1 = split_note(path.read_text())
    assert body1[: body1.rfind("\n## 知識連結\n")] == core0 and fm1 == fm0  # idempotent re-link
    path.write_text(
        original.replace('content_sha256: "%s"' % fm0["content_sha256"], 'content_sha256: "%s"' % ("0" * 64))
    )
    tampered = path.read_bytes()
    p = project(k, "--replace")
    assert p.returncode != 0 and "content_sha256 mismatch" in p.stderr
    assert path.read_bytes() == tampered


def test_research_links_block_regeneration_idempotent(kvault):
    k = kvault
    assert project(k).returncode == 0
    rl = [str(k["v"] / "_kb" / "ops" / "compile" / "research_links.py")]
    envv = dict(os.environ, NEXUS_VAULT=str(k["v"]))
    company = k["v"] / "_kb" / "wiki" / "companies" / "aaa-alpha.md"
    # wipe the generated block, then regenerate twice
    text = company.read_text()
    start = text.index("-->\n", text.index("research-timeline:start")) + 4
    company.write_text(text[:start] + "stale\n" + text[text.index("<!-- research-timeline:end") :])
    dry = subprocess.run([PY, *rl], env=envv, capture_output=True, text=True)
    assert dry.returncode == 0 and "dry-run: 1 company block(s)" in dry.stdout and "stale" in company.read_text()
    first = subprocess.run([PY, *rl, "--apply"], env=envv, capture_output=True, text=True)
    assert first.returncode == 0 and "applied: 1 company block(s)" in first.stdout, first.stderr
    snap = {p: p.read_bytes() for p in (k["v"] / "_kb" / "wiki").rglob("*.md")}
    second = subprocess.run([PY, *rl, "--apply"], env=envv, capture_output=True, text=True)
    assert second.returncode == 0 and "applied: 0 company block(s) changed" in second.stdout
    assert {p: p.read_bytes() for p in (k["v"] / "_kb" / "wiki").rglob("*.md")} == snap  # no change, no extra log line
    assert company.read_text() == text


def test_evidence_ref_keeps_remote_host_paths():
    sys.path.insert(0, str(HERE))
    import project_vault

    assert project_vault.evidence_ref("stormtrooper:research-runner/state/runs/20261022-1/") == (
        "stormtrooper:research-runner/state/runs/20261022-1/"
    )
    local = project_vault.evidence_ref(str(HERE))
    assert local.startswith("aquarium/") or local == str(HERE)  # aquarium-relative when under the aquarium root
