"""SEED-002 C2 claim prechecks, ported from local_dev/scripts/research/test_offline_tooling.py (stdlib only)."""

import pytest

from poseidon.decision_loop.claim_lint import check_numbers, extract_quantities, lint_atomic


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
        "TSMC (TWSE: 2330, NYSE: TSM) on Oct. 15, 2026",
        "3Q26 vs 2Q26 and FQ1-27 guidance, 1H26",
        "LPDDR6, HBM4E and DDR5 in fiscal Q4 2026",
        "2026年10月16日，預計於2027年10月1日完成",
        "2330 財報後與 0050 同幅下跌",
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
        ),
        ("鴻海 2025 年合併營收為新台幣 8,099 億元。", "2025年合併營收新臺幣8.099兆元。", "mismatch", "scale"),
        (
            "Micron expects server unit growth in the high-teens percentage range in calendar 2026 and 2027, "
            "driven mainly by AI training clusters.",
            "We expect server unit growth in the high-teens percentage range in both calendar 2026 and 2027.",
            "no_numbers",
            None,
        ),
        (
            "美光伺服器 LPDDR SOCAMM 產品組合營收在第四季年增超過一倍。",
            "Our industry leading server LPDDR SOCAMM portfolio revenue more than doubled sequentially in the "
            "fourth quarter of 2026.",
            "ok",
            None,
        ),
        # derivable quantities must not be flagged
        (
            "Micron's fiscal Q4 2026 operating cash flow was more than seven times the level of the same quarter "
            "a year earlier.",
            "Operating cash flow of $43.97 billion versus $25.39 billion for the prior quarter and $5.73 billion "
            "for the same period last year",
            "ok",
            None,
        ),
        (
            "台積電第三季毛利為新台幣 9,923.79 億元，年增 68.6%。",
            "(Unit: NT$ million, except for EPS) 3Q26 Amounta 3Q25 Amount YoY Inc. (Dec.) % Net sales 1,494,243 "
            "989,918 50.9 Gross profit 992,379 588,543 68.6",
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
    result = check_numbers(claim, excerpt)
    assert result["status"] == status, result
    if kind:
        assert result["unmatched"][0]["kind"] == kind


def test_check_numbers_derived_flag_and_period_mismatch():
    result = check_numbers(
        "Operating cash flow was more than seven times the year-ago level.",
        "Operating cash flow of $43.97 billion versus $5.73 billion for the same period last year",
    )
    assert result["matched"][0]["match"] == "derived" and result["matched"][0]["via"][0]["how"] == "A/B"
    result = check_numbers(
        "美光第四季 NAND 營收年增 42%。", "Sequentially, NAND revenue increased 42%. Up 526% year over year."
    )
    assert result["status"] == "ok" and result["period_mismatch"][0]["claim_period"] == "yoy"


@pytest.mark.parametrize(
    "claim,ok",
    [
        (
            "Micron expects server unit growth in the high-teens percentage range in calendar 2026 and 2027, "
            "driven mainly by AI training clusters.",
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
