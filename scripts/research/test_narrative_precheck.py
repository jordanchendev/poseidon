"""Owner decision 2026-10-29 (design 2026-10-29 §4.5, step 7 recalibration): the deterministic period and media checks
that run before JEV on a narrative claim. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_precheck.py
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from claim_precheck import check_periods, extract_periods, quoted_media  # noqa: E402

NANYA = "整體 DRAM 供應相對吃緊，預期將延續至 116 年。"
UMC_NOTICE = (
    "6. Content of the report:\n「...UMC’s 2026 revenue is estimated to grow by nearly 21% year on year, with profit "
    "expected to increase by approximately 133%...」\n7. Cause of occurrence:\nThe Company does not comment on "
    "speculative reports or market rumors."
)


def years(text):
    return [(p["years"], p["at_least"]) for p in extract_periods(text) if p["kind"] == "period" and p["years"]]


# ------------------------------------------------------------------------------------------------ period extraction
@pytest.mark.parametrize(
    "text,expected",
    [
        ("預期將延續至 116 年。", [2027]),
        ("民國116年度", [2027]),
        ("延續至１１６年", [2027]),
        ("FY27 毛利率", [2027]),
        ("2H26 demand", [2026]),
        ("3Q28 出貨", [2028]),
        ("Q4'26 revenue", [2026]),
        ("for calendar 2026 and 2027.", [2026, 2027]),
        ("demand through 2026–2028", [2026, 2027, 2028]),
        ("the floor for gross margins in fiscal 2027.", [2027]),
        ("Micron 2026 會計年度第四季", [2026]),
    ],
)
def test_period_years_in_ad(text, expected):
    assert sorted({y for ys, _ in years(text) for y in ys}) == expected


@pytest.mark.parametrize(
    "text", ["revenue of US$2027 million", "margin was 2026.5%", "第三季營收成長 20%", "1,000 千元"]
)
def test_numbers_that_are_not_years(text):
    assert years(text) == []


@pytest.mark.parametrize(
    "text,at_least",
    [
        ("記憶體供給吃緊會延續到至少 2028 年", True),
        ("至少到 2028 年", True),
        ("2028 年以後才緩解", True),
        ("supply stays tight at least through 2028", True),
        ("記憶體供給吃緊會延續至 2028 年", False),
    ],
)
def test_lower_bound_forms(text, at_least):
    assert years(text) == [([2028], at_least)]


# ------------------------------------------------------------------------------------------------ period gate
@pytest.mark.parametrize(
    "claim,excerpt,status,missing",
    [
        ("記憶體供給吃緊會延續到至少 2028 年", NANYA, "not_covered", [2028]),  # nctl-25: the excerpt ends in 2027
        ("南亞科預期整體 DRAM 供應吃緊將延續至 2027 年", NANYA, "covered", []),  # nctl-26: 民國 116 年 = 2027
        ("記憶體供給吃緊會延續到至少 2028 年", "supply constrained in 2027 and 2030", "covered", []),
        (
            "到 2026 年約有 20% 將由南亞塑膠協助織造",
            "預計至 2027 年，將有 20%由本公司協助織造。",
            "not_covered",
            [2026],
        ),
        ("2027 與 2028 年產業 NAND 位元出貨成長", "For calendar 2027 and 2028, NAND grows", "covered", []),
        ("2027 與 2028 年產業 NAND 位元出貨成長", "For calendar 2027, NAND grows", "not_covered", [2028]),
        ("Micron 2026 會計年度第四季資料中心 SSD 營收", "Data center SSD revenue in fiscal Q4", "no_excerpt_years", []),
        ("無塵室不足是產業限制", "In 2026 we add cleanroom space", "no_claim_years", []),
    ],
)
def test_period_cover(claim, excerpt, status, missing):
    r = check_periods(claim, excerpt)
    assert (r["status"], r["missing"]) == (status, missing), r


# ------------------------------------------------------------------------------------------------ media gate
def test_clarification_notice_restating_the_claims_number_is_restated():
    r = quoted_media("聯電 2026 年營收預估年增近 21%", UMC_NOTICE)
    assert r["restated"] and {m["kind"] for m in r["markers"]} == {"quote", "no_comment", "rumor"}


def test_spaced_chinese_notice_is_restated():
    notice = "一、報 導 內 容：某公司 2026 年營收將成長 15%。二、本公司對市場傳聞不予評論。"
    assert quoted_media("某公司 2026 年營收將成長 15%", notice)["restated"]


@pytest.mark.parametrize(
    "claim,excerpt",
    [
        ("聯電 2026 年營收預估年增近 25%", UMC_NOTICE),  # the claim's number is not the reported one
        ("聯電將擴大新加坡產能", UMC_NOTICE),  # no number to be restated
        ("Revenue grew 21% year over year", "Revenue grew 21% year over year in the quarter."),  # no notice wording
    ],
)
def test_not_restated(claim, excerpt):
    assert not quoted_media(claim, excerpt)["restated"]
