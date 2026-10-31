"""Owner decision 2026-10-29 (design 2026-10-29 §4.5, step 7 recalibration): the narrative judgement runs deterministic
gates before JEV and takes a JEV majority vote. narrative_check.judge_record is the one path for the judgement, the
sweep and the control-set evaluation. No network.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_precheck.py
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import narrative_check  # noqa: E402
from claim_precheck import check_periods, extract_periods, quoted_media  # noqa: E402

NANYA = "整體 DRAM 供應相對吃緊，預期將延續至 116 年。"
UMC_NOTICE = (
    "6. Content of the report:\n「...UMC’s 2026 revenue is estimated to grow by nearly 21% year on year, with profit "
    "expected to increase by approximately 133%...」\n7. Cause of occurrence:\nThe Company does not comment on "
    "speculative reports or market rumors."
)
# post 2 of the §7 fixture (股癌, 2026-10-20) and the record N5 the design table expects from it
ASML_LINE = "- 目前規劃的 27/28 產出無法滿足客戶需求，且新增產能較容易落在 2028 "
N5 = "ASML 目前規劃的 2027 與 2028 產出無法滿足客戶需求"


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
        # acceptance 2026-10-30 N5: a consecutive two-digit slash pair names two years on the excerpt side
        (N5, ASML_LINE, "covered", []),
        (N5, "10/27 法說會：新增產能較容易落在 2028", "not_covered", [2027]),  # a date, not years
        (N5, "目前規劃的 27/29 產出不足，新增產能落在 2028", "not_covered", [2027]),  # not consecutive
        (N5, "良率 1.27/28 或 27/28% 的產出，新增產能落在 2028", "not_covered", [2027]),  # numbers that go on
        ("ASML 的新增產能落在 27/28", "新增產能較容易落在 2028", "no_claim_years", []),  # never on the claim side
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


# ------------------------------------------------------------------------------------------------ judge_record
class Votes:
    """A JEV stand-in that answers each call with the next (label, probability of that label)."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def judge(self, claim, source, locator, symbol, as_of):
        label, p = self.answers[self.calls]
        self.calls += 1
        rest = round((1 - p) / 2, 4)
        probs = {k: p if k == label else rest for k in narrative_check.LABELS}
        return {"label": label, "confidence": p, "probabilities": probs, "latency_ms": 1}


def judge(jev, claim="Micron 預期伺服器記憶體內容成長率略低於先前預期", source="content growth was lower"):
    return narrative_check.judge_record(jev, claim, source, "claims[0]", "2303", "2026-10-29T00:00:00Z")


def test_period_gate_decides_without_jev():
    jev = Votes()
    r = judge(jev, "記憶體供給吃緊會延續到至少 2028 年", NANYA)
    assert (r["label"], r["precheck"], r["votes"], jev.calls) == ("says_nothing", "period", [], 0)


def test_media_gate_decides_without_jev():
    jev = Votes()
    r = judge(jev, "聯電 2026 年營收預估年增近 21%", UMC_NOTICE)
    assert (r["label"], r["precheck"], r["votes"], jev.calls) == ("says_nothing", "media", [], 0)


def test_majority_label_stands():
    jev = Votes(("supports", 0.9), ("says_nothing", 0.6), ("supports", 0.9))
    r = judge(jev)
    assert (r["label"], r["precheck"], r["votes"], jev.calls) == (
        "supports",
        "none",
        ["supports", "says_nothing", "supports"],
        narrative_check.JEV_VOTES,
    )
    assert r["p_label"] == pytest.approx((0.9 + 0.2 + 0.9) / 3, abs=1e-4)


def test_majority_below_half_probability_is_says_nothing():  # nctl-06: supports at 0.47 to 0.49
    r = judge(Votes(("supports", 0.49), ("supports", 0.47), ("says_nothing", 0.48)))
    assert r["label"] == "says_nothing" and r["votes"] == ["supports", "supports", "says_nothing"]


def test_no_majority_is_says_nothing():  # supports still averages 0.52, so only the missing majority decides
    r = judge(Votes(("supports", 0.9), ("contradicts", 0.34), ("says_nothing", 0.34)))
    assert r["p_label"] >= narrative_check.MIN_WIN_PROBABILITY and r["label"] == "says_nothing"


def test_contradicts_majority_stands():
    assert judge(Votes(*[("contradicts", 0.8)] * 3))["label"] == "contradicts"


def test_supported_part_in_record_takes_the_period_gate_not_the_media_gate():
    """Owner decision 2026-10-30 (R3): judgement (a) asks whether supported_part is part of the record's claim. A year
    the record never names cannot be (period); a record that is itself a rumor says nothing about whether it is (media).
    Judgement (b) reads the cited source with every gate, as the whole-claim judgement does."""
    record = "市場傳聞某公司 2026 年營收將成長 15%"
    official = "某公司公告 2026 年營收將成長 15%"
    part = "某公司 2026 年營收將成長 15%"
    assert narrative_check._media_gate(part, record)  # the gate (a) leaves out would fire
    jev = Votes(*[("supports", 0.9)] * (2 * narrative_check.JEV_VOTES))
    r = narrative_check.judge_part(jev, part, record, official, "claims[1]", "2303", "2026-10-29T00:00:00Z")
    assert [(r[k]["label"], r[k]["precheck"]) for k in ("in_record", "backed")] == [("supports", "none")] * 2
    jev = Votes()
    r = narrative_check.judge_part(
        jev, "某公司 2027 年營收將成長 15%", record, official, "claims[1]", "2303", "2026-10-29T00:00:00Z"
    )
    assert (r["in_record"]["precheck"], r["backed"]["precheck"], jev.calls) == ("period", "period", 0)


def test_answer_without_probabilities_is_refused():
    class NoProbabilities(Votes):
        def judge(self, *a):
            return {**super().judge(*a), "probabilities": None}

    with pytest.raises(ValueError, match="probabilities"):
        judge(NoProbabilities(*[("supports", 0.9)] * 3))
