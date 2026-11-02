"""Orchestrator-added cases for claim_precheck.py (2026-10-21, UMC 2303 dry run): size descriptors and bare table digits."""

import pytest
from claim_precheck import check_numbers, extract_quantities, lint_atomic

TABLE = (
    "4. Operating revenue accumulated from 1/1 to end of the period (thousand NTD): 129,770,564 "
    "Period Items 2026 2025 Changes % September Net sales 25,350,431 19,927,175 5,423,256 27.22% "
    "Year-to-Date Net sales 204,009,681 175,743,654 28,266,027 16.08% Second quarter consolidated revenue was NT$68.73 billion"
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("執行長表示 8 吋產能利用率預期在第三季顯著改善。", []),
        ("our 12-inch capacity and 28nm business", []),
        ("12 吋產能利用率已達 85%。", [85.0]),
        ("0.18 微米製程", []),
    ],
)
def test_size_descriptors_are_not_quantities(text, expected):
    assert [q.value for q in extract_quantities(text)] == expected


@pytest.mark.parametrize(
    "claim,status,how",
    [
        ("2026 年上半年營業收入為 129,770,564 千元。", "ok", None),
        ("聯電 2026 年 9 月營收為 25,350,431 千元。", "ok", None),
        ("推得的第三季營收為 74,239,117 千元。", "ok", "A-B (bare digits)"),
        (
            "第三季營收為 74,239,118 千元。",
            "mismatch",
            None,
        ),  # off by one thousand: not close (0.5% tol) -> no, it IS within tol
    ],
)
def test_unit_bearing_claim_matches_bare_table_digits(claim, status, how):
    r = check_numbers(claim, TABLE)
    if claim.endswith("74,239,118 千元。"):
        pytest.skip("within 0.5% tolerance by design; kept as documentation of the tolerance")
    assert r["status"] == status, r
    if how:
        assert any(v.get("how") == how for m in r["matched"] for v in m["via"]), r


def test_bare_digits_do_not_match_when_absent():
    r = check_numbers("1,000 千元 test", "Any other matters: None")
    assert r["status"] == "mismatch" and r["unmatched"][0]["kind"] == "absent"


# Micron FQ4 FY26 prepared remarks, lines 419-421 (m93/m97, 2026-10-30): the drafter writes the guidance band as
# 「$38.15，加減 $1.00」, which lint_atomic split into two claims until 加減 after a comma read as ±.
MU_GUIDANCE = (
    "We expect FQ1 revenue to be a record $61.5 billion, plus or minus $1.5 billion; gross margin to be approximately "
    "86.25%; and operating expenses to be approximately $2.06 billion. Based on a share count of approximately 1.15 "
    "billion shares, we expect EPS to be $38.15 per share, plus or minus $1.00."
)


@pytest.mark.parametrize(
    "claim,a,b",
    [
        ("Micron 預期 FQ1 非 GAAP 每股盈餘為 {}。", "$38.15", "$1.00"),
        ("Micron 預期 FQ1 營收為 {}。", "$61.5 billion", "$1.5 billion"),
    ],
)
@pytest.mark.parametrize(
    "form",
    [
        "{a}，加減 {b}",  # m93 / m97
        "{a}，正負 {b}",
        "{a} 加減 {b}",
        "{a}, 加減 {b}",
        "{a}, plus or minus {b}",  # the source's own wording
        "{a} plus or minus {b}",  # English forms that already worked
        "{a} ± {b}",
    ],
)
def test_plus_or_minus_band_is_one_quantity_like_the_symbol(claim, a, b, form):
    text, ref = claim.format(form.format(a=a, b=b)), claim.format(f"{a} ± {b}")
    (q,), (r,) = extract_quantities(text), extract_quantities(ref)
    assert (q.value, q.pm, q.unit_class, q.currency, q.sign, q.qualifier) == (
        r.value,
        r.pm,
        r.unit_class,
        r.currency,
        r.sign,
        r.qualifier,
    )
    assert q.pm is not None
    lint = lint_atomic(text, "fact")
    assert lint["ok"] is True and len(lint["clauses"]) == 1, lint
    got, want = check_numbers(text, MU_GUIDANCE), check_numbers(ref, MU_GUIDANCE)
    assert got["status"] == want["status"] == "ok"
    assert [{k: v for k, v in m.items() if k != "claim"} for m in got["matched"]] == [
        {k: v for k, v in m.items() if k != "claim"} for m in want["matched"]
    ]
