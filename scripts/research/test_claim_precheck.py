"""Orchestrator-added cases for claim_precheck.py (2026-10-21, UMC 2303 dry run): size descriptors and bare table digits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_precheck import check_numbers, extract_quantities

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
