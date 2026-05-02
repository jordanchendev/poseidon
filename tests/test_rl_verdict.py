"""Phase 90 / Wave 5 — RL verdict emitter tests.

Wave 0 scaffold (Plan 90-01 Task 3). Wave 5 fills in the three-way verdict
emitter (PASS / FAIL / NEEDS-MORE-DATA) and its append-not-clobber semantics
on top of the existing ``scripts/gate_86_verdict.py`` golden-test pattern.
"""

import pytest


def test_three_way_verdict_classification():
    """Verdict classifier returns one of PASS / FAIL / NEEDS-MORE-DATA."""
    pytest.skip("Wave 5 implements")


def test_appends_not_clobbers():
    """Verdict emitter appends to the audit log; never overwrites prior runs."""
    pytest.skip("Wave 5 implements")
