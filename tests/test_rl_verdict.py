"""RL verdict emitter tests.

Scaffolded with placeholders. The wave-5 implementation fills in the three-way verdict
emitter (PASS / FAIL / NEEDS-MORE-DATA) and its append-not-clobber semantics
on top of the existing ``scripts/gate_86_verdict.py`` golden-test pattern.
"""

import pytest


def test_three_way_verdict_classification():
    """Verdict classifier returns one of PASS / FAIL / NEEDS-MORE-DATA."""
    pytest.skip("implementation pending")


def test_appends_not_clobbers():
    """Verdict emitter appends to the audit log; never overwrites prior runs."""
    pytest.skip("implementation pending")
