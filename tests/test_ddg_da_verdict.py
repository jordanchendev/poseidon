"""Phase 92 Plan 92-04 — verdict.md keyword check (scaffold).

DDGDA-03 mandates verdict in {rescue, partial-help, no-effect}. This test
re-reads verdict.md from the smoke run output dir and asserts one of the
three keywords is present.
"""

from __future__ import annotations

import pytest


def test_verdict_contains_keyword():
    """DDGDA-03: verdict.md contains exactly one of rescue / partial-help / no-effect."""
    pytest.skip("scaffold — Plan 92-04 fills body")
