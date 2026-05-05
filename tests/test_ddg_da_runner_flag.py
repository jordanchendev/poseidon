"""Phase 92 Plan 92-02 — AutoResearchRunner.use_ddg_da branch dispatch (scaffolds).

Bodies filled by Plan 92-02. Verifies D-03 (use_ddg_da flag dispatches to
PoseidonDDGDA path) AND D-05 (the dispatch happens BEFORE autoresearch_context()
so _AUTORESEARCH_ACTIVE remains False — DDG-DA's internal mutations would
trip ImmutabilityViolationError if guard were active).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch  # noqa: F401  # used by Plan 92-02 when filling bodies

import pytest

from poseidon.autoresearch.guard import (
    _AUTORESEARCH_ACTIVE,  # noqa: F401  # D-05 invariant — Plan 92-02 asserts .get(False) is False
)


def test_use_ddg_da_default_false_retains_existing_path():
    """D-03: when use_ddg_da is unset, AutoResearchRunner enters with autoresearch_context() (existing behavior)."""
    pytest.skip("scaffold — Plan 92-02 fills body")


def test_use_ddg_da_true_dispatches_to_ddg_da_path():
    """D-03: when use_ddg_da=True, runner branches BEFORE autoresearch_context() to PoseidonDDGDA."""
    pytest.skip("scaffold — Plan 92-02 fills body")


def test_immutability_not_violated_in_ddg_da_path():
    """D-05: assert _AUTORESEARCH_ACTIVE.get(False) is False inside DDG-DA dispatch — must not trip ImmutabilityViolationError."""
    pytest.skip("scaffold — Plan 92-02 fills body")
