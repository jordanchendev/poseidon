---
phase: 10
slug: voting-strategy-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `pytest tests/test_voting_strategy.py tests/test_composite_score.py tests/test_vote_combinator.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_voting_strategy.py tests/test_composite_score.py tests/test_vote_combinator.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | VOTE-02 | unit | `pytest tests/test_vote_combinator.py -x` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | VOTE-04 | unit | `pytest tests/test_composite_score.py -x` | ❌ W0 | ⬜ pending |
| 10-02-01 | 02 | 2 | VOTE-01, VOTE-05, VOTE-06 | unit | `pytest tests/test_voting_strategy.py::TestVotingStrategy tests/test_voting_strategy.py::TestATRTrailingStop tests/test_voting_strategy.py::TestPositionSizing -x` | ❌ W0 | ⬜ pending |
| 10-02-02 | 02 | 2 | VOTE-03 | unit + integration | `pytest tests/test_voting_strategy.py::TestNunchiSignals -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_voting_strategy.py` — stubs for VOTE-01, VOTE-03, VOTE-05, VOTE-06
- [ ] `tests/test_vote_combinator.py` — stubs for VOTE-02
- [ ] `tests/test_composite_score.py` — stubs for VOTE-04

*Existing `tests/conftest.py` provides environment setup. No new shared fixtures needed beyond test-specific feature DataFrames.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
