---
phase: 12
slug: autoresearch-loop
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `python -m pytest tests/backtest/test_autoresearch.py -x -q` |
| **Full suite command** | `python -m pytest tests/backtest/ tests/strategies/ tests/workers/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/backtest/test_autoresearch.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/backtest/ tests/strategies/ tests/workers/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | AUTO-03 | unit | `python -m pytest tests/backtest/test_strategy_mutator.py -x -q` | ❌ W0 | ⬜ pending |
| 12-01-02 | 01 | 1 | AUTO-04 | unit | `python -m pytest tests/backtest/test_immutability_guard.py -x -q` | ❌ W0 | ⬜ pending |
| 12-02-01 | 02 | 2 | AUTO-05 | unit | `python -m pytest tests/workers/test_autoresearch_runner.py -x -q` | ❌ W0 | ⬜ pending |
| 12-02-02 | 02 | 2 | AUTO-06 | integration | `python -m pytest tests/backtest/test_immutability_boundary.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/backtest/test_strategy_mutator.py` — stubs for AUTO-03
- [ ] `tests/backtest/test_immutability_guard.py` — stubs for AUTO-04, AUTO-06
- [ ] `tests/workers/test_autoresearch_runner.py` — stubs for AUTO-05
- [ ] `tests/backtest/test_immutability_boundary.py` — integration stubs for AUTO-06

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 10 consecutive experiments unattended | AUTO-05 | Requires running Celery worker + DB + Redis | Deploy on stormtrooper, start autoresearch task, verify 10+ experiments logged in DB |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
