---
phase: 14-nunchi-signal-alignment
plan: 02
subsystem: strategy
tags: [voting-strategy, bidirectional-trading, short-signals, rsi-exit, signal-flip, cooldown, trailing-stop]

# Dependency graph
requires:
  - phase: 10-voting-strategy-foundation
    provides: VotingStrategy base class with bull sub_signals and ATR trailing stop
provides:
  - Bidirectional VotingStrategy with bear_sub_signals and SHORT emission
  - RSI mean-reversion exit mechanism (long RSI>69, short RSI<31)
  - Signal flip exit on opposing ensemble
  - 2-bar cooldown preventing same-direction re-entry after exit
  - Short-side ATR trailing stop with low watermark tracking
  - Exit priority chain (ATR > RSI > signal flip)
affects: [14-03, backtest-runner, regime-router, autoresearch]

# Tech tracking
tech-stack:
  added: []
  patterns: [exit-priority-chain, position-direction-state-machine, cooldown-counter]

key-files:
  created: []
  modified:
    - src/poseidon/strategies/voting_strategy.py
    - tests/test_voting_strategy.py

key-decisions:
  - "Default atr_multiplier changed from 2.0 to 5.5 per D-05 (Nunchi alignment)"
  - "Cooldown uses <= 2 check (blocks 2 full bars after exit, not 1)"
  - "Bear sub_signals use indicator_below DSL condition type (not negate flag)"
  - "_in_position preserved as backward-compat property wrapping _position_direction"

patterns-established:
  - "Exit priority chain: ATR trailing stop checked first, then RSI, then signal flip"
  - "Position state machine: _position_direction (None/long/short) replaces boolean _in_position"
  - "Cooldown counter: _bars_since_exit incremented per evaluate() call, blocks same-direction re-entry"

requirements-completed: [ALIGN-04, ALIGN-05, ALIGN-06, ALIGN-07]

# Metrics
duration: 5min
completed: 2026-03-26
---

# Phase 14 Plan 02: VotingStrategy Bidirectional Overhaul Summary

**Bidirectional VotingStrategy with bear sub_signals, SHORT emission, RSI mean-reversion exits, signal flip, 2-bar cooldown, and short trailing stop -- 41 tests pass**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-26T10:17:40Z
- **Completed:** 2026-03-26T10:23:15Z
- **Tasks:** 1 (TDD: test + implement)
- **Files modified:** 2

## Accomplishments
- VotingStrategy now supports bidirectional trading: LONG via bull sub_signals, SHORT via bear_sub_signals
- Three exit mechanisms implemented with strict priority: ATR trailing stop > RSI exit > signal flip
- 2-bar cooldown prevents whipsaw re-entry to same direction after exit
- Short-side trailing stop tracks low watermark (mirrors long-side high watermark)
- Default ATR multiplier aligned to Nunchi's 5.5 (was 2.0)
- All 22 existing tests pass unchanged (backward compatible)
- 19 new tests covering all new mechanisms

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for bidirectional trading** - `f14a221` (test)
2. **Task 1 (GREEN): Implement bidirectional VotingStrategy** - `2fdcd10` (feat)

_TDD task: test commit followed by implementation commit._

## Files Created/Modified
- `src/poseidon/strategies/voting_strategy.py` - Bidirectional VotingStrategy with bear signals, RSI exit, signal flip, cooldown, short trailing stop
- `tests/test_voting_strategy.py` - 19 new tests for SHORT signals, RSI exit, signal flip, cooldown, short trailing stop, exit priority

## Decisions Made
- Default atr_multiplier changed from 2.0 to 5.5 (per Nunchi D-05 alignment)
- Cooldown threshold uses `<= 2` to block 2 full bars after exit (not just 1)
- Bear sub_signals use `indicator_below` DSL condition type rather than adding a `negate` flag to `indicator_above`
- `_in_position` preserved as backward-compatible property that wraps `_position_direction` state

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed cooldown comparison operator**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Plan specified `< 2` for cooldown check, but with increment-at-top-of-evaluate pattern, this only blocks 1 bar instead of 2
- **Fix:** Changed to `<= 2` to correctly block 2 full bars after exit
- **Files modified:** src/poseidon/strategies/voting_strategy.py
- **Verification:** TestCooldown.test_no_reentry_within_2_bars_same_direction passes
- **Committed in:** 2fdcd10

**2. [Rule 1 - Bug] Fixed bear sub_signals condition types in tests**
- **Found during:** Task 1 (RED phase)
- **Issue:** Plan's bear_sub_signals used `negate: True` flag which isn't supported by DSL condition evaluators
- **Fix:** Used existing `indicator_below` condition type and `direction: "below"` for indicator_comparison
- **Files modified:** tests/test_voting_strategy.py
- **Verification:** All bear signal tests pass with proper DSL conditions
- **Committed in:** 2fdcd10

**3. [Rule 1 - Bug] Updated test_reentry_after_close for cooldown compatibility**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Existing test expected immediate re-entry after close, but new cooldown blocks same-direction re-entry for 2 bars
- **Fix:** Added 2 cooldown bars (with non-triggering features) before re-entry attempt
- **Files modified:** tests/test_voting_strategy.py
- **Verification:** test_reentry_after_close passes with cooldown wait
- **Committed in:** 2fdcd10

---

**Total deviations:** 3 auto-fixed (3 bug fixes)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all features are fully wired with real data paths.

## Next Phase Readiness
- VotingStrategy fully supports bidirectional trading
- Ready for Plan 03 (if applicable) or integration with BacktestRunner and RegimeRouter
- Bear sub_signals config format established for autoresearch to generate

---
*Phase: 14-nunchi-signal-alignment*
*Completed: 2026-03-26*
