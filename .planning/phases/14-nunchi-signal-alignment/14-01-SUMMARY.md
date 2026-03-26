---
phase: 14-nunchi-signal-alignment
plan: 01
subsystem: backtest
tags: [composite-score, equity-valuation, short-selling, nunchi, metrics]

# Dependency graph
requires:
  - phase: 13-regime-classification-optional-gated
    provides: "compute_composite_score, BacktestPortfolio, nunchi_crypto_1h.json baseline"
provides:
  - "Lenient composite score formula (15% dd threshold, capital turnover ratio)"
  - "Correct short position equity valuation in BacktestPortfolio"
  - "Nunchi-proven config defaults (ATR 5.5, BB 0.85)"
affects: [14-02, autoresearch-loop, regime-search]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Threshold-based penalty: max(0, value - threshold) * coefficient"
    - "Direction-aware position valuation in record_equity_point"

key-files:
  created:
    - tests/test_portfolio_short.py
  modified:
    - src/poseidon/backtest/metrics.py
    - src/poseidon/backtest/portfolio.py
    - src/poseidon/strategies/configs/nunchi_crypto_1h.json
    - tests/test_composite_score.py

key-decisions:
  - "Drawdown penalty uses 15% threshold per Nunchi D-15 -- drawdown under 15% is free"
  - "Turnover penalty uses capital turnover ratio instead of raw trade count per D-16"
  - "Short equity = (entry_price - current_price) * quantity per D-12"
  - "ATR multiplier 5.5 and BB threshold 0.85 from Nunchi proven values"

patterns-established:
  - "Composite score formula: sharpe * trade_factor - dd_penalty - turnover_penalty with lenient thresholds"

requirements-completed: [ALIGN-01, ALIGN-02, ALIGN-03]

# Metrics
duration: 5min
completed: 2026-03-26
---

# Phase 14 Plan 01: Nunchi Signal Alignment - Value Fixes Summary

**Composite score aligned with Nunchi lenient formula (15% dd threshold, capital turnover ratio), short equity valuation fixed, and config defaults updated (ATR 5.5, BB 0.85)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-26T10:17:52Z
- **Completed:** 2026-03-26T10:22:59Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Composite score formula now matches Nunchi: drawdown under 15% incurs zero penalty, turnover uses capital ratio not raw count
- Short positions correctly valued in equity curve: (entry_price - current_price) * quantity
- Nunchi config updated with proven ATR multiplier 5.5 and BB squeeze threshold 0.85
- 16 total tests (12 composite score + 4 short equity) all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix composite score formula and update tests** - `c44a7cd` (feat)
2. **Task 2: Fix short position equity valuation and update Nunchi config defaults** - `b71b68a` (feat)

_Both tasks used TDD: RED (failing tests) -> GREEN (implementation) -> verified._

## Files Created/Modified
- `src/poseidon/backtest/metrics.py` - Updated compute_composite_score with lenient dd penalty and capital turnover ratio
- `src/poseidon/backtest/portfolio.py` - Fixed record_equity_point for short position valuation
- `src/poseidon/strategies/configs/nunchi_crypto_1h.json` - ATR 5.5, BB threshold 0.85
- `tests/test_composite_score.py` - 12 tests covering new formula behavior
- `tests/test_portfolio_short.py` - 4 tests for short/long/mixed equity valuation

## Decisions Made
- Drawdown penalty threshold set to 15% (D-15) -- this is the key change that makes autoresearch less harsh on normal drawdowns
- Turnover penalty uses capital turnover ratio with 500x threshold (D-16) -- most strategies will never trigger this
- Default initial_capital=100000.0 when not provided in metrics dict for backward compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] CostModel requires market parameter**
- **Found during:** Task 2 (test fixture setup)
- **Issue:** CostModel constructor requires `market` field not documented in plan
- **Fix:** Added `market="crypto_spot"` to test fixture
- **Files modified:** tests/test_portfolio_short.py
- **Verification:** All tests pass
- **Committed in:** b71b68a (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Trivial test fixture fix. No scope creep.

## Issues Encountered
None beyond the CostModel fixture fix above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Composite score formula ready for autoresearch optimization loops
- Short equity valuation ready for bear signal backtesting (Plan 14-02)
- Nunchi config defaults ready for VotingStrategy factory consumption

## Self-Check: PASSED

- All 6 files verified to exist
- Both commits (c44a7cd, b71b68a) found in git log
- No stubs or TODOs found in modified files
- All 16 tests passing

---
*Phase: 14-nunchi-signal-alignment*
*Completed: 2026-03-26*
