---
phase: 14-nunchi-signal-alignment
plan: 03
subsystem: backtest
tags: [optuna, regime-router, voting-strategy, bear-signals, param-search]

# Dependency graph
requires:
  - phase: 14-02
    provides: "VotingStrategy with bear_sub_signals, _bear_min_votes, _bear_position_pct, SHORT signal emission"
provides:
  - "VotingStrategyFactory generates bidirectional configs with bull + bear sub_signals"
  - "PARAM_BOUNDS expanded with bear_min_votes, bear_position_pct, and corrected ATR range"
  - "RegimeRouter overrides 4 strategy attributes per regime (bull + bear)"
  - "RegimeSearchPipeline searches 4 params per regime via Optuna"
affects: [autoresearch, regime-classification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inverted bear signal conditions: indicator_below for bearish DSL conditions"
    - "4-param regime search: min_votes, position_pct, bear_min_votes, bear_position_pct"

key-files:
  created: []
  modified:
    - src/poseidon/backtest/voting_strategy_factory.py
    - src/poseidon/strategies/regime_router.py
    - src/poseidon/backtest/regime_search.py
    - src/poseidon/strategies/voting_strategy.py
    - tests/test_voting_strategy_factory.py
    - tests/test_regime.py
    - tests/test_regime_search.py

key-decisions:
  - "Bear sub_signals use indicator_below DSL condition type with inverted thresholds"
  - "BB squeeze threshold corrected from 0.2 to 0.85 in factory per D-14"
  - "ATR multiplier range expanded from (1.5,3.0) to (3.0,8.0) per D-06"
  - "from_config atr_multiplier default changed from 2.0 to 5.5 per D-05"

patterns-established:
  - "Bear signal generation: mirror bull signals with indicator_below and direction=below"
  - "4-param regime override: RegimeRouter mutates both bull and bear attributes"

requirements-completed: [ALIGN-08, ALIGN-09, ALIGN-10]

# Metrics
duration: 7min
completed: 2026-03-26
---

# Phase 14 Plan 03: Factory, Router, Search Bear Support Summary

**Factory generates bidirectional configs with inverted bear signals, RegimeRouter overrides 4 params, and Optuna searches 4 params per regime**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-26T10:27:03Z
- **Completed:** 2026-03-26T10:34:36Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- PARAM_BOUNDS expanded with bear_min_votes (3,6,int), bear_position_pct (0.03,0.12,float), and ATR range corrected to (3.0,8.0)
- Factory _build_config_from_params generates bear_sub_signals with 6 inverted conditions (indicator_below, direction below)
- BB squeeze threshold corrected from 0.2 to 0.85 in factory
- RegimeRouter overrides 4 strategy attributes (min_votes, position_pct, bear_min_votes, bear_position_pct) per regime
- RegimeSearchPipeline searches 4 params per regime via Optuna
- 47 tests pass across all three test files

## Task Commits

Each task was committed atomically:

1. **Task 1: Update PARAM_BOUNDS, factory bear signal generation, and BB/ATR defaults** - `db13eac` (feat)
2. **Task 2: Update RegimeRouter and RegimeSearchPipeline for 4-param bear support** - `d383f60` (feat)

_TDD RED commits included for both tasks before GREEN implementation._

## Files Created/Modified
- `src/poseidon/backtest/voting_strategy_factory.py` - PARAM_BOUNDS expanded, bear signal generation, BB 0.85, ATR default 5.5, to_config_dict with bear fields
- `src/poseidon/strategies/regime_router.py` - DEFAULT_REGIME_CONFIGS with bear params, evaluate() overrides 4 attributes
- `src/poseidon/backtest/regime_search.py` - RegimeSearchConfig with bear ranges, 4-param Optuna objective
- `src/poseidon/strategies/voting_strategy.py` - get_feature_specs handles indicator_below for bear signals
- `tests/test_voting_strategy_factory.py` - 21 tests covering bear signals, PARAM_BOUNDS, BB threshold, round-trip
- `tests/test_regime.py` - 15 tests covering bear regime configs and 4-param overrides
- `tests/test_regime_search.py` - 11 tests covering 4-param search and bear config ranges

## Decisions Made
- Bear sub_signals use `indicator_below` DSL condition type (confirmed exists in conditions.py)
- BB squeeze threshold corrected from 0.2 to 0.85 in factory (matching nunchi_crypto_1h.json config)
- ATR multiplier range expanded from (1.5,3.0) to (3.0,8.0) per D-06 Nunchi alignment
- from_config atr_multiplier default changed from 2.0 to 5.5 per D-05

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed get_feature_specs to handle indicator_below type**
- **Found during:** Task 1 (factory implementation)
- **Issue:** VotingStrategy.get_feature_specs() only checked for `indicator_above` type when extracting cum_return, rsi, macd_histogram features. Bear signals using `indicator_below` would not register their feature specs.
- **Fix:** Changed condition checks from `cond_type == "indicator_above"` to `cond_type in ("indicator_above", "indicator_below")` for all three indicator types.
- **Files modified:** src/poseidon/strategies/voting_strategy.py
- **Verification:** Factory tests pass with bear signal round-trip
- **Committed in:** db13eac (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential for bear signals to have correct feature specs registered. No scope creep.

## Issues Encountered
- Test environment required `uv pip install pytest` since pytest was only available system-wide, not in the worktree venv.

## Known Stubs
None - all bear signal fields are fully wired through factory, router, and search pipeline.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 14 (nunchi-signal-alignment) complete: all 3 plans executed
- Factory, router, and search pipeline fully support bidirectional strategies
- Ready for milestone completion review

---
*Phase: 14-nunchi-signal-alignment*
*Completed: 2026-03-26*
