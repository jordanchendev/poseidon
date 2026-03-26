---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Strategy Pivot
status: Milestone complete
stopped_at: Completed 14-03-PLAN.md
last_updated: "2026-03-26T10:42:25.791Z"
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 12
  completed_plans: 12
---

# State: Poseidon

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-25)
**Core value:** Reliably produce quality trading signals and deliver them to Thalassa
**Current focus:** Phase 14 — nunchi-signal-alignment

## Current Position

Phase: 14
Plan: Not started

## Velocity

- v1.0: 9 phases completed (infrastructure through transformer model)
- v2.0: 4 phases planned (voting strategy -> experiment infra -> autoresearch -> regime)

## Decisions

- [v2.0]: Pure TA prediction confirmed dead end — pivot to rule-based voting + automated search
- [v2.0]: Nunchi auto-researchtrading validated approach: simple signals + automated iteration
- [v2.0]: XGBoost repurposed from direction prediction to regime classification
- [v2.0]: Regime classification gated — must outperform static baseline or auto-disabled
- [v2.0]: Holdout (last 20%) must be locked before any experiments run (irreversible)
- [Phase 10]: Vote combinator uses sum() not all()/any() to avoid short-circuit and get accurate count
- [Phase 10]: cum_return resolution appends d suffix to match FeatureEngine column naming
- [Phase 10]: VotingStrategy sets quantity_pct=0.08 as strategy-level intent; BacktestRunner SizingConfig controls final sizing per D-10
- [Phase 10]: ATR trailing stop evaluated BEFORE vote counting to ensure close signals take priority
- [Phase 11]: Default optimization metric changed from sharpe_ratio to composite_score for BayesianOptimizer
- [Phase 12-autoresearch-loop]: ContextVar + class decorator for immutability guard (D-05 through D-08)
- [Phase 12-autoresearch-loop]: StrategyMutator is pure delegation to VotingStrategyFactory (D-01)
- [Phase 12]: CostModel fallback: zero-cost model for unknown markets instead of KeyError
- [Phase 12]: Per-market failure isolation with continue-on-error pattern for autoresearch loops
- [Phase 13-regime-classification-optional-gated]: RegimeRouter mutates strategy attributes (not re-instantiate) for trailing stop state preservation
- [Phase 13-regime-classification-optional-gated]: Per-regime search varies only min_votes/position_pct (2 params), not full PARAM_BOUNDS
- [Phase 13-regime-classification-optional-gated]: Outperformance gate uses strict comparison (>) -- equal scores disable regime routing
- [Phase 14]: Composite score dd_penalty uses 15% threshold (max(0, dd-0.15)*0.05) per Nunchi D-15
- [Phase 14]: Turnover penalty uses capital turnover ratio, not raw trade count per D-16
- [Phase 14]: Short equity valuation: (entry_price - current_price) * qty per D-12
- [Phase 14]: Default atr_multiplier changed from 2.0 to 5.5 per D-05 (Nunchi alignment)
- [Phase 14]: Cooldown uses <= 2 check (blocks 2 full bars after exit)
- [Phase 14]: Bear sub_signals use indicator_below DSL condition type
- [Phase 14]: Bear sub_signals use indicator_below DSL condition type with inverted thresholds
- [Phase 14]: Factory BB threshold corrected 0.2->0.85, ATR range 1.5-3.0->3.0-8.0, default 2.0->5.5
- [Phase 14]: RegimeRouter+Search expanded from 2 to 4 params per regime (adding bear_min_votes, bear_position_pct)

## Blockers

(None)

## Accumulated Context

### Strategy Pivot Research (2026-03-25)

- **autoresearch** (karpathy): 3-layer pattern — prepare.py (fixed), train.py (AI-modifiable), program.md (guidance)
- **auto-researchtrading** (Nunchi): 103 experiments, Sharpe 2.7->21.4, 6-signal voting, simplification > complexity
- **Key signals**: Momentum x2, EMA crossover, RSI(8), MACD(14,23,9), Bollinger squeeze
- **Voting rule**: 4/6 majority with fixed 8% position sizing
- **Anti-features**: pyramiding, variable sizing, multi-timeframe, funding overlay all hurt performance
- **Realistic target**: Sharpe 1.0-3.0 on properly validated strategies (Nunchi 21.4 is overfit)

### Roadmap Evolution

- Phase 14 added: Nunchi Signal Alignment — autoresearch 0% pass rate root cause traced to implementation gaps vs Nunchi original (ATR 2.0 vs 5.5, missing RSI exit/signal flip/cooldown, BB threshold reversed, composite score formula too harsh, no SHORT support)

## Last Session

- **Stopped at:** Completed 14-03-PLAN.md
- **Resume file:** None
- **Next step:** `/gsd:plan-phase 10`

---
*State created: 2026-03-20*
*Last updated: 2026-03-25 after v2.0 roadmap creation*
