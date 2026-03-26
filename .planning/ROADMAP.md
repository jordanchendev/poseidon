# Roadmap: Poseidon

**Created:** 2026-03-20
**Updated:** 2026-03-26
**Granularity:** standard

## Milestones

- **v1.0 Core Trading Signal Platform** - Phases 1-9 (complete)
- **v2.0 Strategy Pivot -- Rule-Based Voting + Automated Search** - Phases 10-13 (in progress)

<details>
<summary>v1.0 Core Platform (Phases 1-9) -- COMPLETE</summary>

Phases 1-9 delivered the full trading signal platform: Docker infrastructure, data ingestion (4 markets), feature engine, model engine (XGBoost + Transformer), strategy DSL (RuleStrategy + ModelStrategy), backtest engine with walk-forward validation, risk engine, signal delivery via Redis Streams, REST API, volume features, and 5m crypto data. All 37 v1.0 requirements fulfilled.

</details>

## v2.0 Strategy Pivot (Phases 10-13)

**Milestone Goal:** Pivot from ML direction prediction (confirmed dead end) to rule-based voting strategies with automated parameter search. Deliver a working Nunchi-inspired 6-signal voting system, persistent Optuna parameter optimization, an autonomous experiment loop, and optional regime-gated strategy selection.

## Phases

- [x] **Phase 10: Voting Strategy Foundation** - VotingStrategy class, DSL vote type, Nunchi 6-signal config, composite scoring, exit logic (completed 2026-03-26)
- [x] **Phase 11: Experiment Infrastructure** - Optuna RDBStorage, ExperimentTracker, VotingStrategyFactory, holdout protocol, parameter search pipeline (completed 2026-03-26)
- [ ] **Phase 12: AutoResearch Loop** - StrategyMutator, 3-layer architecture enforcement, AutoResearchRunner Celery task, immutability boundary
- [ ] **Phase 13: Regime Classification (Optional, Gated)** - XGBoostRegimeModel classifier, RegimeRouter, outperformance gate vs static baseline

## Phase Details

### Phase 10: Voting Strategy Foundation
**Goal**: Users can run a multi-signal voting strategy through the existing backtest and live pipelines, producing signals when a configurable majority of sub-signals agree
**Depends on**: v1.0 complete (Phase 9)
**Requirements**: VOTE-01, VOTE-02, VOTE-03, VOTE-04, VOTE-05, VOTE-06
**Success Criteria** (what must be TRUE):
  1. VotingStrategy wraps 6 RuleStrategy sub-signals and emits a BUY/SELL signal only when >= min_votes (default 4/6) agree -- verified via backtest on crypto 1h data
  2. DSL condition engine accepts a `vote` condition type with `min_votes` parameter (e.g., `{"vote": {"conditions": [...], "min_votes": 4}}`) and evaluates it correctly, including nested sub-conditions
  3. All six Nunchi-derived signal configs (dual momentum, EMA crossover, RSI(8), MACD(14,23,9), Bollinger squeeze) are loadable as RuleStrategy JSON and produce non-trivial signals on historical crypto data
  4. Composite scoring formula (sharpe * sqrt(min(trades/50,1.0)) - dd_penalty - turnover_penalty, with hard cutoffs at <10 trades / >50% drawdown / >50% capital loss) is callable from backtest metrics and returns expected scores on known inputs
  5. ATR-based trailing stop exit fires correctly during backtest bar-by-bar simulation, and fixed position sizing (default 8%) is applied uniformly to all entries
**Plans:** 2/2 plans complete
Plans:
- [x] 10-01-PLAN.md -- DSL vote combinator, new condition evaluators, column resolution fixes, composite scoring
- [x] 10-02-PLAN.md -- VotingStrategy class with ATR trailing stop, Nunchi 6-signal JSON config, integration

### Phase 11: Experiment Infrastructure
**Goal**: Persistent experiment tracking and parameter search pipeline are operational so that every Optuna trial and experiment run is recorded, resumable, and validated against walk-forward and holdout gates
**Depends on**: Phase 10
**Requirements**: PARM-01, PARM-02, PARM-03, PARM-04, PARM-05, AUTO-01, AUTO-02
**Success Criteria** (what must be TRUE):
  1. Optuna studies persist to PostgreSQL via RDBStorage in a separate `optuna` schema -- studies survive service restarts and are queryable via SQL
  2. ExperimentTracker records every experiment run (config JSON, metrics dict, timestamps) in PostgreSQL with a unique experiment ID, and past runs are retrievable by ID or by date range
  3. VotingStrategyFactory creates a valid VotingStrategy instance from an Optuna trial parameter dict or raw JSON config -- round-trip test: factory output produces identical backtest results to hand-constructed strategy
  4. Walk-forward validation (WFE >= 50%) is enforced as a mandatory gate on all parameter search results -- trials failing WFE are recorded but marked as rejected
  5. Holdout data split (last 20% of each dataset) is defined, documented, and enforced before any experiment runs -- code raises an error if optimization touches holdout date ranges
  6. Per-market/timeframe parameter search runs independently and trial count is capped (50-100 per search) to prevent overfitting
**Plans:** 3/3 plans complete
Plans:
- [x] 11-01-PLAN.md -- ExperimentRecord model, Alembic migration (experiments table + optuna schema), ExperimentTracker repository, HoldoutConfig
- [x] 11-02-PLAN.md -- VotingStrategyFactory with from_config/from_trial, PARAM_BOUNDS, round-trip tests
- [x] 11-03-PLAN.md -- BayesianOptimizer RDBStorage upgrade, ParameterSearchPipeline with WFE gate and holdout enforcement

### Phase 12: AutoResearch Loop
**Goal**: An autonomous experiment runner iterates strategy mutations, evaluates them via backtest, and logs results -- all without modifying the scoring formula, backtest runner, or feature engine code
**Depends on**: Phase 11
**Requirements**: AUTO-03, AUTO-04, AUTO-05, AUTO-06
**Success Criteria** (what must be TRUE):
  1. StrategyMutator generates valid VotingStrategy JSON configs by varying signal parameters within defined bounds -- all generated configs pass Pydantic validation and produce runnable strategies
  2. Three-layer architecture is enforced at runtime: FeatureEngine, BacktestRunner, and RiskEngine code paths are read-only during any autoresearch run; only RuleConfig JSON is mutable
  3. AutoResearchRunner executes as a Celery task that completes a full cycle (mutate config -> backtest -> evaluate -> log) and writes results to ExperimentTracker -- at least 10 consecutive experiments run unattended without error
  4. Immutability boundary is provably enforced: attempting to import or call any evaluation-layer modification API from within the autoresearch context raises an explicit error
**Plans**: TBD

### Phase 13: Regime Classification (Optional, Gated)
**Goal**: Market regime detection selects per-regime VotingStrategy configurations, but only if regime routing demonstrably outperforms the static no-regime baseline on out-of-sample data
**Depends on**: Phase 12
**Requirements**: RGME-01, RGME-02, RGME-03
**Success Criteria** (what must be TRUE):
  1. XGBoostRegimeModel classifies market state into regime categories (trending/ranging/volatile/low-vol) from feature data with measurable accuracy on held-out periods
  2. RegimeRouter maps detected regime to a specific VotingStrategy configuration (min_votes, position size multiplier) and the correct config is applied during backtest and live evaluation
  3. Outperformance gate enforced: regime-conditional strategy must beat static VotingStrategy baseline on OOS data by a positive margin, or regime routing is automatically disabled and the system falls back to the static configuration
**Plans**: TBD

## Progress

**Execution Order:** Phases execute in numeric order: 10 -> 11 -> 12 -> 13

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 10. Voting Strategy Foundation | 2/2 | Complete    | 2026-03-26 |
| 11. Experiment Infrastructure | 3/3 | Complete    | 2026-03-26 |
| 12. AutoResearch Loop | 0/TBD | Not started | - |
| 13. Regime Classification | 0/TBD | Not started | - |

---
*Roadmap created: 2026-03-20*
*Updated for v2.0: 2026-03-25*
*Phase 10 planned: 2026-03-26*
*Phase 11 planned: 2026-03-26*
