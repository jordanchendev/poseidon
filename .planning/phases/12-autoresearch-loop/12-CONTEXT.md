# Phase 12: AutoResearch Loop - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Autonomous experiment runner that iterates strategy mutations, evaluates via backtest, and logs results — all without modifying scoring formula, backtest runner, or feature engine code. Modeled after Nunchi's 103-experiment automated flow (Sharpe 2.7→21.4).

Requirements: AUTO-03, AUTO-04, AUTO-05, AUTO-06

</domain>

<decisions>
## Implementation Decisions

### StrategyMutator — Thin Wrapper Over Existing Pipeline
- **D-01:** StrategyMutator is a thin wrapper, NOT a new search engine. It delegates to VotingStrategyFactory + Optuna (from Phase 11)
- **D-02:** `StrategyMutator.mutate_via_optuna(trial)` delegates to `VotingStrategyFactory.from_trial()` for Bayesian-guided mutation
- **D-03:** `StrategyMutator.mutate_random(seed)` generates random config within PARAM_BOUNDS for baseline comparison
- **D-04:** All generated configs must pass existing Pydantic validation (`VotingStrategy.validate_config()`)

### Three-Layer Architecture Enforcement
- **D-05:** Runtime enforcement via `contextvar` flag `_AUTORESEARCH_ACTIVE` — set True at AutoResearchRunner entry, checked in protected modules
- **D-06:** FeatureEngine, BacktestRunner, RiskEngine raise `ImmutabilityViolationError` on `__setattr__` / method mutation while `_AUTORESEARCH_ACTIVE` is True
- **D-07:** Only RuleConfig JSON (strategy parameters) is mutable during autoresearch runs — enforced at runtime, not linter-only
- **D-08:** NOT using OS-level read-only or import hooks — contextvar + explicit checks are sufficient and testable

### AutoResearchRunner Celery Integration
- **D-09:** Single long-running Celery task `autoresearch_run`, NOT one-task-per-experiment
- **D-10:** Task receives `SearchConfig` + market list, internally loops: per-market calls `ParameterSearchPipeline.run()`
- **D-11:** Heartbeat: update Celery task state after each market completes (progress queryable via API)
- **D-12:** Graceful stop: check Redis flag `autoresearch:stop:{task_id}` between markets — allows external stop without killing worker
- **D-13:** Per-market failure isolation: catch exception + log + continue to next market (don't abort entire run)
- **D-14:** Minimum 10 consecutive experiments must run unattended (AUTO-05 success criteria)

### Experiment Results & Convergence
- **D-15:** After run completes, query ExperimentTracker for `status=passed` (WFE gate passed) experiments
- **D-16:** Rank by `composite_score`, produce `autoresearch_report.json` with top-N configs + summary stats
- **D-17:** NO auto-deployment — human reviews report and decides whether to update production config
- **D-18:** `program.md` guidance layer = SearchConfig parameterization (n_trials, min_wfe, market list) in this phase; more sophisticated AI guidance deferred

### Claude's Discretion
- Exact contextvar implementation pattern
- autoresearch_report.json schema details
- Heartbeat update frequency and format
- Error message wording for ImmutabilityViolationError

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Experiment Infrastructure (Phase 11 — direct dependencies)
- `src/poseidon/backtest/param_search.py` — ParameterSearchPipeline, SearchConfig, SearchResult; the core loop that AutoResearchRunner wraps
- `src/poseidon/backtest/voting_strategy_factory.py` — VotingStrategyFactory, PARAM_BOUNDS; StrategyMutator delegates here
- `src/poseidon/backtest/experiment_tracker.py` — ExperimentTracker CRUD; where results are persisted
- `src/poseidon/backtest/holdout.py` — HoldoutConfig; called by pipeline, not by AutoResearchRunner directly

### Protected Layer (immutability targets)
- `src/poseidon/data/feature_engine.py` — FeatureEngine; must be read-only during autoresearch
- `src/poseidon/backtest/runner.py` — BacktestRunner; must be read-only during autoresearch
- `src/poseidon/risk/engine.py` — RiskEngine; must be read-only during autoresearch

### Strategy Layer (mutable layer)
- `src/poseidon/strategies/voting_strategy.py` — VotingStrategy class; the strategy being mutated
- `src/poseidon/strategies/configs/nunchi_crypto_1h.json` — Nunchi baseline config; reference for valid structure

### Celery Infrastructure
- `src/poseidon/workers/cpu_tasks.py` — Existing Celery tasks; pattern reference for autoresearch_run task
- `src/poseidon/workers/celery_app.py` — Celery app configuration

### Requirements
- `.planning/REQUIREMENTS.md` §AUTO-03..AUTO-06 — Acceptance criteria for this phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ParameterSearchPipeline.run()` — Already orchestrates holdout→Optuna→WFE→logging per market; AutoResearchRunner wraps this
- `VotingStrategyFactory.from_trial()` — Already handles Optuna suggest API; StrategyMutator delegates here
- `PARAM_BOUNDS` — Already defines searchable parameter space; StrategyMutator.mutate_random() samples from this
- `ExperimentTracker.save()/query_by_*()` — Already handles persistence; report generation queries this
- `SearchConfig` dataclass — Already parameterizes n_trials, min_wfe, holdout, walk_forward; reuse for autoresearch config

### Established Patterns
- Celery tasks in `cpu_tasks.py` follow: import → configure → execute → persist pattern
- `BacktestRunner` and `ParameterSearchPipeline` use dataclass configs (not dict-based)
- Error handling: catch-log-continue for per-item failures (e.g., per-market in pipeline)

### Integration Points
- `AutoResearchRunner` → `ParameterSearchPipeline.run()` (per market)
- `StrategyMutator` → `VotingStrategyFactory.from_trial()` + `.from_config()`
- `autoresearch_run` Celery task → registered in `cpu_tasks.py` or new `research_tasks.py`
- Immutability guard → `contextvar` checked in FeatureEngine/BacktestRunner/RiskEngine

</code_context>

<specifics>
## Specific Ideas

- 參考 Nunchi 103 次自動實驗模式：每個 market 跑完整 Optuna study，而非全部 market 混在一起
- Nunchi 的成功（Sharpe 2.7→21.4）來自自動化迭代 + WFE 驗證，這個 pipeline 已在 Phase 11 建好
- report 格式參考 Nunchi 的結果整理：per-market best config + composite score + WFE pass rate

</specifics>

<deferred>
## Deferred Ideas

- AI-driven guidance layer (sophisticated program.md that adapts search strategy) — future phase
- Auto-deployment of best config to production — future phase, needs safety gate
- Multi-timeframe simultaneous search — current focus is per-market sequential
- Regime-aware autoresearch — Phase 13 (gated on regime classifier)

</deferred>

---

*Phase: 12-autoresearch-loop*
*Context gathered: 2026-03-26*
