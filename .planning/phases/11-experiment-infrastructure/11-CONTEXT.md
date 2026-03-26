# Phase 11: Experiment Infrastructure - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Persistent experiment tracking and parameter search pipeline. Optuna RDBStorage for study persistence, ExperimentTracker for experiment logging, VotingStrategyFactory for trial-to-strategy conversion, holdout protocol enforcement, and per-market/timeframe parameter search with WFE gate. Scope: infrastructure only — automated iteration loop is Phase 12.

</domain>

<decisions>
## Implementation Decisions

### Optuna Persistence
- **D-01:** Upgrade existing `BayesianOptimizer` — add optional `storage` parameter to `optimize()`. When provided, `optuna.create_study(storage=storage)` persists to PostgreSQL. Without it, remains in-memory (backward compatible).
- **D-02:** Use Poseidon's existing PostgreSQL instance with a dedicated `optuna` schema (`CREATE SCHEMA optuna`). Alembic migration creates the schema; Optuna manages its own tables within.
- **D-03:** Optimization metric switches from `sharpe_ratio` (current default) to `composite_score` (Phase 10's D-08) for all VotingStrategy parameter searches.

### ExperimentTracker
- **D-04:** New SQLAlchemy model `Experiment` in `src/poseidon/backtest/` with fields: `id (UUID)`, `study_name`, `config_json (JSONB)`, `metrics_json (JSONB)`, `composite_score`, `wfe_score`, `status (passed/rejected/running)`, `market`, `interval`, `created_at`, `updated_at`.
- **D-05:** Optional Optuna linkage via `optuna_study_name` + `optuna_trial_number` columns (no foreign key — Optuna manages its own tables).
- **D-06:** Repository pattern matching existing `BacktestRepository`: `save()`, `get_by_id()`, `list_by_date_range()`, `list_by_market()`.
- **D-07:** Trials failing WFE < 50% are recorded with `status=rejected` (not discarded — data is valuable for analysis).

### Holdout Protocol
- **D-08:** Percentage-based split: last 20% of each dataset reserved as holdout. Not a fixed date — adapts to different market/timeframe data lengths.
- **D-09:** `HoldoutConfig` dataclass with `holdout_pct=0.20` and `locked=True` flag. Once holdout boundary is computed for a study, it's persisted in ExperimentTracker metadata and cannot change.
- **D-10:** Runtime enforcement: `BayesianOptimizer.optimize()` checks if OHLCV data exceeds holdout boundary. Raises `HoldoutViolationError` if optimization data touches holdout range.
- **D-11:** Walk-forward validation (WFE >= 50%) is a mandatory gate. Trials pass WFE before being marked `status=passed`.

### VotingStrategyFactory & Search Scope
- **D-12:** `VotingStrategyFactory.from_trial(trial: optuna.Trial) -> VotingStrategy` — uses Optuna suggest API to generate parameter values within defined bounds.
- **D-13:** Searchable parameters: signal periods (RSI period, EMA short/long, MACD fast/slow/signal, Bollinger period), min_votes (3-6), ATR multiplier (1.5-3.0), position_pct (0.05-0.15).
- **D-14:** Not searchable (fixed): signal types (6 Nunchi signals), scoring formula (D-08 locked), exit logic type (ATR trailing stop).
- **D-15:** Each `(market, interval)` combination gets its own Optuna study. 50-100 trials per study (configurable, not hardcoded).

### Claude's Discretion
- Exact Alembic migration structure for optuna schema
- ExperimentTracker index strategy (by market+interval, by date, by score)
- VotingStrategyFactory parameter bound definitions (exact min/max for each signal period)
- Whether HoldoutConfig is stored in config.json or per-study in DB
- API endpoints for experiment queries (if any needed this phase)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Optimizer
- `src/poseidon/backtest/optimizer.py` — Current BayesianOptimizer with in-memory Optuna (upgrade target)
- `src/poseidon/backtest/runner.py` — BacktestRunner used by optimizer for each trial

### Strategy Layer (Phase 10 outputs)
- `src/poseidon/strategies/voting_strategy.py` — VotingStrategy class that factory must produce
- `src/poseidon/strategies/configs/nunchi_crypto_1h.json` — Baseline config to parameterize
- `src/poseidon/backtest/metrics.py` — `compute_composite_score()` (optimization target)

### Existing Patterns
- `src/poseidon/backtest/repository.py` — BacktestRepository pattern to follow for ExperimentTracker
- `src/poseidon/backtest/walk_forward.py` — WalkForwardAnalyzer with WFE computation (gate logic)
- `src/poseidon/backtest/schemas.py` — BacktestConfig/BacktestResult Pydantic models

### Database
- `alembic/` — Migration directory for schema changes

### Research
- `.planning/research/ARCHITECTURE.md` — 3-layer pattern (immutable/mutable/guidance)
- `.planning/research/PITFALLS.md` — Overfitting warnings, WFE thresholds

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BayesianOptimizer`: Already wraps Optuna — only needs `storage` parameter addition
- `BacktestRepository`: Repository pattern with SQLAlchemy — ExperimentTracker follows same pattern
- `WalkForwardAnalyzer.analyze()`: Returns WFE score — directly usable as gate check
- `compute_composite_score()`: Ready to use as optimization metric (Phase 10)
- `VotingStrategy`: Accepts config dict — factory just needs to produce valid config from trial params

### Established Patterns
- Repository pattern: `__init__(session)`, `save()`, `get_by_id()`, `list_*()` methods
- Alembic migrations: one migration file per phase, `upgrade()/downgrade()` functions
- Pydantic schemas: Config/Result pairs for validation
- SQLAlchemy models: declarative base with `__tablename__`, JSONB for flexible fields

### Integration Points
- `backtest/optimizer.py`: Add `storage` param to BayesianOptimizer
- `backtest/__init__.py`: Export new classes (ExperimentTracker, VotingStrategyFactory, HoldoutConfig)
- `alembic/versions/`: New migration for experiments table + optuna schema
- `backtest/walk_forward.py`: WFE check integrated into parameter search pipeline

</code_context>

<specifics>
## Specific Ideas

- Phase 10 CONTEXT D-08: composite score is THE optimization metric — this phase wires it into the actual optimization pipeline
- Nunchi 103 experiments reference: the parameter search here is the systematic version of what Nunchi did manually
- ExperimentTracker is the bridge to Phase 12 AutoResearch — it must log enough data for the mutation loop to read past results

</specifics>

<deferred>
## Deferred Ideas

- StrategyMutator (automated config mutation) — Phase 12
- AutoResearchRunner Celery task — Phase 12
- 3-layer immutability enforcement — Phase 12
- Regime-conditional parameter sets — Phase 13
- optuna-dashboard web UI — v3 (DASH-01)
- API endpoints for experiment browsing — can add later if needed

</deferred>

---

*Phase: 11-experiment-infrastructure*
*Context gathered: 2026-03-26*
