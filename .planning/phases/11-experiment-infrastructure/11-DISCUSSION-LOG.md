# Phase 11: Experiment Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 11-experiment-infrastructure
**Areas discussed:** Optuna persistence, ExperimentTracker design, Holdout protocol, Search parameter scope

---

## Gray Area Selection

User selected "Other" — requested Claude's recommendations for all 4 areas ("這些面向，你有任何建議的設計嗎？我沒有什麼想法").

## Optuna Persistence

| Option | Description | Selected |
|--------|-------------|----------|
| Upgrade existing BayesianOptimizer | Add storage param, backward compatible | ✓ |
| New PersistentOptimizer class | Separate class for DB-backed optimization | |

**User's choice:** Upgrade existing (Claude's recommendation)
**Notes:** Minimal change (~3-5 lines), maintains backward compatibility

## ExperimentTracker Design

| Option | Description | Selected |
|--------|-------------|----------|
| New SQLAlchemy model + Repository | Follows BacktestRepository pattern | ✓ |
| Extend BacktestRepository | Add experiment fields to existing model | |

**User's choice:** New model + Repository (Claude's recommendation)
**Notes:** Separate concern, JSONB for config/metrics flexibility

## Holdout Protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Percentage-based (last 20%) | Adapts to data length | ✓ |
| Fixed date cutoff | Same date across all markets | |

**User's choice:** Percentage-based (Claude's recommendation)
**Notes:** Different markets have different data ranges, percentage adapts automatically

## Search Parameter Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Signal periods + min_votes + ATR + position_pct | Comprehensive but bounded | ✓ |
| Signal periods only | Minimal search space | |

**User's choice:** Comprehensive (Claude's recommendation)
**Notes:** 6 signal types fixed, scoring formula fixed, only parameters vary

## Claude's Discretion

- Alembic migration structure
- ExperimentTracker index strategy
- VotingStrategyFactory exact parameter bounds
- HoldoutConfig storage location
- API endpoints for experiments (if needed)

## Deferred Ideas

- StrategyMutator — Phase 12
- AutoResearchRunner — Phase 12
- optuna-dashboard — v3
