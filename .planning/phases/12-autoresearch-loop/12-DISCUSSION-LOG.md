# Phase 12: AutoResearch Loop - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-03-26
**Phase:** 12-autoresearch-loop
**Mode:** discuss (interactive)
**Areas discussed:** StrategyMutator, Three-Layer Immutability, Celery Integration, Result Handling

## Discussion Flow

### Gray Areas Presented
User selected: all 4 areas + asked for Claude's recommendations + reference Nunchi flow.

### Area 1: StrategyMutator 變異策略
- **Recommendation:** Thin wrapper over VotingStrategyFactory, not a new search engine
- **Rationale:** Phase 11 ParameterSearchPipeline already handles Optuna-based mutation via from_trial(); StrategyMutator adds mutate_random() for baseline comparison
- **User response:** Accepted as-is

### Area 2: Three-Layer Architecture Immutability
- **Recommendation:** Runtime enforcement via contextvar + __setattr__ checks; raise ImmutabilityViolationError
- **Alternatives considered:** Linter-only (insufficient for "provably enforced"), OS-level read-only (over-engineering)
- **User response:** Accepted as-is

### Area 3: AutoResearchRunner Celery Integration
- **Recommendation:** Single long-running Celery task with per-market heartbeat and Redis-based graceful stop
- **Alternatives considered:** One task per experiment (too much overhead), manual orchestration (no Celery)
- **User response:** Accepted as-is

### Area 4: Experiment Results & Convergence
- **Recommendation:** Auto-rank by composite_score, produce report JSON, human decides deployment
- **Alternatives considered:** Auto-deploy best config (too risky), no report (insufficient visibility)
- **User response:** Accepted as-is

## Corrections Made
No corrections — all recommendations confirmed.
