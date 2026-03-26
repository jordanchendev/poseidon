# Phase 12: AutoResearch Loop - Research

**Researched:** 2026-03-26
**Domain:** Autonomous experiment runner (Celery task + contextvar immutability + strategy mutation)
**Confidence:** HIGH

## Summary

Phase 12 wraps the Phase 11 `ParameterSearchPipeline` in an autonomous Celery task that iterates across markets, generates strategy mutations, and logs results to `ExperimentTracker`. The architecture is well-constrained: `StrategyMutator` is a thin wrapper over existing `VotingStrategyFactory` + Optuna, and the three-layer immutability enforcement uses Python's built-in `contextvars` module to guard `FeatureEngine`, `BacktestRunner`, and `RiskEngine` from mutation during autoresearch runs.

All key building blocks exist in the codebase. `ParameterSearchPipeline.run()` already orchestrates holdout -> Optuna -> WFE -> logging per market. `VotingStrategyFactory.from_trial()` handles Optuna suggest API. `ExperimentTracker` handles persistence. The primary new code is: (1) `StrategyMutator` as a delegation wrapper, (2) `_AUTORESEARCH_ACTIVE` contextvar + guard decorators on protected classes, (3) `autoresearch_run` Celery task with heartbeat/graceful-stop, and (4) report generation querying ExperimentTracker results.

**Primary recommendation:** Build four modules in sequence -- immutability guard (contextvar + mixin), StrategyMutator (thin wrapper), AutoResearchRunner (Celery task), and report generator. All follow existing codebase patterns closely.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: StrategyMutator is a thin wrapper, NOT a new search engine. Delegates to VotingStrategyFactory + Optuna (from Phase 11)
- D-02: `StrategyMutator.mutate_via_optuna(trial)` delegates to `VotingStrategyFactory.from_trial()` for Bayesian-guided mutation
- D-03: `StrategyMutator.mutate_random(seed)` generates random config within PARAM_BOUNDS for baseline comparison
- D-04: All generated configs must pass existing Pydantic validation (`VotingStrategy.validate_config()`)
- D-05: Runtime enforcement via `contextvar` flag `_AUTORESEARCH_ACTIVE` -- set True at AutoResearchRunner entry, checked in protected modules
- D-06: FeatureEngine, BacktestRunner, RiskEngine raise `ImmutabilityViolationError` on `__setattr__` / method mutation while `_AUTORESEARCH_ACTIVE` is True
- D-07: Only RuleConfig JSON (strategy parameters) is mutable during autoresearch runs -- enforced at runtime, not linter-only
- D-08: NOT using OS-level read-only or import hooks -- contextvar + explicit checks are sufficient and testable
- D-09: Single long-running Celery task `autoresearch_run`, NOT one-task-per-experiment
- D-10: Task receives `SearchConfig` + market list, internally loops: per-market calls `ParameterSearchPipeline.run()`
- D-11: Heartbeat: update Celery task state after each market completes (progress queryable via API)
- D-12: Graceful stop: check Redis flag `autoresearch:stop:{task_id}` between markets -- allows external stop without killing worker
- D-13: Per-market failure isolation: catch exception + log + continue to next market (don't abort entire run)
- D-14: Minimum 10 consecutive experiments must run unattended (AUTO-05 success criteria)
- D-15: After run completes, query ExperimentTracker for `status=passed` (WFE gate passed) experiments
- D-16: Rank by `composite_score`, produce `autoresearch_report.json` with top-N configs + summary stats
- D-17: NO auto-deployment -- human reviews report and decides whether to update production config
- D-18: `program.md` guidance layer = SearchConfig parameterization (n_trials, min_wfe, market list) in this phase; more sophisticated AI guidance deferred

### Claude's Discretion
- Exact contextvar implementation pattern
- autoresearch_report.json schema details
- Heartbeat update frequency and format
- Error message wording for ImmutabilityViolationError

### Deferred Ideas (OUT OF SCOPE)
- AI-driven guidance layer (sophisticated program.md that adapts search strategy) -- future phase
- Auto-deployment of best config to production -- future phase, needs safety gate
- Multi-timeframe simultaneous search -- current focus is per-market sequential
- Regime-aware autoresearch -- Phase 13 (gated on regime classifier)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUTO-03 | StrategyMutator varies strategy parameters within defined bounds (signal periods, thresholds, vote counts) | Delegates to existing VotingStrategyFactory + PARAM_BOUNDS; mutate_via_optuna wraps from_trial(), mutate_random samples PARAM_BOUNDS |
| AUTO-04 | 3-layer architecture enforced: immutable layer (FeatureEngine+BacktestRunner+RiskEngine), mutable layer (strategy JSON config), guidance layer (program.md) | contextvar `_AUTORESEARCH_ACTIVE` + `__setattr__` guard on protected classes; only RuleConfig JSON is mutable |
| AUTO-05 | AutoResearchRunner as Celery task that iterates: mutate config -> backtest -> evaluate -> log -> repeat | Single long-running `autoresearch_run` task wrapping ParameterSearchPipeline.run() per market with heartbeat and graceful stop |
| AUTO-06 | Immutability boundary enforced -- autoresearch cannot modify scoring formula, backtest runner, or feature engine code | ImmutabilityViolationError raised on __setattr__ of protected classes while contextvar is True; provably testable |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| contextvars (stdlib) | Python 3.13 | Immutability flag propagation | Built-in, thread/async safe, no dependency needed |
| celery | >=5.4 (already in project) | Long-running autoresearch task | Already used for all workers |
| redis | >=5.0 (already in project) | Graceful stop flag, Celery broker | Already used as broker and backend |
| optuna | >=4.8 (already in project) | Bayesian parameter search | Already integrated in Phase 11 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic-settings | >=2.0 (already in project) | AutoResearchConfig validation | For SearchConfig extensions if needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| contextvar | import hooks / `sys.modules` patching | contextvar is simpler, testable, explicit; import hooks are brittle |
| contextvar | OS-level file permission | Not testable in unit tests, breaks normal development |
| Single long task | One task per experiment | Single task avoids Celery overhead per trial, easier heartbeat/stop logic |

**Installation:**
No new packages needed. All dependencies already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure
```
src/poseidon/
├── autoresearch/
│   ├── __init__.py
│   ├── guard.py              # _AUTORESEARCH_ACTIVE contextvar + ImmutabilityViolationError + guard mixin
│   ├── mutator.py            # StrategyMutator (thin wrapper over VotingStrategyFactory)
│   ├── runner.py             # AutoResearchRunner class
│   └── report.py             # Report generation from ExperimentTracker results
├── workers/
│   └── cpu_tasks.py          # Add autoresearch_run task (or new research_tasks.py)
```

### Pattern 1: ContextVar Immutability Guard
**What:** A `contextvars.ContextVar` named `_AUTORESEARCH_ACTIVE` (default `False`) that protected classes check before allowing attribute mutation.
**When to use:** Set to `True` at AutoResearchRunner entry, reset on exit. Protected classes raise `ImmutabilityViolationError` when guard is active.
**Example:**
```python
# src/poseidon/autoresearch/guard.py
import contextvars

_AUTORESEARCH_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_AUTORESEARCH_ACTIVE", default=False
)


class ImmutabilityViolationError(RuntimeError):
    """Raised when autoresearch code attempts to mutate a protected object."""
    pass


def autoresearch_guard(cls):
    """Class decorator that prevents __setattr__ on instances while autoresearch is active.

    Allows initial construction (__init__) by tracking a _initializing flag.
    After __init__ completes, mutation is blocked when _AUTORESEARCH_ACTIVE is True.
    """
    original_init = cls.__init__

    def guarded_init(self, *args, **kwargs):
        object.__setattr__(self, '_ar_initialized', False)
        original_init(self, *args, **kwargs)
        object.__setattr__(self, '_ar_initialized', True)

    def guarded_setattr(self, name, value):
        if (
            getattr(self, '_ar_initialized', False)
            and _AUTORESEARCH_ACTIVE.get(False)
            and not name.startswith('_ar_')
        ):
            raise ImmutabilityViolationError(
                f"Cannot modify {type(self).__name__}.{name} during autoresearch run. "
                f"Only strategy JSON config is mutable."
            )
        object.__setattr__(self, name, value)

    cls.__init__ = guarded_init
    cls.__setattr__ = guarded_setattr
    return cls
```

### Pattern 2: StrategyMutator as Thin Wrapper
**What:** Delegates to `VotingStrategyFactory.from_trial()` and `VotingStrategyFactory.from_config()` with `PARAM_BOUNDS` for random sampling.
**When to use:** Called by `ParameterSearchPipeline` internally (Optuna path) or directly for random baseline.
**Example:**
```python
# src/poseidon/autoresearch/mutator.py
import random
from poseidon.backtest.voting_strategy_factory import (
    PARAM_BOUNDS, VotingStrategyFactory, _build_config_from_params,
)

class StrategyMutator:
    """Thin wrapper over VotingStrategyFactory for autoresearch mutations."""

    @staticmethod
    def mutate_via_optuna(trial, *, symbol: str, market: str, interval: str):
        """Bayesian-guided mutation via Optuna trial suggest API."""
        return VotingStrategyFactory.from_trial(
            trial, symbol=symbol, market=market, interval=interval,
        )

    @staticmethod
    def mutate_random(seed: int, *, symbol: str, market: str, interval: str) -> dict:
        """Generate random config within PARAM_BOUNDS for baseline comparison."""
        rng = random.Random(seed)
        params = {}
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            if ptype == "int":
                params[name] = rng.randint(int(low), int(high))
            else:
                params[name] = rng.uniform(float(low), float(high))
        config = _build_config_from_params(
            params, symbol=symbol, market=market, interval=interval,
        )
        # Validate via existing VotingStrategy validation
        strategy = VotingStrategyFactory.from_config(config)
        strategy.validate_config()
        return config
```

### Pattern 3: Long-Running Celery Task with Heartbeat + Graceful Stop
**What:** Single `autoresearch_run` task that loops per-market, calling `ParameterSearchPipeline.run()`, with heartbeat updates and Redis-based stop flag.
**When to use:** This is the main entry point for the autoresearch loop.
**Example:**
```python
# In cpu_tasks.py or new research_tasks.py
@celery_app.task(name="poseidon.workers.cpu_tasks.autoresearch_run", bind=True)
def autoresearch_run(self, search_config: dict, markets: list[dict]) -> dict:
    """Run autoresearch across multiple markets.

    Args:
        search_config: SearchConfig as dict (n_trials, min_wfe, etc.)
        markets: List of {symbol, market, interval} dicts.
    """
    from poseidon.autoresearch.guard import _AUTORESEARCH_ACTIVE

    token = _AUTORESEARCH_ACTIVE.set(True)
    try:
        results = []
        for i, mkt in enumerate(markets):
            # Check graceful stop flag
            stop_key = f"autoresearch:stop:{self.request.id}"
            if redis_client.get(stop_key):
                break

            # Heartbeat update
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(markets), "market": mkt["symbol"]},
            )

            try:
                result = pipeline.run(ohlcv, mkt["symbol"], mkt["market"], mkt["interval"], cfg)
                results.append(result)
            except Exception as exc:
                logger.error("Market %s failed: %s", mkt["symbol"], exc)
                results.append({"market": mkt["symbol"], "error": str(exc)})

        # Generate report
        report = generate_report(tracker, study_names=[r.study_name for r in results if hasattr(r, 'study_name')])
        return {"status": "completed", "markets_processed": len(results), "report": report}
    finally:
        _AUTORESEARCH_ACTIVE.reset(token)
```

### Pattern 4: Existing Celery Task Pattern (from `cpu_tasks.py`)
**What:** The project uses `SessionLocal()` for DB sessions, `bind=True` for self access, try/finally for cleanup.
**When to use:** Follow this exact pattern for the new autoresearch task.
**Key observations from existing code:**
- Tasks are decorated with `@celery_app.task(name="...", bind=True)`
- DB sessions are created via `SessionLocal()` and closed in `finally`
- Error handling: catch per-item, log, continue (see `fetch_market_data` pattern)
- No existing usage of `self.update_state()` in the codebase yet -- this will be new for heartbeat

### Anti-Patterns to Avoid
- **One Celery task per experiment**: Creates massive overhead from task dispatch, serialization, result backend writes. Per D-09: use single long-running task.
- **Modifying protected classes during autoresearch**: The contextvar guard exists to prevent this. Any attempt to modify FeatureEngine/BacktestRunner/RiskEngine attributes will raise ImmutabilityViolationError.
- **Forgetting to reset contextvar**: Always use `token = var.set(True)` / `var.reset(token)` in try/finally to ensure cleanup.
- **Auto-deploying results**: Per D-17, human must review report first.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parameter search | Custom search loop | `ParameterSearchPipeline.run()` | Already orchestrates holdout->Optuna->WFE->logging |
| Strategy creation from params | Custom config builder | `VotingStrategyFactory.from_trial()` + `from_config()` | Already handles Optuna suggest API and config construction |
| Experiment persistence | Custom DB writes | `ExperimentTracker.save()` | Already handles full experiment record persistence |
| Config validation | Custom validators | `VotingStrategy.validate_config()` | Already validates sub-signals, min_votes, symbol, market |
| Bayesian optimization | Custom optimizer | Optuna via `BayesianOptimizer` | Already integrated with RDBStorage in Phase 11 |

**Key insight:** Phase 12 is primarily a composition phase -- almost all core logic exists from Phase 11. The new code is orchestration (Celery task loop), safety (contextvar guard), and reporting (query + format).

## Common Pitfalls

### Pitfall 1: ContextVar Not Propagating to Worker Threads
**What goes wrong:** If Celery uses threading (prefork is default, but can be configured), contextvars propagate correctly in Python 3.12+ with asyncio but NOT automatically to raw threads.
**Why it happens:** ContextVar values are per-context. In prefork (default), each worker is a separate process and gets its own context -- no issue. But if using eventlet/gevent, context propagation may not work as expected.
**How to avoid:** Verify Celery is using prefork pool (default). Set the contextvar at the start of the task function body, which runs in the worker process.
**Warning signs:** ImmutabilityViolationError not raised during tests that run in a separate thread.

### Pitfall 2: Forgetting to Reset ContextVar on Exception
**What goes wrong:** If AutoResearchRunner throws and contextvar isn't reset, subsequent tasks in the same worker process will still have `_AUTORESEARCH_ACTIVE=True`.
**Why it happens:** Not using try/finally for contextvar reset.
**How to avoid:** Always use `token = _AUTORESEARCH_ACTIVE.set(True)` and `_AUTORESEARCH_ACTIVE.reset(token)` in a try/finally block. Consider a context manager for cleaner syntax.
**Warning signs:** Normal backtest tasks failing with ImmutabilityViolationError after an autoresearch run crashes.

### Pitfall 3: Guard Blocking Object Construction
**What goes wrong:** `__setattr__` guard fires during `__init__` of protected classes, preventing normal construction.
**Why it happens:** Guard doesn't distinguish between initialization and post-init mutation.
**How to avoid:** Use an `_ar_initialized` flag set via `object.__setattr__` before and after `__init__`, or apply guard only to a specific list of protected attributes rather than all attributes.
**Warning signs:** `ImmutabilityViolationError` raised when creating FeatureEngine/BacktestRunner/RiskEngine instances during autoresearch.

### Pitfall 4: Celery JSON Serialization of SearchConfig
**What goes wrong:** Celery's JSON serializer can't handle dataclass instances, datetime objects, or custom types directly.
**Why it happens:** `SearchConfig` is a dataclass with nested dataclasses (`HoldoutConfig`, `WalkForwardConfig`).
**How to avoid:** Serialize `SearchConfig` to a plain dict before passing to the Celery task. Reconstruct inside the task body.
**Warning signs:** `kombu.exceptions.EncodeError` when dispatching the autoresearch task.

### Pitfall 5: DB Session Lifetime in Long-Running Task
**What goes wrong:** A single DB session held for hours accumulates stale state, memory, and potentially loses connection.
**Why it happens:** Long-running task with one SessionLocal() for the entire run.
**How to avoid:** Create a fresh session per market iteration. Use `SessionLocal()` in a context manager per loop iteration. Commit/close after each market completes.
**Warning signs:** `OperationalError: server closed the connection unexpectedly` after running for a while.

### Pitfall 6: Redis Stop Flag Not Cleaned Up
**What goes wrong:** Old stop flags from previous runs persist, causing new runs to immediately stop.
**Why it happens:** Stop flag `autoresearch:stop:{task_id}` not deleted after task completes or at task start.
**How to avoid:** Clean up the stop flag in the finally block of the task. Also, set a TTL (e.g., 24h) on the Redis key when creating it.
**Warning signs:** New autoresearch tasks immediately stop without processing any markets.

## Code Examples

### ContextVar Context Manager
```python
# Cleaner alternative to manual set/reset
from contextlib import contextmanager

@contextmanager
def autoresearch_context():
    """Context manager that activates the autoresearch immutability guard."""
    token = _AUTORESEARCH_ACTIVE.set(True)
    try:
        yield
    finally:
        _AUTORESEARCH_ACTIVE.reset(token)
```

### ExperimentTracker Query for Report (needs new method)
```python
# ExperimentTracker currently has list_by_market() which sorts by composite_score desc.
# For report generation, need to filter by status="passed" as well.
# Recommend adding a query method or filtering in report generator:

def query_passed_by_study(self, study_name: str, limit: int = 10) -> list[ExperimentRecord]:
    return (
        self._db.query(ExperimentRecord)
        .filter(
            ExperimentRecord.study_name == study_name,
            ExperimentRecord.status == "passed",
        )
        .order_by(ExperimentRecord.composite_score.desc().nulls_last())
        .limit(limit)
        .all()
    )
```

### Report Schema (Claude's discretion)
```python
# autoresearch_report.json schema
{
    "run_id": "celery-task-uuid",
    "started_at": "2026-03-26T10:00:00Z",
    "completed_at": "2026-03-26T12:30:00Z",
    "markets_processed": 3,
    "markets_failed": 0,
    "total_experiments": 150,
    "passed_experiments": 42,
    "per_market": [
        {
            "symbol": "BTCUSDT",
            "market": "crypto_spot",
            "interval": "1h",
            "total_trials": 50,
            "passed_trials": 15,
            "best_composite_score": 1.85,
            "best_config": { ... },
            "wfe_pass_rate": 0.30
        }
    ],
    "top_configs": [
        {
            "rank": 1,
            "symbol": "BTCUSDT",
            "composite_score": 1.85,
            "wfe_score": 0.65,
            "config": { ... }
        }
    ]
}
```

### Celery Heartbeat Update
```python
# Celery self.update_state() for progress tracking
# Requires bind=True on the task decorator
self.update_state(
    state="PROGRESS",
    meta={
        "current_market": i + 1,
        "total_markets": len(markets),
        "symbol": mkt["symbol"],
        "market": mkt["market"],
        "completed_markets": [r["symbol"] for r in completed],
    },
)
# Client can query via: AsyncResult(task_id).state / .info
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual parameter tuning | Automated Optuna search (Phase 11) | Phase 11 | ParameterSearchPipeline handles full pipeline |
| Separate Optuna script | Integrated Celery task | Phase 12 (this phase) | Runs unattended, queryable progress |
| No immutability guarantee | ContextVar runtime guard | Phase 12 (this phase) | Provable separation of concerns |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `pytest tests/test_autoresearch.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUTO-03 | StrategyMutator generates valid configs via Optuna and random | unit | `pytest tests/test_strategy_mutator.py -x` | Wave 0 |
| AUTO-03 | All generated configs pass validate_config() | unit | `pytest tests/test_strategy_mutator.py::test_all_configs_validate -x` | Wave 0 |
| AUTO-04 | Protected classes raise ImmutabilityViolationError during autoresearch | unit | `pytest tests/test_autoresearch_guard.py -x` | Wave 0 |
| AUTO-04 | Only RuleConfig JSON is mutable | unit | `pytest tests/test_autoresearch_guard.py::test_strategy_config_mutable -x` | Wave 0 |
| AUTO-05 | AutoResearchRunner completes full cycle (mutate->backtest->evaluate->log) | integration | `pytest tests/test_autoresearch_runner.py -x` | Wave 0 |
| AUTO-05 | 10+ consecutive experiments run without error | integration | `pytest tests/test_autoresearch_runner.py::test_10_consecutive -x` | Wave 0 |
| AUTO-06 | Import/call of evaluation-layer modification raises error | unit | `pytest tests/test_autoresearch_guard.py::test_immutability_enforcement -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_autoresearch*.py tests/test_strategy_mutator.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_autoresearch_guard.py` -- covers AUTO-04, AUTO-06 (contextvar guard + ImmutabilityViolationError)
- [ ] `tests/test_strategy_mutator.py` -- covers AUTO-03 (mutate_via_optuna, mutate_random, validation)
- [ ] `tests/test_autoresearch_runner.py` -- covers AUTO-05 (full cycle, heartbeat, graceful stop, 10 consecutive)
- [ ] `tests/test_autoresearch_report.py` -- covers D-15/D-16 (report generation from ExperimentTracker)

## Open Questions

1. **ExperimentTracker query for report**
   - What we know: `list_by_market()` exists and sorts by composite_score desc, but doesn't filter by status
   - What's unclear: Whether to add a new method to ExperimentTracker or filter in the report generator
   - Recommendation: Add `query_passed()` or `query_by_status()` method to ExperimentTracker for reusability

2. **Celery task routing for autoresearch**
   - What we know: CPU tasks route to `cpu` queue, GPU tasks to `gpu` queue
   - What's unclear: Whether autoresearch should use existing cpu queue or a dedicated queue
   - Recommendation: Use `cpu` queue initially (autoresearch is CPU-bound). If contention becomes an issue, add `research` queue later

3. **Guard decorator vs. metaclass**
   - What we know: Decorator approach is simpler, metaclass is more thorough
   - What's unclear: Whether `__setattr__` guard alone is sufficient or if `__delattr__` should also be guarded
   - Recommendation: Start with `__setattr__` only (per D-06). Add `__delattr__` if testing reveals gaps

## Sources

### Primary (HIGH confidence)
- `src/poseidon/backtest/param_search.py` -- ParameterSearchPipeline, SearchConfig, SearchResult (full read)
- `src/poseidon/backtest/voting_strategy_factory.py` -- VotingStrategyFactory, PARAM_BOUNDS (full read)
- `src/poseidon/backtest/experiment_tracker.py` -- ExperimentTracker CRUD methods (full read)
- `src/poseidon/workers/cpu_tasks.py` -- Celery task patterns (full read)
- `src/poseidon/workers/celery_app.py` -- Celery configuration (full read)
- `src/poseidon/data/feature_engine.py` -- FeatureEngine class structure (partial read)
- `src/poseidon/backtest/runner.py` -- BacktestRunner class structure (partial read)
- `src/poseidon/risk/engine.py` -- RiskEngine class structure (partial read)
- Python 3.13 stdlib `contextvars` module -- thread/async safe context propagation

### Secondary (MEDIUM confidence)
- `src/poseidon/strategies/voting_strategy.py` -- validate_config() method (grep verified)
- `src/poseidon/strategies/configs/nunchi_crypto_1h.json` -- Reference config structure (full read)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new dependencies
- Architecture: HIGH -- follows established codebase patterns, all building blocks exist
- Pitfalls: HIGH -- identified from direct code reading (SessionLocal lifecycle, contextvar propagation, Celery serialization)
- Immutability pattern: HIGH -- contextvars is stdlib, well-documented, testable

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable -- no external dependency changes expected)
