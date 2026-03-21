# Phase 7: API & Integration - Research

**Researched:** 2026-03-21
**Domain:** FastAPI REST API design, endpoint wiring, error handling patterns
**Confidence:** HIGH

## Summary

Phase 7 is an integration phase that exposes the full REST API surface by adding missing endpoint groups (strategies, models, backtests, signals) to the existing FastAPI application and enhancing the health endpoint. The core domain logic already exists in well-structured modules from Phases 1-6. The API layer follows the project's established pattern: thin router modules in `src/poseidon/api/` that delegate to domain services, with Pydantic schemas for request/response validation.

The primary work is: (1) creating 4 new router modules (strategies, models, backtests, signals), (2) enhancing the existing health router to report GPU/Celery/data status, (3) establishing a consistent error response format, and (4) wiring everything into `main.py` with proper auth dependencies. No new libraries are needed -- the existing stack (FastAPI, Pydantic, SQLAlchemy, Celery, Redis) provides everything required.

**Primary recommendation:** Follow the existing `api/risk.py` pattern (inline Pydantic schemas + thin router + dependency injection via `get_db`) for all new endpoints. Use FastAPI exception handlers for the consistent error format. Long-running operations (training, backtesting, optimization) must dispatch Celery tasks and return 202 with a task_id, following the existing `api/data.py` pattern.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-01 | REST endpoints for data management, strategies, models, backtests, risk rules, signals, and health check | All endpoint groups mapped to design doc spec; existing modules provide domain logic; new routers follow established patterns |
</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version Spec | Purpose | Why Standard |
|---------|-------------|---------|--------------|
| FastAPI | >=0.115 | REST framework | Already in use, async-first, auto OpenAPI |
| Pydantic | >=2.0 (via pydantic-settings) | Request/response validation | Already used for all schemas |
| SQLAlchemy | >=2.0 | ORM, database access | Already used for all models |
| Celery | >=5.4 | Async task dispatch | Already used for data fetching |
| Redis | >=5.0 | Celery broker + Streams | Already used for signal delivery |
| httpx | >=0.27 (dev dep) | Test client for FastAPI | Already in dev dependencies |
| uvicorn | >=0.30 | ASGI server | Already in use |

### Supporting (already installed)
| Library | Version Spec | Purpose | When to Use |
|---------|-------------|---------|-------------|
| pytest | >=8.0 | Testing | Endpoint integration tests |
| pytest-asyncio | >=0.23 | Async test support | Testing async endpoints |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Inline Pydantic schemas per router | Centralized schemas.py | Per-router is cleaner for large API; matches risk.py pattern |
| Celery task dispatch for backtests | Synchronous backtest run | Backtests can take minutes; async is required |

**Installation:** No new packages needed. Everything is already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure (additions in bold)
```
src/poseidon/
├── api/
│   ├── __init__.py
│   ├── auth.py              # Existing: X-API-Key verification
│   ├── data.py              # Existing: fetch, backfill, status
│   ├── health.py            # Enhance: GPU, Celery, data freshness
│   ├── risk.py              # Existing: CRUD for risk rules
│   ├── sentiment.py         # Existing: sentiment CRUD
│   ├── strategies.py        # NEW: strategy CRUD + activate/deactivate
│   ├── models.py            # NEW: model lifecycle + train/predict
│   ├── backtests.py         # NEW: run/optimize/results
│   ├── signals.py           # NEW: list/detail
│   └── errors.py            # NEW: consistent error handler
├── main.py                  # Wire new routers
└── ...
```

### Pattern 1: Thin Router with Domain Delegation
**What:** Each API file is a thin routing layer that validates input (Pydantic), calls domain services, and returns validated output. No business logic in routers.
**When to use:** All endpoints.
**Example (from existing `api/risk.py`):**
```python
# Router defines Pydantic schemas inline
class RiskRuleCreate(PydanticBase):
    rule_type: str
    name: str
    enabled: bool = True

# Endpoint delegates to ORM operations
@router.post("", response_model=RiskRuleResponse, status_code=201)
async def create_rule(body: RiskRuleCreate, db: Session = Depends(get_db)):
    record = RiskRuleRecord(...)
    db.add(record)
    db.commit()
    return record
```

### Pattern 2: Async Task Dispatch for Long-Running Operations
**What:** POST endpoints that trigger long-running work return 202 with a Celery task_id. Client polls for status.
**When to use:** Model training, backtests, parameter optimization.
**Example (from existing `api/data.py`):**
```python
@router.post("/fetch", response_model=MessageResponse, status_code=202)
async def trigger_fetch(request: FetchRequest):
    task = fetch_market_data.delay(request.market, request.interval)
    return MessageResponse(message=f"Task dispatched", task_id=task.id)
```

### Pattern 3: Consistent Error Response Format
**What:** All error responses use a single JSON structure: `{"detail": "...", "error_code": "...", "status_code": 4xx/5xx}`.
**When to use:** Every error path -- validation errors, not found, business rule violations.
**Example:**
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "errors": exc.errors(),
        },
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
        },
    )
```

### Pattern 4: Strategy Persistence via DB (no strategy registry module)
**What:** Strategies are stored as DB records. The design doc mentions `strategies/registry.py` but it does not exist and is not needed at the API level. Strategies are reconstructed from DB config when evaluated.
**When to use:** Strategy CRUD endpoints. A `StrategyRecord` table stores name, type, config (JSONB), symbol, market, interval, active status.
**Note:** This requires a new SQLAlchemy model and Alembic migration.

### Anti-Patterns to Avoid
- **Business logic in routers:** Keep routers thin. Complex operations (like reconstructing a strategy from DB config + running backtest) should be in service functions or domain modules.
- **Synchronous long-running endpoints:** Never run backtests or training synchronously -- always dispatch to Celery.
- **Inconsistent auth:** All new routers except health must use the existing `dependencies=secured` pattern.
- **Returning ORM models directly:** Always define explicit Pydantic response models with `model_config = {"from_attributes": True}`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Request validation | Custom validation logic | Pydantic models with Field constraints | Already integrated, auto-generates OpenAPI docs |
| Error responses | Per-endpoint error formatting | FastAPI exception handlers (app-level) | Consistent format, DRY, catches validation errors too |
| API documentation | Separate docs | FastAPI auto-generated OpenAPI (Swagger) | Already configured via `app = FastAPI(title="Poseidon")` |
| Async task status | Custom polling mechanism | `celery_app.AsyncResult(task_id).state` | Built into Celery, returns PENDING/STARTED/SUCCESS/FAILURE |
| Celery queue length | Custom Redis querying | `celery_app.control.inspect().active_queues()` | Celery inspect API handles connection details |
| GPU status check | Custom NVIDIA tooling | `torch.cuda.is_available()` + `torch.cuda.mem_get_info()` | PyTorch already imported in GPU worker context |

## Common Pitfalls

### Pitfall 1: Missing Strategy Persistence Layer
**What goes wrong:** The design doc assumes strategies can be CRUD'd via API, but there is no `StrategyRecord` SQLAlchemy model or DB table. Without it, strategies only exist in memory.
**Why it happens:** Phases 4 (Strategy Layer) focused on the domain logic (BaseStrategy, ModelStrategy, RuleStrategy), not API persistence.
**How to avoid:** Create a `StrategyRecord` model with columns: id (UUID), name, strategy_type (model/rule), config (JSONB), symbol, market, interval, active (bool), created_at, updated_at. Add Alembic migration.
**Warning signs:** Trying to build strategy CRUD without a DB table.

### Pitfall 2: Celery Task Routing for GPU vs CPU
**What goes wrong:** Training tasks dispatched to wrong queue. Model training needs GPU queue, backtests need CPU queue.
**Why it happens:** New Celery tasks must be registered in `celery_app.conf.task_routes`.
**How to avoid:** Define new tasks in appropriate worker modules (`gpu_tasks.py` for training, `cpu_tasks.py` for backtests) and ensure task names match route patterns.
**Warning signs:** Tasks stuck in queue or running on wrong worker type.

### Pitfall 3: Risk Router Prefix Inconsistency
**What goes wrong:** The existing risk router is mounted at `/api/risk-rules` (with `/api/` prefix), while all other routers use no `/api/` prefix (e.g., `/data`, `/sentiment`). The design doc specifies `/risk/rules`.
**Why it happens:** Risk router was added in Phase 5 with a different URL pattern.
**How to avoid:** Decide on one convention. Recommendation: keep existing risk router at `/api/risk-rules` for backward compatibility, or normalize all new endpoints under `/api/` prefix. The design doc uses no `/api/` prefix.
**Warning signs:** Inconsistent URL patterns across endpoints.

### Pitfall 4: Health Endpoint Auth
**What goes wrong:** Health endpoint is intentionally unauthenticated (for Docker healthcheck), but the enhanced version queries DB/Redis/Celery. Must keep it unauthenticated but consider what info to expose.
**Why it happens:** Design doc says health reports GPU status, queue lengths, data freshness -- all of which require service connections.
**How to avoid:** Keep health unauthenticated with basic status. Optionally add a `/health/detailed` or similar that requires auth for the full report. Or keep it all unauthenticated since it's internal-only.
**Warning signs:** Docker healthcheck failing because it can't provide API key.

### Pitfall 5: Backtest/Optimization Result Retrieval
**What goes wrong:** Backtest runs via Celery are async. The result needs to be stored in DB (BacktestRecord) for later retrieval, not just in Celery's result backend.
**Why it happens:** Celery results expire. DB persistence is needed for historical comparison.
**How to avoid:** Celery task writes BacktestRecord to DB upon completion. API retrieval queries DB, not Celery result backend.
**Warning signs:** Results disappearing after Celery result TTL expires.

### Pitfall 6: Model Training Endpoint Needs Feature + Data Context
**What goes wrong:** `POST /models/train` needs to know which features, symbol, market, date range, and hyperparams to use. The API request must provide enough context.
**Why it happens:** Training requires complex orchestration: load data, compute features, train model, save artifacts, update DB status.
**How to avoid:** Design the training request schema carefully. The Celery task handles orchestration: load OHLCV -> FeatureEngine -> model.train() -> save artifacts -> transition status.
**Warning signs:** Incomplete training request schema that requires follow-up calls.

## Code Examples

### Verified Pattern: Strategy CRUD Router Structure
```python
# Source: Based on existing api/risk.py pattern + design doc spec
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel as PydanticBase, ConfigDict, Field
from sqlalchemy.orm import Session
from poseidon.models.base import get_db
import uuid
from datetime import datetime

router = APIRouter()

# --- Pydantic schemas (inline per router, matching project pattern) ---

class StrategyCreate(PydanticBase):
    name: str
    strategy_type: str  # "model" or "rule"
    config: dict  # DSL config for rule, model params for model
    symbol: str
    market: str
    interval: str = "1d"

class StrategyResponse(PydanticBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    strategy_type: str
    config: dict
    symbol: str
    market: str
    interval: str
    active: bool
    created_at: datetime
    updated_at: datetime

# --- Endpoints ---

@router.get("", response_model=list[StrategyResponse])
async def list_strategies(db: Session = Depends(get_db)):
    # Query StrategyRecord, return all
    ...

@router.post("/{strategy_id}/activate", response_model=StrategyResponse)
async def activate_strategy(strategy_id: uuid.UUID, db: Session = Depends(get_db)):
    # Set active=True, return updated record
    ...
```

### Verified Pattern: Enhanced Health Endpoint
```python
# Source: Based on existing api/health.py + Celery inspect API + design doc requirements
import redis
from celery import Celery
from sqlalchemy import func, select
from poseidon.workers.celery_app import celery_app
from poseidon.core.config import settings
from poseidon.models.ohlcv import OHLCV
from poseidon.models.base import SessionLocal

@router.get("/health")
async def health():
    status = {"status": "ok", "components": {}}

    # 1. DB check
    try:
        db = SessionLocal()
        db.execute(select(func.now()))
        status["components"]["database"] = "ok"
        # Data freshness: latest OHLCV timestamp
        latest = db.execute(
            select(func.max(OHLCV.time))
        ).scalar()
        status["components"]["data_freshness"] = latest.isoformat() if latest else None
        db.close()
    except Exception as e:
        status["components"]["database"] = f"error: {e}"

    # 2. Redis check
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        status["components"]["redis"] = "ok"
    except Exception as e:
        status["components"]["redis"] = f"error: {e}"

    # 3. Celery queue lengths
    try:
        inspect = celery_app.control.inspect(timeout=2.0)
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        status["components"]["celery"] = {
            "active_tasks": sum(len(v) for v in active.values()),
            "reserved_tasks": sum(len(v) for v in reserved.values()),
        }
    except Exception as e:
        status["components"]["celery"] = f"error: {e}"

    # 4. GPU status (best-effort)
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            status["components"]["gpu"] = {
                "available": True,
                "free_mb": free // (1024 * 1024),
                "total_mb": total // (1024 * 1024),
            }
        else:
            status["components"]["gpu"] = {"available": False}
    except ImportError:
        status["components"]["gpu"] = {"available": False, "note": "torch not installed"}

    # Overall status
    errors = [k for k, v in status["components"].items() if isinstance(v, str) and v.startswith("error")]
    if errors:
        status["status"] = "degraded"

    return status
```

### Verified Pattern: Backtest Task Dispatch
```python
# Source: Based on existing api/data.py + backtest/runner.py
@router.post("/run", response_model=MessageResponse, status_code=202)
async def run_backtest(request: BacktestRunRequest):
    """Dispatch backtest to CPU worker, return task_id."""
    task = run_backtest_task.delay(
        strategy_id=str(request.strategy_id),
        start_date=request.start_date.isoformat() if request.start_date else None,
        end_date=request.end_date.isoformat() if request.end_date else None,
        initial_capital=request.initial_capital,
    )
    return MessageResponse(
        message=f"Backtest dispatched for strategy {request.strategy_id}",
        task_id=task.id,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FastAPI 0.x with Pydantic v1 | FastAPI 0.115+ with Pydantic v2 | 2023 | Use `model_config = ConfigDict(...)` not `class Config:` |
| `@app.exception_handler` with dict returns | `JSONResponse` with proper content type | Current | Consistent error format |
| Celery result backend for long-term storage | Write results to DB, use Celery only for task dispatch | Current | Results persist beyond Celery TTL |

**Deprecated/outdated:**
- Pydantic v1 syntax (`class Config:`, `Field(...)` with `schema_extra`): Project already uses v2 throughout
- `response_model_exclude_unset` on individual endpoints: Better to use Pydantic model defaults

## Open Questions

1. **Strategy DB persistence model**
   - What we know: No `StrategyRecord` table exists. Strategies are only in-memory objects (ModelStrategy, RuleStrategy).
   - What's unclear: Whether to create a new Alembic migration in this phase or if it should have been part of Phase 4.
   - Recommendation: Create a `StrategyRecord` model and migration as the first task of Phase 7. This is a prerequisite for strategy CRUD.

2. **URL prefix convention**
   - What we know: Risk router uses `/api/risk-rules`, others use `/data`, `/sentiment` (no `/api/` prefix). Design doc uses no prefix.
   - What's unclear: Whether to normalize existing routes or follow mixed convention.
   - Recommendation: Add new routers without `/api/` prefix (matching design doc: `/strategies`, `/models`, `/backtest`, `/signals`) and leave existing risk router as-is for backward compatibility. Note this means risk rules are at `/api/risk-rules` while the design doc says `/risk/rules`.

3. **Model training API orchestration**
   - What we know: Training involves OHLCV loading, feature computation, model training, artifact storage, and status transitions. ModelManager exists.
   - What's unclear: Whether a complete training Celery task exists yet (only `cpu_tasks.py` has data tasks; no `gpu_tasks.py` found).
   - Recommendation: Create `gpu_tasks.py` with a `train_model` task as part of this phase.

4. **Portfolio endpoint data source**
   - What we know: Design doc lists `GET /risk/portfolio`. VirtualPortfolio is in-memory, rebuilt from DB signals on restart.
   - What's unclear: Whether to query VirtualPortfolio (in API process) or query VirtualPositionRecord from DB.
   - Recommendation: Query DB directly (VirtualPositionRecord table) for the API endpoint, since the in-memory portfolio lives in the worker process, not the API process.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 + pytest-asyncio >= 0.23 |
| Config file | `pyproject.toml` ([tool.pytest.ini_options]) |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v --cov=poseidon` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01a | Strategy CRUD endpoints return correct status codes | integration | `pytest tests/test_api_strategies.py -x` | Wave 0 |
| API-01b | Model lifecycle endpoints (train dispatch, shadow, activate) | integration | `pytest tests/test_api_models.py -x` | Wave 0 |
| API-01c | Backtest run/optimize dispatch returns 202 + task_id | integration | `pytest tests/test_api_backtests.py -x` | Wave 0 |
| API-01d | Signal list/detail with filtering | integration | `pytest tests/test_api_signals.py -x` | Wave 0 |
| API-01e | Health endpoint reports all component statuses | integration | `pytest tests/test_api_health.py -x` | Wave 0 |
| API-01f | Error responses follow consistent JSON format | unit | `pytest tests/test_api_errors.py -x` | Wave 0 |
| API-01g | All secured endpoints reject unauthenticated requests | integration | `pytest tests/test_api_auth.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v --cov=poseidon`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_api_strategies.py` -- covers API-01a
- [ ] `tests/test_api_models.py` -- covers API-01b
- [ ] `tests/test_api_backtests.py` -- covers API-01c
- [ ] `tests/test_api_signals.py` -- covers API-01d
- [ ] `tests/test_api_health.py` -- covers API-01e
- [ ] `tests/test_api_errors.py` -- covers API-01f
- [ ] `tests/test_api_auth.py` -- covers API-01g (verify all routers require auth)

## Endpoint Inventory

### Already Exists (from Phases 1, 5)
| Endpoint | Router File | Auth | Status |
|----------|-------------|------|--------|
| `GET /health` | `api/health.py` | No | Needs enhancement (GPU, Celery, freshness) |
| `POST /data/fetch` | `api/data.py` | Yes | Complete |
| `POST /data/backfill` | `api/data.py` | Yes | Complete |
| `GET /data/backfill/status` | `api/data.py` | Yes | Complete |
| `POST /sentiment` | `api/sentiment.py` | Yes | Complete |
| `GET /sentiment` | `api/sentiment.py` | Yes | Complete |
| `GET /api/risk-rules/types` | `api/risk.py` | No* | Complete (missing auth) |
| `GET /api/risk-rules` | `api/risk.py` | No* | Complete (missing auth) |
| `POST /api/risk-rules` | `api/risk.py` | No* | Complete (missing auth) |
| `GET /api/risk-rules/{id}` | `api/risk.py` | No* | Complete (missing auth) |
| `PATCH /api/risk-rules/{id}` | `api/risk.py` | No* | Complete (missing auth) |
| `DELETE /api/risk-rules/{id}` | `api/risk.py` | No* | Complete (missing auth) |

*Note: Risk router in `main.py` is mounted WITHOUT `dependencies=secured`. This needs fixing.

### To Build (Phase 7)
| Endpoint | Design Doc Spec | Router File | Notes |
|----------|----------------|-------------|-------|
| `POST /strategies` | Create strategy | `api/strategies.py` | Needs StrategyRecord model |
| `GET /strategies` | List strategies | `api/strategies.py` | |
| `GET /strategies/{id}` | Get details | `api/strategies.py` | |
| `PUT /strategies/{id}` | Update strategy | `api/strategies.py` | |
| `DELETE /strategies/{id}` | Delete strategy | `api/strategies.py` | |
| `POST /strategies/{id}/activate` | Activate | `api/strategies.py` | Set active=True |
| `POST /strategies/{id}/deactivate` | Deactivate | `api/strategies.py` | Set active=False |
| `POST /models/train` | Start training | `api/models.py` | Dispatch GPU Celery task |
| `GET /models` | List models | `api/models.py` | Uses ModelManager |
| `GET /models/{id}` | Get details+metrics | `api/models.py` | Uses ModelManager |
| `POST /models/{id}/shadow` | Shadow mode | `api/models.py` | Lifecycle transition |
| `POST /models/{id}/activate` | Activate model | `api/models.py` | Lifecycle transition |
| `POST /models/{id}/predict` | Request prediction | `api/models.py` | May need async dispatch |
| `POST /backtest/run` | Run backtest | `api/backtests.py` | Dispatch CPU Celery task |
| `POST /backtest/optimize` | Run optimization | `api/backtests.py` | Dispatch CPU Celery task |
| `GET /backtest/{id}` | Get results | `api/backtests.py` | Query BacktestRecord |
| `GET /backtest` | List backtests | `api/backtests.py` | Query BacktestRecord |
| `GET /risk/portfolio` | Virtual portfolio | `api/risk.py` (extend) | Query VirtualPositionRecord |
| `GET /signals` | List signals | `api/signals.py` | Filter by market/symbol/status |
| `GET /signals/{id}` | Signal detail | `api/signals.py` | Query SignalRecord |

### Enhancements to Existing
| Endpoint | Change | Details |
|----------|--------|---------|
| `GET /health` | Enhance response | Add GPU status, Celery queue lengths, data freshness |
| Risk router mount | Fix auth | Add `dependencies=secured` in main.py |
| `GET /data/symbols` | New endpoint | Design doc lists it; not yet implemented |
| `GET /data/status` | New endpoint | Design doc lists it; data freshness per market |

## Key Implementation Dependencies

### New DB Model Required: StrategyRecord
```python
# models/strategy.py (NEW)
class StrategyRecord(Base):
    __tablename__ = "strategies"
    id: UUID PK
    name: str (unique)
    strategy_type: str  # "model" or "rule"
    config: JSONB  # RuleConfig JSON for rules, model params for model strategies
    symbol: str
    market: str
    interval: str
    active: bool (default False)
    model_version_id: UUID | None  # FK to model_versions for model strategies
    created_at: TIMESTAMPTZ
    updated_at: TIMESTAMPTZ
```
This requires a new Alembic migration (006).

### New Celery Tasks Required
1. `gpu_tasks.py::train_model` -- orchestrates model training pipeline
2. `cpu_tasks.py::run_backtest_task` -- orchestrates backtest execution and DB persistence
3. `cpu_tasks.py::run_optimization_task` -- orchestrates parameter optimization

### Error Handler Registration
```python
# In main.py
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from poseidon.api.errors import http_exception_handler, validation_exception_handler

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
```

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/poseidon/api/*.py`, `src/poseidon/main.py` -- established patterns
- Design doc: `docs/poseidon-design.md` -- API endpoint specifications (lines 701-745)
- `pyproject.toml` -- dependency versions and project config

### Secondary (MEDIUM confidence)
- FastAPI official patterns for exception handlers, dependency injection, router mounting
- Celery inspect API for queue/worker status

### Tertiary (LOW confidence)
- GPU status via `torch.cuda` -- depends on torch being importable in API process (may only be available in GPU worker)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all patterns established in codebase
- Architecture: HIGH -- follows existing patterns exactly, design doc is prescriptive
- Pitfalls: HIGH -- identified from direct codebase analysis (missing strategy model, auth gap on risk router, Celery task routing)
- Endpoint inventory: HIGH -- exhaustive diff between design doc and existing code

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable -- no fast-moving dependencies)
