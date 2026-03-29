# Phase 18: Live Signal Pipeline -- Predict API + Strategy Beat Evaluation - Research

**Researched:** 2026-03-28
**Domain:** Celery task pipeline, FastAPI async endpoints, ML model inference, signal generation
**Confidence:** HIGH

## Summary

Phase 18 connects the existing model inference and strategy evaluation infrastructure to the live signal pipeline. The codebase already has all the building blocks: `BaseModel.predict()` returns DataFrames with prediction/confidence columns, `ModelStrategy.evaluate()` converts predictions to `Signal` objects, `RuleStrategy.evaluate()` evaluates DSL conditions against feature data, `VotingStrategy` implements multi-signal voting, `SignalPipeline.process()` handles risk check -> persist -> deliver, and `SignalRepository.save()` persists signals. What's missing is the wiring: the predict API endpoint is a placeholder, there is no GPU task for prediction, there is no Beat schedule for strategy evaluation, and the health check incorrectly tries to import torch in the API container.

The key architectural insight is that prediction dispatch must follow the same pattern as `train_model` -- API creates a request, Celery GPU task does the heavy lifting (model loading, feature computation, inference), and the result flows through the existing `SignalPipeline`. For rule/voting strategies, evaluation runs on the CPU worker since no GPU is needed. The downstream risk pipeline trigger (VaR recalculation + portfolio exposure update) should use Celery's `link` or explicit task chaining after signal persistence.

**Primary recommendation:** Implement 4 new Celery tasks (`run_prediction`, `evaluate_active_strategies`, `trigger_risk_update`, `check_gpu_health`) and rewire the predict API endpoint + health check. No new DB models or migrations needed -- all existing tables suffice.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRED-01 | POST /models/{version_id}/predict loads trained model, runs inference, returns prediction with confidence | Rewire placeholder endpoint to dispatch `run_prediction` GPU task; model loading via `BaseModel.load()` + `get_model()` registry already works in `run_model_backtest` |
| PRED-02 | Predictions meeting confidence threshold auto-convert to Signal objects via SignalRepository | `ModelStrategy.evaluate()` already does prediction->Signal conversion; pipe output through `SignalPipeline.process()` for risk+persist+deliver |
| PRED-03 | Active rule strategies evaluated by Celery Beat on configurable schedule after data fetch | New `evaluate_active_strategies` CPU task querying `StrategyRecord.active=True`, reconstruct strategy from config, run through `SignalPipeline`; trigger via `link` callback from `fetch_market_data` |
| PRED-04 | New signal generation triggers downstream risk pipeline: VaR snapshot + portfolio exposure | Chain `trigger_risk_update` task after signal persistence; calls existing `compute_var_snapshot("historical")` + portfolio rebuild |
| PRED-05 | Health endpoint reports GPU worker torch/CUDA via Celery worker ping, not local import | Replace `import torch` block with `celery_app.control.inspect().ping()` to GPU worker, then broadcast a custom probe task |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| celery | >=5.4 | Task queue for GPU/CPU workers | Already used for train_model, fetch_market_data |
| celery-redbeat | >=2.2 | Persistent Beat schedule in Redis | Already configured in celery_app.py |
| fastapi | >=0.115 | REST API framework | Already used for all endpoints |
| sqlalchemy | >=2.0 | ORM for strategy/signal/model records | Already used throughout |
| pydantic | >=2.0 | Request/response validation | Already used throughout |

### No New Dependencies Required
All Phase 18 functionality can be built with existing dependencies. No new packages needed.

## Architecture Patterns

### Recommended Project Structure (changes only)
```
src/poseidon/
  workers/
    gpu_tasks.py          # ADD: run_prediction task
    cpu_tasks.py          # ADD: evaluate_active_strategies, trigger_risk_update tasks
  api/
    models.py             # MODIFY: predict endpoint dispatches Celery task
    health.py             # MODIFY: GPU check via worker ping
  core/
    config.py             # ADD: predict_confidence_threshold, strategy_eval_schedule settings
```

### Pattern 1: Predict API -> GPU Task Dispatch (same as train_model)
**What:** API endpoint validates request, dispatches async Celery task to GPU queue, returns task_id
**When to use:** For model prediction, which requires GPU (Transformer) or model artifacts (XGBoost)
**Example:**
```python
# api/models.py - predict endpoint
@router.post("/{version_id}/predict", response_model=MessageResponse, status_code=202)
async def predict(version_id: uuid.UUID, body: PredictRequest, db: Session = Depends(get_db)):
    manager = ModelManager(db)
    mv = manager.get_version(version_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    if mv.status not in ("ready", "shadow", "active"):
        raise HTTPException(status_code=400, detail=f"Model version status '{mv.status}' cannot predict")

    task = run_prediction.delay(
        version_id=str(version_id),
        symbol=body.symbol,
        market=body.market,
        interval=body.interval,
    )
    return MessageResponse(message=f"Prediction dispatched for {mv.name} v{mv.version}", task_id=task.id)
```

### Pattern 2: GPU Task -- Model Load + Predict + Signal Pipeline
**What:** GPU worker loads model from artifacts, computes features, runs inference, pipes through SignalPipeline
**When to use:** For the `run_prediction` Celery task
**Example:**
```python
# workers/gpu_tasks.py - run_prediction task
@celery_app.task(name="poseidon.workers.gpu_tasks.run_prediction", bind=True, max_retries=0)
def run_prediction(self, version_id: str, symbol: str, market: str, interval: str = "1d") -> dict:
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.storage import read_ohlcv
    from poseidon.ml.manager import ModelManager
    from poseidon.ml.registry import get_model
    from poseidon.models.base import SessionLocal
    from poseidon.risk.pipeline import SignalPipeline
    from poseidon.strategies.model_strategy import ModelStrategy

    session = SessionLocal()
    try:
        manager = ModelManager(session)
        mv = manager.get_version(UUID(version_id))
        # Load model from artifacts
        model_cls = get_model(mv.name)
        model_instance = model_cls.load(Path(mv.artifact_path))

        # Load latest OHLCV + compute features
        ohlcv_df = read_ohlcv(session, symbol, market, interval)
        engine = FeatureEngine()
        features_df = engine.compute_from_df(ohlcv_df)

        # Run prediction via ModelStrategy
        strategy = ModelStrategy(name=mv.name, model=model_instance, ...)
        raw_signals = strategy.evaluate(features_df)

        # Filter by confidence threshold, pipe through SignalPipeline
        pipeline = SignalPipeline(session)
        results = []
        for sig in raw_signals:
            if sig.confidence >= settings.predict_confidence_threshold:
                processed = pipeline.process(sig)
                results.append({"signal_id": str(processed.id), "status": processed.status.value})

        return {"version_id": version_id, "signals_generated": len(results), "results": results}
    finally:
        session.close()
```

### Pattern 3: Strategy Beat Evaluation (CPU Task)
**What:** Celery Beat triggers evaluation of all active strategies after data fetch completes
**When to use:** For PRED-03 -- scheduled strategy evaluation
**Example:**
```python
# workers/cpu_tasks.py - evaluate_active_strategies
@celery_app.task(name="poseidon.workers.cpu_tasks.evaluate_active_strategies")
def evaluate_active_strategies(market: str, interval: str) -> dict:
    """Evaluate all active strategies for a given market/interval.

    Triggered as callback after fetch_market_data completes.
    """
    session = SessionLocal()
    try:
        # Query active strategies matching this market/interval
        strategies = session.query(StrategyRecord).filter(
            StrategyRecord.active == True,
            StrategyRecord.market == market,
            StrategyRecord.interval == interval,
        ).all()

        pipeline = SignalPipeline(session)
        engine = FeatureEngine()
        total_signals = 0

        for record in strategies:
            if record.strategy_type == "rule":
                strategy = RuleStrategy(config=record.config, strategy_id=record.id)
            elif record.strategy_type == "voting":
                strategy = VotingStrategy(config=record.config, strategy_id=record.id)
            else:
                continue  # model strategies go through GPU predict path

            features = engine.compute(record.symbol, record.market, record.interval)
            raw_signals = strategy.evaluate(features)
            for sig in raw_signals:
                pipeline.process(sig)
                total_signals += 1

        return {"market": market, "interval": interval, "strategies_evaluated": len(strategies), "signals": total_signals}
    finally:
        session.close()
```

### Pattern 4: Downstream Risk Pipeline Trigger
**What:** After signal generation, trigger VaR snapshot recalculation and portfolio exposure update
**When to use:** For PRED-04 -- risk pipeline as downstream of signal generation
**Implementation options:**
1. **Celery chain/link** -- `evaluate_active_strategies.apply_async(link=trigger_risk_update.si())`
2. **Explicit call at end of evaluate_active_strategies** -- simpler, synchronous within the same task
3. **Beat schedule** -- already runs VaR hourly, but PRED-04 wants it triggered on new signals

**Recommended: Option 1 (Celery link)** -- decouples signal generation from risk computation, allows independent retry.

### Pattern 5: Health Check via Worker Ping
**What:** Replace local `import torch` with Celery inspector ping to GPU worker
**When to use:** PRED-05 -- correct GPU health reporting
**Example:**
```python
# api/health.py - GPU check section
try:
    inspect = celery_app.control.inspect(timeout=3.0)
    ping_result = inspect.ping() or {}
    gpu_workers = {k: v for k, v in ping_result.items() if "gpu" in k.lower()}
    if gpu_workers:
        # Optionally: broadcast a lightweight probe task to check torch/CUDA
        components["gpu"] = {"available": True, "workers": list(gpu_workers.keys())}
    else:
        components["gpu"] = {"available": False, "note": "no GPU workers responding"}
except Exception as e:
    components["gpu"] = {"available": False, "note": f"inspect failed: {e}"}
```

### Anti-Patterns to Avoid
- **Loading model in API container:** Models with torch need GPU worker; API should only dispatch tasks
- **Synchronous prediction:** predict endpoint should be async (202 Accepted) -- model loading takes seconds
- **Local torch import for health check:** API container doesn't have torch; use worker ping instead
- **Strategy evaluation without feature computation:** Always call `FeatureEngine.compute()` or `compute_from_df()` first
- **Ignoring read_ohlcv index format:** `read_ohlcv` returns DataFrame with time as INDEX, not column -- features computation handles this, but downstream code must not assume time is a column

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Prediction -> Signal conversion | Custom mapping code | `ModelStrategy.evaluate()` | Already handles hold filtering, position tracking, direction changes |
| Signal risk checking + persistence | Manual SQL + risk checks | `SignalPipeline.process()` | Handles hot-reload rules, risk evaluation, persist, deliver, commit |
| Strategy reconstruction from DB | Manual if/elif chain | `VotingStrategyFactory.from_config()` for voting; `RuleStrategy(config=record.config)` for rule | Factory already handles config parsing and validation |
| Feature computation for live data | Raw SQL + pandas | `FeatureEngine.compute(symbol, market, interval)` | Loads from DB, computes all features, handles index format |
| Task scheduling | Manual cron in code | `celery_app.conf.beat_schedule` + `celery.chain` | RedBeat persists schedule in Redis, chain/link for callbacks |

## Common Pitfalls

### Pitfall 1: read_ohlcv Returns Time as Index
**What goes wrong:** Code assumes `time` is a column but it's actually the DataFrame index
**Why it happens:** `read_ohlcv` sets time as index; FeatureEngine works fine with this, but any code accessing `df["time"]` will KeyError
**How to avoid:** Use `df.index` for time values, or call `df.reset_index()` if you need time as a column
**Warning signs:** KeyError on "time" column access

### Pitfall 2: VotingStrategy Config Must Include symbol/market
**What goes wrong:** VotingStrategy.__init__ reads `config.get("symbol", "")` -- empty string breaks things
**Why it happens:** StrategyRecord stores config in JSONB but symbol/market are separate columns
**How to avoid:** When reconstructing VotingStrategy from StrategyRecord, merge record.symbol/market into config dict before passing to constructor
**Warning signs:** Empty symbol/market in generated signals

### Pitfall 3: API Container Has No torch
**What goes wrong:** Any code path in the API container that imports torch will fail
**Why it happens:** torch is only in the `gpu` optional dependency group, only installed in GPU worker
**How to avoid:** All torch-dependent code must run in GPU worker tasks; API only dispatches tasks
**Warning signs:** ImportError for torch in API logs

### Pitfall 4: SignalPipeline Commits Transaction
**What goes wrong:** `SignalPipeline.process()` calls `self._db.commit()` -- if you create a new session in the task, this is fine, but if you share a session, intermediate state may be committed
**Why it happens:** Pipeline is designed for one-signal-at-a-time processing with its own transaction boundary
**How to avoid:** Use a dedicated session for SignalPipeline, or ensure all work within a single pipeline.process() call is idempotent
**Warning signs:** Partial commits on errors

### Pitfall 5: Strategy Types in DB Include "voting" But API Only Validates "model"/"rule"
**What goes wrong:** `_VALID_STRATEGY_TYPES = {"model", "rule"}` in strategies.py -- creating a voting strategy via API will 400
**Why it happens:** VotingStrategy was added in Phase 10 but strategy API was Phase 7
**How to avoid:** Add "voting" to `_VALID_STRATEGY_TYPES` in strategies.py
**Warning signs:** 400 error when trying to create a voting strategy via API

### Pitfall 6: Model Status Gating for Prediction
**What goes wrong:** Attempting to predict with a model in "training" or "failed" status
**Why it happens:** Current placeholder doesn't check status
**How to avoid:** Validate `mv.status in ("ready", "shadow", "active")` before dispatching prediction task
**Warning signs:** Model load failures in GPU worker

### Pitfall 7: Celery Link Routing
**What goes wrong:** Linked task runs on wrong queue (e.g., risk update runs on GPU queue)
**Why it happens:** Celery `link` inherits the parent task's queue by default
**How to avoid:** Use `.si()` (immutable signature) with explicit `queue="cpu"` when linking tasks across queues
**Warning signs:** CPU task appearing in GPU worker logs

## Code Examples

### Reconstructing Strategy from StrategyRecord
```python
# Source: existing patterns in gpu_tasks.py and voting_strategy.py
def reconstruct_strategy(record: StrategyRecord) -> BaseStrategy:
    """Reconstruct a strategy instance from a DB record."""
    if record.strategy_type == "rule":
        return RuleStrategy(
            config=record.config,
            strategy_id=record.id,
        )
    elif record.strategy_type == "voting":
        # Merge symbol/market from record into config
        config = {**record.config, "symbol": record.symbol, "market": record.market, "interval": record.interval}
        return VotingStrategy(config=config, strategy_id=record.id)
    else:
        raise ValueError(f"Cannot evaluate strategy_type={record.strategy_type} on CPU")
```

### Celery Chain: Fetch -> Evaluate -> Risk Update
```python
# Source: celery docs on chain/link
from celery import chain

# In beat_schedule or as callback from fetch_market_data:
chain(
    fetch_market_data.s("crypto_spot", "1h"),
    evaluate_active_strategies.s(),  # receives fetch result as first arg
    trigger_risk_update.si(),         # immutable -- doesn't need fetch result
).apply_async()
```

### GPU Health Probe Task
```python
# workers/gpu_tasks.py
@celery_app.task(name="poseidon.workers.gpu_tasks.gpu_health_probe")
def gpu_health_probe() -> dict:
    """Lightweight task to report GPU/torch status from worker."""
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_free_mb": torch.cuda.mem_get_info()[0] // (1024*1024) if torch.cuda.is_available() else 0,
        }
    except ImportError:
        return {"torch_version": None, "cuda_available": False}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Placeholder predict endpoint | GPU task dispatch + SignalPipeline | Phase 18 | Enables real-time prediction flow |
| No Beat strategy evaluation | Post-fetch callback evaluation | Phase 18 | Active strategies auto-generate signals |
| Local torch import for health | Worker ping + probe task | Phase 18 | Correct GPU status in API container |

## Project Constraints (from CLAUDE.md)

- **Language:** Reply in Traditional Chinese (zh-TW); code and technical terms in English
- **Package manager:** Use `uv` (not pip)
- **Testing:** All tests run on stormtrooper via SSH in Docker containers, never locally (no torch/GPU on Mac)
- **Docker:** Do not run docker compose locally -- only on stormtrooper
- **GSD files:** Planning files in `poseidon/.planning/`

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 with pytest-asyncio |
| Config file | pyproject.toml [tool.pytest] section (if exists) or default |
| Quick run command | `ssh stormtrooper "cd ~/Projects/poseidon && docker compose exec gpu-worker python -m pytest tests/test_FILE.py -x"` |
| Full suite command | `ssh stormtrooper "cd ~/Projects/poseidon && docker compose exec gpu-worker python -m pytest tests/ -x"` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRED-01 | Predict endpoint dispatches GPU task and returns 202 | unit (mock celery) | `pytest tests/test_api_models.py::test_predict_dispatch -x` | Partial (file exists, test needs adding) |
| PRED-02 | Predictions above threshold -> Signal via SignalPipeline | unit | `pytest tests/test_predict_signal.py -x` | Wave 0 |
| PRED-03 | Active strategies evaluated by Beat callback | unit | `pytest tests/test_strategy_evaluation.py -x` | Wave 0 |
| PRED-04 | Signal generation triggers VaR recalc | unit | `pytest tests/test_signal_risk_trigger.py -x` | Wave 0 |
| PRED-05 | Health endpoint GPU via worker ping | unit (mock inspect) | `pytest tests/test_api_health.py::test_gpu_worker_ping -x` | Partial (file exists, test needs adding) |

### Sampling Rate
- **Per task commit:** Quick run on specific test file
- **Per wave merge:** Full test suite
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_predict_signal.py` -- covers PRED-01, PRED-02 (GPU task predict + signal conversion)
- [ ] `tests/test_strategy_evaluation.py` -- covers PRED-03 (Beat strategy evaluation task)
- [ ] `tests/test_signal_risk_trigger.py` -- covers PRED-04 (downstream risk trigger)
- [ ] Update `tests/test_api_health.py` -- covers PRED-05 (GPU health via worker ping)
- [ ] Update `tests/test_api_models.py` -- covers PRED-01 (predict endpoint dispatch)

## Open Questions

1. **Synchronous vs Asynchronous predict response**
   - What we know: Train endpoint returns 202 with task_id (async pattern). Predict could follow same pattern.
   - What's unclear: User may want synchronous prediction result (wait for response) for interactive use
   - Recommendation: Start with 202 async (consistent with train). Add a GET /models/{version_id}/predict/{task_id}/result polling endpoint if needed later.

2. **Confidence threshold configuration**
   - What we know: PRED-02 requires filtering by confidence threshold. No threshold exists in config.py yet.
   - What's unclear: What default threshold value? Per-strategy or global?
   - Recommendation: Add `predict_confidence_threshold: float = 0.6` to Settings. Start global, per-strategy can be added later.

3. **Beat schedule for strategy evaluation -- direct schedule vs callback**
   - What we know: Current Beat has no strategy evaluation. Fetch tasks run on schedule (hourly for crypto, daily for stocks).
   - What's unclear: Should strategy evaluation be a fixed Beat schedule or a callback after fetch?
   - Recommendation: Use Celery `link` callback from `fetch_market_data` to `evaluate_active_strategies`. This ensures evaluation happens only after fresh data is available, not on an arbitrary timer.

4. **VotingStrategy state persistence across evaluations**
   - What we know: VotingStrategy tracks `_position_direction`, `_bars_since_exit`, watermarks in memory. A fresh instance on each Beat evaluation loses this state.
   - What's unclear: How to persist voting strategy position state across Beat evaluations
   - Recommendation: Read current position from VirtualPortfolio (already rebuilt from DB). For bars_since_exit and watermarks, store in StrategyRecord.config JSONB or a separate strategy_state table. This is a known limitation to address.

## Sources

### Primary (HIGH confidence)
- `src/poseidon/api/models.py` -- current placeholder predict endpoint (lines 198-220)
- `src/poseidon/workers/gpu_tasks.py` -- train_model and run_model_backtest task patterns
- `src/poseidon/workers/celery_app.py` -- Beat schedule, task routing config
- `src/poseidon/risk/pipeline.py` -- SignalPipeline.process() workflow
- `src/poseidon/signals/repository.py` -- SignalRepository.save()
- `src/poseidon/strategies/model_strategy.py` -- ModelStrategy.evaluate() prediction->signal
- `src/poseidon/strategies/voting_strategy.py` -- VotingStrategy.evaluate()
- `src/poseidon/strategies/rule_strategy.py` -- RuleStrategy.evaluate()
- `src/poseidon/api/health.py` -- current (broken) GPU health check
- `src/poseidon/ml/base.py` -- BaseModel.predict() interface
- `src/poseidon/ml/manager.py` -- ModelManager.get_version(), get_active()
- `src/poseidon/models/strategy.py` -- StrategyRecord with active flag

### Secondary (MEDIUM confidence)
- Celery documentation on task chaining and link callbacks
- RedBeat scheduler behavior for schedule modifications

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- follows existing patterns (train_model task, SignalPipeline, FeatureEngine)
- Pitfalls: HIGH -- identified from direct code inspection of existing codebase

**Research date:** 2026-03-28
**Valid until:** 2026-04-28 (stable -- internal architecture, no external API changes)
