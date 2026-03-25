# Technology Stack

**Project:** Poseidon v2.0 -- Strategy Pivot (Voting + AutoResearch + Regime + Parameter Search)
**Researched:** 2026-03-25
**Mode:** Incremental additions to existing validated stack

## Existing Stack (DO NOT ADD -- Already Present)

Confirmed in `pyproject.toml` and source code. No changes needed for these.

| Existing | Version | Purpose | v2.0 Role |
|----------|---------|---------|-----------|
| FastAPI | >=0.115 | REST API | API endpoints for experiment management |
| Celery + Redis | >=5.4 / >=5.0 | Task queue, scheduling | AutoResearch experiment workers |
| PostgreSQL + SQLAlchemy | >=2.0 | Persistence, ORM | Optuna RDBStorage backend |
| Optuna | >=4.8 | Bayesian optimization | Parameter search + experiment tracking |
| XGBoost | >=2.0 (gpu extra) | ML models | Regime classification (XGBoostRegimeModel exists) |
| scikit-learn | >=1.4 (gpu extra) | ML utilities | Feature scaling, metrics |
| pandas_ta + TA-Lib | >=0.3 / >=0.6 | Technical indicators | FeatureEngine already computes all Nunchi signals |
| pandas | >=2.2 | Data manipulation | Core data pipeline |
| Pydantic | >=2.0 (via FastAPI) | Schema validation | DSL schema, API models |
| Alembic | >=1.13 | DB migrations | Schema changes for experiment tables |

**Existing architecture assets directly reusable in v2.0:**
- `BaseStrategy` ABC with `evaluate(features) -> list[Signal]` -- VotingStrategy will implement this
- `RuleStrategy` evaluates JSON DSL condition trees (`all/any/none` combinators) -- sub-signals for voting
- `FeatureEngine` with `DEFAULT_FEATURES` (SMA/EMA/RSI/MACD/Bollinger/ATR) and `REGIME_FEATURES` (20 volatility features)
- `BacktestRunner` with bar-by-bar event loop -- immutable evaluation layer for autoresearch
- `WalkForwardAnalyzer` with WFE >= 50% threshold -- validation gate for experiments
- `GridSearchOptimizer` + `BayesianOptimizer` (Optuna TPE) -- parameter search already works
- `XGBoostRegimeModel` (3-class: low_vol/medium_vol/high_vol with 20 regime features)
- DSL condition registry: `indicator_above/below`, `price_crosses`, `indicator_crosses`, `price_change_pct`, `volume_spike`
- `compute_metrics()` returns 11 metrics including sharpe_ratio, max_drawdown, win_rate, profit_factor

## Recommended Stack Additions

### 1. optuna-dashboard (Experiment Visualization)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| optuna-dashboard | >=0.20 | Web dashboard for Optuna experiment history | Poseidon already uses Optuna + PostgreSQL. Dashboard connects to the same RDBStorage with zero config. Provides optimization history plots, parameter importance, Pareto frontiers for multi-objective studies. v0.20+ supports PostgreSQL natively. |

**Confidence:** HIGH -- Optuna is already a dependency; dashboard is the official companion.

**What Nunchi does instead:** Flat TSV files (`results.tsv`) with columns: `commit | score | sharpe | max_dd | status | description`. This works for a git-based AI loop where the agent reads its own history, but is inadequate for a production system that needs queryability, visualization, and cross-study comparison.

**What Karpathy does instead:** Same TSV pattern in `results.tsv` with columns: `commit | val_bpb | memory_gb | status | description`. The n-autoresearch fork replaces this with a queryable KV store, acknowledging the flat-file limitation.

**Poseidon's advantage:** Already has PostgreSQL + Optuna. Switching from in-memory to `RDBStorage` is a one-line change. Dashboard deploys as a Docker sidecar.

**Integration:**
```python
# In BayesianOptimizer -- current (in-memory):
study = optuna.create_study(direction="maximize", sampler=sampler)

# Refactored (persistent):
storage = optuna.storages.RDBStorage(
    url="postgresql+psycopg2://poseidon:pass@db:5432/poseidon",
    engine_kwargs={"pool_size": 5},
)
study = optuna.create_study(
    study_name=f"autoresearch_{strategy_name}_{timestamp}",
    direction="maximize",
    sampler=sampler,
    storage=storage,
    load_if_exists=True,  # Resume interrupted studies
)
```

**Docker Compose addition:**
```yaml
optuna-dashboard:
  image: ghcr.io/optuna/optuna-dashboard
  ports:
    - "8080:8080"
  command: >
    optuna-dashboard
    postgresql+psycopg2://poseidon:${POSTGRES_PASSWORD}@db:5432/poseidon
    --host 0.0.0.0 --port 8080
  depends_on:
    - db
```

### 2. hmmlearn (HMM Regime Detection -- Optional Enhancement)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| hmmlearn | >=0.3.3 | Hidden Markov Model for regime label generation | XGBoostRegimeModel treats each bar independently. HMM captures temporal persistence of regime states and transition probabilities. 2025 research confirms XGBoost+HMM hybrid produces more robust regime classification. scikit-learn-compatible API. |

**Confidence:** MEDIUM -- hmmlearn is stable (v0.3.3 has Python 3.12 wheels) but in limited maintenance mode. If it breaks on future Python, fallback to pure XGBoost regime (which already works).

**What Nunchi's regime_mm.py teaches us:** Nunchi uses a simpler 4-regime approach:
- Classifies by annualized volatility thresholds (not ML)
- 4 regimes: low (<0.30), normal (<0.60), high (<1.00), extreme (>=1.00)
- Hysteresis logic: requires 3 consecutive measurements before downshifting
- Each regime maps to different parameters: spread, position size, stop loss

Poseidon should adopt the hysteresis pattern regardless of whether HMM is added:
```python
# From Nunchi regime_mm.py -- prevent rapid regime switching
if target_regime < current_regime:
    consecutive_lower += 1
    if consecutive_lower >= HYSTERESIS_COUNT:  # 3
        current_regime = target_regime
        consecutive_lower = 0
else:
    consecutive_lower = 0
    current_regime = target_regime
```

**Usage pattern:**
- Training time: HMM discovers regime states from feature matrix (unsupervised)
- These HMM-labeled states become training targets for XGBoostRegimeModel
- Inference time: XGBoost predicts regime (fast, no HMM needed at runtime)
- Hysteresis filter applied on top of XGBoost predictions

### 3. No New Dependencies for Core Features

**Voting Strategy:** Pure Python. `VotingStrategy(BaseStrategy)` wraps N `RuleStrategy` instances, counts signals, emits if >= min_votes. Nunchi's strategy.py implements this as `sum(signals) >= MIN_VOTES` where `MIN_VOTES = 4` out of 6 signals. This is trivial logic, not a library need.

**AutoResearch Loop:** Architecture pattern, not a library. The three-layer structure (fixed/variable/guide) maps directly to existing Poseidon components:
- Fixed: `BacktestRunner` + `compute_metrics()` + `WalkForwardAnalyzer`
- Variable: `RuleConfig` JSON (not Python code like Nunchi/Karpathy)
- Guide: `program.md` template rendered with current experiment state

**Composite Scoring Function:** Nunchi's scoring formula (from `prepare.py`):
```
score = sharpe * sqrt(min(trades / 50, 1.0)) - drawdown_penalty - turnover_penalty
# Hard cutoffs: < 10 trades = -999, > 50% drawdown = -999, < 50% final equity = -999
```
This is a new function in `backtest/metrics.py`, not a library.

**Git Audit Trail:** `subprocess.run(["git", "commit", ...])` for atomic experiment commits. Standard library.

**Program.md Template:** Python f-strings or `str.format()` for template rendering. Jinja2 is tempting but overkill -- the template is a single file with simple variable substitution. If templates become complex later, Jinja2 is already a transitive dependency of FastAPI.

## What NOT to Add

| Library | Why Not | Use Instead |
|---------|---------|-------------|
| MLflow | Separate tracking server, overkill for single developer. Optuna RDBStorage covers experiment tracking with zero new infrastructure. | Optuna RDBStorage + optuna-dashboard |
| Weights & Biases | Cloud dependency. Single developer on internal network. | Optuna RDBStorage + optuna-dashboard |
| Neptune | Cloud dependency, same argument as W&B. | Optuna RDBStorage |
| Ray / Dask | Distributed compute framework. Poseidon runs on one machine (stormtrooper) with Celery already available. | Celery workers + Optuna shared storage |
| LangChain / LLM libraries | AutoResearch uses Claude Code as an external agent (like Karpathy/Nunchi), not an embedded LLM. The AI modifies files and runs commands -- no LLM library needed inside Poseidon. | Claude Code CLI called from shell or Celery task |
| iii framework | n-autoresearch uses this for multi-GPU orchestration. Poseidon is single-GPU. | Celery + Optuna |
| scikit-learn VotingClassifier | Designed for ML model ensembles (predict_proba averaging), not rule-based signal voting. Wrong abstraction. | Custom VotingStrategy class |
| backtrader / zipline / bt | Alternative backtesting frameworks. Poseidon's BacktestRunner is already built and integrated. | Existing BacktestRunner |
| scipy | Nunchi uses scipy for stats but only minimally. Poseidon's FeatureEngine handles all needed calculations with pandas/numpy. | Existing pandas/numpy |
| Deep learning regime models | LSTM/Transformer for regime classification is overkill. XGBoost handles 3-4 class classification well. Existing XGBoostRegimeModel validates this. | XGBoost + optional HMM hybrid |
| Jinja2 (explicit) | Template rendering for program.md. Already a transitive dep of FastAPI. Simple variable substitution doesn't need full templating. | Python f-strings. Add Jinja2 explicitly only if templates grow complex. |
| GitPython | Git operations library. subprocess calls to git are simpler and sufficient for atomic commits. | `subprocess.run(["git", "commit", ...])` |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Experiment tracking | Optuna RDBStorage + dashboard | MLflow Tracking Server | Requires separate server process, database, UI. Optuna already in stack and stores params+metrics natively. |
| Experiment tracking | Optuna RDBStorage + dashboard | Flat TSV (Nunchi/Karpathy pattern) | TSV works for autonomous AI loops reading their own history. Lacks queryability, visualization, and resume for a persistent system. |
| Regime detection | XGBoost + optional HMM | Pure threshold (Nunchi pattern) | Threshold works but requires manual calibration per asset/timeframe. XGBoost learns thresholds from data. |
| Regime detection | XGBoost + optional HMM | K-means clustering | Unsupervised. Cluster labels lack semantic meaning. Need manual interpretation every retraining. |
| Regime detection | XGBoost + optional HMM | GMM (Gaussian Mixture) | Better than K-means but still no temporal structure. HMM models state transitions. |
| Voting aggregation | Custom VotingStrategy class | DSL `vote` combinator | DSL evaluates conditions to bool. Voting needs counting + threshold + confidence aggregation. Class-level pattern is cleaner. |
| Parameter search | Optuna multi-objective | Grid search | Already have GridSearchOptimizer but it's exhaustive. Optuna TPE is 10-100x more efficient for multi-dimensional spaces. |
| AutoResearch loop | Celery task + Optuna | Custom scheduler | Celery already handles periodic tasks, retries, worker management. No need for custom implementation. |

## Complete New Dependencies Summary

### Production (add to `[project.optional-dependencies].gpu`)

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| hmmlearn | >=0.3.3 | ~2MB | HMM regime detection (optional enhancement) |

### Development (add to `[project.optional-dependencies].dev`)

| Package | Version | Size | Purpose |
|---------|---------|------|---------|
| optuna-dashboard | >=0.20 | ~15MB | Experiment visualization web UI |

### Installation

```bash
# On stormtrooper (GPU production server)
uv add --optional gpu hmmlearn>=0.3.3

# Development/monitoring (any machine with access to PostgreSQL)
uv add --dev optuna-dashboard>=0.20
```

**Total new package dependencies: 2** (one optional, one dev-only).

## Key Architectural Insight from Reference Repos

Both Karpathy's autoresearch and Nunchi's auto-researchtrading follow the same three-layer pattern:

| Layer | Karpathy (LLM training) | Nunchi (trading) | Poseidon Equivalent |
|-------|------------------------|------------------|---------------------|
| **Fixed evaluation** | `prepare.py`: BPB metric, tokenizer, data splits (IMMUTABLE) | `prepare.py` + `backtest.py`: scoring, slippage, fees (IMMUTABLE) | `BacktestRunner` + `compute_metrics()` + `WalkForwardAnalyzer` (IMMUTABLE during autoresearch) |
| **Mutable experiment** | `train.py`: model architecture, optimizer, hyperparams (Python code) | `strategy.py`: signal logic, voting thresholds, indicator params (Python code) | `RuleConfig` JSON: condition trees, thresholds, indicator periods (structured data, NOT code) |
| **Agent instructions** | `program.md`: constraints, research directions, scoring rules | `program.md`: constraints, signal suggestions, scoring formula | NEW: `program.md` template (generated per run with current regime, best params, history) |
| **Result logging** | `results.tsv`: commit, val_bpb, memory, status, description | `results.tsv`: commit, score, sharpe, max_dd, status, description | Optuna RDBStorage: all trial params, metrics, user attributes in PostgreSQL |

**Poseidon's structural advantage over both repos:** The mutable layer is JSON data (`RuleConfig`), not Python source code. This means:
1. Parameter mutations are bounded (can't accidentally break imports or syntax)
2. Validation is built-in (Pydantic schema enforcement on every config)
3. Rollback is trivial (restore previous JSON, no git reset needed)
4. The search space is explicitly defined (Optuna param_space maps directly to JSON fields)

**Nunchi's key insight for Poseidon:** The 6-signal voting strategy's evolution (documented in STRATEGIES.md across 103 experiments) shows that **simplification beats complexity**:
- Removing pyramiding: +0.4 score
- Removing funding boost: +0.7 score
- Removing BTC lead-lag filter: +0.4 score
- Removing variable sizing: +1.7 score (largest single gain)
- RSI period 14->8: +5.0 score (largest parameter change gain)

**Implication for Poseidon's autoresearch:** The search space should include ablation (removing signals/parameters) alongside additive exploration. The n-autoresearch fork explicitly implements this with adaptive search modes: explore, exploit, combine, ablation.

## Scoring Function Design (From Nunchi's prepare.py)

Poseidon needs a composite scoring function for autoresearch. Based on Nunchi's validated approach:

```python
def compute_composite_score(metrics: dict, target_trades: int = 50) -> float:
    """Nunchi-inspired composite score for autoresearch optimization.

    Combines Sharpe with trade frequency weighting and risk penalties.
    Hard cutoffs prevent degenerate strategies from scoring.
    """
    sharpe = metrics["sharpe_ratio"]
    max_dd = metrics["max_drawdown"]
    trade_count = metrics["trade_count"]
    total_return = metrics["total_return"]

    # Hard cutoffs (from Nunchi prepare.py)
    if trade_count < 10:
        return -999.0
    if max_dd > 0.50:
        return -999.0
    if total_return < -0.50:
        return -999.0

    # Trade frequency factor: sqrt(min(trades/target, 1.0))
    trade_factor = (min(trade_count / target_trades, 1.0)) ** 0.5

    # Drawdown penalty (from Nunchi: penalize > 15%)
    dd_penalty = max(0, (max_dd - 0.15) * 10)

    # Turnover penalty (if available)
    turnover = metrics.get("annual_turnover", 0)
    turnover_penalty = max(0, (turnover - 50) * 0.01)

    return sharpe * trade_factor - dd_penalty - turnover_penalty
```

This is a code addition to `backtest/metrics.py`, not a new dependency.

## Sources

- [Nunchi auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading) -- strategy.py (6-signal voting), prepare.py (scoring function), program.md (agent instructions), results.tsv (experiment logging), regime_mm.py (4-regime hysteresis), STRATEGIES.md (103-experiment evolution history)
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) -- Three-layer pattern (prepare.py/train.py/program.md), autonomous loop protocol, simplicity criterion, results.tsv format
- [n-autoresearch](https://github.com/iii-hq/n-autoresearch) -- Structured experiment tracking, adaptive search modes (explore/exploit/combine/ablation), multi-GPU orchestration patterns
- [Optuna RDB Storage docs](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html) -- PostgreSQL backend configuration, study persistence, distributed optimization
- [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) -- Real-time visualization, PostgreSQL support
- [Optuna Multi-Objective tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html) -- Pareto optimization for Sharpe + drawdown
- [hmmlearn 0.3.3](https://pypi.org/project/hmmlearn/) -- Python 3.12 wheels, scikit-learn-compatible API
- [Regime-switching XGBoost (2025)](https://arxiv.org/html/2510.03236v1) -- Distributional spectral clustering + XGBoost for regime assignment
- Poseidon codebase: `pyproject.toml`, `backtest/optimizer.py`, `backtest/metrics.py`, `backtest/walk_forward.py`, `ml/implementations/xgboost_regime.py`, `strategies/dsl/`, `strategies/rule_strategy.py`, `strategies/base.py`, `data/features/`, `data/feature_engine.py`

---
*Stack research for: Poseidon v2.0 Strategy Pivot*
*Researched: 2026-03-25*
