# Phase 13: Regime Classification (Optional, Gated) - Research

**Researched:** 2026-03-26
**Domain:** Market regime detection, gated feature enablement, per-regime parameter routing
**Confidence:** HIGH

## Summary

Phase 13 builds a regime-conditional trading system on top of existing, well-tested infrastructure. The XGBoostRegimeModel (3-class volatility classifier) already exists with full BaseModel ABC compliance. VotingStrategy already accepts `min_votes` and `position_pct` as constructor params. The holdout, composite scoring, and experiment tracking infrastructure from Phase 11/12 are all in place.

The core new work is: (1) a label generation utility that assigns `low_vol/medium_vol/high_vol` labels via percentile splitting on `realized_vol_20`, (2) a `RegimeRouter` class that wraps VotingStrategy and swaps `min_votes`/`position_pct` based on regime predictions, (3) a per-regime Optuna search that extends ParameterSearchPipeline, and (4) an outperformance gate that compares regime-routed vs. static baseline on holdout data.

**Primary recommendation:** Build RegimeRouter as a thin wrapper/decorator around VotingStrategy that intercepts evaluate() calls, queries the regime model for current regime, and applies per-regime config overrides (min_votes, position_pct only). The gate comparison is a simple paired backtest: run the same holdout data through both static and regime-routed strategies, compare composite_score.

## Project Constraints (from CLAUDE.md)

- **Language:** Replies in Traditional Chinese (zh-TW), code/technical terms in English
- **Package manager:** `uv` (not pip)
- **Testing:** All tests run on stormtrooper via SSH, never locally (no torch/GPU on Mac)
- **Docker:** Do not run docker compose locally -- only on stormtrooper
- **GSD files:** Planning files in `poseidon/.planning/`

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Keep existing 3-class volatility taxonomy: low_vol / medium_vol / high_vol. Do not expand to 4-class (trending/ranging/volatile/low-vol) -- start simple, expand later if proven useful.
- **D-02:** Label generation via percentile-based splitting on `realized_vol_20`: <33rd percentile = low_vol, 33-66th = medium_vol, >66th = high_vol. Simple, transparent, no extra model needed.
- **D-03:** Use existing `XGBoostRegimeModel` (already implemented in `src/poseidon/ml/implementations/xgboost_regime.py`) and existing regime features (`src/poseidon/data/features/regime.py`).
- **D-04:** RegimeRouter adjusts only two parameters per regime: `min_votes` and `position_pct`. Sub-signal composition and other parameters remain static across regimes.
- **D-05:** Sensible defaults: high_vol -> min_votes=5, position_pct=0.05 (conservative); medium_vol -> min_votes=4, position_pct=0.08 (standard); low_vol -> min_votes=3, position_pct=0.10 (aggressive).
- **D-06:** Per-regime parameters also searchable via Optuna (using Phase 11 pipeline). Defaults serve as starting point, Optuna can discover better values.
- **D-07:** Gate metric: `composite_score` (Phase 10 D-05/D-06). Regime-conditional strategy's composite_score must be strictly greater than static VotingStrategy baseline on OOS data.
- **D-08:** Auto-disable mechanism: `RegimeRouter` has `enabled: bool` flag. If OOS test fails, set `enabled=False` -- router passes through to static config. Model and config are preserved (not deleted), just bypassed.
- **D-09:** Gate evaluation runs on holdout data (last 20%) after regime model training and per-regime parameter search complete.
- **D-10:** Regime model trained independently and once -- produces a static model. AutoResearch per-regime parameter search uses this static model's predictions, does NOT retrain regime model per trial.
- **D-11:** Share Phase 11 HoldoutConfig (last 20%). Regime model trains on first 80%, outperformance gate tests on last 20% OOS. Consistent with strategy optimization holdout.
- **D-12:** Flow: (1) Generate percentile labels on training data -> (2) Train XGBoostRegimeModel -> (3) Per-regime Optuna search for min_votes/position_pct -> (4) Run gate comparison on holdout -> (5) Enable or disable.

### Claude's Discretion
- Exact percentile thresholds (33/66 as starting point, may adjust)
- RegimeRouter class structure and interface design
- How per-regime Optuna search integrates with existing ParameterSearchPipeline
- Gate comparison test methodology details (e.g., paired or unpaired comparison)
- API endpoints for regime model management (if any needed this phase)

### Deferred Ideas (OUT OF SCOPE)
- 4-class regime (trending/ranging/volatile/low-vol) -- if 3-class effective, expand later
- Per-regime different sub-signal combinations -- too complex, only adjust parameters
- Regime model automatic retraining -- future consideration
- Regime-aware autoresearch (Phase 12 deferred item) -- this phase builds router, naturally usable later
- AI-driven regime detection (deep learning / HMM) -- future option to replace XGBoost
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RGME-01 | XGBoostRegimeModel classifies market regime from feature data | Existing XGBoostRegimeModel already implements train/predict/validate/save/load. Need label generator utility and training pipeline integration. |
| RGME-02 | RegimeRouter selects VotingStrategy configuration based on detected regime | New RegimeRouter class wrapping VotingStrategy; per-regime config map with defaults (D-05); Optuna search for per-regime params (D-06). |
| RGME-03 | Outperformance gate -- regime routing must beat static no-regime baseline on OOS data, auto-disabled if fails | Gate comparison using composite_score on holdout data; enabled/disabled flag on RegimeRouter; paired backtest methodology. |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| xgboost | (project dep) | 3-class regime classifier | Already used via XGBoostRegimeModel |
| optuna | (project dep) | Per-regime parameter search | Already used in ParameterSearchPipeline |
| pandas | (project dep) | Feature data, percentile computation | Core data layer |
| numpy | (project dep) | Numerical operations | Core dependency |
| joblib | (project dep) | Model serialization | Already used in XGBoostRegimeModel.save/load |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | (project dep) | classification_report for regime model validation metrics | Optional, for detailed per-class metrics |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Percentile labels | K-means clustering | K-means is unstable across runs; percentile is deterministic (D-02 locks percentile) |
| XGBoost classifier | HMM / deep learning | Much more complex; deferred per CONTEXT.md |
| Per-regime Optuna | Manual grid search | Optuna is already integrated; no reason to downgrade |

**Installation:** No new packages required. All dependencies already in project.

## Architecture Patterns

### Recommended Project Structure
```
src/poseidon/
├── ml/
│   └── implementations/
│       └── xgboost_regime.py          # Existing -- no changes needed
├── data/
│   └── features/
│       └── regime.py                   # Existing -- no changes needed
├── strategies/
│   ├── voting_strategy.py              # Existing -- no changes needed
│   └── regime_router.py               # NEW: RegimeRouter wrapper
├── backtest/
│   ├── regime_labels.py               # NEW: Percentile label generator
│   ├── regime_search.py               # NEW: Per-regime Optuna search
│   └── regime_gate.py                 # NEW: Outperformance gate evaluation
│   ├── param_search.py                # Existing -- reuse, don't modify
│   ├── holdout.py                     # Existing -- reuse HoldoutConfig
│   └── metrics.py                     # Existing -- reuse compute_composite_score
└── ...
```

### Pattern 1: RegimeRouter as Strategy Wrapper
**What:** RegimeRouter wraps a VotingStrategy and intercepts evaluate() to apply per-regime config overrides before delegation.
**When to use:** When regime model predictions should modify strategy behavior without changing the strategy interface.
**Example:**
```python
class RegimeRouter:
    """Wraps VotingStrategy, applies per-regime config overrides."""

    def __init__(
        self,
        base_config: dict,
        regime_model: XGBoostRegimeModel,
        regime_configs: dict[str, dict],  # {"low_vol": {"min_votes": 3, "position_pct": 0.10}, ...}
        enabled: bool = True,
    ):
        self.base_config = base_config
        self.regime_model = regime_model
        self.regime_configs = regime_configs
        self.enabled = enabled

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        config = copy.deepcopy(self.base_config)

        if self.enabled:
            regime_pred = self.regime_model.predict(features)
            current_regime = regime_pred.iloc[-1]["prediction"]
            if current_regime in self.regime_configs:
                config.update(self.regime_configs[current_regime])

        strategy = VotingStrategyFactory.from_config(config)
        return strategy.evaluate(features)
```

### Pattern 2: Percentile-Based Label Generation
**What:** Generate regime labels from `realized_vol_20` using percentile thresholds computed on training data only.
**When to use:** Before training XGBoostRegimeModel -- converts continuous vol into categorical targets.
**Example:**
```python
def generate_regime_labels(
    ohlcv: pd.DataFrame,
    feature_engine: FeatureEngine,
    low_pct: float = 33.0,
    high_pct: float = 66.0,
) -> pd.Series:
    """Generate regime labels from realized_vol_20 percentiles."""
    features = feature_engine.compute(ohlcv)
    vol = features["realized_vol_20"].dropna()

    low_threshold = np.percentile(vol, low_pct)
    high_threshold = np.percentile(vol, high_pct)

    labels = pd.Series(1, index=vol.index, name="regime_label")  # default medium_vol
    labels[vol < low_threshold] = 0  # low_vol
    labels[vol > high_threshold] = 2  # high_vol
    return labels
```

### Pattern 3: Gated Feature with Pass-Through Fallback
**What:** The gate pattern: run both strategies on holdout, compare, enable or disable.
**When to use:** After regime model training and per-regime parameter search complete.
**Example:**
```python
def evaluate_regime_gate(
    ohlcv_holdout: pd.DataFrame,
    static_config: dict,
    regime_router: RegimeRouter,
    feature_engine: FeatureEngine,
    risk_engine: RiskEngine,
    cost_model: CostModel,
    initial_capital: float,
) -> bool:
    """Return True if regime routing beats static baseline on holdout."""
    # Run static baseline backtest
    static_strategy = VotingStrategyFactory.from_config(static_config)
    static_result = run_backtest(static_strategy, ohlcv_holdout, ...)
    static_score = compute_composite_score(static_result.metrics)

    # Run regime-routed backtest
    regime_router.enabled = True
    regime_result = run_backtest(regime_router, ohlcv_holdout, ...)
    regime_score = compute_composite_score(regime_result.metrics)

    passed = regime_score > static_score
    regime_router.enabled = passed
    return passed
```

### Anti-Patterns to Avoid
- **Retraining regime model per Optuna trial:** D-10 explicitly forbids this. Train once, use static predictions for all trials.
- **Modifying VotingStrategy class:** RegimeRouter wraps it externally; VotingStrategy code stays unchanged.
- **Computing percentile thresholds on holdout data:** Thresholds must be computed on training data only to prevent data leakage.
- **Deleting model/config on gate failure:** D-08 says preserve model, just set `enabled=False`.
- **Adjusting sub-signal composition per regime:** D-04 locks this -- only min_votes and position_pct change.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Percentile computation | Custom histogram binning | `np.percentile()` on `realized_vol_20` | Exact, deterministic, one line |
| Parameter search | Custom regime-aware optimizer | Extend existing `ParameterSearchPipeline` with regime context | WFE gate, experiment tracking already built |
| Composite scoring | New regime-specific metric | `compute_composite_score()` from metrics.py | D-07 locks this as the gate metric |
| Strategy instantiation | Custom VotingStrategy builder | `VotingStrategyFactory.from_config()` | Already handles all config-to-strategy mapping |
| Holdout splitting | Custom date logic | `HoldoutConfig.compute_boundary()` | D-11 shares Phase 11 holdout protocol |

**Key insight:** This phase is primarily an integration/orchestration task. Almost all building blocks exist. The new code is glue: label generation, routing logic, search orchestration, and gate evaluation.

## Common Pitfalls

### Pitfall 1: Data Leakage in Percentile Computation
**What goes wrong:** Computing percentile thresholds on full dataset (including holdout) contaminates regime labels.
**Why it happens:** Easy to pass full OHLCV to label generator without splitting first.
**How to avoid:** Compute percentiles ONLY on `ohlcv[ohlcv.index < holdout_boundary]`. Store thresholds with model for inference.
**Warning signs:** Regime model accuracy suspiciously high (>80%) on holdout.

### Pitfall 2: Regime Model Retrained Per Trial
**What goes wrong:** If per-regime Optuna search retrains the regime model inside the objective function, each trial sees different regime labels -> inconsistent search.
**Why it happens:** Natural to put training inside the trial loop.
**How to avoid:** D-10: Train regime model ONCE before search. Pass pre-computed regime predictions to the search objective.
**Warning signs:** Search takes orders of magnitude longer than expected; regime labels differ between trials.

### Pitfall 3: RegimeRouter Not BaseStrategy-Compatible
**What goes wrong:** If RegimeRouter doesn't duck-type as BaseStrategy, BacktestRunner can't use it.
**Why it happens:** BacktestRunner.run() expects a strategy with `evaluate(features) -> list[Signal]` and `reset()`.
**How to avoid:** Either inherit from BaseStrategy or implement the same interface (evaluate, reset, validate_config).
**Warning signs:** TypeError when passing RegimeRouter to BacktestRunner.

### Pitfall 4: Trailing Stop State Leakage Between Regimes
**What goes wrong:** If RegimeRouter creates a new VotingStrategy per bar (to apply current regime config), trailing stop state (in_position, high_watermark) is lost.
**Why it happens:** VotingStrategy stores position state internally; re-instantiation resets it.
**How to avoid:** RegimeRouter should hold a single VotingStrategy instance and dynamically update `_min_votes` and `_position_pct` attributes rather than re-instantiating. Or maintain separate state that persists across regime changes.
**Warning signs:** Trailing stops never trigger; strategies re-enter positions immediately after exit.

### Pitfall 5: Gate Comparison on Insufficient Holdout Data
**What goes wrong:** Holdout is last 20% of data. If that's only ~50 bars, composite_score cutoff (trade_count < 10) may reject both strategies.
**Why it happens:** Short datasets or high-frequency filtering.
**How to avoid:** Verify holdout has enough bars for meaningful backtest (>200 bars for 1h data). Log holdout size and warn if too small.
**Warning signs:** Both static and regime scores are 0.0 (hard cutoff triggered).

### Pitfall 6: Optuna Search Space Explosion
**What goes wrong:** If per-regime search runs independent Optuna studies with full PARAM_BOUNDS, you're searching 3x the parameter space with the same trial budget.
**Why it happens:** Each regime gets its own study.
**How to avoid:** Per-regime search should ONLY vary `min_votes` and `position_pct` (2 params, not 12). Signal parameters stay fixed from the best static config. Total trial budget stays manageable (e.g., 30 trials per regime = 90 total).
**Warning signs:** Search takes hours; results don't converge.

## Code Examples

### Label Generation with Threshold Persistence
```python
# Source: Project patterns + D-02
import json
import numpy as np
import pandas as pd
from poseidon.data.feature_engine import FeatureEngine

def generate_regime_labels(
    features: pd.DataFrame,
    low_pct: float = 33.0,
    high_pct: float = 66.0,
) -> tuple[pd.Series, dict]:
    """Generate regime labels and return thresholds for persistence.

    Returns:
        (labels_series, thresholds_dict) where thresholds can be saved with model.
    """
    vol = features["realized_vol_20"].dropna()
    low_threshold = float(np.percentile(vol, low_pct))
    high_threshold = float(np.percentile(vol, high_pct))

    labels = pd.Series(1, index=vol.index, dtype=int)  # medium_vol default
    labels[vol < low_threshold] = 0  # low_vol
    labels[vol > high_threshold] = 2  # high_vol

    thresholds = {
        "low_pct": low_pct,
        "high_pct": high_pct,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    return labels, thresholds
```

### Per-Regime Optuna Objective (min_votes + position_pct only)
```python
# Source: Extending existing ParameterSearchPipeline pattern
def regime_objective(
    trial: optuna.Trial,
    regime: str,
    base_config: dict,
    ohlcv_train: pd.DataFrame,
    regime_predictions: pd.Series,
) -> float:
    """Optuna objective that only varies min_votes and position_pct for a single regime."""
    min_votes = trial.suggest_int("min_votes", 2, 6)
    position_pct = trial.suggest_float("position_pct", 0.03, 0.15)

    # Build config with regime-specific overrides
    config = copy.deepcopy(base_config)
    config["min_votes"] = min_votes
    config["position_pct"] = position_pct

    # Filter training data to bars where regime == target regime
    regime_mask = regime_predictions == regime
    regime_ohlcv = ohlcv_train[regime_mask]

    if len(regime_ohlcv) < 50:  # not enough data for meaningful backtest
        return 0.0

    strategy = VotingStrategyFactory.from_config(config)
    result = run_backtest(strategy, regime_ohlcv, ...)
    return compute_composite_score(result.metrics)
```

### RegimeRouter with State Preservation
```python
# Source: Project VotingStrategy pattern
class RegimeRouter(BaseStrategy):
    """Routes VotingStrategy config based on regime model predictions."""

    strategy_type = StrategyType.VOTING  # same type for backtest compatibility

    def __init__(
        self,
        base_config: dict,
        regime_model: XGBoostRegimeModel,
        regime_configs: dict[str, dict],
        enabled: bool = True,
    ):
        self._base_config = base_config
        self._regime_model = regime_model
        self._regime_configs = regime_configs  # {"low_vol": {"min_votes": 3, ...}, ...}
        self.enabled = enabled

        # Create underlying strategy once to preserve state
        self._strategy = VotingStrategyFactory.from_config(base_config)

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        if self.enabled and not features.empty:
            regime_pred = self._regime_model.predict(features)
            current_regime = regime_pred.iloc[-1]["prediction"]
            overrides = self._regime_configs.get(current_regime, {})
            # Apply overrides directly to strategy instance
            if "min_votes" in overrides:
                self._strategy._min_votes = overrides["min_votes"]
            if "position_pct" in overrides:
                self._strategy._position_pct = overrides["position_pct"]
        else:
            # Fallback to base config values
            self._strategy._min_votes = self._base_config.get("min_votes", 4)
            self._strategy._position_pct = self._base_config.get("position_pct", 0.08)

        return self._strategy.evaluate(features)

    def reset(self) -> None:
        self._strategy.reset()

    def validate_config(self) -> bool:
        return self._strategy.validate_config()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| HMM-based regime detection | XGBoost classifier with engineered features | Industry trend ~2020+ | Simpler, more interpretable, faster training |
| K-means on returns | Percentile-based on realized vol | Poseidon D-02 | Deterministic, no clustering instability |
| Complex regime taxonomies (4-8 classes) | Simple 3-class (low/med/high vol) | Poseidon D-01 | Reduces overfitting risk, easier to validate |
| Adaptive everything per regime | Only min_votes + position_pct per regime | Poseidon D-04 | Nunchi experiments: simplicity wins |

**Deprecated/outdated:**
- Multi-class regime with different signal compositions: D-04 explicitly defers this
- Deep learning regime detection: Deferred per CONTEXT.md

## Open Questions

1. **Per-regime backtest methodology: filter bars or full sequence?**
   - What we know: Per-regime Optuna search needs to evaluate regime-specific params
   - What's unclear: Should we filter OHLCV to only bars matching the target regime (sparse), or run full backtest but only count performance during target regime bars?
   - Recommendation: Run full backtest with RegimeRouter active -- this naturally applies per-regime params at correct times and preserves trailing stop continuity. Score the full result, not per-regime slices.

2. **Threshold persistence format**
   - What we know: Percentile thresholds are computed on training data and must be reused at inference
   - What's unclear: Store with regime model artifacts or separately?
   - Recommendation: Save `thresholds.json` alongside `model.pkl` in regime model artifact directory. XGBoostRegimeModel.save/load already uses a directory; add one more file.

3. **RegimeRouter and BacktestRunner compatibility**
   - What we know: BacktestRunner expects `BaseStrategy` interface
   - What's unclear: Whether RegimeRouter should subclass BaseStrategy or duck-type
   - Recommendation: Subclass BaseStrategy for type safety. Set `strategy_type = StrategyType.VOTING` since it wraps a VotingStrategy.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pyproject.toml) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_regime.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RGME-01 | Regime model classifies 3 classes from feature data with measurable accuracy | unit | `pytest tests/test_regime.py::test_regime_model_classifies_3_classes -x` | Wave 0 |
| RGME-01 | Label generation produces correct percentile-based splits | unit | `pytest tests/test_regime.py::test_label_generation_percentiles -x` | Wave 0 |
| RGME-02 | RegimeRouter applies correct per-regime config | unit | `pytest tests/test_regime.py::test_regime_router_applies_config -x` | Wave 0 |
| RGME-02 | RegimeRouter preserves trailing stop state across regime changes | unit | `pytest tests/test_regime.py::test_regime_router_preserves_state -x` | Wave 0 |
| RGME-03 | Gate enables routing when regime beats static | unit | `pytest tests/test_regime.py::test_gate_enables_on_outperformance -x` | Wave 0 |
| RGME-03 | Gate disables routing when static beats regime | unit | `pytest tests/test_regime.py::test_gate_disables_on_underperformance -x` | Wave 0 |
| RGME-03 | Disabled RegimeRouter passes through to static config | unit | `pytest tests/test_regime.py::test_disabled_router_passthrough -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_regime.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_regime.py` -- covers RGME-01, RGME-02, RGME-03
- [ ] Test fixtures: mock XGBoostRegimeModel, sample OHLCV with known volatility patterns, mock BacktestRunner results

*(Note: tests must run on stormtrooper, not locally)*

## Sources

### Primary (HIGH confidence)
- `src/poseidon/ml/implementations/xgboost_regime.py` -- existing 3-class regime model, full interface
- `src/poseidon/data/features/regime.py` -- 4 regime features already registered
- `src/poseidon/strategies/voting_strategy.py` -- VotingStrategy with min_votes/position_pct constructor params
- `src/poseidon/backtest/param_search.py` -- ParameterSearchPipeline architecture
- `src/poseidon/backtest/voting_strategy_factory.py` -- VotingStrategyFactory + PARAM_BOUNDS
- `src/poseidon/backtest/holdout.py` -- HoldoutConfig with 20% split
- `src/poseidon/backtest/metrics.py` -- compute_composite_score() formula
- `src/poseidon/backtest/experiment_tracker.py` -- ExperimentTracker persistence
- `src/poseidon/autoresearch/runner.py` -- AutoResearchRunner orchestration pattern

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions D-01 through D-12 -- locked implementation decisions

### Tertiary (LOW confidence)
- None -- all findings verified against existing codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new dependencies
- Architecture: HIGH -- patterns derived from existing codebase (VotingStrategy, ParameterSearchPipeline, BacktestRunner)
- Pitfalls: HIGH -- identified from code review of actual VotingStrategy state management and data flow

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable -- no external dependency changes expected)
