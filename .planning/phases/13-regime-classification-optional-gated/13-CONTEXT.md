# Phase 13: Regime Classification (Optional, Gated) - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Market regime detection selects per-regime VotingStrategy configurations, but only if regime routing demonstrably outperforms the static no-regime baseline on out-of-sample data. If the gate fails, regime routing is disabled and the system falls back to static configuration.

Requirements: RGME-01, RGME-02, RGME-03

</domain>

<decisions>
## Implementation Decisions

### Regime Classification
- **D-01:** Keep existing 3-class volatility taxonomy: low_vol / medium_vol / high_vol. Do not expand to 4-class (trending/ranging/volatile/low-vol) — start simple, expand later if proven useful.
- **D-02:** Label generation via percentile-based splitting on `realized_vol_20`: <33rd percentile = low_vol, 33-66th = medium_vol, >66th = high_vol. Simple, transparent, no extra model needed.
- **D-03:** Use existing `XGBoostRegimeModel` (already implemented in `src/poseidon/ml/implementations/xgboost_regime.py`) and existing regime features (`src/poseidon/data/features/regime.py`).

### RegimeRouter Behavior
- **D-04:** RegimeRouter adjusts only two parameters per regime: `min_votes` and `position_pct`. Sub-signal composition and other parameters remain static across regimes.
- **D-05:** Sensible defaults: high_vol → min_votes=5, position_pct=0.05 (conservative); medium_vol → min_votes=4, position_pct=0.08 (standard); low_vol → min_votes=3, position_pct=0.10 (aggressive).
- **D-06:** Per-regime parameters also searchable via Optuna (using Phase 11 pipeline). Defaults serve as starting point, Optuna can discover better values.

### Outperformance Gate
- **D-07:** Gate metric: `composite_score` (Phase 10 D-05/D-06). Regime-conditional strategy's composite_score must be strictly greater than static VotingStrategy baseline on OOS data.
- **D-08:** Auto-disable mechanism: `RegimeRouter` has `enabled: bool` flag. If OOS test fails, set `enabled=False` — router passes through to static config. Model and config are preserved (not deleted), just bypassed.
- **D-09:** Gate evaluation runs on holdout data (last 20%) after regime model training and per-regime parameter search complete.

### Training & Evaluation Flow
- **D-10:** Regime model trained independently and once — produces a static model. AutoResearch per-regime parameter search uses this static model's predictions, does NOT retrain regime model per trial.
- **D-11:** Share Phase 11 HoldoutConfig (last 20%). Regime model trains on first 80%, outperformance gate tests on last 20% OOS. Consistent with strategy optimization holdout.
- **D-12:** Flow: (1) Generate percentile labels on training data → (2) Train XGBoostRegimeModel → (3) Per-regime Optuna search for min_votes/position_pct → (4) Run gate comparison on holdout → (5) Enable or disable.

### Claude's Discretion
- Exact percentile thresholds (33/66 as starting point, may adjust)
- RegimeRouter class structure and interface design
- How per-regime Optuna search integrates with existing ParameterSearchPipeline
- Gate comparison test methodology details (e.g., paired or unpaired comparison)
- API endpoints for regime model management (if any needed this phase)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Regime Model (existing code)
- `src/poseidon/ml/implementations/xgboost_regime.py` — XGBoostRegimeModel with 3-class classifier, train/predict/validate/save/load interface
- `src/poseidon/data/features/regime.py` — Regime features: VolatilityRatio, RealizedVolatility, VolatilityOfVolatility, ReturnAutocorrelation
- `src/poseidon/ml/base.py` — BaseModel ABC interface

### Strategy Layer (integration targets)
- `src/poseidon/strategies/voting_strategy.py` — VotingStrategy with configurable min_votes, position_pct, ATR trailing stop
- `src/poseidon/strategies/configs/nunchi_crypto_1h.json` — Baseline static config

### Experiment Infrastructure (Phase 11 — reuse)
- `src/poseidon/backtest/param_search.py` — ParameterSearchPipeline, SearchConfig; extend for per-regime search
- `src/poseidon/backtest/voting_strategy_factory.py` — VotingStrategyFactory, PARAM_BOUNDS
- `src/poseidon/backtest/holdout.py` — HoldoutConfig (shared holdout split)
- `src/poseidon/backtest/experiment_tracker.py` — ExperimentTracker for logging regime experiments

### Scoring
- `src/poseidon/backtest/metrics.py` — compute_composite_score() (gate comparison metric)

### Requirements
- `.planning/REQUIREMENTS.md` §RGME-01..RGME-03 — Acceptance criteria for this phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `XGBoostRegimeModel`: Already implements BaseModel ABC with train/predict/validate/save/load — ready to use as-is
- Regime features (vol_ratio, realized_vol, vol_of_vol, return_autocorr): Already registered in FeatureEngine
- `VotingStrategy`: Already accepts `min_votes` and `position_pct` as constructor params — RegimeRouter just needs to pass different values
- `ParameterSearchPipeline`: Can be extended to do per-regime parameter search
- `HoldoutConfig`: Shared holdout protocol already enforced
- `compute_composite_score()`: Gate comparison metric already available

### Established Patterns
- BaseModel ABC: train/predict/validate/save/load — XGBoostRegimeModel follows this
- ParameterSearchPipeline: Optuna → WFE gate → ExperimentTracker — extend for regime-aware search
- Config-driven strategies: VotingStrategy from JSON config — RegimeRouter maps regime → config override

### Integration Points
- `RegimeRouter` wraps `VotingStrategy` — intercepts evaluate() calls to apply per-regime config
- Regime model predict() called at evaluation time to determine current regime
- Per-regime Optuna search extends ParameterSearchPipeline with regime-conditional objective
- Gate comparison: run backtest twice (with/without regime routing) on holdout data, compare composite_score

</code_context>

<specifics>
## Specific Ideas

- 這是 gated 功能 — 整個設計圍繞「證明有效才啟用」的原則
- 先用最簡單的 3-class volatility 測試假設，如果有效再考慮擴展分類
- Percentile-based labels 避免了 k-means 的不穩定性和 HMM 的複雜度
- RegimeRouter 只調兩個參數（min_votes, position_pct），最小化 overfitting 風險

</specifics>

<deferred>
## Deferred Ideas

- 4-class regime (trending/ranging/volatile/low-vol) — 如果 3-class 有效再擴展
- Per-regime 不同 sub-signal 組合 — 複雜度太高，先只調參數
- Regime model 自動重訓練 — 未來考慮定期重訓
- Regime-aware autoresearch (Phase 12 deferred item) — 本 phase 建好 router 後自然可用
- AI-driven regime detection (deep learning / HMM) — 未來取代 XGBoost 的選項

</deferred>

---

*Phase: 13-regime-classification-optional-gated*
*Context gathered: 2026-03-26*
