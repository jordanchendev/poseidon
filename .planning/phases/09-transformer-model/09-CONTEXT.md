# Phase 9: Transformer Model - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement a Transformer-based deep learning model for directional prediction (long/short/hold), following the existing BaseModel ABC pattern. The model plugs into the same ModelRegistry, uses the same FeatureEngine output, and produces the same prediction+confidence DataFrame as XGBoost. Training uses GPU (shared with Triton), inference can run on CPU or GPU.

</domain>

<decisions>
## Implementation Decisions

### Model architecture
- **D-01:** Use **PatchTST** (Patch Time Series Transformer) — lightweight Transformer variant purpose-built for time-series forecasting
- Rationale: ~9GB VRAM headroom with Triton sharing on RTX 4070 Ti SUPER (16GB). PatchTST is VRAM-efficient (~2-4GB for training), has strong benchmarks on time-series tasks, and avoids the overhead of heavier architectures like Temporal Fusion Transformer
- Implementation: encoder-only architecture with patching (groups consecutive time steps into patches for attention)

### Input representation
- **D-02:** Sliding window of existing FeatureEngine feature vectors as model input
- Reuse the same 20 DEFAULT_FEATURES from FeatureEngine (SMA, EMA, RSI, MACD, BB, ATR, returns, volatility, volume)
- Configurable lookback window (e.g., 60 time steps) → each sample is a 2D tensor [lookback_window, num_features]
- No separate feature engineering — the Transformer sees the same features as XGBoost, but with temporal context

### Training infrastructure
- **D-03:** Mixed precision training (fp16) with small batch sizes to fit within ~4GB VRAM
- PyTorch already in project dependencies (`torch>=2.0` in pyproject.toml)
- Training scheduled off-peak to avoid GPU contention with Triton (~7GB VRAM)
- Support CPU fallback: if CUDA not available, train on CPU (slower but functional)
- Use PyTorch's native `torch.amp` for automatic mixed precision

### Prediction output
- **D-04:** Same 3-class output (long/short/hold) with confidence scores, matching XGBoost contract
- `predict()` returns `DataFrame[prediction: str, confidence: float]` — identical to XGBoostModel
- Confidence from softmax probabilities (same approach as XGBoost's predict_proba)
- Maintains full compatibility with ModelStrategy, backtest engine, and risk engine

### Model persistence
- **D-05:** Save/load via `torch.save()` / `torch.load()` for model weights, JSON for feature list and hyperparams
- Follow XGBoost pattern: `model.pt` (weights), `features.json` (feature list), `metadata.json` (hyperparams, training metrics)
- Same directory-based artifact structure used by ArtifactManager

### Claude's Discretion
- PatchTST hyperparameter defaults (patch length, number of heads, number of layers, d_model, dropout)
- Lookback window size default
- Learning rate schedule and optimizer choice
- Data normalization strategy (per-feature standardization, etc.)
- Early stopping criteria
- Test strategy and synthetic data generation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Model engine (follow existing patterns)
- `src/poseidon/ml/base.py` — BaseModel ABC with 7 required methods (train/predict/validate/save/load/get_default_params/get_feature_list)
- `src/poseidon/ml/implementations/xgboost_model.py` — Reference implementation to follow: @register_model, DEFAULT_FEATURES, LABEL_MAP, save/load pattern
- `src/poseidon/ml/registry.py` — @register_model decorator pattern for auto-registration

### Feature engine (input data)
- `src/poseidon/data/feature_engine.py` — FeatureEngine and DEFAULT_FEATURES list (source of input features)
- `src/poseidon/data/features/` — All feature modules (technical, returns, volatility, volume)

### Model lifecycle and artifacts
- `src/poseidon/ml/lifecycle.py` — Model lifecycle state machine
- `src/poseidon/ml/artifacts.py` — ArtifactManager for versioned model storage
- `src/poseidon/ml/manager.py` — ModelManager orchestration

### Architecture reference
- `../docs/poseidon-design.md` — Overall architecture, model engine design section

### Infrastructure constraints
- `.planning/PROJECT.md` — GPU constraints: RTX 4070 Ti SUPER 16GB, Triton uses ~7GB, ~9GB headroom

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BaseModel` ABC: 7 abstract methods, all well-defined — TransformerModel just needs to implement them
- `@register_model` decorator: same pattern as `@register_feature` — one decorator to register
- `LABEL_MAP` / `REVERSE_LABEL_MAP` in xgboost_model.py: can be extracted to shared constants or duplicated (only 3 entries)
- `DEFAULT_FEATURES` list: 20 features, reusable as TransformerModel input specification
- `ArtifactManager`: handles versioned directory creation, symlink for active version

### Established Patterns
- Model implementation file lives in `src/poseidon/ml/implementations/`
- `__init__.py` imports all implementations for auto-registration
- Graceful degradation: XGBoost uses `_HAS_XGBOOST` flag for optional dependency — Transformer should use similar `_HAS_TORCH` pattern (though torch is already a required dep)
- `train()` returns metrics dict, `predict()` returns DataFrame with prediction+confidence columns
- `save()` writes to Path directory, `load()` is classmethod that returns instance

### Integration Points
- `src/poseidon/ml/implementations/__init__.py` — needs import of new transformer module
- `ModelManager` — already supports any BaseModel implementation via registry
- `ModelStrategy` — calls `predict()` on any registered model, no changes needed
- Celery GPU worker — already configured for GPU tasks, Transformer training tasks would use same worker

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Follow XGBoost implementation as closely as possible for consistency.

</specifics>

<deferred>
## Deferred Ideas

- Multi-horizon prediction (predict 1h, 4h, 1d ahead simultaneously) — could be its own enhancement phase
- Ensemble model combining XGBoost + Transformer predictions — future phase
- Attention visualization for interpretability — nice-to-have, not needed for v1
- Transfer learning across markets (pre-train on crypto, fine-tune on TW stocks) — research topic
- RL-based model implementation — separate phase per REQUIREMENTS.md

</deferred>

---

*Phase: 09-transformer-model*
*Context gathered: 2026-03-22*
