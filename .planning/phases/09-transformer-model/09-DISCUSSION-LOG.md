# Phase 9: Transformer Model - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-22
**Phase:** 09-transformer-model
**Areas discussed:** Model architecture, Input representation, Training infrastructure, Prediction output
**Mode:** --auto --analyze (all decisions auto-selected with trade-off analysis)

---

## Model Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| PatchTST (lightweight) | Purpose-built for time-series, low VRAM, fast inference | ✓ |
| Temporal Fusion Transformer | Interpretable attention, built for forecasting, heavy VRAM (~4-6GB) | |
| Vanilla Transformer encoder | Well-understood, flexible, not optimized for time-series | |
| iTransformer | Channel-independent, SOTA on benchmarks, very new | |

**User's choice:** [auto] PatchTST (lightweight) — recommended default
**Notes:** Best fit for ~9GB VRAM constraint with Triton sharing, purpose-built for time-series forecasting

---

## Input Representation

| Option | Description | Selected |
|--------|-------------|----------|
| Sliding window of feature vectors | Reuses FeatureEngine output, simple, consistent with XGBoost | ✓ |
| Raw OHLCV + learned embeddings | Model learns representations, duplicates feature engineering | |
| Patched time-series (PatchTST native) | Efficient, captures local patterns, requires reshaping | |

**User's choice:** [auto] Sliding window of feature vectors — recommended default
**Notes:** Reuses existing 20 DEFAULT_FEATURES, maintains consistency with XGBoost pipeline

---

## Training Infrastructure

| Option | Description | Selected |
|--------|-------------|----------|
| Mixed precision (fp16) + small batch | ~50% VRAM savings, fits in ~4GB | ✓ |
| Full precision (fp32) | Maximum stability, ~8GB VRAM, tight with Triton | |
| CPU-only training | No VRAM contention, very slow | |

**User's choice:** [auto] Mixed precision (fp16) + small batch — recommended default
**Notes:** ~4GB VRAM fits comfortably alongside Triton's ~7GB on 16GB card

---

## Prediction Output

| Option | Description | Selected |
|--------|-------------|----------|
| Same 3-class (long/short/hold) | Compatible with existing Strategy layer, consistent with XGBoost | ✓ |
| Regression (future return prediction) | More information, needs threshold logic | |
| Both (multi-head) | Maximum flexibility, more complexity | |

**User's choice:** [auto] Same 3-class (long/short/hold) — recommended default
**Notes:** Maintains full compatibility with ModelStrategy, backtest, and risk engine

---

## Claude's Discretion

- PatchTST hyperparameters (patch length, heads, layers, d_model, dropout)
- Lookback window size
- Learning rate schedule and optimizer
- Data normalization strategy
- Early stopping criteria
- Test strategy

## Deferred Ideas

- Multi-horizon prediction — future enhancement
- Ensemble model (XGBoost + Transformer) — future phase
- Attention visualization — nice-to-have
- Transfer learning across markets — research topic
- RL-based model — separate phase
