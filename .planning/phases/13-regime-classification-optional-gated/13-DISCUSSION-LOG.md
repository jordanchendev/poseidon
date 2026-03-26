# Phase 13: Regime Classification (Optional, Gated) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 13-Regime Classification (Optional, Gated)
**Areas discussed:** Regime 分類方式, RegimeRouter 行為, Outperformance gate, 訓練與評估流程

---

## Regime 分類方式

| Option | Description | Selected |
|--------|-------------|----------|
| 保持 3-class volatility | 用現有 low/medium/high_vol，簡單直接。先用最簡單的分類測試假設是否有效。 | ✓ |
| 改 4-class regime | trending/ranging/volatile/low-vol，更精細但需新增 trend/momentum features 和更複雜的 label 產生邏輯。 | |

**User's choice:** 保持 3-class volatility
**Notes:** Regime 是 gated 功能，先用最簡單的分類測試核心假設。

### Label 產生方式

| Option | Description | Selected |
|--------|-------------|----------|
| Percentile-based | 用 realized_vol_20 的歷史分位數切割：<33%=low, 33-66%=medium, >66%=high。簡單透明。 | ✓ |
| K-means clustering | 對多個 vol features 做 k-means(k=3)，讓數據自行分群。更 data-driven 但可能不穩定。 | |

**User's choice:** Percentile-based
**Notes:** 無額外說明。

---

## RegimeRouter 行為

### 調整範圍

| Option | Description | Selected |
|--------|-------------|----------|
| 只調 min_votes + position_pct | high_vol 時更保守（提高 min_votes、降低 position_pct），low_vol 時更積極。最簡單，只調兩個參數。 | ✓ |
| 完整的 per-regime config | 每個 regime 有完全獨立的 VotingStrategy JSON config（不同 signal 參數、不同 sub-signal 組合）。更彈性但複雜度高。 | |

**User's choice:** 只調 min_votes + position_pct
**Notes:** 無額外說明。

### 配置來源

| Option | Description | Selected |
|--------|-------------|----------|
| 預設寫死 + Optuna 搜索 | 先用合理預設值，但支援透過 Optuna 搜尋 per-regime 最佳參數。用現有 Phase 11 pipeline。 | ✓ |
| 純預設寫死 | 手動設定每個 regime 的參數，不做自動搜索。最簡單但可能不是最佳。 | |

**User's choice:** 預設寫死 + Optuna 搜索
**Notes:** 無額外說明。

---

## Outperformance Gate

### Gate Metric

| Option | Description | Selected |
|--------|-------------|----------|
| composite_score | 用 Phase 10 定義的 composite_score 作為唯一評比指標。與全專案指標一致。 | ✓ |
| 多指標綜合 | composite_score + max_drawdown + win_rate 等多個指標都要優於或等於 baseline。更嚴格但更難通過。 | |

**User's choice:** composite_score
**Notes:** 無額外說明。

### Auto-disable 機制

| Option | Description | Selected |
|--------|-------------|----------|
| Config flag + 回溯默認 | RegimeRouter 有 enabled=True/False flag。OOS 測試輸給 baseline 時自動設 enabled=False，跳過 regime routing，直接用 static VotingStrategy。不刪除 model，只是停用。 | ✓ |
| 完全移除 | Gate 失敗時刪除 regime model 和 router 配置。彈性低但乾淨。 | |

**User's choice:** Config flag + 回溯默認
**Notes:** 無額外說明。

---

## 訓練與評估流程

### 訓練整合方式

| Option | Description | Selected |
|--------|-------------|----------|
| 獨立訓練 + 靜態模型 | Regime model 先獨立訓練一次，產生靜態模型。AutoResearch 跨 per-regime 搜索時直接使用這個靜態模型的預測結果，不重新訓練 regime model。 | ✓ |
| 每次搜索都重新訓練 | 每次 Optuna trial 都包含 regime model 重新訓練。理論上更好但極耗時。 | |

**User's choice:** 獨立訓練 + 靜態模型
**Notes:** 無額外說明。

### Holdout 策略

| Option | Description | Selected |
|--------|-------------|----------|
| 共用 Phase 11 holdout | 沿用 Phase 11 的 HoldoutConfig (last 20%)。Regime model 訓練用前 80%，outperformance gate 在 last 20% OOS 上測。確保一致性。 | ✓ |
| 獨立 holdout | Regime model 有自己的 holdout split，與策略優化的 holdout 獨立。更嚴謹但可能造成小樣本問題。 | |

**User's choice:** 共用 Phase 11 holdout
**Notes:** 無額外說明。

---

## Claude's Discretion

- Exact percentile thresholds (33/66 as starting point)
- RegimeRouter class structure and interface design
- Per-regime Optuna search integration with existing ParameterSearchPipeline
- Gate comparison test methodology details
- API endpoints for regime model management

## Deferred Ideas

- 4-class regime (trending/ranging/volatile/low-vol) — 如果 3-class 有效再擴展
- Per-regime 不同 sub-signal 組合 — 複雜度太高
- Regime model 自動重訓練 — 未來考慮
- AI-driven regime detection (deep learning / HMM) — 未來選項
