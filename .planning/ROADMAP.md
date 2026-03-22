# Roadmap: Poseidon

**Created:** 2026-03-20
**Granularity:** standard
**Milestone:** v1 — Core Trading Signal Platform

## Phases

### Phase 1: Infrastructure & Data Foundation
**Goal:** Stand up the deployment skeleton, database schemas, and data ingestion pipeline so all downstream modules have real market data to work with.
**Requirements:** INFRA-01, INFRA-02, INFRA-03, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, API-02
**Success Criteria:**
1. `docker compose up` starts all 6 services (api, beat, gpu-worker, cpu-worker, redis, postgres) and they pass health checks.
2. Celery Beat triggers a scheduled data fetch job that survives a container restart.
3. OHLCV data for at least one symbol per market (TW stock, TW futures, US stock, crypto spot) is stored in TimescaleDB with automatic compression policy configured for 7-day threshold.
4. Fundamentals JSONB row can be written and read back via storage layer.
5. Sentiment score can be received via API endpoint and persisted.
6. All endpoints reject requests without a valid `X-API-Key` header.
7. All timestamps in the database are stored as TIMESTAMPTZ in UTC.

### Phase 2: Feature Engine
**Goal:** Compute technical indicators and derived features on-the-fly from raw OHLCV data so that models and strategies have a standardized feature matrix.
**Requirements:** FEAT-01, FEAT-02
**Success Criteria:**
1. `FeatureEngine.compute()` returns a wide DataFrame with correct MA, RSI, MACD, Bollinger, ATR, returns, and volatility columns for a given symbol/market/interval range.
2. A new custom feature can be registered via the `BaseFeature` ABC and appears in the feature registry without modifying existing code.
3. Training, prediction, and backtesting paths all invoke the same `FeatureEngine` instance (no duplicated computation logic).

### Phase 3: Model Engine
**Goal:** Provide a model-agnostic training, versioning, and lifecycle management system so ML models can be trained, validated, and deployed.
**Requirements:** MOD-01, MOD-02, MOD-03, MOD-04
**Success Criteria:**
1. A concrete model (e.g., XGBoost stub) implements `BaseModel` ABC with all 7 required methods and can train on feature data.
2. Model Registry can register, look up by name, and list versions of a model.
3. Model lifecycle transitions (training -> ready -> shadow -> active -> retired, and training -> failed) are enforced; invalid transitions raise errors.
4. Model artifacts are saved to versioned filesystem directories and the `active` symlink correctly points to the promoted version.
5. `predict()` returns a DataFrame with `prediction` and `confidence` columns, not Signal objects.

### Phase 4: Strategy Layer
**Goal:** Unify model-based and rule-based strategies under a single interface, including the JSON DSL condition engine for rule strategies.
**Requirements:** STRAT-01, STRAT-02, STRAT-03, STRAT-04
**Plans:** 3 plans
Plans:
- [ ] 04-01-PLAN.md — Signal schema + BaseStrategy ABC contracts
- [ ] 04-02-PLAN.md — ModelStrategy wraps BaseModel, converts predictions to Signals
- [ ] 04-03-PLAN.md — RuleStrategy + DSL condition engine (schema, conditions, executor)
**Success Criteria:**
1. `BaseStrategy` ABC defines a shared interface and both `ModelStrategy` and `RuleStrategy` implement it.
2. `ModelStrategy` wraps a `BaseModel`, calls `predict()`, and converts the raw DataFrame output into standardized `Signal` objects.
3. `RuleStrategy` parses a JSON DSL document and evaluates conditions against a feature DataFrame to produce signals.
4. DSL `all`, `any`, and `none` combinators work correctly, including at least two levels of nesting.

### Phase 5: Risk Engine & Signal Delivery
**Goal:** Ensure every signal passes through configurable risk checks before being reliably delivered to Thalassa via Redis Streams.
**Requirements:** RISK-01, RISK-02, RISK-03, SIG-01, SIG-02
**Plans:** 4 plans
Plans:
- [ ] 05-01-PLAN.md — DB schemas (SignalRecord, RiskRuleRecord, VirtualPositionRecord) + Alembic migration 004
- [ ] 05-02-PLAN.md — Risk engine (BaseRule ABC, 5 concrete rules, RiskEngine chain, VirtualPortfolio)
- [ ] 05-03-PLAN.md — Signal delivery (Redis Streams publisher, consumer group, retention, SignalRepository)
- [ ] 05-04-PLAN.md — Integration (SignalPipeline wiring, risk rule CRUD API)
**Success Criteria:**
1. Risk engine applies a chain of `BaseRule` checks (position control, loss control, frequency control, confidence threshold, leverage cap) and correctly rejects signals that violate any rule.
2. Virtual portfolio state is persisted in PostgreSQL and can be rebuilt from signal history after a restart.
3. Risk rules can be updated via API without restarting the service, and changes take effect on the next signal evaluation.
4. Passed signals are written to the correct Redis Stream (`poseidon:signals:{market}`) and can be consumed via a consumer group with acknowledgment.
5. Rejected signals are recorded in the database with the rejection reason but are not pushed to Redis Streams.
6. Signal format includes all required fields: action, confidence, instrument-specific JSONB params, and supports spot/futures/perpetual/option instrument types.
7. Redis Stream retention is configured at 7 days.

### Phase 6: Backtest Engine
**Goal:** Enable realistic historical strategy evaluation using the exact same pipeline (features, strategy, risk) as live prediction, with walk-forward validation and parameter optimization.
**Requirements:** BT-01, BT-02, BT-03, BT-04, BT-05
**Plans:** 4 plans
Plans:
- [ ] 06-01-PLAN.md — DB schemas (BacktestRecord, BacktestTradeRecord, BacktestEquityRecord) + Alembic migration 005 + CostModel + BacktestPortfolio + metrics
- [ ] 06-02-PLAN.md — BacktestRunner bar-by-bar simulation using existing FeatureEngine + Strategy + RiskEngine pipeline
- [ ] 06-03-PLAN.md — Walk-forward analysis engine with WFE calculation and flagging
- [ ] 06-04-PLAN.md — Parameter optimization (Grid Search + Bayesian/Optuna)
**Success Criteria:**
1. Backtest runner uses the identical `FeatureEngine` + `Strategy.evaluate()` + `RiskEngine` code path as live prediction — no separate backtest-only logic.
2. Virtual portfolio simulator correctly applies market-specific fee schedules (TW stock tax rates for stock/ETF/day trade, crypto maker/taker) and configurable slippage.
3. Walk-forward analysis completes with rolling windows and reports WFE; strategies with WFE < 50% or fewer than 30 trades per OOS segment are flagged.
4. Grid Search and Bayesian Optimization produce ranked parameter sets with associated backtest metrics.
5. Backtest trades and equity curves are stored in separate, queryable tables (not JSONB blobs) enabling cross-backtest comparison queries.

### Phase 7: API & Integration
**Goal:** Expose the full REST API surface for Thalassa to manage data, models, strategies, backtests, risk rules, and signals — completing the end-to-end platform.
**Requirements:** API-01
**Plans:** 5 plans
Plans:
- [ ] 07-01-PLAN.md — StrategyRecord DB model + migration 006 + strategy CRUD API + error handler module
- [ ] 07-02-PLAN.md — Model API (train/shadow/activate/predict) + GPU Celery tasks
- [ ] 07-03-PLAN.md — Backtest API (run/optimize/results) + backtest/optimization Celery tasks
- [ ] 07-04-PLAN.md — Signal API (list/detail) + enhanced health endpoint
- [ ] 07-05-PLAN.md — Integration wiring (all routers in main.py, auth on all, error handlers, portfolio endpoint)
**Success Criteria:**
1. All endpoint groups are functional: data management, strategies (CRUD + activate/deactivate), models (train/shadow/activate/predict), backtests (run/optimize/results), risk rules (list/update/portfolio), signals (list/detail), and health check.
2. Thalassa can execute the full signal flow via API: create strategy -> run backtest -> activate strategy -> receive signals on Redis Stream.
3. Health endpoint reports GPU status, Celery queue lengths, and data freshness.
4. Error responses follow a consistent JSON format with appropriate HTTP status codes.

### Phase 8: Volume Features & 5m Crypto Data

**Goal:** Add volume-based features (volume_sma, volume_ratio, obv) to the feature engine and enable 5-minute candle interval for crypto spot data via CCXT, with updated batch configuration for 5m data ingestion.
**Requirements:** PHASE8-01, PHASE8-02, PHASE8-03
**Depends on:** Phase 7
**Plans:** 2 plans

Plans:
- [x] 08-01-PLAN.md — Volume features (VolumeSMA, VolumeRatio, OBV) + DEFAULT_FEATURES updates + tests
- [x] 08-02-PLAN.md — 5m crypto interval config + BATCH_DAYS_5M + BaseFetcher docstring

**Success Criteria:**
1. FeatureEngine computes volume_sma, volume_ratio, and obv features from OHLCV data via the standard BaseFeature pattern.
2. All 3 volume features are included in both FeatureEngine and XGBoost DEFAULT_FEATURES lists.
3. crypto_spot intervals in symbols.yaml include "5m" alongside "1d" and "1h".
4. Backfill pagination for 5m candles uses BATCH_DAYS_5M with ~3 days per batch.
5. All existing and new tests pass.

### Phase 9: Transformer Model

**Goal:** Implement a PatchTST (Patch Time Series Transformer) deep learning model as a second BaseModel implementation, using the same FeatureEngine output and producing the same prediction+confidence DataFrame contract as XGBoost, with mixed precision GPU training and CPU fallback.
**Requirements:** TRANS-01, TRANS-02, TRANS-03, TRANS-04, TRANS-05
**Depends on:** Phase 8
**Plans:** 2 plans

Plans:
- [x] 09-01-PLAN.md — PatchTST architecture + TimeSeriesDataset + TransformerModel (all 7 BaseModel methods) + registry registration
- [ ] 09-02-PLAN.md — Comprehensive test suite for TransformerModel (PatchTST forward, dataset, contract, save/load, registry)

**Success Criteria:**
1. TransformerModel implements all 7 BaseModel ABC methods (train/predict/validate/save/load/get_default_params/get_feature_list) and is registered via `@register_model`.
2. PatchTST encoder-only architecture with patching correctly processes [batch, lookback_window, num_features] tensors and outputs [batch, 3] logits.
3. Training uses mixed precision (fp16) via `torch.amp` when CUDA available, falls back to CPU gracefully.
4. `predict()` returns DataFrame with "prediction" (long/short/hold) and "confidence" (0.0-1.0) columns, identical contract to XGBoostModel.
5. Model persistence saves `model.pt` + `features.json` + `metadata.json`, and `load()` restores a working model that can predict.
6. All new tests pass and no existing tests are broken.

---
*Roadmap created: 2026-03-20*
