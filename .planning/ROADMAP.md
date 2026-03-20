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
**Success Criteria:**
1. `BaseStrategy` ABC defines a shared interface and both `ModelStrategy` and `RuleStrategy` implement it.
2. `ModelStrategy` wraps a `BaseModel`, calls `predict()`, and converts the raw DataFrame output into standardized `Signal` objects.
3. `RuleStrategy` parses a JSON DSL document and evaluates conditions against a feature DataFrame to produce signals.
4. DSL `all`, `any`, and `none` combinators work correctly, including at least two levels of nesting.

### Phase 5: Risk Engine & Signal Delivery
**Goal:** Ensure every signal passes through configurable risk checks before being reliably delivered to Thalassa via Redis Streams.
**Requirements:** RISK-01, RISK-02, RISK-03, SIG-01, SIG-02
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
**Success Criteria:**
1. Backtest runner uses the identical `FeatureEngine` + `Strategy.evaluate()` + `RiskEngine` code path as live prediction — no separate backtest-only logic.
2. Virtual portfolio simulator correctly applies market-specific fee schedules (TW stock tax rates for stock/ETF/day trade, crypto maker/taker) and configurable slippage.
3. Walk-forward analysis completes with rolling windows and reports WFE; strategies with WFE < 50% or fewer than 30 trades per OOS segment are flagged.
4. Grid Search and Bayesian Optimization produce ranked parameter sets with associated backtest metrics.
5. Backtest trades and equity curves are stored in separate, queryable tables (not JSONB blobs) enabling cross-backtest comparison queries.

### Phase 7: API & Integration
**Goal:** Expose the full REST API surface for Thalassa to manage data, models, strategies, backtests, risk rules, and signals — completing the end-to-end platform.
**Requirements:** API-01
**Success Criteria:**
1. All endpoint groups are functional: data management, strategies (CRUD + activate/deactivate), models (train/shadow/activate/predict), backtests (run/optimize/results), risk rules (list/update/portfolio), signals (list/detail), and health check.
2. Thalassa can execute the full signal flow via API: create strategy -> run backtest -> activate strategy -> receive signals on Redis Stream.
3. Health endpoint reports GPU status, Celery queue lengths, and data freshness.
4. Error responses follow a consistent JSON format with appropriate HTTP status codes.

---
*Roadmap created: 2026-03-20*
