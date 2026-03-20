# Requirements: Poseidon

**Defined:** 2026-03-20
**Core Value:** Reliably produce quality trading signals and deliver them to Thalassa for human review before manual execution.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Infrastructure

- [ ] **INFRA-01**: Docker Compose deploys 6 services (api, beat, gpu-worker, cpu-worker, redis, postgres/TimescaleDB)
- [ ] **INFRA-02**: Celery Beat runs persistent scheduled tasks (data fetching, cleanup)
- [ ] **INFRA-03**: API key authentication via X-API-Key header on all endpoints

### Data

- [ ] **DATA-01**: Fetcher can pull OHLCV data from FinMind (TW stocks, TW futures)
- [ ] **DATA-02**: Fetcher can pull OHLCV data from yfinance (US stocks)
- [ ] **DATA-03**: Fetcher can pull OHLCV data from CCXT/Binance (crypto spot)
- [ ] **DATA-04**: OHLCV stored in TimescaleDB hypertable with automatic compression after 7 days
- [ ] **DATA-05**: Fundamentals stored in PostgreSQL with JSONB flexible fields
- [ ] **DATA-06**: Sentiment scores received from Thalassa via API and stored

### Feature Engine

- [ ] **FEAT-01**: FeatureEngine computes technical indicators on-the-fly from OHLCV (MA, RSI, MACD, Bollinger, ATR, returns, volatility)
- [ ] **FEAT-02**: BaseFeature ABC allows registering and computing new feature types

### Model Engine

- [ ] **MOD-01**: BaseModel ABC with train/predict/validate/save/load/get_default_params/get_feature_list interface
- [ ] **MOD-02**: Model Registry for registration, lookup, and version management
- [ ] **MOD-03**: Model lifecycle: training → failed → ready → shadow → active → retired
- [ ] **MOD-04**: Model artifacts stored on filesystem with versioned directories and symlink for active version

### Strategy

- [ ] **STRAT-01**: BaseStrategy ABC unifies model-based and rule-based strategies under one interface
- [ ] **STRAT-02**: ModelStrategy wraps BaseModel, converts raw predictions (DataFrame) to standardized Signals
- [ ] **STRAT-03**: RuleStrategy parses and executes JSON DSL conditions against live/historical data
- [ ] **STRAT-04**: DSL supports all/any/none boolean condition combinators with nesting

### Backtest

- [ ] **BT-01**: Backtest engine shares exact same FeatureEngine + Strategy + Risk pipeline as live prediction
- [ ] **BT-02**: Virtual portfolio simulator with configurable fees and slippage per market (TW stock tax rates for stock/ETF/day trade, crypto maker/taker, etc.)
- [ ] **BT-03**: Walk-forward analysis with WFE (Walk-Forward Efficiency) ≥ 50% pass criteria and minimum 30 trades per OOS segment
- [ ] **BT-04**: Parameter optimization via Grid Search and Bayesian Optimization
- [ ] **BT-05**: Backtest trades and equity curves stored in separate queryable tables (not JSONB blobs)

### Risk

- [ ] **RISK-01**: BaseRule ABC risk engine with chain-of-responsibility pattern (position control, loss control, frequency control, confidence threshold, leverage cap)
- [ ] **RISK-02**: Virtual portfolio persisted in PostgreSQL, rebuilt from signal history on restart
- [ ] **RISK-03**: Risk rules stored in DB and configurable via API without service restart

### Signal

- [ ] **SIG-01**: Standardized Signal format with action (long/short/close/hold), confidence, instrument-specific params (JSONB), supporting spot/futures/perpetual/option types
- [ ] **SIG-02**: Redis Streams delivery with consumer groups, acknowledgment, and replay on Thalassa reconnect (7-day retention)

### API

- [ ] **API-01**: REST endpoints for data management, strategies, models, backtests, risk rules, signals, and health check
- [ ] **API-02**: All timestamps stored in UTC (TIMESTAMPTZ), market-specific timezone conversions at application layer

## v2 Requirements

Deferred to future release. Schema supports these from day one.

### Crypto Derivatives

- **DERIV-01**: Fetch crypto perpetual contract data via CCXT (separate from spot)
- **DERIV-02**: Handle leverage and funding rate in signal params and backtest cost models

## v3 Requirements

### Options

- **OPT-01**: Fetch TW options (TXO) and US stock options data from appropriate providers
- **OPT-02**: Compute Greeks (delta, gamma, theta, vega) for options strategies
- **OPT-03**: Add options-specific conditions to strategy DSL (option_type, strike, expiry)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Order execution / auto-trading | Poseidon produces signals only; execution layer is a future independent service |
| Text extraction / NLP | Triton handles media-to-text; Thalassa handles sentiment scoring |
| Web scraping | Thalassa's responsibility |
| Telegram / notifications | Thalassa handles user communication |
| Real-time tick data (Shioaji) | Deferred; FinMind daily data sufficient for Phase 1 |
| Feature caching / storage table | On-the-fly computation sufficient at personal scale |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | 1 | not started |
| INFRA-02 | 1 | not started |
| INFRA-03 | 1 | not started |
| DATA-01 | 1 | not started |
| DATA-02 | 1 | not started |
| DATA-03 | 1 | not started |
| DATA-04 | 1 | not started |
| DATA-05 | 1 | not started |
| DATA-06 | 1 | not started |
| FEAT-01 | 2 | not started |
| FEAT-02 | 2 | not started |
| MOD-01 | 3 | not started |
| MOD-02 | 3 | not started |
| MOD-03 | 3 | not started |
| MOD-04 | 3 | not started |
| STRAT-01 | 4 | not started |
| STRAT-02 | 4 | not started |
| STRAT-03 | 4 | not started |
| STRAT-04 | 4 | not started |
| BT-01 | 6 | not started |
| BT-02 | 6 | not started |
| BT-03 | 6 | not started |
| BT-04 | 6 | not started |
| BT-05 | 6 | not started |
| RISK-01 | 5 | not started |
| RISK-02 | 5 | not started |
| RISK-03 | 5 | not started |
| SIG-01 | 5 | not started |
| SIG-02 | 5 | not started |
| API-01 | 7 | not started |
| API-02 | 1 | not started |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0 ✅

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-20 after roadmap creation*
