# Requirements: Poseidon

**Defined:** 2026-03-20
**Core Value:** Reliably produce quality trading signals and deliver them to Thalassa for human review before manual execution.

## v1.0 Requirements (Milestone v1.0 — Core Platform)

Requirements from initial release. All phases (1-9) complete.

### Infrastructure

- [x] **INFRA-01**: Docker Compose deploys 6 services (api, beat, gpu-worker, cpu-worker, redis, postgres/TimescaleDB)
- [x] **INFRA-02**: Celery Beat runs persistent scheduled tasks (data fetching, cleanup)
- [x] **INFRA-03**: API key authentication via X-API-Key header on all endpoints

### Data

- [x] **DATA-01**: Fetcher can pull OHLCV data from FinMind (TW stocks, TW futures)
- [x] **DATA-02**: Fetcher can pull OHLCV data from yfinance (US stocks)
- [x] **DATA-03**: Fetcher can pull OHLCV data from CCXT/Binance (crypto spot)
- [x] **DATA-04**: OHLCV stored in TimescaleDB hypertable with automatic compression after 7 days
- [x] **DATA-05**: Fundamentals stored in PostgreSQL with JSONB flexible fields
- [x] **DATA-06**: Sentiment scores received from Thalassa via API and stored

### Feature Engine

- [x] **FEAT-01**: FeatureEngine computes technical indicators on-the-fly from OHLCV (MA, RSI, MACD, Bollinger, ATR, returns, volatility)
- [x] **FEAT-02**: BaseFeature ABC allows registering and computing new feature types

### Model Engine

- [x] **MOD-01**: BaseModel ABC with train/predict/validate/save/load/get_default_params/get_feature_list interface
- [x] **MOD-02**: Model Registry for registration, lookup, and version management
- [x] **MOD-03**: Model lifecycle: training -> failed -> ready -> shadow -> active -> retired
- [x] **MOD-04**: Model artifacts stored on filesystem with versioned directories and symlink for active version

### Transformer Model

- [x] **TRANS-01**: TransformerModel implements BaseModel ABC with all 7 required methods
- [x] **TRANS-02**: PatchTST encoder-only Transformer architecture with patching, registered via @register_model
- [x] **TRANS-03**: Mixed precision training (fp16) via torch.amp when CUDA available, with CPU fallback
- [x] **TRANS-04**: predict() returns DataFrame with prediction and confidence columns, identical contract to XGBoostModel
- [x] **TRANS-05**: Model persistence via torch.save + JSON, compatible with ArtifactManager

### Strategy

- [x] **STRAT-01**: BaseStrategy ABC unifies model-based and rule-based strategies under one interface
- [x] **STRAT-02**: ModelStrategy wraps BaseModel, converts raw predictions (DataFrame) to standardized Signals
- [x] **STRAT-03**: RuleStrategy parses and executes JSON DSL conditions against live/historical data
- [x] **STRAT-04**: DSL supports all/any/none boolean condition combinators with nesting

### Backtest

- [x] **BT-01**: Backtest engine shares exact same FeatureEngine + Strategy + Risk pipeline as live prediction
- [x] **BT-02**: Virtual portfolio simulator with configurable fees and slippage per market
- [x] **BT-03**: Walk-forward analysis with WFE >= 50% pass criteria and minimum 30 trades per OOS segment
- [x] **BT-04**: Parameter optimization via Grid Search and Bayesian Optimization
- [x] **BT-05**: Backtest trades and equity curves stored in separate queryable tables

### Risk

- [x] **RISK-01**: BaseRule ABC risk engine with chain-of-responsibility pattern
- [x] **RISK-02**: Virtual portfolio persisted in PostgreSQL, rebuilt from signal history on restart
- [x] **RISK-03**: Risk rules stored in DB and configurable via API without service restart

### Signal

- [x] **SIG-01**: Standardized Signal format with action, confidence, instrument-specific params (JSONB)
- [x] **SIG-02**: Redis Streams delivery with consumer groups, acknowledgment, and replay (7-day retention)

### API

- [x] **API-01**: REST endpoints for data management, strategies, models, backtests, risk rules, signals, and health check
- [x] **API-02**: All timestamps stored in UTC (TIMESTAMPTZ), market-specific timezone conversions at application layer

### Volume & 5m Data

- [x] **PHASE8-01**: Volume features (volume_sma, volume_ratio, obv) computed via BaseFeature pattern
- [x] **PHASE8-02**: 5m crypto interval support with BATCH_DAYS_5M pagination
- [x] **PHASE8-03**: All volume features in DEFAULT_FEATURES lists

## v2.0 Requirements (Milestone v2.0 — Strategy Pivot)

Strategy pivot from ML direction prediction to rule-based voting + automated parameter search.

### Voting Strategy

- [x] **VOTE-01**: VotingStrategy extends BaseStrategy, accepts N child RuleStrategy instances, emits signal when >= min_votes threshold (default 4/6)
- [x] **VOTE-02**: DSL condition engine supports new `vote` condition type with `min_votes` parameter, enabling "M of N conditions true" without combinatorial explosion
- [x] **VOTE-03**: Six Nunchi-derived signal strategies implemented as RuleStrategy JSON configs: dual Momentum, EMA crossover, RSI(8), MACD(14,23,9), Bollinger squeeze
- [x] **VOTE-04**: Composite scoring calculates confidence from vote count and individual signal strengths, with hard cutoffs for low-quality signals
- [x] **VOTE-05**: ATR-based trailing stop exit logic integrated into VotingStrategy evaluation
- [x] **VOTE-06**: Fixed position sizing (default 8%, configurable) applied uniformly — no adaptive sizing mechanisms

### AutoResearch Framework

- [ ] **AUTO-01**: ExperimentTracker persists experiment runs (config, metrics, timestamps) in PostgreSQL with unique experiment IDs
- [ ] **AUTO-02**: VotingStrategyFactory generates VotingStrategy instances from JSON config files (the "variable layer")
- [x] **AUTO-03**: StrategyMutator varies strategy parameters within defined bounds (signal periods, thresholds, vote counts)
- [x] **AUTO-04**: 3-layer architecture enforced: immutable layer (FeatureEngine+BacktestRunner+RiskEngine), mutable layer (strategy JSON config), guidance layer (program.md)
- [x] **AUTO-05**: AutoResearchRunner as Celery task that iterates: mutate config -> backtest -> evaluate -> log -> repeat
- [x] **AUTO-06**: Immutability boundary enforced — autoresearch cannot modify scoring formula, backtest runner, or feature engine code

### Regime Classification (Optional — Gated)

- [x] **RGME-01**: XGBoostRegimeModel classifies market regime (trending/ranging/volatile/low-vol) from feature data
- [x] **RGME-02**: RegimeRouter selects VotingStrategy configuration based on detected regime
- [x] **RGME-03**: Outperformance gate — regime routing must beat static no-regime baseline on OOS data, auto-disabled if fails

### Parameter Search

- [x] **PARM-01**: Optuna studies persist to PostgreSQL via RDBStorage (not in-memory)
- [x] **PARM-02**: Walk-forward validation is mandatory gate for all parameter search results (WFE >= 50%)
- [ ] **PARM-03**: Holdout data split defined and locked before any experiments run (irreversible decision point)
- [x] **PARM-04**: Per-market/timeframe parameter search discovers optimal signal parameters independently
- [x] **PARM-05**: Trial count limited (50-100 per search) to prevent overfitting on finite data

### Nunchi Signal Alignment

- [x] **ALIGN-01**: Composite score formula uses lenient drawdown penalty (max(0, dd-0.15)*0.05) and capital turnover ratio penalty instead of quadratic dd and raw trade count
- [x] **ALIGN-02**: BacktestPortfolio equity curve correctly values short positions using entry_price - current_price direction
- [x] **ALIGN-03**: Nunchi baseline config uses ATR multiplier 5.5 and BB squeeze threshold 0.85
- [x] **ALIGN-04**: VotingStrategy supports separate bear_sub_signals list with independent bear_min_votes threshold, emitting SHORT signals
- [x] **ALIGN-05**: VotingStrategy implements RSI mean-reversion exit (long exit at RSI > 69, short exit at RSI < 31)
- [x] **ALIGN-06**: VotingStrategy implements signal flip exit (opposing ensemble fires -> close and reverse) and 2-bar cooldown
- [x] **ALIGN-07**: ATR trailing stop works bidirectionally: high watermark for longs, low watermark for shorts
- [x] **ALIGN-08**: PARAM_BOUNDS expanded with bear_min_votes, bear_position_pct, and ATR range (3.0, 8.0); factory generates bear_sub_signals with inverted conditions
- [x] **ALIGN-09**: RegimeRouter overrides 4 strategy attributes per regime (min_votes, position_pct, bear_min_votes, bear_position_pct)
- [x] **ALIGN-10**: RegimeSearchPipeline searches 4 params per regime instead of 2

## v3.0 Requirements (Milestone v3.0 — Risk Management + Data Quality + Live Pipeline)

### Live Signal Pipeline

- [ ] **PRED-01**: POST /models/{version_id}/predict loads trained model on GPU worker, runs inference, returns 202 with task_id
- [ ] **PRED-02**: Predictions meeting confidence threshold auto-convert to Signal objects via SignalPipeline (risk check + persist + deliver)
- [ ] **PRED-03**: Active rule/voting strategies evaluated by Celery Beat on configurable schedule after data fetch
- [ ] **PRED-04**: New signal generation triggers downstream risk pipeline: VaR snapshot recalculation
- [ ] **PRED-05**: Health endpoint reports GPU worker torch/CUDA status via Celery worker ping, not local torch import

## Future Requirements

### Crypto Derivatives (v3)

- **DERIV-01**: Fetch crypto perpetual contract data via CCXT (separate from spot)
- **DERIV-02**: Handle leverage and funding rate in signal params and backtest cost models

### Options (v4)

- **OPT-01**: Fetch TW options (TXO) and US stock options data from appropriate providers
- **OPT-02**: Compute Greeks (delta, gamma, theta, vega) for options strategies
- **OPT-03**: Add options-specific conditions to strategy DSL (option_type, strike, expiry)

### Experiment Dashboard (v3)

- **DASH-01**: optuna-dashboard integration for visual experiment result exploration

## Out of Scope

| Feature | Reason |
|---------|--------|
| Order execution / auto-trading | Poseidon produces signals only; execution layer is a future independent service |
| Text extraction / NLP | Triton handles media-to-text; Thalassa handles sentiment scoring |
| Web scraping | Thalassa's responsibility |
| Telegram / notifications | Thalassa handles user communication |
| ML direction prediction | Confirmed dead end across BTC/ETH, 1d/1h, Transformer/XGBoost |
| Adaptive position sizing | Nunchi 103 experiments: fixed sizing outperforms all adaptive methods |
| Complex signal filters (correlation, multi-timeframe) | Nunchi: removing "smart" features improved score +52% |
| optuna-dashboard (this milestone) | Nice-to-have, can query DB directly; deferred to v3 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| VOTE-01 | Phase 10 | Complete |
| VOTE-02 | Phase 10 | Complete |
| VOTE-03 | Phase 10 | Complete |
| VOTE-04 | Phase 10 | Complete |
| VOTE-05 | Phase 10 | Complete |
| VOTE-06 | Phase 10 | Complete |
| AUTO-01 | Phase 11 | Pending |
| AUTO-02 | Phase 11 | Pending |
| AUTO-03 | Phase 12 | Complete |
| AUTO-04 | Phase 12 | Complete |
| AUTO-05 | Phase 12 | Complete |
| AUTO-06 | Phase 12 | Complete |
| RGME-01 | Phase 13 | Complete |
| RGME-02 | Phase 13 | Complete |
| RGME-03 | Phase 13 | Complete |
| PARM-01 | Phase 11 | Complete |
| PARM-02 | Phase 11 | Complete |
| PARM-03 | Phase 11 | Pending |
| PARM-04 | Phase 11 | Complete |
| PARM-05 | Phase 11 | Complete |
| ALIGN-01 | Phase 14 | Complete |
| ALIGN-02 | Phase 14 | Complete |
| ALIGN-03 | Phase 14 | Complete |
| ALIGN-04 | Phase 14 | Complete |
| ALIGN-05 | Phase 14 | Complete |
| ALIGN-06 | Phase 14 | Complete |
| ALIGN-07 | Phase 14 | Complete |
| ALIGN-08 | Phase 14 | Complete |
| ALIGN-09 | Phase 14 | Complete |
| ALIGN-10 | Phase 14 | Complete |
| PRED-01 | Phase 18 | Planned |
| PRED-02 | Phase 18 | Planned |
| PRED-03 | Phase 18 | Planned |
| PRED-04 | Phase 18 | Planned |
| PRED-05 | Phase 18 | Planned |

**Coverage:**
- v2.0 requirements: 30 total
- Mapped to phases: 30/30
- Unmapped: 0
- v3.0 requirements: 5 total (PRED-01 through PRED-05)
- Mapped to phases: 5/5
- Unmapped: 0

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-28 after Phase 18 planning*
