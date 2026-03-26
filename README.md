# Poseidon

Trading signal platform that fetches multi-market data, computes features, trains ML models, evaluates rule-based strategies, runs backtests with walk-forward analysis, applies risk controls, and delivers standardized trading signals via Redis Streams.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  FastAPI     │    │ Celery Beat  │    │  Consumer   │
│  (API :8001) │    │ (scheduler)  │    │ (external)  │
└──────┬───────┘    └──────┬───────┘    └──────▲──────┘
       │                   │                   │
       │            ┌──────▼───────┐    Redis Streams
       │            │ CPU Worker   │───────────┘
       │            │ (fetch/back- │
       │            │  test/rules) │
       │            └──────────────┘
       │            ┌──────────────┐
       │            │ GPU Worker   │
       │            │ (train/pred) │
       │            └──────────────┘
       │
  ┌────▼───────┐    ┌───────────┐
  │TimescaleDB │    │   Redis   │
  │(PostgreSQL)│    │ (broker)  │
  └────────────┘    └───────────┘
```

### Project Structure

```
src/poseidon/
├── api/           # FastAPI routers
├── backtest/      # Backtest runner, walk-forward, optimizer
├── core/          # Config, schemas, events
├── data/          # Fetchers (FinMind, yfinance, CCXT), feature engine
├── ml/            # Model registry, training, artifacts
├── models/        # SQLAlchemy database models
├── risk/          # Risk rules engine, virtual portfolio
├── signals/       # Signal generation & Redis Streams delivery
├── strategies/    # Rule DSL, model strategy, voting strategy, regime router
├── autoresearch/  # Autonomous parameter search runner, guard, report
└── workers/       # Celery task definitions
```

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env: set POSEIDON_API_KEY and POSEIDON_FINMIND_TOKEN

# 2. Start all services
docker compose up -d

# API available at http://localhost:8001
```

## Supported Markets

| Market | Symbols | Data Source | Interval |
|--------|---------|------------|----------|
| Taiwan Stocks | 22 (TSMC, MediaTek, Hon Hai...) | FinMind | 1d |
| Taiwan Futures | 4 (TX, MTX, TE, TF) | FinMind | 1d |
| US Stocks | 21 (AAPL, NVDA, TSLA...) | yfinance | 1d |
| Crypto Spot | 10 (BTC, ETH, SOL...) | CCXT | 1d, 1h |

Symbol watchlist defined in `config/symbols.yaml`. Can be modified via API.

## API Endpoints

All endpoints except `/health` require `X-API-Key` header.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (no auth) |
| POST | `/data/fetch` | Trigger market data fetch |
| POST | `/data/backfill` | Start historical backfill |
| GET | `/data/backfill/status` | Check backfill progress |
| GET/POST | `/sentiment` | Sentiment scores |
| GET/POST | `/strategies` | Strategy CRUD |
| POST | `/strategies/{id}/activate` | Activate strategy |
| GET/POST | `/api/risk-rules` | Risk rule management |
| GET | `/api/risk-rules/portfolio` | Virtual portfolio state |
| GET | `/models` | List model versions |
| POST | `/models/train` | Train new model |
| POST | `/models/{id}/predict` | Run prediction |
| POST | `/backtest/run` | Execute backtest |
| POST | `/backtest/optimize` | Hyperparameter optimization |
| POST | `/autoresearch/run` | Dispatch autoresearch (returns task_id) |
| GET | `/autoresearch/status/{task_id}` | Poll autoresearch progress |
| POST | `/autoresearch/stop/{task_id}` | Graceful stop |
| GET | `/autoresearch/experiments` | Query experiment results |
| GET | `/signals` | Signal history |

## Configuration

### Environment Variables

```bash
POSEIDON_API_KEY=<secret>              # API authentication key
POSEIDON_FINMIND_TOKEN=<token>         # FinMind API token (Taiwan market data)
# Set automatically by docker-compose:
# POSEIDON_DATABASE_URL=postgresql://poseidon:poseidon@postgres:5432/poseidon
# POSEIDON_REDIS_URL=redis://redis:6379/0
```

## Docker Services

| Service | Image | Purpose |
|---------|-------|---------|
| api | Dockerfile | FastAPI server (port 8001) |
| beat | Dockerfile | Celery Beat scheduler |
| cpu-worker | Dockerfile | Data fetch, backtest, rules (4 concurrency) |
| gpu-worker | Dockerfile.gpu | Model training & prediction (GPU) |
| postgres | timescale/timescaledb:latest-pg16 | TimescaleDB database |
| redis | redis:7-alpine | Message broker & signal delivery |

## Development

```bash
# Install dependencies
uv sync --extra dev

# Run tests
pytest

# Run API locally
uvicorn poseidon.main:app --reload --port 8000

# Lint
ruff check src/ tests/
```

### GPU Dependencies

```bash
uv sync --extra gpu  # PyTorch + XGBoost
```

## v2.0: Strategy Pivot — Rule-Based Voting + AutoResearch

v2.0 放棄 ML 方向預測，改用簡單信號投票 + 自動化迭代搜索。

### VotingStrategy (Phase 10)
6-signal 投票策略：Momentum×2, EMA crossover, RSI, MACD histogram, Bollinger squeeze。`min_votes` 多數決 + ATR trailing stop 出場。

### AutoResearch (Phase 11-12)
自動化參數搜索框架：
- **ParameterSearchPipeline** — Optuna TPE + holdout split + WFE validation
- **AutoResearchRunner** — 跨市場批次搜索，per-market 失敗隔離
- **StrategyMutator** — 隨機 + Optuna 變異策略參數
- **ExperimentTracker** — 實驗結果持久化到 PostgreSQL
- **Immutability Guard** — contextvar 保護 FeatureEngine/BacktestRunner 不被 autoresearch 修改

### Regime Classification (Phase 13)
市場狀態分類系統（gated — 不贏就關）：
- **RegimeRouter** — 包裝 VotingStrategy，根據 regime 動態調整 min_votes/position_pct
- **Regime Labels** — 百分位 realized_vol_20 分 3 類 (low/medium/high vol)
- **RegimeSearchPipeline** — Per-regime Optuna 搜索（只調 2 參數）
- **Outperformance Gate** — holdout 上嚴格比較，不贏就 bypass

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Modular monolith | Single developer, operational simplicity |
| Features computed on-the-fly | No feature store needed; avoids EAV anti-pattern |
| Redis Streams for signals | Guaranteed delivery with consumer groups |
| Celery Beat (redbeat) | Persistent scheduling, survives restarts |
| Strategy DSL | JSON condition trees with boolean combinators |
| Walk-forward backtest | WFE >= 50% threshold for overfitting detection |
| TimescaleDB | Native time-series compression for OHLCV data |
| Voting > ML prediction | Simple rule voting + automated search beats complex ML (v2.0) |
| Outperformance gate | Regime routing must strictly beat static baseline or auto-disable |
| Dynamic feature specs | VotingStrategy tells FeatureEngine exactly what it needs |
