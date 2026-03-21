# Poseidon

Trading signal platform for the OpenClaw investment analysis system. Fetches multi-market data, computes features, trains ML models, evaluates rule-based strategies, runs backtests with walk-forward analysis, applies risk controls, and delivers standardized signals via Redis Streams.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  FastAPI     │    │ Celery Beat  │    │ Thalassa    │
│  (API :8001) │    │ (scheduler)  │    │ (consumer)  │
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
├── strategies/    # Rule DSL, model strategy, rule strategy
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

Symbol watchlist defined in `config/symbols.yaml`. Thalassa can modify via API.

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
