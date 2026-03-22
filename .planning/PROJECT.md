# Poseidon

## What This Is

Poseidon is a trading signal platform — part of the OpenClaw (龍蝦) investment analysis system. It fetches market data across four markets (Taiwan stocks, Taiwan futures, US stocks, cryptocurrency), computes features, trains/deploys ML models, evaluates rule-based strategies from natural language, runs backtests, applies risk controls, and pushes standardized trading signals to the Thalassa orchestrator via Redis Streams. A personal tool for one developer (Jordan Chen), running on a home GPU server (stormtrooper).

## Core Value

Reliably produce quality trading signals and deliver them to Thalassa for human review before manual execution.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Fetch and store OHLCV data for TW stocks, TW futures, US stocks, crypto spot
- [ ] Compute technical indicators and features on-the-fly from raw data
- [ ] Train, version, and deploy ML models (model-agnostic: XGBoost, LSTM, RL, etc.)
- [ ] Define and evaluate rule-based strategies via JSON DSL with boolean combinators
- [ ] Run backtests with realistic cost models, slippage, and walk-forward analysis
- [ ] Apply pre-signal risk checks (position limits, drawdown, frequency, leverage)
- [ ] Push standardized signals to Thalassa via Redis Streams with guaranteed delivery
- [ ] Expose REST API for Thalassa to manage data, models, strategies, backtests, and risk
- [ ] Accept natural language strategies from Thalassa (LLM translates to DSL)
- [ ] Support multiple time granularities (minute to daily) for different strategies

### Out of Scope

- Order execution / auto-trading — Poseidon only produces signals, not trades
- Text extraction / NLP — Triton handles media-to-text, Thalassa handles sentiment scoring
- Web scraping — Thalassa's responsibility
- Notification / Telegram — Thalassa handles user communication
- Options trading (Phase 3) — schema supports it, implementation deferred
- Crypto perpetual contracts (Phase 2) — schema supports it, implementation deferred

## Context

**OpenClaw ecosystem:**
- **Triton** (complete): Media-to-text GPU worker (FastAPI + Celery + Redis + PostgreSQL). Running on stormtrooper.
- **Thalassa** (not started): Orchestrator / lobster core. Decides what to process, does web scraping, NLP sentiment, talks to user via Telegram.
- **Poseidon** (this project): Trading signal platform.

**Signal flow:** Poseidon → Redis Streams → Thalassa → Telegram → User manually trades

**Sentiment flow:** Triton (text) → Thalassa (NLP) → Poseidon (receives score as a feature, no raw text)

**Infrastructure:**
- stormtrooper: Ubuntu 24.04, i7-8700, 32GB RAM, RTX 4070 Ti SUPER 16GB VRAM
- Triton uses ~7GB VRAM. Poseidon shares the same GPU (schedule training off-peak, future: second GPU).
- Poseidon gets its own Redis and PostgreSQL instances (independent from Triton).

**Design spec:** Full design document at `../docs/poseidon-design.md` covering architecture, data layer, feature engine, model engine, strategy DSL, backtest engine, risk engine, signal schema, API design, security, error handling, timezone handling, data lifecycle, and scheduling.

**Industry research completed:** Web research on feature stores (wide table > EAV), Redis Streams (confirmed over Kafka for this scale), backtesting (event-driven, shared pipeline), strategy DSL (JSON condition trees with boolean combinators), model registry (custom PostgreSQL over MLflow), risk engines (chain-of-responsibility pattern).

## Constraints

- **Tech stack**: FastAPI + Celery + Redis + PostgreSQL/TimescaleDB + Docker Compose — must match Triton for consistency
- **GPU**: Shared RTX 4070 Ti SUPER with Triton (~9GB headroom). ML prediction on CPU for most models, GPU only for DL training/inference.
- **Data providers**: Free tier only — FinMind (TW), yfinance (US), CCXT/Binance (crypto)
- **Single developer**: Architecture must be simple enough for one person to maintain
- **Network**: Internal only — Poseidon API not exposed publicly, API key auth for Thalassa

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Modular monolith (not microservices) | Single developer, operational simplicity | — Pending |
| Features computed on-the-fly (no feature store table) | Data volume too small to justify storage; eliminates EAV anti-pattern | — Pending |
| Redis Streams (not pub/sub) | Guaranteed delivery, consumer groups, replay on reconnect | — Pending |
| Celery Beat (not APScheduler) | Survives container restarts, persistent scheduling | — Pending |
| BaseModel.predict() returns DataFrame, not Signal | Decouples models from signal schema; ModelStrategy wrapper converts | — Pending |
| Model lifecycle includes shadow state | Champion-challenger testing before activation; future-proofs for auto-trading | — Pending |
| Strategy DSL with all/any/none combinators | Enables complex rule composition from natural language | — Pending |
| Walk-forward uses WFE ≥ 50% (not Sharpe threshold) | Industry standard; better overfitting detection | — Pending |
| Backtest trades/equity in separate tables (not JSONB) | Enables cross-backtest comparison queries | — Pending |
| Virtual portfolio persisted in PostgreSQL | Survives restarts; rebuilt from signal history | — Pending |

## Current State

Phase 9 complete — PatchTST Transformer model implemented (582 lines), all 7 BaseModel methods, mixed precision training, 57 tests passing. This is the last phase in milestone v1.0.

---
*Last updated: 2026-03-22 after Phase 9 completion*
