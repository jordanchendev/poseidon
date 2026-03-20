# Phase 1: Infrastructure & Data Foundation - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Stand up the deployment skeleton (Docker Compose with 6 services), database schemas (TimescaleDB hypertable for OHLCV, PostgreSQL for fundamentals and sentiment), and data ingestion pipeline (fetchers for 4 markets with scheduled updates) so all downstream modules have real market data to work with.

</domain>

<decisions>
## Implementation Decisions

### Symbol watchlist scope
- Broad coverage: 20+ symbols per market
- Config file seed list + API-managed (Thalassa can add/remove symbols via API later)
- Taiwan stocks: mixed — large-cap (權值) + mid/small-cap
- US stocks: Claude's discretion based on liquidity and sector diversity
- Taiwan futures: Claude's discretion based on FinMind availability
- Crypto spot: mainstream coins (top 10 by market cap — BTC, ETH, SOL, BNB, XRP, etc.)

### Fetch interval scope
- Taiwan stocks / US stocks / Taiwan futures: **daily (1d) only**
- Crypto spot: **daily (1d) + hourly (1h)**
- Rationale: Free APIs (FinMind, yfinance) are most reliable at daily granularity. Binance API is stable and free for hourly. Intraday for traditional markets added in future phases.

### Historical data backfill
- Target: **5+ years** of historical data (subject to API availability per market)
- Backfill strategy: batch by month/quarter to respect rate limits
- FinMind: ~600 requests/day free tier, spread across hours, one symbol per task
- yfinance: 1-2 second interval between requests, 3 retries on failure
- Binance (CCXT): 1200 req/min limit, 1h K-lines max 1000 per request (~41 days), segment accordingly
- Track backfill progress per symbol (resume from checkpoint on failure, don't restart)
- Idempotent writes: UPSERT (ON CONFLICT DO UPDATE) to prevent duplicates

### Data fetch scheduling (Celery Beat)
- Initial launch: trigger backfill task for all symbols
- Taiwan stocks/futures: daily after market close (13:45 UTC+8)
- US stocks: daily after market close (16:15 US/Eastern)
- Crypto spot: hourly (latest 1h candle) + daily (1d candle)
- No minute-level scheduling in Phase 1 (no active strategies yet)

### Fundamentals data
- Schema only — create the JSONB table, confirm read/write works
- No actual data fetching in Phase 1; deferred to when Feature Engine needs it

### Sentiment endpoint
- Build POST endpoint + schema (ready for Thalassa integration)
- Include mock sentiment data generator: **price-correlated pseudo-sentiment** (positive when price rises, negative when drops, with noise)
- Allows downstream phases to test strategies that include sentiment features

### Claude's Discretion
- Specific symbol selection within the 20+ per market guidelines
- Docker Compose networking and volume configuration details
- Alembic migration strategy and initial migration structure
- Config file format for symbol watchlist (YAML, TOML, or JSON)
- Error handling details for fetcher failures (retry logic, logging)
- Health check endpoint implementation
- API key middleware implementation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture & data layer
- `../docs/poseidon-design.md` — Full design document covering architecture, data layer schemas (OHLCV, fundamentals, sentiment tables), fetcher interfaces, Docker services, tech stack, scheduling, error handling, timezone handling, data lifecycle, and retention policies

### Project context
- `poseidon/.planning/PROJECT.md` — Project vision, constraints, key decisions, OpenClaw ecosystem context
- `poseidon/.planning/REQUIREMENTS.md` — Phase 1 requirements: INFRA-01/02/03, DATA-01/02/03/04/05/06, API-02

### Reference implementation
- `triton/docker-compose.yml` — Triton's Docker Compose (reference for service structure, but Poseidon uses TimescaleDB and adds Beat)
- `triton/pyproject.toml` — Triton's build config (reference for uv + pyproject.toml pattern)
- `triton/src/triton/` — Triton's source layout (reference for src/ layout convention)
- `triton/alembic/` — Triton's Alembic setup (reference for migration pattern)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- No existing Poseidon code — greenfield project
- Triton's patterns serve as reference but no shared code between services

### Established Patterns (from Triton)
- **Build tooling**: uv + pyproject.toml with optional dependency groups (gpu, dev)
- **Source layout**: src/{service_name}/ with `__init__.py`
- **API layer**: FastAPI with route modules under api/
- **Workers**: Celery with separate queue-based workers (gpu, cpu)
- **Database**: SQLAlchemy + Alembic migrations, pydantic-settings for config
- **Docker**: Multi-service compose with healthchecks, volume mounts, restart policies
- **Testing**: pytest + httpx + pytest-asyncio

### Integration Points
- Poseidon gets its own Redis and PostgreSQL instances (independent from Triton)
- TimescaleDB image replaces postgres:16-alpine (for hypertable support)
- Celery Beat service added (Triton doesn't have one)
- Module structure follows design spec (api/, core/, data/, strategies/, models/, signals/, backtest/, risk/, workers/) not Triton's flatter layout

</code_context>

<specifics>
## Specific Ideas

- Triton is running well — no known antipatterns to avoid. Follow its conventions where applicable.
- Mock sentiment should correlate with price movements (not pure random) so downstream strategy testing produces meaningful results.
- Backfill must be resumable — track per-symbol progress, resume from checkpoint on failure.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-infrastructure-data-foundation*
*Context gathered: 2026-03-20*
