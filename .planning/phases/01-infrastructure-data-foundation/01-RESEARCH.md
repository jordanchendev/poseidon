# Phase 1: Infrastructure & Data Foundation - Research

**Date:** 2026-03-20
**Phase:** 01-infrastructure-data-foundation
**Purpose:** Technical research to inform planning. Answers: "What do I need to know to PLAN this phase well?"

---

## 1. TimescaleDB Setup

### Docker Image

Use the official TimescaleDB image based on PostgreSQL 16:

```yaml
postgres:
  image: timescale/timescaledb:latest-pg16
```

The `timescale/timescaledb:latest-pg16` image is a drop-in replacement for `postgres:16-alpine`. It includes PostgreSQL 16 with the TimescaleDB extension pre-installed. No separate installation step is needed. The image supports the same environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`), the same data volume path (`/var/lib/postgresql/data`), and the same healthcheck command (`pg_isready`).

**Image size consideration:** The TimescaleDB image is larger than `postgres:16-alpine` (~400MB vs ~80MB). This is acceptable for a server deployment.

### Enabling the Extension

TimescaleDB must be explicitly enabled per database. In the first Alembic migration:

```python
from alembic import op

def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
```

This must run before any `create_hypertable()` call. The `CASCADE` flag installs any required dependencies.

### Creating Hypertables

After creating the OHLCV table as a regular PostgreSQL table, convert it to a hypertable:

```python
def upgrade():
    # 1. Create the regular table first
    op.create_table(
        "ohlcv",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("open", sa.Numeric, nullable=False),
        sa.Column("high", sa.Numeric, nullable=False),
        sa.Column("low", sa.Numeric, nullable=False),
        sa.Column("close", sa.Numeric, nullable=False),
        sa.Column("volume", sa.Numeric, nullable=False),
    )
    # Composite primary key
    op.create_primary_key("pk_ohlcv", "ohlcv", ["time", "symbol", "market", "interval"])

    # 2. Convert to hypertable
    op.execute(
        "SELECT create_hypertable('ohlcv', 'time', "
        "chunk_time_interval => INTERVAL '1 month', "
        "if_not_exists => TRUE)"
    )
```

**Key parameters for `create_hypertable`:**
- `chunk_time_interval`: Controls chunk size. `1 month` is appropriate for daily data across ~80 symbols. Each chunk will hold ~1,600 rows (80 symbols x 20 trading days) for daily, more for hourly crypto. This is a good balance between chunk count and query performance.
- `if_not_exists`: Prevents errors on re-run (idempotency).

**Pitfall: Primary keys and hypertables.** TimescaleDB requires the partitioning column (`time`) to be part of any unique constraint or primary key. The composite PK `(time, symbol, market, interval)` satisfies this. Do NOT create a separate `id` serial column as PK -- it will conflict with the hypertable requirement.

### Compression Policy

TimescaleDB native compression is configured via a policy after the hypertable exists:

```python
def upgrade():
    # Enable compression on the hypertable
    op.execute("""
        ALTER TABLE ohlcv SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, market, interval',
            timescaledb.compress_orderby = 'time DESC'
        )
    """)
    # Auto-compress chunks older than 7 days
    op.execute(
        "SELECT add_compression_policy('ohlcv', INTERVAL '7 days')"
    )
```

**`compress_segmentby`:** Columns used for filtering. Queries like `WHERE symbol = '2330' AND market = 'tw_stock'` remain fast because compressed data is organized by these segments.

**`compress_orderby`:** Determines physical ordering within segments. `time DESC` optimizes for "get the latest data" queries, which are the most common access pattern (latest candle, recent N candles for feature computation).

**Compression ratio:** TimescaleDB documentation reports 90-95% compression for time-series data with this configuration. For OHLCV data with numeric columns, expect approximately 90% compression.

**Transparent to queries:** Compressed and uncompressed chunks are queried identically. No application code changes needed. The only difference: INSERTs into compressed chunks require decompression first (handled automatically by TimescaleDB, but with a performance cost). This is why the 7-day policy matters -- recent data (where backfill UPSERTs might touch) stays uncompressed.

### Alembic + TimescaleDB Interaction

**Known issue:** Alembic's `--autogenerate` does not understand TimescaleDB-specific DDL (hypertable creation, compression policies). These must be written as raw `op.execute()` calls in manual migration steps.

**Recommended migration structure:**
1. Migration 001: `CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE`
2. Migration 002: Create tables (ohlcv, fundamentals, sentiment, backfill_progress) + convert to hypertable + set compression policy
3. Migration 003+: Future schema changes

**Downgrade consideration:** `op.execute("SELECT remove_compression_policy('ohlcv', if_exists => TRUE)")` followed by table drop. Hypertable conversion is not directly reversible -- the downgrade should drop and recreate as a regular table if needed.

### Indexes

TimescaleDB automatically creates an index on the partitioning column (`time`). Additional indexes to create:

```sql
-- Most common query pattern: "get OHLCV for a specific symbol/market/interval in a time range"
CREATE INDEX idx_ohlcv_symbol_market_interval_time
ON ohlcv (symbol, market, interval, time DESC);
```

TimescaleDB uses the chunk exclusion optimization: queries with a `WHERE time > X` clause automatically skip irrelevant chunks. This means the time-based index is extremely efficient even with millions of rows.

---

## 2. Data Provider APIs

### 2.1 FinMind (Taiwan Stocks + Futures)

**API endpoint:** `https://api.finmindtrade.com/api/v4/data`

**Authentication:** Free tier requires an API token. Register at finmindtrade.com. Token passed as `token` parameter.

**Rate limits (free tier):**
- 600 requests per day (hard limit)
- No per-second rate limit documented, but should add 0.5-1s delay between requests to be respectful

**Key datasets:**
| Dataset | `dataset` parameter | Fields |
|---------|-------------------|--------|
| TW Stock Daily | `TaiwanStockPrice` | date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover |
| TW Futures Daily | `TaiwanFuturesDaily` | date, FuturesID, Open, High, Low, Close, Volume, ... |

**Data format:** JSON response with a `data` array of objects. Dates as strings (`YYYY-MM-DD`).

**Historical data availability:** Data goes back to ~2000 for major stocks, ~2005 for futures. More than sufficient for the 5-year target.

**Request parameters:**
```python
params = {
    "dataset": "TaiwanStockPrice",
    "data_id": "2330",       # symbol
    "start_date": "2021-01-01",
    "end_date": "2021-12-31",
    "token": FINMIND_TOKEN,
}
```

**Backfill strategy with 600/day limit:**
- 20+ TW stocks + futures symbols, 5 years of data
- Batch by year: 1 request per symbol per year = ~5 requests per symbol
- 20 symbols x 5 requests = 100 requests for full TW stock backfill
- Add futures (~5 symbols x 5 requests = 25)
- Total: ~125 requests. Well within the 600/day limit for initial backfill in a single day
- For ongoing daily fetches: 1 request per symbol per day (~25 requests). Negligible.

**Column mapping to OHLCV schema:**
```python
FINMIND_COLUMN_MAP = {
    "date": "time",          # needs timezone attach (Asia/Taipei -> UTC)
    "open": "open",
    "max": "high",           # FinMind uses "max" not "high"
    "min": "low",            # FinMind uses "min" not "low"
    "close": "close",
    "Trading_Volume": "volume",
}
```

**Pitfall: FinMind field names.** `max`/`min` instead of `high`/`low`. `Trading_Volume` is capitalized. Each fetcher must normalize to the canonical OHLCV schema.

**Pitfall: FinMind date handling.** Dates are naive (no timezone). Taiwan market times must be treated as `Asia/Taipei` and converted to UTC before storage. A daily candle for 2024-01-15 in Taiwan should be stored as `2024-01-15T05:30:00+00:00` (market close at 13:30 local = 05:30 UTC).

**Pitfall: FinMind free tier reliability.** The free tier occasionally returns empty responses or HTTP 500 during high traffic. Retry logic (3 retries with exponential backoff) is essential.

### 2.2 yfinance (US Stocks)

**Library:** `yfinance` (Python package, wraps Yahoo Finance unofficial API)

**Authentication:** None required. No API key.

**Rate limits:** Unofficial API, no documented rate limits. In practice:
- Too-fast requests trigger HTTP 429 or soft bans (empty responses)
- Safe interval: 1-2 seconds between requests
- Session-level throttling: use a single `yfinance.download()` call for multiple symbols when possible

**Bulk download advantage:**
```python
import yfinance as yf
# Download multiple symbols in one call (much more efficient)
data = yf.download(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start="2019-01-01",
    end="2024-12-31",
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    threads=False,  # single-threaded to avoid rate limits
)
```

Using `yf.download()` with multiple tickers batches requests internally. This is the preferred approach for both backfill and daily updates.

**Historical data availability:**
- Daily data: goes back to IPO date for most stocks (decades). No 5-year limitation.
- `auto_adjust=True`: Returns adjusted OHLCV (accounts for splits/dividends). Recommended for consistency.

**Data format:** Returns a pandas DataFrame with DatetimeIndex (tz-aware, US/Eastern) and columns: Open, High, Low, Close, Volume.

**Column mapping:**
```python
YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
# DatetimeIndex -> "time" column, already tz-aware
```

**Pitfall: yfinance reliability.** yfinance wraps an unofficial API that Yahoo can change at any time. Common failure modes:
1. Empty DataFrame returned (symbol not found or API issue) -- check `df.empty` before processing
2. Missing trading days (holidays not always clean) -- don't treat gaps as errors
3. Occasional HTTP errors -- retry with backoff
4. Yahoo sometimes returns data with 1-day lag -- the daily fetch schedule (after market close) mitigates this

**Pitfall: Volume data type.** yfinance returns volume as float64 (not int). This is fine for the DECIMAL column in the schema but worth noting.

**Pitfall: Adjusted prices.** With `auto_adjust=True`, historical prices are retroactively adjusted for splits. This means the same historical date can return different values over time as new splits occur. For backtesting consistency, store the adjusted values at fetch time and note when the data was last refreshed. For Phase 1 this is acceptable; true point-in-time accuracy is a Phase 2+ concern.

### 2.3 CCXT / Binance (Crypto Spot)

**Library:** `ccxt` (unified cryptocurrency exchange API)

**Authentication:** Public endpoints only (OHLCV is public). No API key required for kline data.

**Rate limits (Binance):**
- 1200 requests per minute (weight-based, klines are weight 1-2)
- Individual IP rate limit, not account-based
- Very generous for our use case

**Kline endpoint:**
```python
import ccxt

exchange = ccxt.binance({
    "enableRateLimit": True,  # Built-in rate limiter
})

# Fetch 1h klines
ohlcv = exchange.fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe="1h",
    since=exchange.parse8601("2021-01-01T00:00:00Z"),
    limit=1000,  # Max 1000 candles per request
)
```

**Return format:** List of lists: `[[timestamp_ms, open, high, low, close, volume], ...]`

**Historical data pagination:**
- Max 1000 candles per request
- For 1h data: 1000 candles = ~41 days
- For 1d data: 1000 candles = ~2.7 years
- 5 years of hourly data for one symbol: ~43,800 candles = ~44 requests
- 10 crypto symbols x 44 requests = ~440 requests (completes in < 1 minute with rate limiting)
- 5 years of daily data: 2-3 requests per symbol (trivial)

**Pagination pattern:**
```python
async def fetch_all_ohlcv(exchange, symbol, timeframe, since, until):
    """Paginate through historical kline data."""
    all_data = []
    current_since = since
    while current_since < until:
        batch = exchange.fetch_ohlcv(
            symbol, timeframe, since=current_since, limit=1000
        )
        if not batch:
            break
        all_data.extend(batch)
        # Next page starts after the last candle
        current_since = batch[-1][0] + 1  # +1ms to avoid overlap
    return all_data
```

**Column mapping:**
```python
# CCXT returns: [timestamp_ms, open, high, low, close, volume]
# timestamp is milliseconds since epoch (UTC)
```

**Pitfall: CCXT sync vs async.** CCXT offers both sync and async clients. Since Celery tasks are synchronous by default, use the synchronous client (`ccxt.binance`), not the async version (`ccxt.async_support.binance`). The rate limiter works in both modes.

**Pitfall: Symbol format.** CCXT uses `BTC/USDT` format, but the design schema stores just the symbol code. Need a mapping: `BTC/USDT` -> symbol=`BTCUSDT`, market=`crypto_spot`. Define this in the symbol config.

**Pitfall: Binance delisted pairs.** Some trading pairs get delisted. Historical data may be unavailable for delisted pairs. Stick to top-10 coins by market cap to minimize this risk.

**Pitfall: Volume meaning.** Binance volume is in base currency (e.g., BTC for BTC/USDT). This is consistent with the OHLCV schema's generic `volume` column, but worth documenting.

---

## 3. Celery Beat Persistence

### The Problem

The default Celery Beat scheduler stores its schedule in a local `celerybeat-schedule` file (shelve database). This file is ephemeral in Docker containers -- if the container restarts, the schedule state (last run times, pending tasks) is lost. Tasks might re-run or be skipped.

### Options Evaluated

| Scheduler | Persistence | Dependencies | Complexity | Django Required |
|-----------|------------|--------------|------------|-----------------|
| Default (shelve file) | Volume mount | None | Low | No |
| celery-redbeat | Redis | `celery-redbeat` | Low | No |
| django-celery-beat | PostgreSQL | Django ORM | High | Yes |
| Custom DB scheduler | PostgreSQL | Custom code | Medium | No |

### Recommendation: celery-redbeat

**`celery-redbeat`** is the best fit for this stack:

1. **No Django dependency.** django-celery-beat requires Django ORM, which conflicts with our SQLAlchemy + FastAPI stack.
2. **Redis is already in the stack.** No additional infrastructure.
3. **True persistence.** Schedule state stored in Redis with configurable key prefix. Survives container restarts as long as Redis data is persisted (volume mount).
4. **Dynamic schedule support.** Schedules can be added/modified at runtime via the RedBeat API -- useful for future dynamic strategy scheduling.
5. **Minimal code change.** Drop-in replacement for the default scheduler.

**Installation:**
```
celery-redbeat>=2.2
```

**Configuration:**
```python
celery_app.conf.update(
    # RedBeat scheduler
    beat_scheduler="redbeat.RedBeatScheduler",
    redbeat_redis_url=settings.redis_url,
    redbeat_key_prefix="poseidon:redbeat:",

    # Beat schedule definition
    beat_schedule={
        "fetch-tw-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=6, minute=0),  # 14:00 UTC+8 = 06:00 UTC
            "args": ["tw_stock", "1d"],
        },
        "fetch-us-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=21, minute=30),  # 16:30 US/Eastern ~ 21:30 UTC
            "args": ["us_stock", "1d"],
        },
        "fetch-crypto-hourly": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(minute=5),  # Every hour at :05
            "args": ["crypto_spot", "1h"],
        },
        "fetch-crypto-daily": {
            "task": "poseidon.workers.cpu_tasks.fetch_market_data",
            "schedule": crontab(hour=0, minute=15),  # 00:15 UTC
            "args": ["crypto_spot", "1d"],
        },
    },
)
```

**Docker Beat service:**
```yaml
beat:
  build: .
  command: celery -A poseidon.workers.celery_app beat --loglevel=info
  # No --scheduler flag needed if beat_scheduler is set in config
  environment:
    - DATABASE_URL=...
    - REDIS_URL=...
  depends_on:
    redis:
      condition: service_started
  restart: unless-stopped
```

**Pitfall: Timezone handling in crontab.** Celery uses UTC by default (`timezone="UTC"` in config). All crontab schedules should be specified in UTC. The design spec says "After TW market close (13:45 UTC+8)" which is 05:45 UTC -- schedule the fetch for 06:00 UTC to allow a buffer. Similarly, "After US market close (16:15 US/Eastern)" varies with DST: 21:15 UTC (EST) or 20:15 UTC (EDT). Use 21:30 UTC as a safe buffer that works in both seasons.

**Pitfall: Exactly-once scheduling.** RedBeat uses Redis locks to ensure only one Beat instance runs the schedule. If two Beat containers start accidentally, the lock prevents duplicate task dispatching. However, ensure only one Beat service is defined in docker-compose.

**Alternative considered: Volume-mounted shelve file.** This works (`-v ./beat-schedule:/app/celerybeat-schedule`) but has drawbacks: (a) file corruption on unclean shutdown, (b) no runtime schedule modification, (c) no visibility into schedule state without accessing the container filesystem. RedBeat is worth the small dependency for these benefits.

---

## 4. Backfill Architecture

### Progress Tracking Table

Create a `backfill_progress` table to track per-symbol, per-interval backfill state:

```python
class BackfillProgress(Base):
    __tablename__ = "backfill_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    market = Column(String(32), nullable=False)
    interval = Column(String(8), nullable=False)
    last_fetched_date = Column(DateTime(timezone=True), nullable=True)
    target_start_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    # status: pending | in_progress | completed | failed
    error_message = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "market", "interval", name="uq_backfill_symbol_market_interval"),
    )
```

### Backfill Strategy

**Initialization flow:**
1. On first startup (or manual trigger via API), create `backfill_progress` rows for every symbol/interval combination with `status=pending` and `target_start_date` = 5 years ago.
2. Celery task picks up pending/failed rows and processes them.
3. Each batch fetches one time segment (e.g., 1 year for FinMind, 1000 candles for CCXT), writes to DB via UPSERT, updates `last_fetched_date`.
4. If interrupted, the task resumes from `last_fetched_date` on next run.

**Idempotent writes (UPSERT):**
```sql
INSERT INTO ohlcv (time, symbol, market, instrument, interval, open, high, low, close, volume)
VALUES (...)
ON CONFLICT (time, symbol, market, interval) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume;
```

In SQLAlchemy with PostgreSQL dialect:
```python
from sqlalchemy.dialects.postgresql import insert

stmt = insert(OHLCVTable).values(rows)
stmt = stmt.on_conflict_do_update(
    constraint="pk_ohlcv",
    set_={
        "open": stmt.excluded.open,
        "high": stmt.excluded.high,
        "low": stmt.excluded.low,
        "close": stmt.excluded.close,
        "volume": stmt.excluded.volume,
    },
)
session.execute(stmt)
```

### Batching Strategy Per Provider

| Provider | Batch Size | Rate Limiting | Estimated Time (Full Backfill) |
|----------|-----------|---------------|-------------------------------|
| FinMind | 1 year per request | 1s delay between requests | ~125 requests = ~3 min |
| yfinance | All 5 years in one `yf.download()` per symbol (or bulk) | 2s delay between symbols | ~20 symbols x 2s = ~40s (or one bulk call) |
| CCXT (1d) | 1000 days per request (~2.7y) | Built-in rate limiter | ~20 requests = ~30s |
| CCXT (1h) | 1000 hours per request (~41d) | Built-in rate limiter | ~440 requests = ~10 min |

**Total estimated backfill time: < 15 minutes.** This is fast enough to run as a one-time Celery task on startup.

### Backfill Celery Task Design

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def backfill_symbol(self, symbol: str, market: str, interval: str):
    """Backfill historical data for a single symbol. Resumable."""
    progress = get_or_create_progress(symbol, market, interval)
    if progress.status == "completed":
        return

    progress.status = "in_progress"
    save_progress(progress)

    fetcher = get_fetcher(market)
    start = progress.last_fetched_date or progress.target_start_date
    end = datetime.now(timezone.utc)

    try:
        # Fetch in batches (provider-specific batch size)
        for batch_start, batch_end in iter_batches(start, end, market):
            df = fetcher.fetch_ohlcv(symbol, interval, batch_start, batch_end)
            if not df.empty:
                upsert_ohlcv(df, symbol, market, interval)
                progress.last_fetched_date = batch_end
                save_progress(progress)

        progress.status = "completed"
        save_progress(progress)

    except Exception as exc:
        progress.status = "failed"
        progress.error_message = str(exc)
        save_progress(progress)
        raise self.retry(exc=exc)
```

**Key design decisions:**
- One Celery task per symbol (not per market). This allows fine-grained progress tracking and parallel execution across CPU worker processes.
- Progress saved after each batch, not just at the end. This enables true resumability.
- Failed tasks retry 3 times with 60-second delay. After max retries, the progress row stays in `failed` state for manual investigation.

### Triggering Backfill

Two trigger mechanisms:
1. **API endpoint:** `POST /data/backfill` -- trigger backfill for all symbols or a specific symbol. Useful for manual control.
2. **Startup check:** A Celery task that runs on Beat startup, checks for any `pending` or `failed` backfill rows, and re-dispatches them. This handles the "first launch" case.

---

## 5. Docker Compose Structure

### 6 Services

```yaml
services:
  api:          # FastAPI (REST API)
  beat:         # Celery Beat (persistent scheduler via RedBeat)
  gpu-worker:   # Celery worker (GPU queue: model training/inference)
  cpu-worker:   # Celery worker (CPU queue: data fetching, backtesting, features)
  redis:        # Message broker + result backend + RedBeat storage
  postgres:     # TimescaleDB (OHLCV hypertable + PostgreSQL tables)
```

### Full Compose Skeleton

```yaml
services:
  api:
    build: .
    ports:
      - "8001:8000"  # 8001 on host to avoid conflict with Triton's 8000
    environment:
      - DATABASE_URL=postgresql://poseidon:poseidon@postgres:5432/poseidon
      - REDIS_URL=redis://redis:6379/0
      - API_KEY=${POSEIDON_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  beat:
    build: .
    command: celery -A poseidon.workers.celery_app beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://poseidon:poseidon@postgres:5432/poseidon
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  gpu-worker:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    command: celery -A poseidon.workers.celery_app worker -Q gpu -c 1 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://poseidon:poseidon@postgres:5432/poseidon
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - model-artifacts:/data/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  cpu-worker:
    build: .
    command: celery -A poseidon.workers.celery_app worker -Q cpu -c 4 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://poseidon:poseidon@postgres:5432/poseidon
      - REDIS_URL=redis://redis:6379/0
      - FINMIND_TOKEN=${FINMIND_TOKEN}
    volumes:
      - model-artifacts:/data/models
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - poseidon-redis:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  postgres:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_USER: poseidon
      POSTGRES_PASSWORD: poseidon
      POSTGRES_DB: poseidon
    volumes:
      - poseidon-pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U poseidon"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  poseidon-redis:
  poseidon-pgdata:
  model-artifacts:
```

### Key Design Points

**Port mapping:** Use 8001 on the host to avoid conflict with Triton's port 8000. The API container internally still runs on 8000.

**Redis AOF persistence:** The `--appendonly yes` flag enables Redis AOF (Append-Only File) persistence. This is critical because RedBeat stores schedule state in Redis. Without persistence, a Redis restart loses all schedule state.

**Named volumes vs host mounts:** Triton uses host path mounts (`/data/postgres`). For Poseidon, use named Docker volumes (`poseidon-pgdata`) to keep data separate and avoid collision with Triton's data directories. The naming convention `poseidon-*` makes it clear which volumes belong to which service.

**GPU passthrough:** Identical to Triton's pattern. The `deploy.resources.reservations.devices` block passes the NVIDIA GPU to the container. Requires `nvidia-container-toolkit` on the host (already installed for Triton). Only the gpu-worker needs GPU access.

**Healthchecks:**
- `postgres`: `pg_isready` (same as Triton)
- `redis`: `redis-cli ping` (standard Redis healthcheck)
- `api`: HTTP check on `/health` endpoint
- Workers and Beat do not need healthchecks (they auto-reconnect to Redis/Postgres, and `restart: unless-stopped` handles crashes)

**Environment variables via `.env` file:** Sensitive values (`POSEIDON_API_KEY`, `FINMIND_TOKEN`) should live in a `.env` file loaded by Docker Compose. The compose file references them as `${VAR_NAME}`.

**Network isolation:** Docker Compose creates a default network for all services. The API is the only service with a port mapping to the host. Redis and Postgres are not exposed outside the Docker network.

**Phase 1 GPU worker:** The GPU worker is defined in docker-compose but will do minimal work in Phase 1 (no model training yet). It still needs to start for the healthcheck/service count success criterion. Give it a basic Dockerfile that installs the base dependencies without ML libraries. The Dockerfile.gpu with ML dependencies can be created in Phase 3.

---

## 6. API Key Auth Middleware

### FastAPI Dependency Injection Pattern

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from poseidon.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate X-API-Key header against configured key."""
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    return api_key
```

### Applying Globally

Two approaches:

**Option A: Router-level dependency (recommended)**
```python
# main.py
from poseidon.api.auth import verify_api_key

app = FastAPI(title="Poseidon", dependencies=[Depends(verify_api_key)])

# Health endpoint excluded from auth (for Docker healthcheck)
health_router = APIRouter()

@health_router.get("/health")
async def health():
    return {"status": "ok"}

app.include_router(health_router)  # No auth dependency
app.include_router(data_router, prefix="/data", tags=["data"])  # Inherits app-level auth
```

**Option B: Middleware approach**
A raw middleware that checks headers before routing. More flexible but loses OpenAPI schema integration.

**Recommendation: Option A.** It integrates cleanly with FastAPI's OpenAPI/Swagger UI (the "Authorize" button works), and the health endpoint can be explicitly excluded by adding it before the global dependency takes effect or by using a separate app mount.

**Excluding the health endpoint from auth:**

The cleanest approach is to not use `dependencies=[Depends(verify_api_key)]` at the app level, but instead apply it to each router:

```python
app = FastAPI(title="Poseidon")

# No auth
app.include_router(health_router, tags=["health"])

# With auth
secured = [Depends(verify_api_key)]
app.include_router(data_router, prefix="/data", tags=["data"], dependencies=secured)
app.include_router(sentiment_router, prefix="/sentiment", tags=["sentiment"], dependencies=secured)
```

This is more explicit and avoids the need to "un-apply" a global dependency.

### Configuration

```python
class Settings(BaseSettings):
    api_key: str = ""  # Required in production, empty fails all auth checks

    model_config = {"env_prefix": "POSEIDON_", "env_file": ".env"}
```

With `env_prefix = "POSEIDON_"`, the environment variable is `POSEIDON_API_KEY`. This avoids collisions with Triton's environment variables.

---

## 7. Project Structure

### Source Layout

Following the design spec module layout and Triton's `src/` convention:

```
poseidon/
├── .planning/              # Planning documents (existing)
├── src/
│   └── poseidon/
│       ├── __init__.py
│       ├── api/            # REST endpoints (thin routing layer)
│       │   ├── __init__.py
│       │   ├── auth.py     # API key dependency
│       │   ├── data.py     # /data endpoints
│       │   ├── sentiment.py # /sentiment endpoints
│       │   └── health.py   # /health endpoint
│       ├── core/           # Shared: config, schemas, events
│       │   ├── __init__.py
│       │   ├── config.py   # pydantic-settings Settings
│       │   └── schemas.py  # Pydantic models for API request/response
│       ├── data/           # Data layer: fetchers, storage
│       │   ├── __init__.py
│       │   ├── fetchers/
│       │   │   ├── __init__.py
│       │   │   ├── base.py     # BaseFetcher ABC
│       │   │   ├── finmind.py  # FinMind fetcher
│       │   │   ├── yfinance.py # yfinance fetcher
│       │   │   └── ccxt.py     # CCXT/Binance fetcher
│       │   └── storage.py  # DB read/write for OHLCV, fundamentals, sentiment
│       ├── models/         # SQLAlchemy models (DB tables)
│       │   ├── __init__.py
│       │   ├── ohlcv.py
│       │   ├── fundamentals.py
│       │   ├── sentiment.py
│       │   └── backfill.py
│       ├── workers/        # Celery app + tasks
│       │   ├── __init__.py
│       │   ├── celery_app.py
│       │   └── cpu_tasks.py
│       ├── strategies/     # Phase 4 (empty __init__.py for now)
│       ├── signals/        # Phase 5 (empty __init__.py for now)
│       ├── backtest/       # Phase 6 (empty __init__.py for now)
│       ├── risk/           # Phase 5 (empty __init__.py for now)
│       └── main.py         # FastAPI app entry point
├── alembic/
│   ├── env.py
│   ├── versions/           # Migration files
│   └── alembic.ini
├── tests/
│   ├── conftest.py
│   ├── test_api/
│   ├── test_fetchers/
│   ├── test_storage/
│   └── test_workers/
├── config/
│   └── symbols.yaml        # Symbol watchlist seed
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.gpu
├── pyproject.toml
├── alembic.ini
└── .env.example
```

### Module Naming Clarification

The design spec has a `models/` directory for the ML model engine (BaseModel, registry, trainer). The Phase 1 `models/` directory holds SQLAlchemy ORM models (database table definitions). These are the same directory in the design spec -- the ML model engine classes will be added in Phase 3 alongside the ORM models, or the ORM models can live under a `models/db/` subdirectory if separation is needed. For Phase 1, `models/` contains only ORM definitions. This is a naming collision to be aware of and resolve during Phase 3 planning.

**Recommendation:** Use `models/` for SQLAlchemy ORM models (the database schema). When Phase 3 arrives, the ML model engine can live under `models/engine/` or the design spec's `models/` layout can be the organizing principle with DB models under `models/db.py` or similar. For now, keep it simple.

### Entry Points

| Service | Entry Point | Command |
|---------|------------|---------|
| API | `poseidon.main:app` | `uvicorn poseidon.main:app --host 0.0.0.0 --port 8000` |
| Beat | `poseidon.workers.celery_app` | `celery -A poseidon.workers.celery_app beat --loglevel=info` |
| GPU Worker | `poseidon.workers.celery_app` | `celery -A poseidon.workers.celery_app worker -Q gpu -c 1 --loglevel=info` |
| CPU Worker | `poseidon.workers.celery_app` | `celery -A poseidon.workers.celery_app worker -Q cpu -c 4 --loglevel=info` |

### Symbol Watchlist Configuration

YAML is the most readable format for a seed list:

```yaml
# config/symbols.yaml
markets:
  tw_stock:
    instrument: spot
    interval: ["1d"]
    symbols:
      - { id: "2330", name: "TSMC" }
      - { id: "2317", name: "Hon Hai" }
      - { id: "2454", name: "MediaTek" }
      # ... 20+ symbols

  tw_futures:
    instrument: futures
    interval: ["1d"]
    symbols:
      - { id: "TX", name: "TAIEX Futures" }
      - { id: "MTX", name: "Mini TAIEX" }
      # ...

  us_stock:
    instrument: spot
    interval: ["1d"]
    symbols:
      - { id: "AAPL", name: "Apple" }
      - { id: "MSFT", name: "Microsoft" }
      - { id: "GOOGL", name: "Alphabet" }
      # ... 20+ symbols

  crypto_spot:
    instrument: spot
    interval: ["1d", "1h"]
    symbols:
      - { id: "BTCUSDT", name: "Bitcoin", ccxt_symbol: "BTC/USDT" }
      - { id: "ETHUSDT", name: "Ethereum", ccxt_symbol: "ETH/USDT" }
      # ... 10+ symbols
```

The YAML file is the seed. Thalassa can add/remove symbols via API later. On startup, the system loads the YAML and creates any missing symbols in the watchlist table (or uses the YAML directly if no DB watchlist table is needed in Phase 1).

### pyproject.toml

```toml
[project]
name = "poseidon"
version = "0.1.0"
description = "Trading signal platform for OpenClaw"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "psycopg2-binary>=2.9",
    "pydantic-settings>=2.0",
    "celery[redis]>=5.4",
    "celery-redbeat>=2.2",
    "redis>=5.0",
    "pandas>=2.2",
    "yfinance>=0.2.40",
    "ccxt>=4.0",
    "requests>=2.31",          # For FinMind API
    "pyyaml>=6.0",             # For symbol config
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0",
    "xgboost>=2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "pytest-cov>=5.0",
    "ruff>=0.5",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 120
target-version = "py311"
```

---

## 8. Testing Strategy

### Testing Pyramid for Phase 1

| Layer | What to Test | Tools | Count (est.) |
|-------|-------------|-------|-------------|
| Unit | Fetcher data normalization, storage UPSERT logic, config parsing, auth | pytest, mock | ~30 tests |
| Integration | Fetcher + real API (limited), storage + TimescaleDB, Celery task execution | pytest, testcontainers or Docker fixtures | ~15 tests |
| E2E | docker compose up, health check, full fetch-store cycle | pytest + httpx, docker compose | ~5 tests |

### Testing Fetchers (Mocking External APIs)

Fetchers have two concerns: (1) HTTP call to external API, (2) data normalization to OHLCV schema.

**Strategy: Mock the HTTP layer, test normalization with fixture data.**

```python
# tests/test_fetchers/test_finmind.py
import pytest
from unittest.mock import patch, MagicMock

from poseidon.data.fetchers.finmind import FinMindFetcher

# Fixture: real FinMind API response shape
FINMIND_RESPONSE = {
    "msg": "success",
    "status": 200,
    "data": [
        {"date": "2024-01-02", "stock_id": "2330", "open": 590.0, "max": 595.0,
         "min": 588.0, "close": 593.0, "Trading_Volume": 25000000},
        {"date": "2024-01-03", "stock_id": "2330", "open": 593.0, "max": 598.0,
         "min": 591.0, "close": 596.0, "Trading_Volume": 28000000},
    ],
}

@patch("poseidon.data.fetchers.finmind.requests.get")
def test_finmind_fetch_normalizes_columns(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = FINMIND_RESPONSE
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    fetcher = FinMindFetcher(token="test")
    df = fetcher.fetch_ohlcv("2330", "1d", "2024-01-01", "2024-01-31")

    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.iloc[0]["high"] == 595.0  # "max" -> "high"
    assert df.iloc[0]["low"] == 588.0   # "min" -> "low"
```

**Each fetcher test file includes:**
1. Fixture data matching the real API response shape (captured once from real calls)
2. Tests for column normalization
3. Tests for timezone conversion
4. Tests for empty response handling
5. Tests for error response handling (HTTP errors, malformed data)

### Testing Database Operations

**Option A: testcontainers-python (recommended)**

```python
# tests/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def db_engine():
    with PostgresContainer(
        image="timescale/timescaledb:latest-pg16",
        user="test",
        password="test",
        dbname="test_poseidon",
    ) as pg:
        engine = create_engine(pg.get_connection_url())
        # Run Alembic migrations
        run_migrations(engine)
        yield engine
```

This spins up a real TimescaleDB container for the test session. Pros: tests hypertable/compression behavior realistically. Cons: slower (container startup ~5-10s), requires Docker.

**Option B: SQLite in-memory (not viable)**

SQLite does not support TimescaleDB extensions, `ON CONFLICT DO UPDATE`, or `TIMESTAMPTZ`. Not suitable for this project.

**Option C: Shared Docker Compose test database**

Use the same `docker-compose.yml` but with a separate test database:
```bash
# Before tests
docker compose exec postgres createdb -U poseidon test_poseidon
# Run tests against test_poseidon
# After tests
docker compose exec postgres dropdb -U poseidon test_poseidon
```

This is simpler than testcontainers but requires the Docker Compose stack to be running.

**Recommendation:** Use testcontainers for CI-like isolation. For local development, use the shared Docker Compose database approach (faster iteration).

### Testing Celery Tasks

```python
# tests/test_workers/test_cpu_tasks.py
import pytest
from unittest.mock import patch, MagicMock
from poseidon.workers.cpu_tasks import backfill_symbol

@patch("poseidon.workers.cpu_tasks.get_fetcher")
@patch("poseidon.workers.cpu_tasks.upsert_ohlcv")
@patch("poseidon.workers.cpu_tasks.get_or_create_progress")
def test_backfill_symbol_success(mock_progress, mock_upsert, mock_fetcher):
    """Test backfill task completes and updates progress."""
    mock_progress.return_value = MagicMock(
        status="pending",
        last_fetched_date=None,
        target_start_date=datetime(2019, 1, 1, tzinfo=timezone.utc),
    )
    mock_fetcher.return_value.fetch_ohlcv.return_value = sample_df

    # Call task directly (not via Celery, to avoid broker dependency)
    backfill_symbol("2330", "tw_stock", "1d")

    mock_upsert.assert_called_once()
    assert mock_progress.return_value.status == "completed"
```

**Key testing principle for Celery tasks:** Call the task function directly in unit tests (bypass the Celery broker). Integration tests that verify the full Celery task lifecycle (dispatch, execution, result) require a running Redis broker -- use this sparingly.

### Testing API Endpoints

```python
# tests/test_api/test_health.py
from fastapi.testclient import TestClient
from poseidon.main import app

client = TestClient(app)

def test_health_no_auth():
    """Health endpoint should work without API key."""
    response = client.get("/health")
    assert response.status_code == 200

def test_data_requires_auth():
    """Data endpoints should reject requests without API key."""
    response = client.get("/data/symbols")
    assert response.status_code == 401  # or 403

def test_data_with_valid_key():
    """Data endpoints should accept valid API key."""
    response = client.get("/data/symbols", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
```

Use `TestClient` (which wraps httpx) for synchronous testing of FastAPI endpoints. Override the settings dependency to inject a test API key.

### Test Configuration

```python
# tests/conftest.py
import pytest
from poseidon.core.config import Settings

@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Override settings for tests."""
    monkeypatch.setenv("POSEIDON_API_KEY", "test-key")
    monkeypatch.setenv("POSEIDON_DATABASE_URL", "postgresql://test:test@localhost:5432/test_poseidon")
```

---

## 9. Dependency Version Matrix

Summary of all Phase 1 dependencies with recommended versions:

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.115 | REST API framework |
| uvicorn[standard] | >=0.30 | ASGI server |
| sqlalchemy | >=2.0 | ORM and database toolkit |
| alembic | >=1.13 | Database migrations |
| psycopg2-binary | >=2.9 | PostgreSQL driver |
| pydantic-settings | >=2.0 | Configuration management |
| celery[redis] | >=5.4 | Task queue |
| celery-redbeat | >=2.2 | Persistent Beat scheduler |
| redis | >=5.0 | Redis client |
| pandas | >=2.2 | Data manipulation |
| yfinance | >=0.2.40 | US stock data |
| ccxt | >=4.0 | Crypto exchange data |
| requests | >=2.31 | HTTP client (FinMind) |
| pyyaml | >=6.0 | Symbol config parsing |
| timescale/timescaledb | latest-pg16 | Docker image |
| redis (Docker) | 7-alpine | Docker image |

---

## 10. Potential Pitfalls & Risk Mitigations

### High Risk

| Risk | Impact | Mitigation |
|------|--------|-----------|
| yfinance API breaking change | US stock data unavailable | Pin version, have a manual CSV import fallback, monitor yfinance GitHub issues |
| FinMind free tier rate limit too restrictive | Backfill takes multiple days | Batch efficiently (~125 requests needed, well within 600/day), cache responses |
| TimescaleDB compression blocks UPSERT on old data | Backfill of already-compressed time ranges fails | Set compression policy to 7 days; backfill only touches recent data after initial load |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|-----------|
| RedBeat Redis key corruption | Schedules lost | Redis AOF persistence, schedule defined in code (re-created on restart) |
| Alembic + TimescaleDB DDL drift | autogenerate misses hypertable changes | Manual migration steps for all TimescaleDB-specific DDL; never rely solely on autogenerate |
| Docker volume data loss | All market data lost | Named volumes with backup strategy; data is re-fetchable (inconvenient but not catastrophic) |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CCXT Binance API changes | Crypto data fetch fails | CCXT abstracts Binance API; update CCXT version |
| Port conflict with Triton | API unreachable | Use port 8001 for Poseidon |
| GPU contention with Triton | Training slow | Phase 1 has no GPU training; address in Phase 3 |

---

## 11. Validation Architecture

How to verify each success criterion is met.

### Criterion 1: docker compose up starts 6 services with health checks

**Validation command:**
```bash
docker compose up -d
# Wait for all services to be healthy
docker compose ps --format "table {{.Service}}\t{{.Status}}"
```

**Expected output:** All 6 services (api, beat, gpu-worker, cpu-worker, redis, postgres) show "Up" or "Up (healthy)" status.

**Automated test:**
```python
def test_all_services_running():
    result = subprocess.run(
        ["docker", "compose", "ps", "--services", "--filter", "status=running"],
        capture_output=True, text=True
    )
    running = set(result.stdout.strip().split("\n"))
    expected = {"api", "beat", "gpu-worker", "cpu-worker", "redis", "postgres"}
    assert running == expected
```

**Health check verification:**
```bash
# API health
curl -s http://localhost:8001/health | jq .
# Redis health
docker compose exec redis redis-cli ping
# Postgres/TimescaleDB health
docker compose exec postgres pg_isready -U poseidon
# TimescaleDB extension loaded
docker compose exec postgres psql -U poseidon -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"
```

### Criterion 2: Celery Beat triggers scheduled data fetch surviving container restart

**Validation steps:**
1. Start the stack: `docker compose up -d`
2. Wait for Beat to register schedules (check logs): `docker compose logs beat --tail 20`
3. Verify RedBeat keys in Redis:
   ```bash
   docker compose exec redis redis-cli KEYS "poseidon:redbeat:*"
   ```
4. Restart the Beat container:
   ```bash
   docker compose restart beat
   ```
5. Verify RedBeat keys persist after restart:
   ```bash
   docker compose exec redis redis-cli KEYS "poseidon:redbeat:*"
   ```
6. Trigger a manual task to verify the pipeline works:
   ```bash
   curl -X POST http://localhost:8001/data/fetch \
     -H "X-API-Key: $POSEIDON_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"market": "crypto_spot", "symbol": "BTCUSDT", "interval": "1d"}'
   ```
7. Check cpu-worker logs for task execution: `docker compose logs cpu-worker --tail 20`

**Automated test:**
```python
def test_beat_survives_restart():
    # Record Redis keys before restart
    keys_before = get_redis_keys("poseidon:redbeat:*")
    assert len(keys_before) > 0

    subprocess.run(["docker", "compose", "restart", "beat"])
    time.sleep(10)  # Wait for Beat to reinitialize

    keys_after = get_redis_keys("poseidon:redbeat:*")
    assert len(keys_after) > 0
    # Schedule names should match
    assert keys_before == keys_after
```

### Criterion 3: OHLCV data for at least one symbol per market stored in TimescaleDB with 7-day compression

**Validation steps:**
1. Trigger backfill or fetch for one symbol per market
2. Query the hypertable:
   ```sql
   -- Verify data exists for each market
   SELECT market, symbol, interval, COUNT(*), MIN(time), MAX(time)
   FROM ohlcv
   GROUP BY market, symbol, interval;

   -- Verify hypertable exists
   SELECT hypertable_name, compression_enabled
   FROM timescaledb_information.hypertables
   WHERE hypertable_name = 'ohlcv';

   -- Verify compression policy
   SELECT hypertable_name, schedule_interval
   FROM timescaledb_information.jobs
   WHERE proc_name = 'policy_compression';
   ```
3. Verify data schema (TIMESTAMPTZ):
   ```sql
   SELECT column_name, data_type
   FROM information_schema.columns
   WHERE table_name = 'ohlcv';
   ```

**Automated test:**
```python
def test_ohlcv_data_per_market(db_session):
    markets = ["tw_stock", "us_stock", "crypto_spot", "tw_futures"]
    for market in markets:
        count = db_session.execute(
            text("SELECT COUNT(*) FROM ohlcv WHERE market = :m"),
            {"m": market}
        ).scalar()
        assert count > 0, f"No OHLCV data for market {market}"

def test_compression_policy(db_session):
    result = db_session.execute(text(
        "SELECT COUNT(*) FROM timescaledb_information.jobs "
        "WHERE proc_name = 'policy_compression' AND hypertable_name = 'ohlcv'"
    )).scalar()
    assert result == 1
```

### Criterion 4: Fundamentals JSONB row writable/readable

**Validation steps:**
```sql
-- Write
INSERT INTO fundamentals (id, symbol, market, date, data)
VALUES (gen_random_uuid(), '2330', 'tw_stock', '2024-01-01',
        '{"eps": 32.5, "pe_ratio": 18.2, "revenue": 2000000}'::jsonb);

-- Read
SELECT * FROM fundamentals WHERE symbol = '2330';

-- Query JSONB field
SELECT symbol, data->>'eps' AS eps FROM fundamentals WHERE data->>'pe_ratio' IS NOT NULL;
```

**Automated test:**
```python
def test_fundamentals_write_read(db_session):
    row = Fundamentals(
        symbol="2330", market="tw_stock",
        date=date(2024, 1, 1),
        data={"eps": 32.5, "pe_ratio": 18.2},
    )
    db_session.add(row)
    db_session.commit()

    result = db_session.query(Fundamentals).filter_by(symbol="2330").first()
    assert result is not None
    assert result.data["eps"] == 32.5
```

### Criterion 5: Sentiment score received via API and persisted

**Validation steps:**
```bash
# POST sentiment score
curl -X POST http://localhost:8001/sentiment \
  -H "X-API-Key: $POSEIDON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "2330",
    "market": "tw_stock",
    "source_type": "news",
    "score": 0.75
  }'

# GET sentiment scores
curl http://localhost:8001/sentiment?symbol=2330 \
  -H "X-API-Key: $POSEIDON_API_KEY"
```

**Automated test:**
```python
def test_sentiment_endpoint(client):
    # POST
    response = client.post(
        "/sentiment",
        json={"symbol": "2330", "market": "tw_stock", "source_type": "news", "score": 0.75},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 201

    # Verify persisted
    response = client.get("/sentiment?symbol=2330", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["score"] == 0.75
```

### Criterion 6: All endpoints reject requests without valid X-API-Key

**Validation steps:**
```bash
# Without key -- should return 401 or 403
curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/data/symbols
# Expected: 401 or 403

# With invalid key
curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: wrong" http://localhost:8001/data/symbols
# Expected: 401

# Health endpoint should work without key
curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health
# Expected: 200
```

**Automated test:**
```python
SECURED_ENDPOINTS = [
    ("GET", "/data/symbols"),
    ("GET", "/data/status"),
    ("POST", "/data/fetch"),
    ("POST", "/sentiment"),
    ("GET", "/sentiment"),
]

@pytest.mark.parametrize("method,path", SECURED_ENDPOINTS)
def test_endpoint_rejects_no_key(client, method, path):
    response = getattr(client, method.lower())(path)
    assert response.status_code in (401, 403)

@pytest.mark.parametrize("method,path", SECURED_ENDPOINTS)
def test_endpoint_rejects_invalid_key(client, method, path):
    response = getattr(client, method.lower())(path, headers={"X-API-Key": "invalid"})
    assert response.status_code == 401

def test_health_no_auth_required(client):
    response = client.get("/health")
    assert response.status_code == 200
```

### Criterion 7: All timestamps stored as TIMESTAMPTZ in UTC

**Validation steps:**
```sql
-- Check column types
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND data_type LIKE '%timestamp%'
ORDER BY table_name, column_name;
-- All should show "timestamp with time zone"

-- Verify UTC storage
SELECT time, time AT TIME ZONE 'UTC' AS utc_time
FROM ohlcv LIMIT 5;
-- Both columns should show the same value (confirming UTC storage)

-- Verify application writes in UTC
SELECT created_at FROM sentiment ORDER BY created_at DESC LIMIT 1;
-- Should show +00 timezone offset
```

**Automated test:**
```python
def test_all_timestamp_columns_are_timestamptz(db_engine):
    """Verify every timestamp column in the schema uses TIMESTAMPTZ."""
    result = db_engine.execute(text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND (data_type LIKE '%timestamp%' OR column_name LIKE '%time%' OR column_name LIKE '%_at')
    """))
    for row in result:
        assert row.data_type == "timestamp with time zone", \
            f"{row.table_name}.{row.column_name} is {row.data_type}, expected TIMESTAMPTZ"
```

---

## RESEARCH COMPLETE

### Key Findings Summary

1. **TimescaleDB** is a drop-in replacement for PostgreSQL. Use `timescale/timescaledb:latest-pg16` Docker image. Hypertable creation and compression policies must be managed via raw SQL in Alembic migrations (not autogenerate). Compression after 7 days with `segmentby = 'symbol, market, interval'` and `orderby = 'time DESC'` matches the access patterns.

2. **Data providers** are all viable for Phase 1 scope. FinMind's 600/day free tier limit is sufficient (estimated ~125 requests for full 5-year backfill). yfinance bulk download is efficient but fragile (unofficial API). CCXT/Binance has generous limits (1200/min) and built-in rate limiting. Each fetcher needs distinct column mapping and timezone normalization logic.

3. **celery-redbeat** is the clear choice for persistent Beat scheduling -- no Django dependency, uses Redis (already in stack), survives container restarts, and supports dynamic schedule modification.

4. **Backfill architecture** uses a `backfill_progress` database table for per-symbol/interval checkpoint tracking. One Celery task per symbol enables parallelism and fine-grained resumability. UPSERT (ON CONFLICT DO UPDATE) ensures idempotent writes. Total initial backfill is estimated at under 15 minutes.

5. **Docker Compose** follows Triton's patterns with three changes: TimescaleDB image replaces postgres, Beat service added, port 8001 to avoid Triton collision. Redis needs AOF persistence (`--appendonly yes`) for RedBeat. Named volumes keep data separate from Triton.

6. **API key auth** uses FastAPI's `Security` dependency with `APIKeyHeader`. Apply per-router (not globally) to exclude the health endpoint cleanly.

7. **Project structure** follows `src/poseidon/` layout with the design spec's module directories created upfront (empty `__init__.py` for future phases). pyproject.toml follows Triton's pattern with Phase 1 dependencies.

8. **Testing** uses three layers: unit tests with mocked HTTP for fetchers, integration tests with testcontainers (TimescaleDB) for storage, and parametrized endpoint tests for auth. Direct task function calls bypass Celery broker in unit tests.

### Dependencies to Install Before Phase 1 Implementation

```
pip install: fastapi uvicorn[standard] sqlalchemy alembic psycopg2-binary pydantic-settings celery[redis] celery-redbeat redis pandas yfinance ccxt requests pyyaml python-multipart
Docker images: timescale/timescaledb:latest-pg16, redis:7-alpine
Dev: pytest pytest-asyncio httpx pytest-cov ruff testcontainers[postgres]
```

### Open Questions for Planning

1. **Phase 1 GPU worker Dockerfile:** Should it be a real GPU image (with CUDA) or a minimal placeholder that just starts Celery? The success criterion says "6 services running" but Phase 1 has no GPU tasks. Recommendation: minimal Dockerfile that shares the base image, with GPU dependencies added in Phase 3.

2. **Watchlist management:** Should the symbol watchlist be a database table (API-managed) or purely config file in Phase 1? The context says "config file seed list + API-managed." Recommendation: YAML seed file loaded on startup into a `watchlist` DB table. API endpoints for CRUD can be added but are not required by any Phase 1 success criterion.

3. **Mock sentiment generator:** The context specifies "price-correlated pseudo-sentiment." This should be a standalone utility/script, not part of the main application code. It writes sentiment scores to the DB that correlate with recent price movements. Timing: implement after OHLCV data is available.

---

*Research completed: 2026-03-20*
*Phase: 01-infrastructure-data-foundation*
*Ready for: /gsd:plan-phase 1*
