# Phase 39 Backfill API + Coverage + Live-Isolation Smoke Runbook

Operator-facing runbook proving the Phase 39 (`backfill-api-coverage`) work
ships end-to-end on stormtrooper. Satisfies Phase 39 ROADMAP success
criteria 4 (real backfill via API), 5 (data_coverage_mv read), and 6
(live-trading worker isolation).

This is the canonical "did Phase 39 actually work in production" smoke
test. Run it after every Phase 39 plan deploy and any time a backfill
regression is suspected.

All commands run on stormtrooper unless explicitly noted.

## 0. Preconditions

Before starting, confirm ALL of the following:

- [ ] Phase 39 plans 39-01, 39-02, 39-03, and 39-04 merged into `poseidon`
      `main` and pulled onto `~/Projects/poseidon`.
- [ ] Containers rebuilt with the Phase 39 migration files baked in:
      `docker compose build api beat backfill-worker cpu-worker`.
- [ ] Database is at the Phase 39 head:

      ```bash
      docker compose exec api uv run alembic current
      # Expected: 022 (head)
      ```

      If you see `020 (head)` or `021 (head)` you are running pre-Phase 39
      schema. Run:

      ```bash
      docker compose exec api uv run alembic upgrade head
      ```

      and re-check.
- [ ] `docker compose ps` shows ALL of the following containers `Up`:
      `api`, `cpu-worker`, `gpu-worker`, `beat`, `backfill-worker`. If
      `backfill-worker` is missing the dedicated queue from plan 39-02 is
      not deployed; bring it up before continuing:

      ```bash
      docker compose up -d backfill-worker
      ```
- [ ] Three Celery worker nodes online with the right queue subscriptions:

      ```bash
      docker compose exec api celery -A poseidon.workers.celery_app inspect active_queues
      # Expected: 3 nodes online
      #   - cpu node with queue 'cpu'
      #   - gpu node with queue 'gpu'
      #   - backfill node with queue 'backfill'
      ```
- [ ] `POSEIDON_API_KEY` from `.env` is exported into your local shell so
      you can hit the API:

      ```bash
      export POSEIDON_API_KEY=$(grep POSEIDON_API_KEY ~/Projects/poseidon/.env | cut -d= -f2)
      ```

## 1. Manual POST `/api/data/backfill`

Trigger a real two-year backfill of `crypto_perp BTCUSDT 4h`. Plain HTTP
POST against the live FastAPI on port 8001 (host port mapping in
`docker-compose.yml`). The endpoint requires explicit `symbols[]` and
`intervals[]` per Phase 39 D-03.

```bash
curl -X POST http://localhost:8001/api/data/backfill \
  -H "X-API-Key: ${POSEIDON_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "market": "crypto_perp",
    "symbols": ["BTCUSDT"],
    "intervals": ["4h"],
    "start": "2024-04-01T00:00:00Z",
    "end":   "2026-04-01T00:00:00Z"
  }'
```

Expected response (HTTP 202):

```json
{
  "job_id": "<uuid>",
  "status": "pending"
}
```

Capture the `job_id` for the next steps:

```bash
JOB_ID="<paste uuid here>"
```

## 2. Poll `/api/data/backfill/{job_id}` and `/api/data/coverage`

### 2a. Job status polling

Watch the job tick from `pending` → `running` → `succeeded` (or
`cancelled` if you abort it). The detail endpoint reads the durable
BackfillJob row directly from Postgres (Phase 39 D-04), not the Celery
result backend, so restarts of `backfill-worker` mid-job will not erase
state.

```bash
watch -n 5 "curl -s -H 'X-API-Key: ${POSEIDON_API_KEY}' \
  http://localhost:8001/api/data/backfill/${JOB_ID} \
  | python -m json.tool"
```

Expected progression of fields:

- `status`: `pending` → `running` → `succeeded`
- `cursor.next_ts`: monotonically advances toward the requested `end`
- `progress.rows_written`: monotonically increases
- `progress.chunks_done`: monotonically increases
- `started_at`: stamped once on the first chunk
- `finished_at`: stamped once when the job transitions to `succeeded`

Notes:

- If the job sits in `running` for a long time and `cursor.next_ts` does
  not move, check `backfill-worker` logs for rate-limit backoff or
  circuit-breaker open events. Phase 39-02 is designed to self-reschedule
  via `backfill_chunk.delay(job_id)` rather than fail the job.
- If you need to stop the job early (e.g. wrong window), POST to the
  cancel endpoint — it cooperatively flips the row to `cancelled` and
  preserves cursor + progress so you can see exactly where it stopped:

  ```bash
  curl -X POST -H "X-API-Key: ${POSEIDON_API_KEY}" \
    http://localhost:8001/api/data/backfill/${JOB_ID}/cancel
  ```

### 2b. Coverage read

The success-path hook in `_run_backfill_chunk` triggers
`coverage_view_refresh.delay()` immediately after the job transitions to
`succeeded`, so the materialized view should reflect the new rows within
seconds, not on the next hourly beat tick.

```bash
curl -s -H "X-API-Key: ${POSEIDON_API_KEY}" \
  "http://localhost:8001/api/data/coverage?market=crypto_perp&symbol=BTCUSDT&interval=4h" \
  | python -m json.tool
```

Expected response (single-element list):

```json
[
  {
    "market": "crypto_perp",
    "symbol": "BTCUSDT",
    "interval": "4h",
    "first_ts": "2024-04-01T00:00:00Z",
    "last_ts":  "2026-04-01T00:00:00Z",
    "row_count": 4380,
    "expected_count": 4380,
    "gap_count": 0,
    "completeness_pct": 1.0,
    "staleness_seconds": <recent>,
    "health": "green"
  }
]
```

Acceptable degraded states (still a passing smoke):

- `health: "yellow"` if `staleness_seconds` > `interval_seconds * 2` but
  `completeness_pct >= 0.95`. Indicates the latest live ingest tick has
  not run yet — not a backfill failure.
- A small non-zero `gap_count` (<1% of `expected_count`) for intervals
  that include exchange downtime windows. Investigate via the `ohlcv`
  table directly if you need exact reasoning.

## 3. Live-trading worker isolation checks

Phase 39 success criterion 6: a long-running backfill must NOT delay the
live perpetual trading workers. While the backfill is still running,
verify both of the following.

### 3a. `perp_liquidation_monitor` stays at 1-minute cadence

```bash
docker compose logs --since 5m cpu-worker \
  | grep -E 'perp_liquidation_monitor.*succeeded' \
  | tail -5
```

Expected: at least 4 successful runs in the last 5 minutes (one every
minute), each `succeeded in <1s` from the timestamps. If the cadence
slips to >2 minutes the backfill is starving the cpu queue and the
isolation guarantee is broken — abort the smoke and investigate.

### 3b. `perp_rebalance` runs on its 4h schedule

If your smoke window crosses an `:05` mark of an even 4h slot
(00:05 / 04:05 / 08:05 / 12:05 / 16:05 / 20:05 UTC):

```bash
docker compose logs --since 1h cpu-worker \
  | grep -E 'perp_rebalance.*(succeeded|received)' \
  | tail -10
```

Expected: `perp_rebalance` `received` and `succeeded` lines with normal
duration (no >5x latency vs the Phase 27 baseline).

### 3c. Backfill task lives on the dedicated worker, not cpu-worker

The whole point of Phase 39-02 is that `backfill_chunk` runs on
`backfill-worker` and never on `cpu-worker`. Verify with two greps:

```bash
# Should show backfill_chunk activity:
docker compose logs --since 5m backfill-worker \
  | grep -E 'backfill_chunk.*(received|succeeded)' \
  | tail -5

# Should NOT show backfill_chunk activity:
docker compose logs --since 5m cpu-worker \
  | grep -E 'backfill_chunk' \
  || echo "OK: no backfill_chunk on cpu-worker"
```

If `cpu-worker` shows any `backfill_chunk` line during the window the
queue routing in `celery_app.task_routes` has regressed and the
isolation contract is broken.

## 4. Dispatcher (Plan 39-04 BACKFILL-05) sanity check

Phase 39-04 ships `historical_backfill_dispatcher` on the cpu queue,
scheduled hourly via Beat. After the first hourly tick (or by manually
firing the task), the dispatcher should create one BackfillJob row per
`(market, symbol, interval)` tuple in `symbols.yaml` that does not yet
have history.

### 4a. Verify the beat entry is registered

```bash
docker compose exec beat celery -A poseidon.workers.celery_app inspect scheduled \
  | grep historical-backfill-dispatcher \
  || docker compose logs --since 1h beat | grep -i "historical-backfill-dispatcher"
```

Expected: at least one match (either in the live schedule introspection
or the most recent beat log line).

### 4b. Manually trigger the dispatcher (no need to wait for the hour)

```bash
docker compose exec api python -c "
from poseidon.workers.cpu_tasks import historical_backfill_dispatcher
result = historical_backfill_dispatcher.apply().result
print(result)
"
```

Expected output is a dict like:

```
{'created': N, 'skipped_first_backfill_done': M, 'skipped_duplicate': K}
```

`N + M + K` should equal the number of `(market, symbol, interval)`
tuples in `symbols.yaml`. After running the dispatcher twice in a row,
the second invocation must report `created == 0` and
`skipped_duplicate == N` from the first run — this is the duplicate
suppression contract from RESEARCH Pitfall 5.

### 4c. Verify the dispatcher actually wrote rows

```bash
docker compose exec api python -c "
from poseidon.models.base import SessionLocal
from poseidon.models.backfill import BackfillJob
s = SessionLocal()
rows = s.query(BackfillJob).filter(BackfillJob.requested_by == 'dispatcher').all()
print(f'dispatcher rows: {len(rows)}')
for r in rows[:5]:
    print(r.market, r.symbols, r.intervals, r.status, r.created_at)
s.close()
"
```

Expected: at least one `dispatcher`-tagged row per missing tuple from
`symbols.yaml`. The dedicated `backfill-worker` should then start
chewing through them on its own cadence.

## 5. Success Checklist

A passing Phase 39 smoke requires ALL of the following:

- [ ] Section 1 POST returned HTTP 202 with a UUID job_id.
- [ ] Section 2a job reached `status="succeeded"` with `finished_at`
      stamped.
- [ ] Section 2a `progress.rows_written` matches the expected candle
      count for the requested window (~4380 for a 2-year 4h backfill).
- [ ] Section 2b `/api/data/coverage` shows the new tuple with
      `health` in `{"green", "yellow"}` and `gap_count` ≤ 1% of
      `expected_count`.
- [ ] Section 3a `perp_liquidation_monitor` ran at least once per minute
      during the entire backfill window.
- [ ] Section 3b `perp_rebalance` (if a 4h boundary was crossed) ran on
      schedule with no >5x latency regression.
- [ ] Section 3c `backfill_chunk` appeared ONLY in `backfill-worker`
      logs, never in `cpu-worker` logs.
- [ ] Section 4 dispatcher created one row per uncovered tuple, and the
      second invocation skipped them all as duplicates.

If any item fails, stop the smoke and file a finding in
`.planning/STATE.md` Blockers with the failing section and the exact
log lines.

## 6. Cleanup

The smoke leaves real data in the `ohlcv` hypertable and durable
BackfillJob rows in Postgres. That is intentional — the next smoke run
will exercise the dispatcher's "skip first_backfill_done=true" path and
cover the resume case naturally.

If you need to start clean (e.g. a re-run from scratch), do NOT drop
`ingest_state` or `ohlcv`. Instead, cancel any in-flight backfill
explicitly so the worker stops cleanly:

```bash
curl -X POST -H "X-API-Key: ${POSEIDON_API_KEY}" \
  http://localhost:8001/api/data/backfill/${JOB_ID}/cancel
```

Then archive the smoke output (curl responses + relevant log greps)
into `local_dev/sessions/` for posterity.
