# Phase 40 Data Health Observability Smoke Runbook

Operator-facing runbook proving the Phase 40 (`data-health-observability`)
work ships end-to-end on stormtrooper. Satisfies Phase 40 ROADMAP success
criteria 1 (gap audit), 2 (Uptime Kuma dead-man's-switch), 3 (Uptime Kuma violation
alert), 4 (CAGG transparency), 5 (Kairos data-health view), and 6 (this
runbook itself).

This is the canonical "did Phase 40 actually work in production" smoke
test. Run it after every Phase 40 plan deploy and any time a freshness or
gap regression is suspected.

All commands run on stormtrooper unless explicitly noted.

## 0. Preconditions

Before starting, confirm ALL of the following:

- [ ] Phase 40 plans 40-01..05 merged into `poseidon` `main` and pulled
      onto `~/Projects/poseidon`.
- [ ] Containers rebuilt with the Phase 40 migration files baked in:

      ```bash
      cd ~/Projects/poseidon
      docker compose build api beat backfill-worker cpu-worker
      ```

- [ ] Database is at the Phase 40 head:

      ```bash
      docker compose exec api uv run alembic current
      # Expected: 024 (head)
      ```

      If you see `022 (head)` or `023 (head)` you are running pre-Phase 40
      schema. Run:

      ```bash
      docker compose exec api uv run alembic upgrade head
      ```

      and re-check. Note: migration 024 creates a TimescaleDB continuous
      aggregate; this requires Postgres extension `timescaledb` to be
      present (it is — see migration 002).

- [ ] `docker compose ps` shows ALL of the following containers `Up`:
      `api`, `cpu-worker`, `gpu-worker`, `beat`, `backfill-worker`.

- [ ] Phase 40 beat schedule entries are registered:

      ```bash
      docker compose exec api celery -A poseidon.workers.celery_app inspect registered \
        | grep -E 'data_gap_audit|ingest_freshness_watchdog'
      ```

      Expected: both `poseidon.workers.cpu_tasks.data_gap_audit` and
      `poseidon.workers.cpu_tasks.ingest_freshness_watchdog` listed.

- [ ] Uptime Kuma running on stormtrooper. Deploy if not:
      ```bash
      docker run -d --restart=always -p 3001:3001 \
        -v uptime-kuma:/app/data --name uptime-kuma louislam/uptime-kuma:1
      ```
      1. Open http://stormtrooper:3001 and complete initial setup (create admin account).
      2. Add Monitor → Type: **Push** → Name: `poseidon-freshness`.
      3. Set Heartbeat Interval = 900 (15 min), Retries = 1.
      4. Setup Notifications → Telegram (Bot Token + Chat ID).
      5. Copy the Push URL from the monitor detail page (format: `http://localhost:3001/api/push/<token>`).

- [ ] `UPTIME_KUMA_PUSH_URL` is set in `~/Projects/poseidon/.env`:

      ```bash
      grep UPTIME_KUMA_PUSH_URL ~/Projects/poseidon/.env
      ```

      If empty, append:

      ```bash
      echo "UPTIME_KUMA_PUSH_URL=http://localhost:3001/api/push/<token>" >> ~/Projects/poseidon/.env
      docker compose up -d cpu-worker beat
      ```

- [ ] `POSEIDON_API_KEY` from `.env` is exported into your local shell:

      ```bash
      export POSEIDON_API_KEY=$(grep POSEIDON_API_KEY ~/Projects/poseidon/.env | cut -d= -f2)
      ```

## 1. Verify Uptime Kuma heartbeat (FRESH-04 dead-man's-switch baseline)

Goal: prove the watchdog is currently pinging Uptime Kuma success. If this
section fails, sections 2 and 3 are meaningless.

Manually fire the watchdog once:

```bash
docker compose exec api python -c "
from poseidon.workers.cpu_tasks import ingest_freshness_watchdog
print(ingest_freshness_watchdog.apply().result)
"
```

Expected output:

```
{'checked': N, 'violations': 0, 'unknown': U, 'monitor_pinged': 'success'}
```

Verify on https://Uptime Kuma that the `poseidon-freshness`
check shows a fresh "Up" event within the last minute.

Also confirm the REST endpoint returns the same freshness data (this is
the data source for the Kairos `/data-health` freshness panel):

```bash
curl -s -H "X-API-Key: ${POSEIDON_API_KEY}" \
  'http://localhost:8001/api/data/freshness' \
  | python -m json.tool
```

Expected: a list of `DataFreshnessResponse` objects with `status: "ok"`
for all tuples that have an SLA defined. Each row shows `market`,
`interval`, `last_successful_ts`, `lag_seconds`, and `sla_seconds`.

## 2. Inject FRESH-03 violation and verify Telegram alert

Goal: verify a stale ingest_state row triggers a /fail ping and Uptime Kuma
fires a Telegram alert.

Pick a sacrificial tuple (use crypto_perp BTCUSDT 4h — it will self-heal
on the next cron tick):

```bash
docker compose exec api python -c "
from datetime import datetime, timezone, timedelta
from poseidon.models.base import SessionLocal
from poseidon.models.ingest_state import IngestState
s = SessionLocal()
row = s.query(IngestState).filter_by(market='crypto_perp', symbol='BTCUSDT', interval='4h').one()
original_ts = row.last_successful_ts
row.last_successful_ts = datetime.now(timezone.utc) - timedelta(hours=24)
s.commit()
print('original ts:', original_ts)
print('forced stale ts:', row.last_successful_ts)
s.close()
"
```

Manually fire the watchdog:

```bash
docker compose exec api python -c "
from poseidon.workers.cpu_tasks import ingest_freshness_watchdog
print(ingest_freshness_watchdog.apply().result)
"
```

Expected output (note `violations: 1` and `monitor_pinged: 'fail'`):

```
{'checked': N, 'violations': 1, 'unknown': U, 'monitor_pinged': 'fail'}
```

Verify on https://Uptime Kuma that the `poseidon-freshness`
check shows a "Down" event AND that you receive a Telegram alert (Uptime Kuma
fires the integration on the first /fail ping after a healthy state).

Restore the original timestamp so live ingest is not stuck:

```bash
docker compose exec api python -c "
from datetime import datetime, timezone
from poseidon.models.base import SessionLocal
from poseidon.models.ingest_state import IngestState
s = SessionLocal()
row = s.query(IngestState).filter_by(market='crypto_perp', symbol='BTCUSDT', interval='4h').one()
row.last_successful_ts = datetime.now(timezone.utc)
s.commit()
s.close()
print('restored')
"
```

Re-fire the watchdog and confirm `violations: 0` + `monitor_pinged: 'success'`.

## 3. Verify FRESH-04 dead-man's-switch by killing Beat

Goal: prove that if Beat dies entirely, Uptime Kuma fires a "dead" alert
automatically WITHOUT any explicit Python error path.

Stop Beat:

```bash
docker compose stop beat
```

Wait at least the configured Uptime Kuma grace period (35 minutes if you set
grace=30, to be safe). Verify on https://Uptime Kuma that the
`poseidon-freshness` check transitions to "Down" automatically
AND that you receive a second Telegram alert ("watchdog dead").

Restart Beat:

```bash
docker compose up -d beat
```

The next 15-minute beat tick should resume the heartbeat. Confirm the
Uptime Kuma check transitions back to "Up".

## 4. Inject a known gap and verify GET /api/data/gaps (COVERAGE-04)

Goal: prove the daily `data_gap_audit` task detects a known gap window
via the LAG-window-function logic and that the `/api/data/gaps`
endpoint surfaces it.

Pick a sacrificial tuple. We will delete a small window of bars and
re-fire the audit.

Record the rows we are about to delete (so we can restore them after):

```bash
docker compose exec api python -c "
from datetime import datetime, timezone
from poseidon.models.base import SessionLocal
from sqlalchemy import text
s = SessionLocal()
rows = s.execute(text('''
    SELECT time, open, high, low, close, volume
    FROM ohlcv
    WHERE market='crypto_perp' AND symbol='ETHUSDT' AND interval='4h'
    ORDER BY time DESC
    LIMIT 12
''')).fetchall()
for r in rows:
    print(r)
s.close()
" | tee /tmp/phase40-deleted-rows.txt
```

Delete the same window:

```bash
docker compose exec api python -c "
from poseidon.models.base import SessionLocal
from sqlalchemy import text
s = SessionLocal()
s.execute(text('''
    DELETE FROM ohlcv
    WHERE market='crypto_perp' AND symbol='ETHUSDT' AND interval='4h'
      AND time IN (
          SELECT time FROM ohlcv
          WHERE market='crypto_perp' AND symbol='ETHUSDT' AND interval='4h'
          ORDER BY time DESC
          LIMIT 12 OFFSET 4
      )
'''))
s.commit()
s.close()
print('deleted')
"
```

Refresh `data_coverage_mv` so the audit's tuple iteration sees the
updated row count:

```bash
docker compose exec api python -c "
from poseidon.workers.backfill_tasks import coverage_view_refresh
print(coverage_view_refresh.apply().result)
"
```

Manually fire the gap audit:

```bash
docker compose exec api python -c "
from poseidon.workers.cpu_tasks import data_gap_audit
print(data_gap_audit.apply().result)
"
```

Expected: `{'scanned': N, 'inserted': >=1, 'healed': 0}` (one new gap
row for the (crypto_perp, ETHUSDT, 4h) tuple).

Query `/api/data/gaps` and confirm the gap window is visible:

```bash
curl -s -H "X-API-Key: ${POSEIDON_API_KEY}" \
  'http://localhost:8001/api/data/gaps?market=crypto_perp&symbol=ETHUSDT&interval=4h' \
  | python -m json.tool
```

Expected: at least one DataGapResponse row with `market=crypto_perp`,
`symbol=ETHUSDT`, `interval=4h`, `missing_bars >= 8`, `healed_at=null`.

Restore the deleted rows from /tmp/phase40-deleted-rows.txt (use a
one-shot Python script: `INSERT INTO ohlcv ... ON CONFLICT DO NOTHING`).
Re-fire `coverage_view_refresh.apply()` and `data_gap_audit.apply()`
and confirm the gap is now `healed_at IS NOT NULL` (the heal lifecycle
from D-07).

## 5. Verify CAGG transparency (CAGG-03 reference parity)

Goal: prove that `read_ohlcv(interval='1d')` for `crypto_perp BTCUSDT`
returns the same values as a manual pandas resample of the 4h source
bars.

```bash
docker compose exec api python -c "
import pandas as pd
from datetime import datetime, timezone, timedelta
from poseidon.models.base import SessionLocal
from poseidon.data.storage import read_ohlcv

s = SessionLocal()
end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
start = end - timedelta(days=30)

cagg = read_ohlcv(s, 'BTCUSDT', 'crypto_perp', '1d', start=start, end=end)
raw_4h = read_ohlcv(s, 'BTCUSDT', 'crypto_perp', '4h', start=start, end=end)

reference = raw_4h.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
}).dropna()

aligned = cagg.reindex(reference.index)
diffs = (aligned - reference).abs()
print('CAGG row count:', len(cagg))
print('Reference row count:', len(reference))
print('Max abs diff per column:')
print(diffs.max())
s.close()
"
```

Expected: `Max abs diff per column` shows zero (or numerical-precision
floats < 1e-9) for `open`, `high`, `low`, `close`, `volume` across the
30-day window. Any non-zero diff is a CAGG bug — investigate.

## 6. Verify the Kairos /data-health view (COVERAGE-05)

Goal: prove the new Risk-route page renders all three panels against
the live stormtrooper API.

From your local Mac:

```bash
open http://stormtrooper:5173/data-health
# OR if behind same-origin reverse proxy:
open https://stormtrooper.example/data-health
```

Confirm visually:

- [ ] Page header reads "Data Health" with the "Risk Route · Phase 40"
      eyebrow
- [ ] Freshness panel renders at the top with green/yellow/red totals
      and per-(market, interval) tiles
- [ ] Coverage matrix renders below with one row per (market, symbol,
      interval) and a green/yellow/red health badge
- [ ] Gap heatmap renders at the bottom with cells colored by
      missing_bars
- [ ] No console errors in the browser devtools
- [ ] Polling: the panels refetch every 30s (or whatever
      getMonitoringQueryOptions("overview") sets)

## 7. Pass / fail checklist

Phase 40 is shippable iff ALL of the following are true:

- [ ] Section 1: Uptime Kuma heartbeat baseline is `success`
- [ ] Section 2: Forced stale ts produced a `/fail` ping AND a Telegram
      alert was received
- [ ] Section 3: Killing Beat for >grace_period produced an automatic
      Uptime Kuma "Down" event AND a Telegram alert (no Python error path)
- [ ] Section 4: Manual `data_gap_audit.apply()` after a DELETE produced
      at least one new `data_gaps` row visible via `/api/data/gaps`
- [ ] Section 4: Restoring the deleted bars and re-firing the audit set
      `healed_at` on the same row
- [ ] Section 5: CAGG values match the manual `pandas.resample('1D')`
      reference within float precision over a 30-day window
- [ ] Section 6: Kairos /data-health renders all three panels with live
      data and no console errors

## 8. Cleanup

- Restore any deleted ohlcv rows (Section 4 should already have done
  this, but double-check via `/api/data/coverage`).
- Confirm the freshness watchdog is back to `monitor_pinged: 'success'`.
- Confirm `docker compose ps` shows all five services `Up`.
- Update STATE.md with the smoke run date so the next operator knows
  when this was last validated.
