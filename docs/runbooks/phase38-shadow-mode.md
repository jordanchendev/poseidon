# Phase 38 Shadow-Mode Runbook — 48h `INGEST_CURSOR_MODE=cursor` Smoke Test

Operator-facing runbook for the 48-hour shadow-mode soak test that flips the
Phase 38 cursor-based self-healing ingest path on in production. Satisfies
Phase 38 success criterion #5 and `DATA-FOUND-03`.

Default production state is `INGEST_CURSOR_MODE=legacy` (byte-identical to
v7.0). This runbook is the only prescribed path for flipping it to `cursor`
on stormtrooper.

## 0. Preconditions

Before starting the window, confirm ALL of the following on stormtrooper:

- [ ] Plans `38-01`, `38-02`, and `38-03` (Task 1 + Task 2) merged into
      `poseidon` `main` and pulled onto `~/Projects/poseidon`.
- [ ] `alembic upgrade head` reports `020 (head)`.
- [ ] `docker compose ps` shows `api`, `cpu-worker`, `gpu-worker`, `beat`
      healthy with `INGEST_CURSOR_MODE` either absent or set to `legacy` in
      `.env`.
- [ ] Phase 38 test suite green via
      `docker compose run --rm -e PYTHONPATH=/app/src qlib-research uv run --with pytest --with pytest-asyncio --with fakeredis python -m pytest tests/unit/test_ingest_cursor.py tests/unit/test_backfill_chunk_idempotent.py tests/unit/test_fetch_market_data_flag.py tests/unit/test_migration_020.py tests/integration/test_cursor_self_heal.py tests/integration/test_backfill_kill_resume.py -x`.
- [ ] On-call rotation aware of the 48h window and the rollback command in
      section 6.

## 1. Baseline Capture (T-2h → T0)

Capture 2 hours of baseline latency on the two perp schedules that MUST
remain within tolerance after the flip:

- `poseidon.workers.cpu_tasks.perp_rebalance` (Beat schedule
  `perp-rebalance-4h`)
- `poseidon.workers.cpu_tasks.perp_liquidation_monitor` (Beat schedule
  `perp-liquidation-monitor`, every minute)

```bash
ssh stormtrooper
cd ~/Projects/poseidon

# Latency via docker logs (works regardless of Flower availability)
docker compose logs --since 2h cpu-worker \
  | grep -E "perp_rebalance|perp_liquidation_monitor" \
  | grep -E "succeeded in|duration" \
  | tee /tmp/phase38-baseline-latency.log

# Row counts — must be stable across runs
docker compose exec -T api python -c "
from poseidon.models.base import SessionLocal
from sqlalchemy import text
s = SessionLocal()
print('ingest_state rows:', s.execute(text('SELECT COUNT(*) FROM ingest_state')).scalar())
for row in s.execute(text('SELECT market, interval, COUNT(*) FROM ohlcv GROUP BY market, interval ORDER BY 1,2')):
    print(row)
s.close()
" | tee /tmp/phase38-baseline-counts.txt
```

Record `p50` / `p95` of both tasks over the 2h window. Save
`/tmp/phase38-baseline-latency.log` and `/tmp/phase38-baseline-counts.txt`
off-box before proceeding.

## 2. Flip Flag to `cursor` (T0)

```bash
ssh stormtrooper
cd ~/Projects/poseidon

# Append the flag (or edit if already present). Case-sensitive; must be
# exactly 'cursor' — any other value raises pydantic.ValidationError at
# worker bootstrap and the container will refuse to start.
echo "INGEST_CURSOR_MODE=cursor" >> .env

# Rebuild-free flag flip: force-recreate the workers + beat to pick up the
# new env. api does not read INGEST_CURSOR_MODE so it can stay up.
docker compose up -d --force-recreate cpu-worker gpu-worker beat

# Tail cursor-mode bootstrap — expect "cursor-mode fetch" log lines
docker compose logs -f --since 1m cpu-worker | grep -iE "cursor|ingest_state"
```

Verify the flag is live inside the container:

```bash
docker compose exec cpu-worker python -c "
from poseidon.core.config import settings
print('ingest_cursor_mode =', settings.ingest_cursor_mode)
"
# Expected: ingest_cursor_mode = cursor
```

**T0** is now stamped. The 48h observation window runs until **T0 + 48h**.

## 3. Observation Window (48h)

Every **6h** for the next 48h, capture the following and append to
`/tmp/phase38-window.log`:

```bash
ssh stormtrooper
cd ~/Projects/poseidon

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

  # 1) Celery latency delta vs baseline
  docker compose logs --since 6h cpu-worker \
    | grep -E "perp_rebalance|perp_liquidation_monitor" \
    | grep -E "succeeded in|duration"

  # 2) ingest_state growth + staleness
  docker compose exec -T api python -c "
from poseidon.models.base import SessionLocal
from sqlalchemy import text
s = SessionLocal()
print('ingest_state rows:', s.execute(text('SELECT COUNT(*) FROM ingest_state')).scalar())
for row in s.execute(text('SELECT symbol, market, interval, last_successful_ts, last_attempt_ts FROM ingest_state ORDER BY last_attempt_ts DESC NULLS LAST LIMIT 10')):
    print(row)
s.close()
"

  # 3) Error log scan — cursor path should produce ZERO ERROR lines
  docker compose logs --since 6h cpu-worker \
    | grep -iE "ERROR.*(cursor|ingest_state|UniqueViolation)" || echo "no cursor errors"

  # 4) Gap audit on ohlcv hypertable (delta vs baseline)
  docker compose exec -T api python -c "
from poseidon.models.base import SessionLocal
from sqlalchemy import text
s = SessionLocal()
for row in s.execute(text('SELECT symbol, market, interval, MIN(time), MAX(time), COUNT(*) FROM ohlcv GROUP BY 1,2,3 ORDER BY 1,2,3')):
    print(row)
s.close()
"
} >> /tmp/phase38-window.log 2>&1
```

Between 6h checkpoints, on-call should watch for alerts on any of:

- `perp_rebalance` latency p95 > baseline × 1.2
- `perp_liquidation_monitor` latency p95 > baseline × 1.2
- ERROR lines in `cpu-worker` / `gpu-worker` logs mentioning `cursor`,
  `ingest_state`, or `UniqueViolation`
- New gaps in the `ohlcv` hypertable vs baseline

## 4. Success Criteria (T0 + 48h)

All of the following MUST be TRUE before declaring the window passed:

1. `perp_rebalance` p95 within ±20% of baseline.
2. `perp_liquidation_monitor` p95 within ±20% of baseline.
3. **Zero** ERROR log lines related to `cursor`, `ingest_state`, or
   `UniqueViolation` over the entire 48h window.
4. `ingest_state` table populated with a row for every live
   `(symbol, market, interval)` tuple that `poseidon.data.symbols`
   configures.
5. `last_attempt_ts` within 2× the shortest Beat interval for every tuple
   (e.g. crypto 1h should have updated within ~2h).
6. Zero new gaps in the `ohlcv` hypertable vs the baseline gap audit from
   section 1.

If any criterion fails, proceed immediately to section 6 rollback.

## 5. Sign-off Checklist

- [ ] Baseline captured and archived off stormtrooper.
- [ ] Flag flip executed at documented T0 timestamp.
- [ ] 8 observation checkpoints captured (every 6h, T0..T0+48h).
- [ ] `perp_rebalance` p95 Δ ≤ 20%.
- [ ] `perp_liquidation_monitor` p95 Δ ≤ 20%.
- [ ] Zero cursor-path ERROR lines.
- [ ] `ingest_state` covers all live tuples.
- [ ] Zero new `ohlcv` gaps.
- [ ] Decision recorded in `.planning/STATE.md` (promote / rollback).

## 6. Rollback Procedure

If ANY success criterion fails, or at any point during the window on-call
judgment says rollback, execute immediately:

```bash
ssh stormtrooper
cd ~/Projects/poseidon

# Remove the cursor flag (or set explicitly to legacy)
sed -i '/^INGEST_CURSOR_MODE=cursor$/d' .env
echo "INGEST_CURSOR_MODE=legacy" >> .env

# Force workers + beat to re-read .env
docker compose up -d --force-recreate cpu-worker gpu-worker beat

# Verify
docker compose exec cpu-worker python -c "
from poseidon.core.config import settings
print('ingest_cursor_mode =', settings.ingest_cursor_mode)
"
# Expected: ingest_cursor_mode = legacy
```

Existing `ingest_state` rows remain in place — the legacy `fetch_market_data`
branch does not touch them, so they are harmless and retained for the next
attempt at the window. Do NOT drop `ingest_state` on rollback.

Then file an incident note in `.planning/STATE.md` with:

- T0 timestamp of the failed window
- Which criterion failed (exact numbers)
- Root cause investigation plan (usually a Phase 38 code fix before
  re-attempting)
