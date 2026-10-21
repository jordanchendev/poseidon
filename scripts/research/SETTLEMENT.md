# Settlement ledger (`settle_theme.py`, algorithm `settle-v1`; gap B6)

Scores every completed research revision against what the market did by its own review horizon, so the
research loop has a hit/miss record instead of only theses. Offline, SQLite only, same ledger file that
`research_theme.py` writes. Nothing in Poseidon, Thalassa or the vault is touched.

```bash
cd poseidon/.worktrees/phase101-shadow-fix
export PYTHONPATH=src; S=../../../local_dev/scripts/research/settle_theme.py
../../.venv/bin/python $S settle  --ledger L.sqlite --csv bars.csv [--as-of 2026-10-22T08:00:00Z] [--force] [--benchmark 0050]
../../.venv/bin/python $S flag    --ledger L.sqlite --revision <id> --note "SMA5 crossed SMA20 on 10-23" [--clear]
../../.venv/bin/python $S report  --ledger L.sqlite [--as-of ...] [--out report.md]
../../.venv/bin/python $S pending --ledger L.sqlite [--as-of ...]
```

`--csv` is the same `time,symbol,open,high,low,close,volume` file `research_theme.py --csv` consumes.
`--as-of` defaults to now (UTC); bars after it are ignored, so a settlement can be replayed point-in-time.

## What is read

From each `research_revisions` row with `status = 'completed'`, joined to its `data_manifests` row:

| field | source |
|---|---|
| `stance`, `thesis`, `invalidation_conditions` | `research_json` |
| `decided_as_of` | `research_json.as_of` (= manifest `as_of`) |
| `horizon_time` | leading ISO time of the free-text `research_json.next_review_at`. Date only → 23:59:59Z that day; no offset → UTC; no leading ISO time → never settled (listed by `pending`) |
| subject symbol | manifest request `symbol`; if absent (sector scope) → equal-weight basket of every frozen `ohlcv` symbol except the benchmark |
| benchmark | `benchmark` of a frozen `price_facts` payload if present, else `--benchmark` (default `0050`) |

`catalysts` are not used for settlement (they already constrain `next_review_at` in `research_theme.py`).

## Return math (reused, not copied)

`settle_theme.py` imports `research_theme._event_returns` and `load_bars`; nothing is re-implemented.
It is the price-facts-v2 "since" window with `since = decided_as_of`, closes truncated at the horizon:

- base = last close with bar time <= `decided_as_of`; end = last close with bar time <= `horizon_time`.
- `abs_return_pct = round((end/base - 1) * 100, 2)`; same for the benchmark (`bench_return_pct`).
- `rel_return_pct = round((sym - bench) * 100, 2)` on unrounded fractions, exactly like `stats()`'s
  `rel_ret_*` fields. A test asserts equality with `stats()["rel_ret_since_event_pct"]` on the same window.
- Basket: equal-weight mean of the unrounded member fractions, then the same formula.

Guards (each leaves the row `unsettled` with a `detail`, and `settle` retries it next run):
- no bar strictly after `decided_as_of` up to the horizon;
- the CSV does not reach the horizon: there must be a bar after `horizon_time`, or the last bar <= horizon must be on
  the horizon's UTC date (a weekend horizon therefore waits for the next trading bar);
- the CSV base close differs from the close frozen in the manifest at the same bar time (> 1e-6), e.g. a
  dividend-adjusted feed. If the base bar was not frozen, the row settles but `detail` says "not cross-checked";
- unknown stance, missing symbol or missing bars.

`bars_sha256` = Poseidon `content_sha256` of `{symbol: [[time, close], ...]}` for base..horizon of every symbol used.

## Outcome thresholds (settle-v1)

Band `FLAT_BAND_PP = 1.0` percentage points of **relative** return.

| stance (case-insensitive) | direction | hit | miss | flat |
|---|---|---|---|---|
| long, bullish, buy, overweight, accumulate | +1 | rel > +1.0 | rel < -1.0 | -1.0 <= rel <= +1.0 |
| short, bearish, sell, underweight, avoid, reduce | -1 | rel < -1.0 | rel > +1.0 | -1.0 <= rel <= +1.0 |
| watch, hold, neutral | 0 | \|rel\| <= 1.0 | \|rel\| > 1.0 | never |
| anything else | — | — | — | `unsettled` ("unknown stance") |

All three existing revisions (74f09b1c, 04f5d757, c8f1512d) use `watch`; the directional words are a
forward vocabulary, not observed usage. A neutral stance is scored as "no relative move worth acting on
happened": it is a hit when the subject stayed within ±1 pp of the benchmark and a miss when it moved
out of the band in either direction (the sign is in `rel_return_pct`).

## Table `research_settlements` (created if missing)

`revision_id` (PK), `scope_key`, `symbol` (comma-joined for a basket), `benchmark`, `stance`, `direction`,
`decided_as_of`, `horizon_time`, `horizon_bar_time`, `settle_as_of`, `settled_at` (wall clock),
`settle_algorithm`, `bars_sha256`, `abs_return_pct`, `bench_return_pct`, `rel_return_pct`,
`outcome` ∈ {hit, miss, flat, unsettled, superseded}, `detail` (machine reason), `invalidation_flag`, `note` (human),
`flagged_at`, `outcome_before_supersede`, `superseded_by`, `content_sha256` (= `content_sha256` of every other column;
the two supersede columns are left out of the hash while NULL, so rows written before they existed keep their hash).

A ledger whose table predates the supersede columns is rebuilt in place on first open (SQLite cannot change the outcome
CHECK): rows are copied unchanged.

- Idempotent: `settle` skips a row whose outcome is not `unsettled` unless `--force`. Revisions whose
  horizon is still in the future get no row (`pending` and `report` show them as unsettled / awaiting horizon).
- `flag` is the only way to set `invalidation_flag`; it requires `--note`, works before or after settlement
  (creates an `unsettled` row if needed), accepts dashed or hex ids, and survives `settle --force`.
  `--clear` unsets it and records the note.

## Superseded revisions (design-note C3)

- A completed revision R is **superseded** when another completed revision S in the same ledger, created no earlier than R,
  carries `supersedes_revision_id` naming R in its `research_json` (fallback key `supersedes`; dashed or hex ids). Equal
  `created_at` is allowed because SQLite `CURRENT_TIMESTAMP` has one-second resolution (a strict "later" check was flaky);
  `research_theme.py` already refuses a pointer that is not a completed revision at the time S is written. The first such S in
  ledger order wins; pointers to unknown or earlier-created revisions are ignored.
- `settle` writes the row with `outcome = superseded`, `superseded_by = <successor id>` and `outcome_before_supersede` =
  what settle-v1 measured (`hit|miss|flat|unsettled`), or NULL when R was superseded before its horizon (never measured;
  `detail` says so). It is filled in later runs once the horizon passes. An already-settled row is converted without `--force`.
- Superseded rows are excluded from hit / miss / flat counts, the hit rate (`hit / (hit + miss + flat)`) and the mean
  relative return; `report` shows a `superseded` count per group, `superseded (was hit)` per revision and a `superseded_by` column.
- The successor itself is scored normally against its own horizon.

## Deliberately not done

- No vault projection or Markdown written next to revisions; `report` prints to stdout / `--out`. Whether and
  where settlements are projected is the orchestrator's decision.
- No automatic evaluation of `invalidation_conditions` (they are free text; a human flags).
- No Poseidon table, migration or API; this is a stopgap until Poseidon owns settlements.
- The session fixtures and `gold-set-v0.json` are untouched.

## Blind spots

- The 1 pp band is fixed regardless of horizon length or volatility: for the current one-day horizons it is
  roughly a 1-sigma daily move; for multi-week horizons it is too tight. No volatility scaling.
- Neutral (`watch`) scoring is a convention; a watch revision is really "wait for a trigger", so its hit rate
  says little about research quality. Directional calls are where the ledger becomes informative.
- Supersede is explicit only: a scope revised twice before the same horizon without `supersedes_revision_id` (74f09b1c and
  c8f1512d, both 2330 watch with horizon 2026-10-21T06:40Z) still counts twice. `previous_revision_id` alone does not
  supersede, because a successor may be a follow-up rather than a correction.
- Return is close-to-close on whatever the CSV holds. Unadjusted data across an ex-dividend date understates the
  return (the base-close cross-check catches a *re-adjusted* history, not an unadjusted one).
- Equal-weight basket for sector scopes is an assumption; the revision does not state weights.
- Only the first ISO time in `next_review_at` is used; later text ("fundamental side: December") is ignored.
- `settled_at` is wall clock, so `--force` always changes `content_sha256` even with identical inputs.
- Small sample: three `watch` revisions so far; any hit rate is anecdotal until directional stances accumulate.
- Supersede ordering uses `created_at` at one-second resolution with ties allowed; two revisions written in the same second
  that point at each other would both be marked superseded (research_theme.py makes this practically impossible, since the
  target must already be completed).
