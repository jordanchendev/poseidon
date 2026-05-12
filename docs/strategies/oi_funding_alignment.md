# OI / Funding Rate Alignment to 1m Bar Timeline

**Status:** Locked by test contract.

## Context

Aquarium v17.0 validates a 1-minute crypto liquidation hunting maker thesis on
BTC/ETH USDT-margined perps. The strategy consumes three streams that arrive
at three different cadences:

| Stream        | Native cadence       | Source                                 |
| ------------- | -------------------- | -------------------------------------- |
| OHLCV         | 1 minute             | Thalassa `/api/v1/ohlcv`               |
| Open Interest | ~5 minutes (Binance) | Thalassa `/api/v1/open-interest`       |
| Funding rate  | 8 hours (Binance)    | Thalassa `/api/v1/funding-rates`       |

The strategy needs all three aligned onto the 1-minute bar timeline.
**Alignment runs in the Poseidon strategy feature pipeline**
(`poseidon/src/poseidon/data/features/`), NOT in the Thalassa data layer.
Thalassa serves raw cadence data only — it remains a frozen contract
(single CCXT/Binance source, single `ohlcv` table, raw timestamps).
Time-aware alignment is a strategy concern; pushing it into Thalassa would
spread bar-interval-dependent logic across two repos and couple the data
service to consumer-specific assumptions.

This doc specifies the two ffill rules (for OI and funding) and
points at the helper functions and unit tests that lock them.

## Open Interest (5-min cadence) → 1m bars

**Rule:** Last-known forward-fill with a **15-bar staleness cap**.

- **Native cadence:** ~5 minutes. Verifiable on macminim4:
  ```
  curl 'http://192.168.31.241:8001/api/v1/open-interest?symbol=BTCUSDT&start=...'
  ```
- **Forward-fill behavior:** at any 1m bar `T`, the strategy sees the most
  recent OI snapshot with timestamp `<= T`, but only if that snapshot is no
  more than **15 bars (= 15 minutes = 3 OI snapshot cycles)** old. Past the
  cap, the bar's OI feature is `NaN` and the strategy MUST skip zone
  identification on `NaN` OI rows (no signal emitted).
- **Why 15 bars (= 3 cycles):** one cycle of grace would mask a single late
  snapshot; three cycles distinguishes ordinary jitter from a real Thalassa
  fetch outage. Beyond 15 minutes the OI value is no longer representative
  of current positioning and using it would silently leak stale state into
  zone identification.

**Helper:**

```
poseidon/src/poseidon/data/features/open_interest.py
  _align_oi_to_index(
      oi_series: pd.Series,
      target_index: pd.Index,
      method: str = "ffill",
      *,
      staleness_limit_bars: int | None = None,
  ) -> pd.Series
```

The kwarg-only `staleness_limit_bars` parameter is **additive**. Default
`None` preserves the legacy unbounded ffill semantics so all existing 4H
callers (`OIChange`, `OIBuildup`, `OICostBasis` at 4H granularity) remain
unchanged. The 1m strategy pipeline must explicitly opt in by passing
`staleness_limit_bars=15`.

**Caller propagation status:** the internal feature classes
(`OIChange.compute`, `OIBuildup.compute`, `OICostBasis.compute`) currently do
not see a `bar_interval` argument, so they cannot auto-detect 1m vs 4H. A
module-level `TODO` in `open_interest.py` documents that the 1m strategy
pipeline must thread `staleness_limit_bars=15` through (either by adding
`bar_interval` to `BaseFeature.compute` or by passing the value via
`**kwargs`). Pipeline plumbing is follow-up scope; the helper-level kwarg is
already locked.

**Tests:** `poseidon/tests/test_oi_staleness_cap.py`

| Test                                                        | Locks                                                                 |
| ----------------------------------------------------------- | --------------------------------------------------------------------- |
| `test_staleness_limit_bars_kwarg_signature_default_none`    | Signature contract: kwarg exists, default `None` preserved            |
| `test_staleness_limit_default_none_unbounded_ffill`         | Backward-compat: 4H callers unaffected (no false NaN)                 |
| `test_staleness_limit_15_caps_at_nan`                       | Bar at last_snapshot + 16 min becomes `NaN` under cap=15              |
| `test_staleness_limit_15_within_window`                     | Bar at last_snapshot + 15 min still inherits (no false NaN)           |

## Funding Rate (8h cadence) → 1m bars

**Rule:** Last-known forward-fill across the full 8h window. **No
amortization. No interpolation.**

- **Native cadence:** 8 hours (Binance perp standard, settled at 00:00,
  08:00, 16:00 UTC).
- **Forward-fill behavior:** every 1m bar in `[funding_event_t,
  funding_event_{t+1})` sees the same rate — the rate settled at
  `funding_event_t`. That is exactly **480 bars** (8h × 60min) per window.
- **Forbidden alternatives:**
  - **Amortization** (`rate / 480` per bar): turns a discrete settlement
    into a continuous drag; would corrupt funding hurdle accounting and
    silently shrink the per-bar view of crowded positioning.
  - **Linear interpolation** (smooth ramp from `event_t` rate to
    `event_{t+1}` rate): leaks future information into pre-event bars.
- **Boundary rule:** the bar at the exact funding-event timestamp sees the
  **NEW** rate. The minute before sees the OLD rate. This is verified
  empirically — `pd.Series.reindex(method="ffill")` includes the matching
  index value, so a bar stamped exactly `08:00` lines up with the `08:00`
  event and inherits its rate.

**Helper:**

```
poseidon/src/poseidon/data/features/funding_rate.py
  FundingRateDaily.compute(...)  # uses series.reindex(ohlcv.index, method="ffill")
```

`FundingRateDaily.compute` was already correct; only the lock-down tests
below were added.

**Used by:** the `funding_hurdle` filter (rate_per_8h = 0.0001 → annualized
~10.95%, locked by the gate config). Funding hurdle is one of the 4 frozen
gate criteria for verdict evaluation.

**Tests:** `poseidon/tests/test_funding_forward_fill.py`

| Test                                                       | Locks                                                                   |
| ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| `test_funding_held_constant_across_8h_window`              | All 480 bars between two events see the same rate                       |
| `test_funding_no_amortize_no_interpolate`                  | Bar at the 4-hour midpoint sees the FULL rate, not `rate/480` or smoothed |
| `test_funding_boundary_event_timestamp_uses_new_rate`      | Bar at exact 08:00 sees the NEW rate; 07:59 still sees OLD              |

The funding tests exercise `pd.Series.reindex(method="ffill")` directly
rather than instantiating `FundingRateDaily` — the invariant under test is
the pandas primitive that `FundingRateDaily.compute` calls (line 59 of
`funding_rate.py`). Decoupling from `funding_data` DataFrame plumbing keeps
the lock-down test focused on the alignment contract itself.

## Architecture Decision

Alignment runs in the Poseidon strategy feature pipeline
(`poseidon/src/poseidon/data/features/`), NOT in the Thalassa data layer
(`/api/v1/open-interest`, `/api/v1/funding-rates`). Thalassa returns raw
cadence; Poseidon performs the time-aware alignment per-strategy.

Rationale:

1. **Thalassa is a raw data source** — a single CCXT/Binance fetcher with a
   single `ohlcv` table. Adding ffill semantics would couple Thalassa to
   consumer-side bar-interval assumptions.
2. **Alignment is bar-interval dependent.** A 4H consumer wants unbounded
   ffill (4H bars are sparse enough that any cap inside Thalassa would
   misfire); the 1m consumer wants a 15-bar cap. Pushing this into Thalassa
   forces a per-consumer config knob that doesn't belong there.
3. **Time-aware logic in one place.** Keeping ffill rules in
   `poseidon/src/poseidon/data/features/` means future maintainers find
   both the rule and its tests adjacent to the strategy that uses them.

## References

**Source:**

- `poseidon/src/poseidon/data/features/open_interest.py` — `_align_oi_to_index` helper
- `poseidon/src/poseidon/data/features/funding_rate.py` — `FundingRateDaily.compute`

**Tests:**

- `poseidon/tests/test_oi_staleness_cap.py` — 4 tests for the OI cap contract
- `poseidon/tests/test_funding_forward_fill.py` — 3 tests for the funding ffill contract
