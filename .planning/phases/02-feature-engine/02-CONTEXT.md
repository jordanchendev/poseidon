# Phase 2: Feature Engine - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Compute technical indicators and derived features on-the-fly from raw OHLCV data so that models and strategies have a standardized feature matrix. The FeatureEngine is the single computation entry point shared by training, prediction, and backtesting — no duplicated logic.

</domain>

<decisions>
## Implementation Decisions

### Library choice
- **No external indicator library** — implement core indicators natively with pandas/numpy
- Rationale: Only ~10 core indicators needed (MA, RSI, MACD, Bollinger, ATR, returns, volatility). These are straightforward to implement. Avoids adding pandas-ta dependency (150+ indicators we don't need). Full control over computation, easier to debug and test.
- pandas and numpy are already installed (Phase 1 dependencies)

### Feature module location
- **`src/poseidon/data/features/`** — under the data module, per design doc layout
- Not a separate top-level module because features are tightly coupled to data loading
- `feature_engine.py` at `src/poseidon/data/feature_engine.py` (single entry point)

### BaseFeature ABC design
- `compute(ohlcv: DataFrame, **params) -> Series | DataFrame`
- Features return Series for single-column output, DataFrame for multi-column (e.g., Bollinger Bands → upper/middle/lower)
- Feature naming convention: `{indicator}_{period}` (e.g., `sma_20`, `rsi_14`, `bb_20`)
- Column names in output follow this convention automatically

### FeatureEngine design
- **Not a singleton** — instantiated per computation call or passed as dependency
- Accepts a `Session` (or creates one) to load OHLCV/fundamentals/sentiment from DB
- `compute(symbol, market, interval, start, end, features) -> DataFrame`
- Returns wide DataFrame: original OHLCV columns + computed feature columns
- NaN handling: leave NaN from rolling windows (caller decides to dropna or not)

### Feature registry
- Simple dict-based registry with decorator `@register_feature`
- Auto-discovery: all features in `features/` submodules are imported in `__init__.py`
- New feature = create class extending BaseFeature + decorate with `@register_feature`
- No YAML config — registry is code-based for simplicity

### Dependency resolution
- **Not needed for Phase 2** — all core indicators compute directly from OHLCV
- Bollinger Bands internally computes SMA (doesn't depend on a separate SMA feature)
- Cross-market and fundamental features are deferred to when Phase 3+ actually needs them
- Keep the design extensible but don't build the DAG resolver until required (YAGNI)

### Feature categories for Phase 2
1. **Technical indicators**: SMA, EMA, RSI, MACD (line + signal + histogram), Bollinger Bands (upper/middle/lower), ATR
2. **Returns**: daily return, cumulative return, log return
3. **Volatility**: standard (close-to-close), Parkinson (high-low range), Garman-Klass (OHLC)

### Deferred to later phases
- Fundamental features (PE ratio, EPS growth) — needs actual fundamentals data first
- Sentiment features (aggregated score, by-source) — needs Thalassa integration
- Cross-market features (BTC correlation, TWD impact) — needs multi-symbol loading pattern
- Feature caching — on-the-fly computation sufficient at personal scale

### Claude's Discretion
- Exact parameter defaults for each indicator (MA periods, RSI period, etc.)
- Internal implementation details of each indicator
- Test strategy and test data generation
- Error handling for edge cases (insufficient data, all-zero volume, etc.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture & feature engine
- `../docs/poseidon-design.md` (lines 184-233) — Feature Engine design: categories, BaseFeature interface, FeatureEngine interface, future caching strategy
- `../docs/poseidon-design.md` (lines 134-143) — data/features/ directory structure

### Phase 1 integration points
- `src/poseidon/data/storage.py` — read_ohlcv(session, symbol, market, interval, start, end) -> DataFrame[time, open, high, low, close, volume]
- `src/poseidon/data/storage.py` — read_fundamentals(), read_sentiment() (for future feature types)
- `src/poseidon/models/base.py` — SessionLocal, get_db() patterns
- `src/poseidon/core/config.py` — Settings class

### Requirements
- `poseidon/.planning/REQUIREMENTS.md` — FEAT-01 (technical indicators), FEAT-02 (extensible BaseFeature ABC)

</canonical_refs>

<code_context>
## Existing Code Insights

### Phase 1 Integration Surface
- `read_ohlcv()` returns DataFrame with columns: time, open, high, low, close, volume (all float except time which is datetime UTC)
- Data is sorted by time ascending
- Empty DataFrame returned when no data
- SessionLocal pattern for DB access in workers/tasks
- get_db() generator for FastAPI dependency injection

### Key Constraints
- No new PyPI dependencies needed (pandas + numpy sufficient)
- Features computed on-the-fly, not stored
- Training, prediction, and backtesting share same FeatureEngine
- All timestamps are UTC

</code_context>

<specifics>
## Specific Ideas

- Keep feature implementations simple and readable — one class per indicator file is overkill; group related indicators (e.g., all MAs in one file, all volatility measures in one file)
- The `compute()` method on FeatureEngine should accept a list of feature specs like `[("sma", {"period": 20}), ("rsi", {"period": 14})]` rather than just feature names, since the same indicator can have different parameters
- Consider a `compute_default()` convenience method that computes a standard set of features with default params

</specifics>

<deferred>
## Deferred Ideas

- Feature importance analysis (Phase 3 — when models consume features)
- Feature versioning (not needed until feature computation logic changes)
- Parallel feature computation (premature optimization at personal scale)

</deferred>

---

*Phase: 02-feature-engine*
*Context gathered: 2026-03-21*
