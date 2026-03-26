# Phase 14: Nunchi Signal Alignment - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Align VotingStrategy with Nunchi auto-research proven logic. Fix all implementation gaps that caused 0% autoresearch pass rate: exit mechanisms, ATR multiplier, BB threshold, composite score formula, SHORT support, and RegimeRouter bear parameter support.

</domain>

<decisions>
## Implementation Decisions

### Exit Mechanisms
- **D-01:** RSI exit added inside VotingStrategy.evaluate() — exit longs at RSI > 69, exit shorts at RSI < 31. RSI period uses the same `rsi_period` param from sub_signals config.
- **D-02:** Signal flip — when opposing ensemble fires (bear fires while in long, or bull fires while in short), reverse position immediately (emit CLOSE then opposite entry).
- **D-03:** Cooldown — 2-bar minimum between exit and re-entry to same direction. New state field `_bars_since_exit` tracks this. Cooldown resets on `reset()`.
- **D-04:** Exit priority order: (1) ATR trailing stop, (2) RSI mean-reversion exit, (3) Signal flip. First matching exit wins per bar.

### ATR Multiplier
- **D-05:** Default ATR multiplier changes from 2.0 to 5.5 (Nunchi proven value — "let winners run").
- **D-06:** PARAM_BOUNDS `atr_multiplier` range changes from (1.5, 3.0) to (3.0, 8.0) to include Nunchi-proven range.

### SHORT Signal Support
- **D-07:** VotingStrategy gets two separate sub_signal lists: `sub_signals` (bull) and `bear_sub_signals` (bear). Each evaluated independently with separate `min_votes` / `bear_min_votes` thresholds.
- **D-08:** Bear sub_signals are inverted conditions: RSI < 50, EMA short < long, negative momentum, MACD histogram < 0. Same indicators, opposite direction.
- **D-09:** VotingStrategy emits SignalAction.SHORT when bear ensemble >= bear_min_votes and not in position. Uses `bear_position_pct` for sizing.
- **D-10:** Trailing stop for shorts: tracks `_position_low_watermark`, exits when `close > low_watermark + atr_multiplier * atr_val` (price rises above stop).
- **D-11:** New VotingStrategy attributes: `_bear_min_votes`, `_bear_position_pct`, `_position_direction` ("long"/"short"/None), `_bars_since_exit`, `_last_vote_direction`.

### BacktestPortfolio Short Fix
- **D-12:** `record_equity_point()` must correctly value short positions: `entry_value - (current_price - entry_price) * quantity` instead of `quantity * current_price`.
- **D-13:** `_execute_short()` already exists in BacktestPortfolio — verify it handles PnL correctly for closing shorts.

### BB Squeeze Threshold
- **D-14:** BB squeeze threshold changes from 0.2 to 0.85 in both `_build_config_from_params()` and `nunchi_crypto_1h.json`. Condition logic (`percentile < threshold`) stays the same — only the threshold value changes.

### Composite Score Formula
- **D-15:** DD penalty changes from `max_drawdown ** 2` to `max(0, max_drawdown - 0.15) * 0.05` — drawdown under 15% incurs no penalty.
- **D-16:** Turnover penalty changes from `max(0, (trade_count - 200) / 1000)` to `max(0, trade_count * avg_trade_value / capital - 500) * 0.001` — based on capital turnover ratio, not raw trade count.
- **D-17:** Hard cutoffs remain: trade_count < 10, max_drawdown > 0.50, total_return < -0.50 → score 0.0.
- **D-18:** Historical experiment data in DB is all 0-scored — no backward compatibility concern with formula change.

### RegimeRouter Bear Parameters
- **D-19:** RegimeRouter overrides 4 attributes instead of 2: `_min_votes`, `_position_pct`, `_bear_min_votes`, `_bear_position_pct`.
- **D-20:** DEFAULT_REGIME_CONFIGS expands with bear params per regime:
  - high_vol: min_votes=5, position_pct=0.05, bear_min_votes=4, bear_position_pct=0.04
  - medium_vol: min_votes=4, position_pct=0.08, bear_min_votes=4, bear_position_pct=0.06
  - low_vol: min_votes=3, position_pct=0.10, bear_min_votes=5, bear_position_pct=0.05
- **D-21:** RegimeSearchPipeline searches 4 params per regime instead of 2 (add bear_min_votes and bear_position_pct).

### Parameter Search Space
- **D-22:** PARAM_BOUNDS adds `bear_min_votes: (3, 6, "int")` and `bear_position_pct: (0.03, 0.12, "float")`.
- **D-23:** `_build_config_from_params()` generates both `sub_signals` (bull) and `bear_sub_signals` (bear) from the same indicator params but with inverted conditions.

### Claude's Discretion
- Exact RSI exit threshold values (69/31 from Nunchi, but searchable if beneficial)
- Whether cooldown applies per-direction (bull cooldown separate from bear cooldown) or globally
- Short position trailing stop watermark initialization details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### VotingStrategy and signals
- `src/poseidon/strategies/voting_strategy.py` — Current evaluate(), exit logic, state management
- `src/poseidon/strategies/base.py` — BaseStrategy ABC, StrategyType enum
- `src/poseidon/signals/schemas.py` — SignalAction (LONG, SHORT, CLOSE, HOLD), Signal dataclass

### Condition evaluation
- `src/poseidon/strategies/dsl/conditions.py` — resolve_column_name(), bollinger_width_percentile evaluator (line 179)
- `src/poseidon/strategies/dsl/executor.py` — evaluate_condition() dispatcher

### Factory and search
- `src/poseidon/backtest/voting_strategy_factory.py` — PARAM_BOUNDS, _build_config_from_params(), from_trial()
- `src/poseidon/backtest/optimizer.py` — BayesianOptimizer
- `src/poseidon/strategies/configs/nunchi_crypto_1h.json` — Baseline config

### Scoring and portfolio
- `src/poseidon/backtest/metrics.py` — compute_composite_score() (lines 89-113)
- `src/poseidon/backtest/portfolio.py` — BacktestPortfolio, record_equity_point(), _execute_short()

### Regime
- `src/poseidon/strategies/regime_router.py` — RegimeRouter, DEFAULT_REGIME_CONFIGS
- `src/poseidon/backtest/regime_search.py` — RegimeSearchPipeline (per-regime Optuna)

### Existing tests
- `tests/test_voting_strategy.py` — VotingStrategy tests
- `tests/test_regime.py` — RegimeRouter tests
- `tests/test_regime_search.py` — RegimeSearch + gate tests
- `tests/test_composite_score.py` — Composite score tests

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SignalAction.SHORT` already exists in signals/schemas.py — no schema change needed
- `BacktestPortfolio._execute_short()` already exists — needs verification but basic structure is there
- `evaluate_condition()` already supports all needed condition types (indicator_above, indicator_below, indicator_comparison, bollinger_width_percentile)
- `get_feature_specs()` on VotingStrategy dynamically generates feature specs — bear signals can reuse the same mechanism

### Established Patterns
- Exit logic lives in VotingStrategy.evaluate(), not BacktestRunner (Phase 10 D-09)
- BacktestRunner is read-only during autoresearch (Phase 12 immutability guard)
- RegimeRouter delegates to VotingStrategy.evaluate() — new behaviors propagate automatically
- TDD pattern: tests first, then implementation

### Integration Points
- RegimeRouter.evaluate() calls self._strategy.evaluate() → new exits + SHORT propagate
- BayesianOptimizer.optimize() uses strategy_factory callback → factory must produce strategies with bear_sub_signals
- RegimeSearchPipeline varies only override params → must expand to 4 params
- compute_composite_score() called by optimizer, search pipeline, and outperformance gate → formula change affects all

</code_context>

<specifics>
## Specific Ideas

- Nunchi ATR 5.5× proven across 103 experiments — "holds winners much longer" (+1.0 Sharpe vs tighter stops)
- Nunchi "removing strength scaling" and "eliminating pyramiding" each independently improved Sharpe — simplicity wins
- BB squeeze at 85th percentile means signal fires most of the time except when bands are extremely wide — acts more as a filter than a trigger

</specifics>

<deferred>
## Deferred Ideas

- RSI exit thresholds (69/31) as searchable params — could add to PARAM_BOUNDS in future phase
- Per-direction cooldown tracking — start with global cooldown, refine later if needed
- Multi-timeframe signal confirmation — Nunchi explicitly found this hurts performance

</deferred>

---

*Phase: 14-nunchi-signal-alignment*
*Context gathered: 2026-03-26*
