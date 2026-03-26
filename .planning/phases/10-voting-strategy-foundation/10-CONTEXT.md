# Phase 10: Voting Strategy Foundation - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Multi-signal voting strategy producing signals when a configurable majority of sub-signals agree. Includes DSL `vote` condition type, 6 Nunchi-derived signal configs, composite scoring function, ATR trailing stop exit, and fixed position sizing. Scope: crypto_spot 1h only — other markets/timeframes handled by Phase 11 parameter search.

</domain>

<decisions>
## Implementation Decisions

### VotingStrategy Architecture
- **D-01:** VotingStrategy is a new BaseStrategy subclass using composition pattern — wraps N child RuleStrategy instances
- **D-02:** `evaluate()` calls each child strategy, collects votes (triggered = 1 vote), emits signal when >= `min_votes` threshold (default 4/6)
- **D-03:** DSL executor gets new `vote` combinator at same level as `all/any/none` — syntax: `{"vote": {"conditions": [...], "min_votes": 4}}`
- **D-04:** VotingStrategy can be expressed as pure DSL JSON (RuleConfig with vote combinator), maintaining compatibility with existing RuleStrategy pipeline

### Composite Scoring
- **D-05:** New function `compute_composite_score(metrics: dict) -> float` alongside existing `compute_metrics()` — does NOT modify existing function
- **D-06:** Formula: `sharpe * sqrt(min(trades/50, 1.0)) - dd_penalty - turnover_penalty`
- **D-07:** Hard cutoffs: <10 trades -> score 0, >50% drawdown -> score 0, >50% capital loss -> score 0
- **D-08:** This composite score becomes the single optimization metric for Phase 12 autoresearch

### Exit Logic & Position Sizing
- **D-09:** ATR trailing stop is VotingStrategy-level logic (not BacktestPortfolio modification) — tracks position high watermark, emits close signal when price drops > N*ATR
- **D-10:** Fixed 8% position sizing maps to `SizingConfig(mode=FIXED_NOTIONAL, notional_pct=0.08)`
- **D-11:** No RSI exit condition — Nunchi research confirms simplification > complexity. Leave for autoresearch to discover if beneficial.
- **D-12:** BacktestPortfolio requires no changes — exit signals come from strategy layer

### Signal Configuration
- **D-13:** Use Nunchi's original parameters without adjustment: RSI(8), MACD(14,23,9), EMA(7/26), Bollinger(20,2)
- **D-14:** Initial scope: `crypto_spot` + `1h` interval only (Nunchi-validated timeframe)
- **D-15:** TW stocks / US stocks / daily parameters deferred to Phase 11 parameter search
- **D-16:** All 6 signals expressed using existing condition evaluators (indicator_above, indicator_below, indicator_crosses, price_crosses)
- **D-17:** New condition evaluator needed: `bollinger_width_percentile` for Bollinger squeeze signal

### Claude's Discretion
- ATR multiplier value (default 2.0, tunable)
- Cooldown period between signals (if needed)
- Exact turnover penalty formula in composite scoring
- Whether VotingStrategy also subclasses from RuleStrategy or only BaseStrategy

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Strategy Layer
- `src/poseidon/strategies/base.py` — BaseStrategy ABC with evaluate() and validate_config() interface
- `src/poseidon/strategies/rule_strategy.py` — RuleStrategy implementation, how DSL configs become strategies
- `src/poseidon/strategies/dsl/schema.py` — RuleConfig and RuleEntry Pydantic models
- `src/poseidon/strategies/dsl/executor.py` — Recursive tree evaluator with all/any/none combinators (add vote here)
- `src/poseidon/strategies/dsl/conditions.py` — CONDITION_REGISTRY with @register_condition pattern

### Backtest Layer
- `src/poseidon/backtest/metrics.py` — compute_metrics() returns sharpe, drawdown, trade_count (add composite_score here)
- `src/poseidon/backtest/portfolio.py` — BacktestPortfolio with SizingConfig (FIXED_NOTIONAL for 8%)

### Research
- `.planning/research/FEATURES.md` — Nunchi signal definitions with exact parameters
- `.planning/research/PITFALLS.md` — DSL gap analysis, overfit warnings, regime pitfalls
- `.planning/research/ARCHITECTURE.md` — 3-layer pattern, component boundaries

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CONDITION_REGISTRY` + `@register_condition` decorator: New conditions (bollinger_width_percentile) follow this pattern
- `evaluate_condition()` recursive evaluator: Adding `vote` combinator follows exact same pattern as `all/any/none`
- `RuleStrategy.__init__` accepts `config: RuleConfig | dict`: VotingStrategy can follow same pattern
- `compute_metrics()` returns standard dict: `compute_composite_score()` consumes this dict

### Established Patterns
- Strategy → Signal flow: `evaluate(features) -> list[Signal]` — VotingStrategy must follow this
- DSL schema: `RuleConfig(name, symbol, market, interval, rules)` — VotingStrategy JSON wraps multiple rule configs
- Condition evaluators: `(condition: dict, features: pd.DataFrame, row_idx: int) -> bool`
- SizingConfig: `FIXED_NOTIONAL` mode with `notional_pct` — exactly what fixed 8% needs

### Integration Points
- `strategies/__init__.py`: Register VotingStrategy alongside RuleStrategy and ModelStrategy
- `strategies/dsl/executor.py`: Add `vote` handling in evaluate_condition() combinator section
- `strategies/dsl/conditions.py`: Add bollinger_width_percentile evaluator
- `backtest/metrics.py`: Add compute_composite_score() function
- `api/strategies.py`: May need StrategyType.VOTING enum value

</code_context>

<specifics>
## Specific Ideas

- Nunchi's exact 6 signals: dual Momentum (short period > long period returns), EMA crossover (7/26), RSI(8) overbought/oversold, MACD histogram (14,23,9) sign change, Bollinger squeeze (width percentile < threshold)
- Composite scoring is the bridge to Phase 12 autoresearch — it must be the single metric that autoresearch optimizes
- VotingStrategy should be testable on historical crypto 1h data as a standalone backtest before Phase 11

</specifics>

<deferred>
## Deferred Ideas

- Multi-market parameter adaptation (TW stocks, US stocks, daily) — Phase 11
- Optuna parameter search integration — Phase 11
- Automated experiment iteration — Phase 12
- Regime-conditional strategy selection — Phase 13
- RSI exit condition — may emerge from autoresearch in Phase 12
- Adaptive position sizing — explicitly out of scope (Nunchi: fixed beats adaptive)

</deferred>

---

*Phase: 10-voting-strategy-foundation*
*Context gathered: 2026-03-26*
