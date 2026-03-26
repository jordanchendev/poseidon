# Phase 10: Voting Strategy Foundation - Research

**Researched:** 2026-03-26
**Domain:** Rule-based voting strategy, DSL extension, composite scoring, trailing stop exits
**Confidence:** HIGH

## Summary

Phase 10 builds the VotingStrategy class and supporting infrastructure on top of a well-established codebase. The existing `BaseStrategy` ABC, `RuleStrategy`, DSL condition engine (`all/any/none` combinators), `FeatureEngine` (with all required technical indicators), and `BacktestRunner` (bar-by-bar expanding window) provide a solid foundation. Every component this phase needs to create follows established patterns already in the codebase.

The core work is: (1) a new `VotingStrategy` class that wraps N child condition evaluations and counts votes, (2) a `vote` combinator in the DSL executor alongside existing `all/any/none`, (3) a `bollinger_width_percentile` condition evaluator, (4) six JSON signal configs using Nunchi parameters, (5) `compute_composite_score()` function in metrics.py, and (6) ATR trailing stop logic in VotingStrategy.

All required indicators (EMA, RSI, MACD, Bollinger Bands, ATR, returns) already exist in `FeatureEngine`. The `SizingConfig(mode=FIXED_NOTIONAL, notional_pct=0.08)` exactly maps to the 8% fixed position sizing requirement. `BacktestPortfolio` needs zero modifications. The main risk is getting the vote combinator semantics right and ensuring the ATR trailing stop integrates cleanly with the strategy-level signal flow (not portfolio-level).

**Primary recommendation:** Follow existing patterns exactly -- VotingStrategy subclasses BaseStrategy, vote combinator mirrors all/any/none in executor.py, bollinger_width_percentile follows @register_condition pattern, composite scoring is a pure function next to compute_metrics().

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** VotingStrategy is a new BaseStrategy subclass using composition pattern -- wraps N child RuleStrategy instances
- **D-02:** `evaluate()` calls each child strategy, collects votes (triggered = 1 vote), emits signal when >= `min_votes` threshold (default 4/6)
- **D-03:** DSL executor gets new `vote` combinator at same level as `all/any/none` -- syntax: `{"vote": {"conditions": [...], "min_votes": 4}}`
- **D-04:** VotingStrategy can be expressed as pure DSL JSON (RuleConfig with vote combinator), maintaining compatibility with existing RuleStrategy pipeline
- **D-05:** New function `compute_composite_score(metrics: dict) -> float` alongside existing `compute_metrics()` -- does NOT modify existing function
- **D-06:** Formula: `sharpe * sqrt(min(trades/50, 1.0)) - dd_penalty - turnover_penalty`
- **D-07:** Hard cutoffs: <10 trades -> score 0, >50% drawdown -> score 0, >50% capital loss -> score 0
- **D-08:** This composite score becomes the single optimization metric for Phase 12 autoresearch
- **D-09:** ATR trailing stop is VotingStrategy-level logic (not BacktestPortfolio modification) -- tracks position high watermark, emits close signal when price drops > N*ATR
- **D-10:** Fixed 8% position sizing maps to `SizingConfig(mode=FIXED_NOTIONAL, notional_pct=0.08)`
- **D-11:** No RSI exit condition -- leave for autoresearch to discover if beneficial
- **D-12:** BacktestPortfolio requires no changes -- exit signals come from strategy layer
- **D-13:** Use Nunchi's original parameters without adjustment: RSI(8), MACD(14,23,9), EMA(7/26), Bollinger(20,2)
- **D-14:** Initial scope: `crypto_spot` + `1h` interval only
- **D-15:** TW stocks / US stocks / daily parameters deferred to Phase 11
- **D-16:** All 6 signals expressed using existing condition evaluators (indicator_above, indicator_below, indicator_crosses, price_crosses)
- **D-17:** New condition evaluator needed: `bollinger_width_percentile` for Bollinger squeeze signal

### Claude's Discretion
- ATR multiplier value (default 2.0, tunable)
- Cooldown period between signals (if needed)
- Exact turnover penalty formula in composite scoring
- Whether VotingStrategy also subclasses from RuleStrategy or only BaseStrategy

### Deferred Ideas (OUT OF SCOPE)
- Multi-market parameter adaptation (TW stocks, US stocks, daily) -- Phase 11
- Optuna parameter search integration -- Phase 11
- Automated experiment iteration -- Phase 12
- Regime-conditional strategy selection -- Phase 13
- RSI exit condition -- may emerge from autoresearch in Phase 12
- Adaptive position sizing -- explicitly out of scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VOTE-01 | VotingStrategy extends BaseStrategy, accepts N child RuleStrategy instances, emits signal when >= min_votes threshold (default 4/6) | BaseStrategy ABC at `strategies/base.py` with evaluate() -> list[Signal] interface. RuleStrategy pattern at `strategies/rule_strategy.py` shows how to subclass. Composition pattern: VotingStrategy holds list of child condition dicts, evaluates each, counts True results. |
| VOTE-02 | DSL condition engine supports new `vote` condition type with `min_votes` parameter | `executor.py` already handles all/any/none combinators with identical recursive pattern. Adding `vote` combinator: check `"vote" in condition`, extract `conditions` list and `min_votes`, count True evaluations, return `count >= min_votes`. |
| VOTE-03 | Six Nunchi-derived signal strategies as RuleStrategy JSON configs | All required indicators exist in FeatureEngine: EMA (7, 26), RSI (8 via period param), MACD (14,23,9 via fast/slow/signal params), Bollinger (20, 2). Need `bollinger_width_percentile` condition and momentum return comparison conditions. Existing `indicator_above`, `indicator_below`, `indicator_crosses` cover most signals. |
| VOTE-04 | Composite scoring with hard cutoffs | `compute_metrics()` at `backtest/metrics.py` already returns sharpe_ratio, max_drawdown, trade_count, total_return. New `compute_composite_score(metrics)` consumes this dict. Formula and cutoffs are fully specified in decisions D-05 through D-07. |
| VOTE-05 | ATR-based trailing stop exit logic | ATR feature exists (`atr_{period}` column). VotingStrategy tracks position state internally (high watermark per position). On each evaluate() call, checks if current price dropped > N*ATR below high watermark, emits CLOSE signal. This is strategy-level logic, not portfolio modification (D-09, D-12). |
| VOTE-06 | Fixed position sizing (default 8%) | `SizingConfig(mode=FIXED_NOTIONAL, notional_pct=0.08)` already supported by BacktestPortfolio. BacktestRunner._compute_sizing() handles FIXED_NOTIONAL mode. Zero code changes needed -- just pass correct SizingConfig when creating BacktestRunner. |
</phase_requirements>

## Architecture Patterns

### Recommended Project Structure

New and modified files for this phase:

```
src/poseidon/
  strategies/
    voting_strategy.py        # NEW: VotingStrategy class (VOTE-01, VOTE-05)
    dsl/
      executor.py             # MODIFY: add vote combinator (VOTE-02)
      conditions.py           # MODIFY: add bollinger_width_percentile (VOTE-03)
    configs/                   # NEW directory
      nunchi_crypto_1h.json   # NEW: 6-signal voting config for crypto_spot 1h (VOTE-03)
    __init__.py               # MODIFY: export VotingStrategy
    base.py                   # MODIFY: add StrategyType.VOTING enum
  backtest/
    metrics.py                # MODIFY: add compute_composite_score() (VOTE-04)
tests/
  test_voting_strategy.py     # NEW: VotingStrategy tests
  test_composite_score.py     # NEW: composite scoring tests
  test_vote_combinator.py     # NEW: DSL vote combinator tests
```

### Pattern 1: Vote Combinator in DSL Executor

**What:** Add `vote` as a new combinator alongside `all/any/none` in `evaluate_condition()`.
**When to use:** When the DSL tree contains a `{"vote": {"conditions": [...], "min_votes": N}}` node.

```python
# In executor.py, after the "none" combinator check:
if "vote" in condition:
    vote_config = condition["vote"]
    sub_conditions = vote_config["conditions"]
    min_votes = vote_config.get("min_votes", 4)
    vote_count = sum(
        1 for c in sub_conditions
        if evaluate_condition(c, features, row_idx, max_depth=max_depth, _current_depth=next_depth)
    )
    return vote_count >= min_votes
```

Key difference from `all`: `vote` does NOT short-circuit -- it must evaluate ALL conditions to get the count. This is intentional. The `all()` generator short-circuits on first False; vote needs the full count.

### Pattern 2: VotingStrategy with ATR Trailing Stop

**What:** VotingStrategy subclasses BaseStrategy, holds child condition configs, tracks position state for trailing stop.
**Architecture decision:** VotingStrategy should subclass BaseStrategy directly (not RuleStrategy). Rationale: RuleStrategy has its own evaluate() logic that iterates config.rules. VotingStrategy has fundamentally different evaluation logic (vote counting + trailing stop state). Composition over inheritance -- VotingStrategy uses RuleConfig for serialization but implements its own evaluate().

```python
class VotingStrategy(BaseStrategy):
    strategy_type = StrategyType.VOTING

    def __init__(self, *, config: dict, atr_multiplier: float = 2.0, ...):
        self._sub_conditions: list[dict] = config["sub_signals"]
        self._min_votes: int = config.get("min_votes", 4)
        self._atr_multiplier = atr_multiplier
        self._position_high_watermark: float | None = None
        self._in_position: bool = False

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        row_idx = len(features) - 1
        signals = []

        # Check trailing stop first (if in position)
        if self._in_position:
            close = features.iloc[row_idx]["close"]
            self._position_high_watermark = max(self._position_high_watermark, close)
            atr = features.iloc[row_idx][f"atr_{self._atr_period}"]
            if close < self._position_high_watermark - self._atr_multiplier * atr:
                signals.append(self._make_signal(SignalAction.CLOSE, ...))
                self._in_position = False
                self._position_high_watermark = None
                return signals

        # Count votes from sub-signals
        vote_count = sum(
            1 for cond in self._sub_conditions
            if evaluate_condition(cond, features, row_idx)
        )

        if vote_count >= self._min_votes and not self._in_position:
            signals.append(self._make_signal(SignalAction.LONG, confidence=vote_count/len(self._sub_conditions), ...))
            self._in_position = True
            self._position_high_watermark = features.iloc[row_idx]["close"]

        return signals
```

**Important:** VotingStrategy is stateful (tracks position for trailing stop). This means:
- During backtest bar-by-bar loop, state accumulates naturally
- Strategy must be re-instantiated between separate backtest runs
- The BacktestRunner already creates strategy once and iterates, so this works

### Pattern 3: Bollinger Width Percentile Condition

**What:** New condition evaluator that computes rolling percentile rank of Bollinger Band width.
**Column dependency:** Requires `bb_upper_{period}` and `bb_lower_{period}` columns from BollingerBands feature.

```python
@register_condition("bollinger_width_percentile")
def eval_bollinger_width_percentile(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    params = condition.get("params", {})
    period = params.get("period", 20)
    lookback = params.get("lookback", 168)  # 1 week of hourly bars
    threshold = condition.get("threshold", 0.2)  # below 20th percentile = squeeze

    upper_col = f"bb_upper_{period}"
    lower_col = f"bb_lower_{period}"

    # Compute width series up to current row (no look-ahead)
    start = max(0, row_idx - lookback + 1)
    widths = features[upper_col].iloc[start:row_idx + 1] - features[lower_col].iloc[start:row_idx + 1]

    if len(widths) < 2:
        return False

    current_width = widths.iloc[-1]
    percentile = (widths < current_width).sum() / len(widths)
    return percentile < threshold
```

### Pattern 4: Dual Momentum Signal via Existing Conditions

**What:** Dual momentum compares short-period vs long-period returns. Nunchi uses 12h and 6h return.
**Implementation:** This needs a new condition evaluator or creative use of existing ones.

The existing `price_change_pct` condition checks `return_1d` column. For hourly data, we need `return_Nh` (N-bar returns). The `CumulativeReturn` feature computes `cum_return_{period}d`. We can use this with `indicator_above` to compare short vs long period returns.

Recommended approach: Add `cum_return` to feature specs for the voting strategy (periods 6 and 12 for hourly). Then use `indicator_above` on `cum_return_6d` with threshold 0 (positive momentum) as one sub-signal, and similarly for `cum_return_12d`.

Actually, Nunchi's dual momentum checks `short_return > long_return`. This needs a new condition evaluator:

```python
@register_condition("indicator_comparison")
def eval_indicator_comparison(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Compare two indicator values: indicator_a > indicator_b."""
    params = condition.get("params", {})
    indicator_a = condition.get("indicator_a", "")
    indicator_b = condition.get("indicator_b", "")
    col_a = resolve_column_name(indicator_a, {"period": params.get("period_a")})
    col_b = resolve_column_name(indicator_b, {"period": params.get("period_b")})
    direction = condition.get("direction", "above")

    val_a = float(features.iloc[row_idx][col_a])
    val_b = float(features.iloc[row_idx][col_b])

    if direction == "above":
        return val_a > val_b
    return val_a < val_b
```

### Anti-Patterns to Avoid

- **Modifying BacktestPortfolio for trailing stops:** D-09 and D-12 are explicit -- trailing stop logic lives in VotingStrategy, not portfolio. The strategy emits CLOSE signals; the portfolio just executes them.
- **Using `all` combinator instead of `vote`:** `all` short-circuits. Vote needs full count. They are semantically different.
- **Computing BB width percentile on full series:** Must use expanding/rolling window up to current row only. See Pitfall 2 in research.
- **Hardcoding Nunchi parameters:** All parameters should come from JSON config, not Python code. This enables Phase 11 parameter search.
- **Making VotingStrategy subclass RuleStrategy:** VotingStrategy has different evaluation semantics (vote counting, trailing stop state). Subclassing RuleStrategy would inherit unwanted rule-iteration logic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Position sizing | Custom sizing logic | `SizingConfig(mode=FIXED_NOTIONAL, notional_pct=0.08)` | Already implemented and tested in BacktestPortfolio |
| Indicator computation | Manual RSI/MACD/BB calculation | FeatureEngine with existing feature classes | Tested, no look-ahead bias, handles edge cases |
| Signal schema | Custom signal format | `Signal` from `signals/schemas.py` | Standardized format consumed by risk engine and portfolio |
| Condition evaluation | Custom condition tree parser | Existing `evaluate_condition()` recursive evaluator | Battle-tested with all/any/none, max depth protection |
| Backtest loop | Custom bar iteration | `BacktestRunner.run()` with expanding window | Handles warmup, equity tracking, risk checks |

## Common Pitfalls

### Pitfall 1: Double-Counting Votes
**What goes wrong:** A sub-signal that fires on both BUY and SELL sides counts as 2 votes.
**Why it happens:** Sub-signals designed as full strategies (with both long and short rules) instead of simple boolean directional conditions.
**How to avoid:** Each sub-signal in the voting config is a single directional condition (e.g., "RSI below 30 = bullish vote"), not a complete strategy with multiple rules. One condition = one vote max.
**Warning signs:** Vote count exceeding the number of sub-signals.

### Pitfall 2: Look-Ahead in BB Width Percentile
**What goes wrong:** Percentile rank computed over the entire series instead of expanding window.
**Why it happens:** Natural pandas pattern `series.rank(pct=True)` operates on full series.
**How to avoid:** Use rolling window computation bounded by `row_idx`. The condition evaluator receives `row_idx` and must not access `features.iloc[row_idx+1:]`.
**Warning signs:** Suspiciously good backtest results that degrade in live.

### Pitfall 3: VotingStrategy State Not Reset Between Backtests
**What goes wrong:** Trailing stop state (`_in_position`, `_position_high_watermark`) carries over between separate backtest runs.
**Why it happens:** Reusing the same VotingStrategy instance across multiple BacktestRunner.run() calls.
**How to avoid:** Either instantiate a fresh VotingStrategy per backtest run, or add a `reset()` method called before each run.
**Warning signs:** First bar of a new backtest immediately triggers a trailing stop exit.

### Pitfall 4: Feature Spec Mismatch
**What goes wrong:** VotingStrategy requires features (EMA 7, EMA 26, RSI 8, MACD 14/23/9, BB 20, ATR 14, cum_return 6, cum_return 12) that are not in DEFAULT_FEATURES.
**Why it happens:** DEFAULT_FEATURES has EMA 12/26, RSI 14, standard MACD 12/26/9 -- not the Nunchi-optimized parameters.
**How to avoid:** VotingStrategy must specify its own feature_specs when used with FeatureEngine. Either pass custom specs to BacktestRunner, or the strategy config includes required feature specs.
**Warning signs:** KeyError on column access during condition evaluation (e.g., `ema_7` not found).

### Pitfall 5: MACD Column Name Resolution
**What goes wrong:** MACD with custom parameters (14/23/9) still produces columns named `macd_line`, `macd_signal`, `macd_histogram` (no period in name), but default MACD (12/26/9) also produces the same column names. If both are computed, one overwrites the other.
**Why it happens:** MACD feature class uses fixed column names regardless of parameters.
**How to avoid:** For Phase 10, only use MACD(14,23,9) -- do not include default MACD in feature specs. If both are needed later, MACD feature class needs parameterized column names (Phase 11 concern).
**Warning signs:** MACD values don't match expected output for given parameters.

## Nunchi Signal Definitions (Exact Specifications)

These are the 6 signals to implement as JSON configs:

| # | Signal | Condition Type | Parameters | Bullish When |
|---|--------|---------------|------------|--------------|
| 1 | Short Momentum | indicator_above | cum_return_6d > 0 | 6-bar return positive |
| 2 | Long Momentum | indicator_above | cum_return_12d > 0 | 12-bar return positive |
| 3 | EMA Crossover | indicator_crosses | ema_7 crosses above ema_26 | Fast EMA above slow EMA |
| 4 | RSI(8) | indicator_above | rsi_8 > 50 | RSI above midline |
| 5 | MACD Histogram | indicator_above | macd_histogram > 0 | MACD histogram positive |
| 6 | Bollinger Squeeze | bollinger_width_percentile | bb(20,2), lookback 168, threshold 0.2 | Width below 20th percentile (squeeze) |

**Required FeatureEngine specs for VotingStrategy:**
```python
VOTING_FEATURES: list[tuple[str, dict]] = [
    ("ema", {"period": 7}),
    ("ema", {"period": 26}),
    ("rsi", {"period": 8}),
    ("macd", {"fast_period": 14, "slow_period": 23, "signal_period": 9}),
    ("bollinger", {"period": 20, "num_std": 2.0}),
    ("atr", {"period": 14}),
    ("cum_return", {"period": 6}),
    ("cum_return", {"period": 12}),
    ("returns", {}),
]
```

**Note on Signal 6 (Bollinger Squeeze):** This is a non-directional quality gate. It votes "conditions are favorable for a breakout" regardless of direction. It amplifies existing directional consensus from the other 5 signals.

## Composite Scoring Formula (Detailed)

```python
def compute_composite_score(metrics: dict) -> float:
    """Compute composite optimization score from backtest metrics.

    Formula: sharpe * sqrt(min(trades/50, 1.0)) - dd_penalty - turnover_penalty

    Hard cutoffs (return 0.0 immediately):
    - trade_count < 10
    - max_drawdown > 0.50
    - total_return < -0.50 (>50% capital loss)
    """
    trade_count = metrics.get("trade_count", 0)
    max_drawdown = metrics.get("max_drawdown", 1.0)
    total_return = metrics.get("total_return", -1.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)

    # Hard cutoffs
    if trade_count < 10:
        return 0.0
    if max_drawdown > 0.50:
        return 0.0
    if total_return < -0.50:
        return 0.0

    # Trade count factor: penalizes low trade count
    trade_factor = math.sqrt(min(trade_count / 50.0, 1.0))

    # Drawdown penalty (Claude's discretion: quadratic penalty)
    dd_penalty = max_drawdown ** 2

    # Turnover penalty (Claude's discretion: simple linear)
    # Approximate turnover as 2 * trade_count / bars_count
    # For now, use a mild penalty based on excessive trading
    turnover_penalty = max(0, (trade_count - 200) / 1000.0)  # penalty kicks in above 200 trades

    return sharpe * trade_factor - dd_penalty - turnover_penalty
```

**Design decisions (Claude's discretion):**
- **dd_penalty:** Quadratic (`max_drawdown ** 2`) so 20% DD costs 0.04 but 40% DD costs 0.16. Gentle at moderate levels, harsh at extremes.
- **turnover_penalty:** Linear penalty starting above 200 trades. A strategy trading every bar on hourly data (~8760 trades/year) would get -8.56 penalty, effectively zeroing it. A strategy with 100 trades gets 0 penalty. This discourages churning without penalizing normal trading.

## ATR Trailing Stop Design

**Mechanics:**
1. When VotingStrategy opens a position (vote count >= min_votes), record entry price as initial high watermark.
2. Each subsequent bar, update high watermark to `max(hwm, close)`.
3. If `close < hwm - atr_multiplier * atr_value`, emit CLOSE signal.
4. Reset state after close.

**Default ATR multiplier:** 2.0 (Claude's discretion). This means a trailing stop at 2x ATR below the high watermark. On crypto hourly data, ATR(14) typically represents 1-2% of price, so 2x ATR gives ~2-4% trailing distance. This balances between:
- Too tight (1.0x): Stopped out by normal volatility
- Too loose (3.0x): Gives back too much profit

**Cooldown:** Not implementing a cooldown initially. If signals whipsaw (open, immediately stopped out, reopen), the composite scoring's trade count factor and turnover penalty will naturally penalize this. Let Phase 12 autoresearch discover if cooldown helps.

## Code Examples

### Vote Combinator Addition to executor.py

```python
# In evaluate_condition(), after the "none" block:

if "vote" in condition:
    vote_config = condition["vote"]
    sub_conditions = vote_config.get("conditions", [])
    min_votes = vote_config.get("min_votes", 4)
    if not sub_conditions:
        return False
    vote_count = sum(
        1
        for c in sub_conditions
        if evaluate_condition(
            c, features, row_idx,
            max_depth=max_depth, _current_depth=next_depth,
        )
    )
    return vote_count >= min_votes
```

### Sample Nunchi 6-Signal JSON Config

```json
{
    "name": "nunchi_voting_crypto_1h",
    "description": "6-signal voting strategy (Nunchi-derived) for crypto_spot 1h",
    "symbol": "BTCUSDT",
    "market": "crypto_spot",
    "interval": "1h",
    "rules": [
        {
            "condition": {
                "vote": {
                    "conditions": [
                        {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 6}, "threshold": 0},
                        {"type": "indicator_above", "indicator": "cum_return", "params": {"period": 12}, "threshold": 0},
                        {"type": "indicator_crosses", "indicator": "ema", "params": {"fast": 7, "slow": 26}, "direction": "up"},
                        {"type": "indicator_above", "indicator": "rsi", "params": {"period": 8}, "threshold": 50},
                        {"type": "indicator_above", "indicator": "macd_histogram", "params": {}, "threshold": 0},
                        {"type": "bollinger_width_percentile", "params": {"period": 20, "lookback": 168}, "threshold": 0.2}
                    ],
                    "min_votes": 4
                }
            },
            "action": "long",
            "confidence": 1.0,
            "quantity_pct": 0.08
        }
    ]
}
```

**Note on column resolution:** `cum_return` with period 6 resolves to `cum_return_6d` via `resolve_column_name()`. The `macd_histogram` needs special handling in resolve_column_name -- currently `macd` maps to `macd_line`. Need to add: if indicator is `macd_histogram`, return `"macd_histogram"` directly.

### compute_composite_score() Integration

```python
# In backtest/metrics.py, after compute_metrics():

def compute_composite_score(metrics: dict) -> float:
    import math

    trade_count = metrics.get("trade_count", 0)
    max_drawdown = metrics.get("max_drawdown", 1.0)
    total_return = metrics.get("total_return", -1.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)

    if trade_count < 10 or max_drawdown > 0.50 or total_return < -0.50:
        return 0.0

    trade_factor = math.sqrt(min(trade_count / 50.0, 1.0))
    dd_penalty = max_drawdown ** 2
    turnover_penalty = max(0, (trade_count - 200) / 1000.0)

    return sharpe * trade_factor - dd_penalty - turnover_penalty
```

## Column Resolution Gaps

The existing `resolve_column_name()` function needs extensions for new indicators:

| Indicator | Current Resolution | Needed Resolution | Fix |
|-----------|-------------------|-------------------|-----|
| `macd_histogram` | Not handled (only `macd` -> `macd_line`) | `macd_histogram` | Add case: `if indicator == "macd_histogram": return "macd_histogram"` |
| `cum_return` | Not handled | `cum_return_{period}d` | Standard pattern: `{indicator}_{period}` works if CumulativeReturn output is `cum_return_6d` -- verify column name. Actually CumulativeReturn outputs `cum_return_{period}d` with a `d` suffix. Need to handle this. |
| `macd_signal` | Not handled | `macd_signal` | Add case: `if indicator == "macd_signal": return "macd_signal"` |

**Recommended fix:** Extend `resolve_column_name()` with a special cases dict:

```python
_DIRECT_COLUMN_MAP = {
    "macd_histogram": "macd_histogram",
    "macd_signal": "macd_signal",
    "macd_line": "macd_line",
    "macd": "macd_line",
}
```

And for `cum_return`, since the feature outputs `cum_return_{period}d` (with 'd' suffix), either:
- Option A: Override in resolve_column_name to append 'd' for cum_return
- Option B: Modify CumulativeReturn feature to output `cum_return_{period}` (breaking change, avoid)
- **Recommendation:** Option A -- add special case in resolve_column_name.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ML direction prediction | Rule-based voting | v2.0 pivot (2026-03) | Core architectural change. ML confirmed dead end. |
| Single-strategy signals | Multi-signal voting ensemble | v2.0 pivot | Higher conviction, fewer false signals |
| Manual parameter tuning | Nunchi-validated parameters | Phase 10 | Starting point for automated search in Phase 11-12 |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `pytest tests/test_voting_strategy.py tests/test_composite_score.py tests/test_vote_combinator.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VOTE-01 | VotingStrategy wraps child strategies, emits signal at threshold | unit | `pytest tests/test_voting_strategy.py::TestVotingStrategy -x` | Wave 0 |
| VOTE-02 | DSL vote combinator evaluates M-of-N conditions | unit | `pytest tests/test_vote_combinator.py -x` | Wave 0 |
| VOTE-03 | 6 Nunchi signal configs load and produce signals | unit + integration | `pytest tests/test_voting_strategy.py::TestNunchiSignals -x` | Wave 0 |
| VOTE-04 | Composite scoring formula returns expected values | unit | `pytest tests/test_composite_score.py -x` | Wave 0 |
| VOTE-05 | ATR trailing stop fires correctly | unit | `pytest tests/test_voting_strategy.py::TestATRTrailingStop -x` | Wave 0 |
| VOTE-06 | Fixed 8% position sizing applied | unit | `pytest tests/test_voting_strategy.py::TestPositionSizing -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_voting_strategy.py tests/test_composite_score.py tests/test_vote_combinator.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_voting_strategy.py` -- covers VOTE-01, VOTE-03, VOTE-05, VOTE-06
- [ ] `tests/test_vote_combinator.py` -- covers VOTE-02
- [ ] `tests/test_composite_score.py` -- covers VOTE-04

*(Existing `tests/conftest.py` provides environment setup. No new shared fixtures needed beyond test-specific feature DataFrames.)*

## Open Questions

1. **EMA Crossover vs EMA Position for Voting**
   - What we know: `indicator_crosses` detects the exact bar where EMA(7) crosses above EMA(26). This fires once, then goes silent until the next crossover.
   - What's unclear: For voting, should signal 3 be "EMA(7) crossed above EMA(26) recently" or "EMA(7) is currently above EMA(26)"? Nunchi's code uses the latter (position, not cross).
   - Recommendation: Use `indicator_above` with fast=ema_7 vs slow=ema_26 (currently above), not `indicator_crosses` (just crossed). This requires the `indicator_comparison` condition evaluator described above. If `indicator_crosses` is used, the vote would only count on the exact crossover bar, making it nearly useless for a voting system.

2. **MACD Histogram Column Resolution**
   - What we know: MACD feature returns columns `macd_line`, `macd_signal`, `macd_histogram`. The `resolve_column_name` function maps `"macd"` to `"macd_line"`.
   - What's unclear: How to reference `macd_histogram` in DSL JSON cleanly.
   - Recommendation: Add `"macd_histogram"` and `"macd_signal"` as direct mappings in resolve_column_name. Then the DSL condition can use `"indicator": "macd_histogram"` directly.

3. **Bearish Voting (SELL signals)**
   - What we know: The 6 signals above define bullish conditions. Nunchi's strategy also has inverse conditions for bearish signals.
   - What's unclear: Should Phase 10 include bearish voting configs, or only bullish?
   - Recommendation: Include both. Each voting config has 2 rules: one for long entry (bullish vote), one for short or close (bearish vote with inverted conditions). The JSON config should have both rules.

## Sources

### Primary (HIGH confidence)
- Poseidon codebase: `strategies/base.py`, `strategies/rule_strategy.py`, `strategies/dsl/executor.py`, `strategies/dsl/conditions.py`, `backtest/metrics.py`, `backtest/portfolio.py`, `backtest/runner.py`, `data/feature_engine.py`, `data/features/technical.py`, `data/features/returns.py`
- `.planning/research/FEATURES.md` -- Nunchi signal definitions
- `.planning/research/PITFALLS.md` -- DSL gaps, overfit warnings

### Secondary (MEDIUM confidence)
- `.planning/phases/10-voting-strategy-foundation/10-CONTEXT.md` -- User decisions and canonical references

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project (pandas, pydantic, numpy)
- Architecture: HIGH -- patterns directly visible in existing codebase
- Pitfalls: HIGH -- documented from Nunchi experiments and codebase analysis

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable codebase, no external dependency changes)
