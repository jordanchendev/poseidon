# Architecture Patterns

**Domain:** Trading signal platform -- strategy pivot (voting + autoresearch + regime + param search)
**Researched:** 2026-03-25

## Recommended Architecture

### The Three-Layer AutoResearch Pattern

Derived from studying both Karpathy's autoresearch and Nunchi's auto-researchtrading repos, adapted for Poseidon's existing architecture.

```
+--------------------------------------------------+
|                  GUIDE LAYER                      |
|  program.md template (generated per experiment)   |
|  - Current best params + score                    |
|  - Experiment history summary                     |
|  - Search space constraints                       |
|  - Current market regime                          |
+--------------------------------------------------+
         |                         ^
         | instructs agent         | reads results
         v                         |
+--------------------------------------------------+
|               VARIABLE LAYER                      |
|  RuleConfig JSON (strategy DSL)                   |
|  - VotingStrategy params (min_votes, thresholds)  |
|  - Sub-signal configs (periods, levels)           |
|  - Regime-specific overrides                      |
|  Each experiment = one JSON mutation              |
+--------------------------------------------------+
         |                         ^
         | feeds into              | returns metrics
         v                         |
+--------------------------------------------------+
|                FIXED LAYER                        |
|  BacktestRunner + compute_metrics() + WFA         |
|  FeatureEngine (all indicator computation)        |
|  CostModel + RiskEngine                           |
|  IMMUTABLE during autoresearch runs               |
+--------------------------------------------------+
```

**Poseidon's key advantage over Karpathy/Nunchi:** The variable layer is structured JSON data, not Python source code. This means:
- Mutations are bounded by Pydantic schema validation
- No risk of syntax errors, import failures, or runtime exceptions from bad mutations
- Search space is explicit: Optuna param_space maps directly to JSON fields
- Rollback = restore previous JSON config (no git reset needed)

### Component Architecture

```
src/poseidon/
  strategies/
    base.py              # BaseStrategy ABC (EXISTING, unchanged)
    rule_strategy.py     # RuleStrategy (EXISTING, unchanged)
    model_strategy.py    # ModelStrategy (EXISTING, unchanged)
    voting_strategy.py   # NEW: VotingStrategy(BaseStrategy)
    dsl/
      schema.py          # RuleConfig (EXISTING, unchanged)
      executor.py        # evaluate_condition() (EXISTING, unchanged)
      conditions.py      # CONDITION_REGISTRY (EXISTING, add bb_squeeze)

  backtest/
    runner.py            # BacktestRunner (EXISTING, immutable for autoresearch)
    metrics.py           # compute_metrics() (EXISTING, add composite_score())
    optimizer.py         # Grid + Bayesian (EXISTING, refactor for RDBStorage)
    walk_forward.py      # WalkForwardAnalyzer (EXISTING, unchanged)

  ml/implementations/
    xgboost_regime.py    # XGBoostRegimeModel (EXISTING, add hysteresis)

  autoresearch/          # NEW module
    runner.py            # ExperimentRunner: orchestrates the loop
    scoring.py           # composite_score() + hard cutoffs
    factory.py           # VotingStrategy factory from params dict
    program.py           # program.md template generation
    search.py            # Adaptive search mode selection
```

### Component Boundaries

| Component | Responsibility | Communicates With | Mutability |
|-----------|---------------|-------------------|------------|
| VotingStrategy | Aggregate N sub-signal evaluations into majority vote | RuleStrategy (sub-signals), FeatureEngine (data), Signal (output) | Config is mutable (params), class is fixed |
| ExperimentRunner | Orchestrate one autoresearch experiment cycle | VotingStrategy factory, BacktestRunner, Optuna storage, Git | Orchestrator only, no business logic |
| VotingStrategyFactory | Create VotingStrategy instances from parameter dicts | VotingStrategy, RuleConfig, Optuna trial params | Stateless factory |
| CompositeScorer | Compute Nunchi-style composite score from metrics | BacktestRunner output, metrics dict | Pure function, no state |
| RegimeMapper | Map regime labels to VotingStrategy configs | XGBoostRegimeModel, VotingStrategy | Config dict, updatable |
| ProgramGenerator | Render program.md from template + experiment state | Optuna study, RegimeMapper, best params | Template + data, stateless |
| SearchModeSelector | Choose explore/exploit/combine/ablation mode | Experiment history from Optuna | Stateful (tracks improvement rate) |

### Data Flow

**Single Experiment Cycle (AutoResearch):**

```
1. ProgramGenerator renders program.md with:
   - Current best: {sharpe: 15.2, params: {...}}
   - History: last 10 experiments with scores
   - Constraints: param bounds, immutable layers
   - Regime: current = "low_vol"

2. AI Agent (Claude Code) reads program.md
   - Proposes parameter mutation
   - Writes new RuleConfig JSON

3. VotingStrategyFactory creates VotingStrategy from JSON
   - Validates via Pydantic schema
   - Creates N RuleStrategy sub-signals

4. BacktestRunner.run(ohlcv) evaluates strategy
   - FeatureEngine computes indicators (fixed)
   - Strategy.evaluate() runs voting logic
   - RiskEngine applies position limits (fixed)
   - Portfolio tracks equity curve (fixed)

5. compute_metrics() + composite_score()
   - Standard 11 metrics + composite score
   - Hard cutoffs applied (< 10 trades = -999)

6. WalkForwardAnalyzer.analyze() validates robustness
   - WFE >= 50% required to "keep"
   - Per-window trade count minimums

7. Optuna records trial:
   - trial.params = {min_votes, rsi_period, ema_fast, ...}
   - trial.value = composite_score
   - trial.user_attrs["metrics"] = full metrics dict
   - trial.user_attrs["wfe"] = walk_forward_efficiency

8. Git commit (if improvement):
   - Atomic commit with experiment description
   - If no improvement: revert JSON to previous best

9. Loop continues (step 1)
```

**Live Signal Generation (Runtime):**

```
1. Celery Beat triggers evaluation task (hourly/daily)
2. FeatureEngine computes indicators from latest OHLCV
3. XGBoostRegimeModel classifies current regime
4. Hysteresis filter: require 3 consecutive same-regime predictions
5. RegimeMapper selects VotingStrategy config for confirmed regime
6. VotingStrategy.evaluate(features) -> list[Signal]
   - Each sub-signal RuleStrategy evaluates independently
   - Count votes in each direction
   - Emit signal if votes >= min_votes for that regime
7. RiskEngine.evaluate(signals) -> approved signals
8. Signal delivery to Redis Streams -> Thalassa
```

## Patterns to Follow

### Pattern 1: Strategy Composition (Voting)

**What:** VotingStrategy wraps N RuleStrategy instances. Each sub-signal evaluates independently. VotingStrategy counts how many signal in the same direction and emits if >= threshold.

**When:** Always for v2.0. This is the core pattern replacing ML direction prediction.

**Why this over DSL combinator:** The DSL evaluates conditions to `bool`. Voting needs:
1. Count how many sub-signals fire (not just all/any)
2. Compute aggregate confidence from individual confidences
3. Support "K of N" semantics (4 of 6, not "all 6" or "any 1")

**Directly from Nunchi's strategy.py:**
```python
# Nunchi voting: 6 signals, need 4 to agree
bull_votes = sum(1 for s in signals if s == 1)
bear_votes = sum(1 for s in signals if s == -1)
if bull_votes >= MIN_VOTES:  # MIN_VOTES = 4
    target_position = position_size
elif bear_votes >= MIN_VOTES:
    target_position = -position_size
```

**Poseidon implementation:**
```python
class VotingStrategy(BaseStrategy):
    strategy_type = StrategyType.RULE

    def __init__(self, sub_strategies: list[RuleStrategy], min_votes: int = 4, ...):
        self.sub_strategies = sub_strategies
        self.min_votes = min_votes

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        long_votes, short_votes = 0, 0
        for sub in self.sub_strategies:
            for sig in sub.evaluate(features):
                if sig.action == SignalAction.LONG:
                    long_votes += 1
                elif sig.action == SignalAction.SHORT:
                    short_votes += 1

        if long_votes >= self.min_votes:
            return [self._make_signal(SignalAction.LONG, long_votes / len(self.sub_strategies))]
        elif short_votes >= self.min_votes:
            return [self._make_signal(SignalAction.SHORT, short_votes / len(self.sub_strategies))]
        return []
```

### Pattern 2: Immutable Evaluation Layer

**What:** During autoresearch, BacktestRunner + FeatureEngine + CostModel + RiskEngine are LOCKED.

**Why:** From Karpathy's program.md: "DO NOT modify prepare.py." From Nunchi's program.md: "Only strategy.py may be modified." The evaluation layer must be constant for experiments to be comparable.

### Pattern 3: Regime-Conditional Configuration with Hysteresis

**What:** Map regime labels to VotingStrategy configs. Apply hysteresis to prevent whipsaw.

**From Nunchi's regime_mm.py:**
```python
# 4 regimes with different trading parameters
REGIMES = {
    "low_vol":  {"spread_bps": 10,  "size_mult": 1.5, "stop": 0.02},
    "normal":   {"spread_bps": 25,  "size_mult": 1.0, "stop": 0.03},
    "high_vol": {"spread_bps": 50,  "size_mult": 0.5, "stop": 0.05},
    "extreme":  {"spread_bps": 100, "size_mult": 0.2, "stop": 0.08},
}

# Hysteresis: require 3 consecutive lower readings to downshift
if target < current_regime:
    consecutive_lower += 1
    if consecutive_lower >= 3:
        current_regime = target
else:
    consecutive_lower = 0
    current_regime = target
```

**Poseidon adaptation:**
```python
REGIME_STRATEGY_MAP = {
    "low_vol":    {"min_votes": 3, "position_pct": 1.0, "atr_stop_mult": 3.0},
    "medium_vol": {"min_votes": 4, "position_pct": 0.7, "atr_stop_mult": 4.5},
    "high_vol":   {"min_votes": 5, "position_pct": 0.3, "atr_stop_mult": 6.0},
}
```

### Pattern 4: Git-Based Experiment Audit Trail

**What:** Each experiment = one atomic git commit. From Karpathy/Nunchi pattern.

**Poseidon adaptation:** Since the mutable layer is JSON, commits contain RuleConfig JSON files rather than Python source:
```python
def commit_experiment(config: dict, score: float, exp_num: int):
    path = f"experiments/exp_{exp_num:04d}.json"
    Path(path).write_text(json.dumps(config, indent=2))
    subprocess.run(["git", "add", path])
    subprocess.run(["git", "commit", "-m", f"exp{exp_num}: score={score:.3f}"])
```

### Pattern 5: Bollinger Squeeze as Non-Directional Quality Gate

**What:** From Nunchi's strategy.py: BB width percentile < 90 indicates volatility compression (squeeze). This signal votes for BOTH bull and bear -- it only amplifies existing directional consensus. It acts as a "breakout imminent" quality filter.

**Why this is important:** Unlike other signals (RSI, MACD, momentum) which vote directionally, BB squeeze is non-directional. It represents entry timing quality, not direction. Adding it to the vote count means entries only happen during squeeze periods, improving signal quality.

**Implementation:** New condition type in DSL registry:
```python
@register_condition("bb_width_percentile_below")
def eval_bb_squeeze(condition, features, row_idx):
    period = condition.get("params", {}).get("period", 7)
    threshold = condition.get("threshold", 0.9)  # 90th percentile
    bb_upper = features.iloc[row_idx][f"bb_upper_{period}"]
    bb_lower = features.iloc[row_idx][f"bb_lower_{period}"]
    bb_middle = features.iloc[row_idx][f"bb_middle_{period}"]
    width = (bb_upper - bb_lower) / bb_middle
    # Compare against rolling percentile rank
    ...
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Mutating the Evaluation Layer
**What:** Changing BacktestRunner/FeatureEngine/CostModel between experiments.
**Why bad:** Makes results incomparable.
**Instead:** Lock all evaluation parameters at autoresearch run start.

### Anti-Pattern 2: Unbounded Code Mutations
**What:** Allowing AI agent to modify arbitrary Python.
**Why bad for Poseidon:** Multi-module architecture. Code mutations can break imports, interfaces.
**Instead:** Constrain mutations to RuleConfig JSON with Pydantic validation.

### Anti-Pattern 3: Optimizing Raw Sharpe Alone
**What:** Using sharpe_ratio as sole optimization target.
**Why bad:** Can be gamed by 2 perfect trades. Nunchi's hard cutoffs prevent this.
**Instead:** Composite score with trade count factor, drawdown penalty, turnover penalty.

### Anti-Pattern 4: Rapid Regime Switching
**What:** Changing strategy config every bar the regime classifier shifts.
**Why bad:** Regime boundaries are noisy. Causes whipsaw.
**Instead:** Hysteresis: 3 consecutive measurements before switching (from Nunchi).

### Anti-Pattern 5: Adding Before Removing
**What:** Starting autoresearch by adding signals/complexity.
**Why bad:** Nunchi's largest gains came from REMOVING: variable sizing (-1.7), funding boost (-0.7), BTC filter (-0.4), pyramiding (-0.4).
**Instead:** Include ablation mode in search strategy.

## Scalability Considerations

| Concern | At v2.0 launch | At 100 experiments | At 1000 experiments |
|---------|-----------------|--------------------|--------------------|
| Experiment storage | Optuna RDBStorage in PostgreSQL | PostgreSQL handles easily | Partition by study_name, archive old studies |
| Backtest compute | Sequential per trial (~5-30s each) | Add Celery workers, Optuna distributed | Prune unpromising trials early (MedianPruner) |
| Feature computation | On each backtest bar | Cache computed features per OHLCV dataset | Precompute feature DataFrame once, reuse across trials |
| Git history | Linear commit chain | Manageable | Squash old experiments, keep winning configs only |
| Dashboard performance | Trivial | Fast with PostgreSQL indexes | Optuna dashboard handles large studies natively |

## Sources

- [Nunchi strategy.py](https://github.com/Nunchi-trade/auto-researchtrading) -- 6-signal voting, MIN_VOTES=4, BB squeeze quality gate
- [Nunchi prepare.py](https://github.com/Nunchi-trade/auto-researchtrading) -- Immutable evaluation layer, composite scoring, hard cutoffs
- [Nunchi regime_mm.py](https://github.com/Nunchi-trade/auto-researchtrading) -- 4-regime classification with hysteresis
- [Nunchi STRATEGIES.md](https://github.com/Nunchi-trade/auto-researchtrading) -- 103-experiment evolution, simplification lesson
- [Karpathy program.md](https://github.com/karpathy/autoresearch/blob/master/program.md) -- Immutable evaluation constraint, simplicity criterion
- [Karpathy prepare.py](https://github.com/karpathy/autoresearch) -- Anti-gaming scoring design
- [n-autoresearch](https://github.com/iii-hq/n-autoresearch) -- Adaptive search modes, structured tracking
- Poseidon codebase: BaseStrategy, RuleStrategy, DSL, BacktestRunner, WalkForwardAnalyzer, XGBoostRegimeModel

---
*Architecture patterns for: Poseidon v2.0 Strategy Pivot*
*Researched: 2026-03-25*
