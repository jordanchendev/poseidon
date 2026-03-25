# Domain Pitfalls

**Domain:** Trading signal platform -- strategy pivot (voting + autoresearch + regime + param search)
**Researched:** 2026-03-25

## Critical Pitfalls

Mistakes that cause rewrites, invalid results, or fundamentally broken strategy logic.

### Pitfall 1: Overfitting Through Optimization (The Biggest Risk)

**What goes wrong:** Optuna finds parameters that produce Sharpe 20+ on training data but Sharpe 0.3 out-of-sample. The autoresearch loop reinforces this by always keeping "improvements" that are actually overfitting.
**Why it happens:** With enough parameters to tune (6 signals x 2-3 params each = 12-18 dimensions), Optuna can always find a configuration that fits historical noise.
**Consequences:** Strategy looks amazing in backtest. Loses money in production. Wastes months of development time chasing phantom alpha.
**Prevention:**
- Walk-forward validation (WFE >= 50%) is the primary defense. Already implemented in `WalkForwardAnalyzer`. Every experiment MUST pass WFE before being "kept."
- Nunchi's composite score penalizes low trade count (`sqrt(min(trades/50, 1.0))`). A strategy that trades twice perfectly is scored nearly zero.
- Hard cutoffs: < 10 trades = instant failure (-999 score). Prevents single-trade outlier strategies.
- Keep parameter count low. Nunchi's final strategy has ~15 tunable params across 6 signals. Karpathy's program.md explicitly values "simplification wins."
**Detection:** WFE < 50% on any OOS window. Large gap between IS and OOS Sharpe (> 2x difference). Trade count drops below 20 per OOS window.

### Pitfall 2: Look-Ahead Bias in Feature Computation

**What goes wrong:** Features accidentally use future data during backtesting. Common when BB width percentile or regime labels are computed on the full dataset instead of expanding-window.
**Why it happens:** Rolling calculations in pandas operate on the full DataFrame by default. The backtester must enforce expanding-window access. Poseidon's `BacktestRunner` already does bar-by-bar expansion, but new features (BB width percentile rank) could bypass this if computed outside the runner.
**Consequences:** Backtest results are meaninglessly optimistic. No way to recover without rerunning everything after fixing the bias.
**Prevention:**
- All features must be computed inside `FeatureEngine` using only data up to the current bar
- BB width percentile rank must use `rolling().rank()` or expanding window, never full-series percentile
- Regime labels must be generated from a model trained only on IS data, never on OOS data
- Walk-forward analysis provides a structural defense: each OOS window uses features computed without OOS data
**Detection:** Suspiciously good OOS performance that degrades sharply in live trading. Validate by manually checking feature values at bar N using only bars [0, N].

### Pitfall 3: Evaluation Layer Drift During AutoResearch

**What goes wrong:** Someone changes BacktestRunner, FeatureEngine, or CostModel parameters between experiments. Results become incomparable. Experiment #47 scored 12.3 under different conditions than experiment #48 at 11.8 -- the "regression" might actually be an improvement.
**Why it happens:** Natural impulse to "fix" or "improve" the evaluation layer mid-run. Karpathy addresses this with "DO NOT modify prepare.py" in bold. Nunchi locks prepare.py and backtest.py.
**Consequences:** Entire experiment history is invalidated. Must restart from scratch.
**Prevention:**
- Hash the fixed layer configuration (FeatureEngine params, CostModel params, BacktestRunner settings) and store with each experiment
- Validate hash matches before comparing experiments
- Version the evaluation config in Optuna study metadata
**Detection:** Config hash mismatch when loading experiment history. Sudden score discontinuities without parameter changes.

### Pitfall 4: Regime Model Circular Dependency

**What goes wrong:** The regime classifier is trained on features that include regime labels. Or the regime is used to filter training data for the regime model itself.
**Why it happens:** When regime labels are used both as training targets AND as feature inputs.
**Consequences:** Regime model learns to predict its own output. 99% accuracy that means nothing.
**Prevention:**
- `REGIME_FEATURES` list in FeatureEngine must NEVER include the regime label itself
- Training: features -> label regimes (HMM or thresholds) -> train XGBoost on features to predict labels
- Inference: only features go in, regime label comes out
- Check `_feature_list` in XGBoostRegimeModel to ensure no regime columns
**Detection:** Perfect accuracy in regime prediction (too good to be true).

## Moderate Pitfalls

### Pitfall 5: Voting Threshold Sensitivity

**What goes wrong:** Small changes to min_votes (4 vs 3) cause dramatic behavior changes.
**Why it happens:** With 6 binary signals, 3/6 allows 20 entry combinations vs 4/6 which allows 15 -- a 25% reduction.
**Prevention:**
- Always optimize min_votes as part of parameter search, not fix manually
- Test sensitivity: run at min_votes = [2, 3, 4, 5, 6] and ensure graceful degradation
- Nunchi's evolution: 3/5 -> 4/5 -> 4/6 as signals were added
**Detection:** Parameter importance analysis shows min_votes as dominant. High WFE variance across values.

### Pitfall 6: RSI/MACD Period Mismatch with Timeframe

**What goes wrong:** Using daily-calibrated periods (RSI-14, MACD 12/26/9) on hourly data.
**Why it happens:** Off-the-shelf defaults are for daily charts. Nunchi discovered RSI 14->8 gave +5 points on hourly.
**Prevention:**
- Nunchi's optimized hourly params: EMA(7, 26), RSI(8), MACD(14, 23, 9), BB(7)
- Start with Nunchi's params as crypto hourly baseline
- Include period as Optuna search parameter for other timeframes
**Detection:** Low trade frequency with standard periods. Compare results at period=14 vs period=8.

### Pitfall 7: Optuna Study Naming Collisions

**What goes wrong:** Two autoresearch runs share an Optuna study. Results from different evaluation configs get mixed.
**Why it happens:** Using `load_if_exists=True` with generic study name.
**Prevention:** Convention: `f"{symbol}_{strategy}_{config_hash}_{timestamp}"`. Store eval config hash in study user_attrs.
**Detection:** Study contains trials with inconsistent user_attrs.

### Pitfall 8: AutoResearch Agent Runaway

**What goes wrong:** AI agent modifies files outside the intended scope.
**Why it happens:** Agent has access to full codebase. Karpathy/Nunchi rely on agent discipline.
**Prevention:**
- Variable layer is JSON, not Python. Agent writes configs, not code.
- Pydantic validation on every config before backtesting
- Sandboxed execution (Docker with read-only source mounts)
- Log all file modifications. Alert on non-JSON changes.
**Detection:** Git diff shows changes outside `experiments/`. Pydantic validation failures.

## Minor Pitfalls

### Pitfall 9: BB Width Percentile Window Size

**What goes wrong:** Too short a lookback window for percentile rank. 20 bars on hourly = 1 day. Squeeze detection needs longer context.
**Prevention:** Use 168+ bars (1 week hourly) for percentile computation.

### Pitfall 10: Signal Counting Edge Cases

**What goes wrong:** A sub-signal returns both CLOSE and new LONG. VotingStrategy counts 2 votes from one source.
**Prevention:** Each sub-strategy contributes at most 1 directional vote. Sub-strategies for voting should be simple condition evaluators, not position-tracking strategies.

### Pitfall 11: Optuna Connection Pool Exhaustion

**What goes wrong:** Parallel Celery workers exhaust PostgreSQL connection pool.
**Prevention:** Configure `engine_kwargs={"pool_size": 5, "max_overflow": 10}` in RDBStorage. Match to Celery concurrency.

### Pitfall 12: Hysteresis Creates Regime Lag

**What goes wrong:** 3-measurement requirement means 3-bar late response to regime shifts.
**Prevention:** Accept as deliberate trade-off. Nunchi uses hysteresis because false switches > lag cost. Apply hysteresis only to DOWNSHIFT. UPSHIFT responds immediately.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| VotingStrategy implementation | Pitfall 10: double-counting signals | Each sub-strategy = 1 vote max. Simple bool evaluation. |
| Nunchi 6-signal porting | Pitfall 6: period mismatch | Use Nunchi hourly params (RSI 8, EMA 7/26, MACD 14/23/9, BB 7) |
| Optuna RDBStorage migration | Pitfall 7: naming collisions | Enforce `{symbol}_{strategy}_{hash}_{timestamp}` from day one |
| Parameter search | Pitfall 1: overfitting | WFE >= 50% gate. Composite score. Hard cutoffs. |
| Regime-to-strategy mapping | Pitfall 4: circular dependency | Regime features and labels strictly separate. |
| AutoResearch loop | Pitfall 3: evaluation drift | Hash fixed-layer config. Validate before comparison. |
| AutoResearch loop | Pitfall 8: agent runaway | JSON variable layer. Pydantic validation. Sandboxed execution. |
| BB squeeze feature | Pitfall 9: window size | 168+ bar lookback for hourly percentile rank. |
| Production deployment | Pitfall 12: hysteresis lag | Immediate upshift, delayed downshift. |

## Lessons from Nunchi's 103-Experiment History

Documented failures from STRATEGIES.md -- not speculative.

| Experiment Range | What Was Tried | Result | Lesson for Poseidon |
|-----------------|----------------|--------|---------------------|
| exp1-exp4 | Simple momentum -> 4-signal voting | +1.5 score | Voting ensemble strictly better than single signals |
| exp15 | Raised min_votes 3/5 -> 4/5 | +1.1 score | Higher conviction reduces false signals |
| exp32 | Added BB squeeze as 6th signal | +0.8 score | Non-directional quality gate improves timing |
| exp41-66 | Removed pyramiding, funding, lead-lag, variable sizing | +3.2 total | **Simplification beats complexity** |
| exp72-77 | RSI period 14 -> 8 | +5.0 score | Default periods wrong for hourly data |
| exp119-251 | Fine-tuning all params | ~0.3 total | Diminishing returns after architecture decisions |

**Autoresearch priority order based on Nunchi evidence:**
1. Signal selection (which signals) -- highest impact
2. Ablation (removing signals) -- second highest
3. Period optimization (RSI 14->8 type) -- third
4. Threshold fine-tuning -- lowest impact, do last

## Sources

- [Nunchi STRATEGIES.md](https://github.com/Nunchi-trade/auto-researchtrading) -- 103-experiment evolution with documented failures and gains
- [Nunchi strategy.py](https://github.com/Nunchi-trade/auto-researchtrading) -- Signal counting, MIN_VOTES, BB squeeze quality gate
- [Nunchi regime_mm.py](https://github.com/Nunchi-trade/auto-researchtrading) -- Hysteresis logic, regime-to-params mapping
- [Karpathy program.md](https://github.com/karpathy/autoresearch/blob/master/program.md) -- "DO NOT modify prepare.py", simplicity criterion
- [n-autoresearch](https://github.com/iii-hq/n-autoresearch) -- Crash recovery, adaptive search
- [Optuna RDBStorage](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html) -- Connection pool configuration
- Poseidon codebase: BacktestRunner, WalkForwardAnalyzer, XGBoostRegimeModel, FeatureEngine

---
*Domain pitfalls for: Poseidon v2.0 Strategy Pivot*
*Researched: 2026-03-25*
