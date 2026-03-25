# Feature Landscape

**Domain:** Trading signal platform -- strategy pivot (voting + autoresearch + regime + param search)
**Researched:** 2026-03-25

## Table Stakes

Features that must work correctly for the v2.0 pivot to deliver value.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| VotingStrategy class | Core pivot from ML direction prediction to rule-based voting. Foundation for all other v2.0 features. | Medium | New `BaseStrategy` subclass wrapping N `RuleStrategy` sub-signals. Implements `evaluate()` -> `list[Signal]`. |
| Nunchi 6-signal configuration | The validated starting point: momentum x2 (12h + 6h returns), EMA crossover (7/26), RSI(8), MACD(14,23,9) histogram, Bollinger squeeze (7-period width percentile). | Low | Each signal maps to existing DSL conditions. EMA crossover = `indicator_crosses`. RSI above/below 50 = `indicator_above/below`. MACD histogram sign = `indicator_above/below` on macd_histogram. |
| Configurable vote threshold | 4/6 default (Nunchi validated), must be tunable: 3/6 aggressive, 5/6 conservative. Different thresholds per regime. | Low | `min_votes` parameter on VotingStrategy. |
| Walk-forward validation integration | VotingStrategy must validate through same pipeline as existing strategies. WFE >= 50%. | Low | VotingStrategy implements BaseStrategy, so WalkForwardAnalyzer works unchanged. Zero code changes in walk_forward.py. |
| Optuna persistent studies | Parameter search results must survive restarts and be queryable. Current BayesianOptimizer loses everything on restart. | Low | Config change: `optuna.storages.RDBStorage(url=settings.DATABASE_URL)` in BayesianOptimizer. One-line refactor. |
| Composite scoring function | Raw sharpe_ratio is insufficient for autoresearch -- must penalize low trade count, high drawdown, and excessive turnover. | Low | New function in `backtest/metrics.py`. Nunchi formula: `sharpe * sqrt(trade_factor) - dd_penalty - turnover_penalty`. |
| Regime-to-strategy mapping | Current regime label should select different VotingStrategy configurations (different thresholds, position sizes). | Medium | XGBoostRegimeModel already classifies regimes. Need a mapping dict: `regime_label -> {min_votes, position_size_multiplier, stop_atr_multiplier}`. |

## Differentiators

Features that make v2.0 significantly more powerful than manual parameter tuning.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| AutoResearch experiment loop | AI agent autonomously iterates VotingStrategy parameters, tests via backtest, keeps improvements. Nunchi achieved Sharpe 2.7 -> 21.4 in 103 unattended experiments. | High | Three-layer architecture (fixed BacktestRunner / variable RuleConfig JSON / guide program.md). Celery task orchestration. Git audit trail. This is the highest-value differentiator. |
| Multi-objective Optuna optimization | Optimize Sharpe AND max_drawdown simultaneously. Produces Pareto frontier of non-dominated strategies. | Low | Built into Optuna >= 3.0. Change `create_study(direction="maximize")` to `create_study(directions=["maximize", "minimize"])`. |
| Adaptive search modes | Based on n-autoresearch: switch between explore (random mutations), exploit (refine best), combine (merge successful params), ablation (remove signals). | Medium | Implement as search strategy selection in the autoresearch loop. Track improvement rate to trigger mode switches. |
| Regime transition alerts | Detect when XGBoostRegimeModel output changes regime (e.g., low_vol -> high_vol). Trigger strategy reconfiguration. | Medium | Compare current vs previous regime prediction. On change: log event, potentially pause signals during transition window. |
| optuna-dashboard deployment | Web UI for experiment history, parameter importance plots, optimization trajectory visualization. | Low | Docker sidecar container. `ghcr.io/optuna/optuna-dashboard` with PostgreSQL connection string. |
| Pruning for fast parameter search | Early-stop unpromising Optuna trials using intermediate walk-forward window results. 30-50% compute savings. | Low | `optuna.pruners.MedianPruner()`. Report intermediate WFE values from each walk-forward window. Prune if consistently below median. |
| Hysteresis regime filtering | Prevent rapid regime switching using Nunchi's pattern: require N consecutive measurements before downshifting regime. Reduces whipsaw. | Low | 3-consecutive-bar filter on regime predictions. Pure Python, no new deps. From Nunchi regime_mm.py. |
| Bollinger squeeze as quality gate | Nunchi's key insight: BB width percentile as a non-directional "breakout imminent" signal. Votes for both bull/bear, only amplifies existing directional agreement. | Low | New condition type `bb_width_percentile_below` in DSL conditions registry. Computes rolling percentile rank of BB width. |
| Per-regime parameter optimization | Run separate Optuna studies for each regime. Low-vol regime may prefer different RSI period than high-vol. | Medium | Create study per regime: `study_name=f"{symbol}_{regime}_{timestamp}"`. Filter training data by regime label before optimization. |

## Anti-Features

Features to explicitly NOT build in v2.0.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| ML direction prediction | Confirmed dead end across BTC/ETH, 1d/1h, Transformer/XGBoost. Pure TA cannot predict price direction. This is the entire motivation for v2.0 pivot. | Rule-based voting signals. ML only for regime classification. |
| Real-time sub-minute signals | Adds latency requirements, WebSocket infrastructure, exchange rate limit management. Nunchi's winning strategy runs on hourly bars. | Stay on 1h/1d timeframes. Voting strategy validated on hourly. |
| LLM-generated strategy code | Tempting but unsafe. LLM-generated Python is untestable at generation time. Karpathy's autoresearch works because train.py is a single file with a clear metric. | Use AI to adjust JSON params within bounded search space. Never generate executable code. The mutable layer is RuleConfig JSON, not Python. |
| Backtesting web UI | Significant frontend effort for a single-developer tool. | optuna-dashboard for experiment viz. REST API + CLI for everything else. Poseidon is an API-first system. |
| Automated position sizing optimization | Conflates strategy quality with risk management. Position sizing belongs in the risk engine, not the strategy optimizer. | Optimize strategy params (which signals, what thresholds) via Optuna. Position sizing stays in RiskEngine with per-regime scaling. |
| Cross-asset correlation strategies | Multi-asset strategies require synchronized data, cross-market timing, and correlation computation. Significant data pipeline complexity. | Single-asset voting strategies per symbol. Cross-asset is v3.0. |
| Pyramiding / variable position sizing | Nunchi's experiment history proves this hurts: removing pyramiding = +0.4, removing variable sizing = +1.7 (largest single improvement). | Fixed position sizes per regime. Let the strategy decide direction, let the risk engine decide size. |
| Multi-timeframe confirmation | Nunchi experimented and removed it: "net harmful." Looking at daily RSI while trading hourly adds noise. | Single-timeframe signals only. Each signal operates on the same interval as the VotingStrategy. |
| Funding rate overlay | Nunchi tried and removed it: "+0.7 when removed." Funding carry adds complexity without improving the voting signal. | Ignore funding for signal generation. Track funding as a data feature but don't include in voting. |

## Feature Dependencies

```
Phase 1: VotingStrategy
    VotingStrategy class -> BaseStrategy ABC (implements interface)
                         -> RuleStrategy (wraps N instances as sub-signals)
                         -> FeatureEngine (computes indicators for sub-signals)
                         -> BacktestRunner (validates through existing pipeline)
    Composite scoring    -> compute_metrics() (extends with composite score)
    BB squeeze condition -> DSL conditions registry (new condition type)

Phase 2: Optuna + Regime
    Optuna RDBStorage    -> BayesianOptimizer (refactored for storage param)
                         -> PostgreSQL (existing, Optuna creates own tables)
    Regime mapping       -> XGBoostRegimeModel (already exists, adds hysteresis)
                         -> VotingStrategy (configured per regime label)
    optuna-dashboard     -> Optuna RDBStorage (reads same database)

Phase 3: Parameter Search
    VotingStrategy factory -> VotingStrategy (parameterized creation from dict)
    Optuna study per regime -> Regime mapping (provides regime labels)
                            -> VotingStrategy factory (creates strategy per params)
                            -> WalkForwardAnalyzer (validation gate per trial)
    Multi-objective         -> BayesianOptimizer (directions=["maximize","minimize"])

Phase 4: AutoResearch Loop
    Experiment runner    -> VotingStrategy factory (creates strategies)
                         -> BacktestRunner (immutable evaluation)
                         -> Optuna RDBStorage (logs all trials)
                         -> Git subprocess (atomic commits per experiment)
    program.md template  -> Current regime (from XGBoostRegimeModel)
                         -> Best params (from Optuna study)
                         -> Experiment history (from Optuna)
    Adaptive search      -> Experiment runner (tracks improvement rate)
                         -> Mode selection (explore/exploit/combine/ablation)
```

## MVP Recommendation

Build in this order based on dependency chain and value delivery:

1. **VotingStrategy class + Nunchi 6-signal config** -- Foundation everything else builds on. Port the exact signals from Nunchi's strategy.py. Validate with BacktestRunner immediately.
2. **Composite scoring function** -- Needed before any optimization can produce meaningful results. Add to `backtest/metrics.py`.
3. **Optuna RDBStorage migration** -- One-line change enables persistent experiment tracking. Do this early so all subsequent work is tracked.
4. **VotingStrategy factory + parameter search** -- Parameterize the 6-signal config (periods, thresholds, min_votes). Run BayesianOptimizer with RDBStorage.
5. **Regime-to-strategy mapping + hysteresis** -- Use existing XGBoostRegimeModel to select VotingStrategy configs per regime.
6. **AutoResearch loop** -- Highest complexity, highest payoff. Only meaningful after all other pieces work.

**Defer to v2.1:**
- HMM-enhanced regime labeling (XGBoost regime already works)
- Soft voting with weighted confidence (hard voting is sufficient)
- Per-regime parameter optimization (requires enough data per regime)
- Adaptive search modes (start with simple random + TPE)

## Sources

- [Nunchi auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading) -- strategy.py (6-signal voting implementation), STRATEGIES.md (103-experiment evolution showing what to remove), prepare.py (composite scoring), regime_mm.py (hysteresis pattern)
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) -- Three-layer architecture, simplicity criterion ("equal results via code deletion is a simplification win")
- [n-autoresearch](https://github.com/iii-hq/n-autoresearch) -- Adaptive search modes (explore/exploit/combine/ablation)
- Poseidon codebase analysis -- BaseStrategy, RuleStrategy, DSL conditions, FeatureEngine features, BacktestRunner, WalkForwardAnalyzer, XGBoostRegimeModel

---
*Feature landscape for: Poseidon v2.0 Strategy Pivot*
*Researched: 2026-03-25*
