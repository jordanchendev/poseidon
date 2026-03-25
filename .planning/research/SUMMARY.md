# Project Research Summary

**Project:** Poseidon v2.0 — Strategy Pivot (Voting + AutoResearch + Regime + Parameter Search)
**Domain:** Algorithmic trading signal platform — rule-based voting strategies with autonomous parameter optimization
**Researched:** 2026-03-25
**Confidence:** HIGH (grounded in 103-experiment empirical record from Nunchi, Karpathy autoresearch patterns, and direct Poseidon codebase analysis)

## Executive Summary

Poseidon v2.0 is a strategic pivot from ML-based price direction prediction (confirmed dead end across BTC/ETH, 1d/1h, Transformer/XGBoost) to rule-based voting strategies with autonomous parameter search. The reference model is Nunchi's auto-researchtrading: a 6-signal voting system where a majority threshold triggers entries, which evolved from Sharpe 2.7 to 21.4 across 103 unattended experiments using an AI-driven autoresearch loop. The critical lesson from those 103 experiments is that **simplification consistently beats complexity** — removing seven "smart" features (pyramiding, funding overlay, BTC lead-lag filter, variable position sizing, multi-timeframe confirmation) improved the composite score by +52%. Poseidon already has approximately 80% of the required infrastructure in place: BacktestRunner, BayesianOptimizer with Optuna, WalkForwardAnalyzer, XGBoostRegimeModel, FeatureEngine, and the RuleStrategy DSL. The v2.0 work is primarily integration and extension, not greenfield construction. Only two new packages are needed (both optional).

The architecture follows a strict three-layer pattern used independently by both Karpathy's autoresearch and Nunchi's trading repo: a fixed evaluation layer (BacktestRunner + FeatureEngine + WalkForwardAnalyzer, immutable during all experiments), a variable layer (RuleConfig JSON mutated per experiment), and a guide layer (program.md rendered with current best params and experiment history for the AI agent). Poseidon's structural advantage over both reference projects is that its mutable layer is Pydantic-validated JSON, not Python source code — mutations are bounded by schema, cannot break imports, and are trivially reversible by restoring a previous JSON. One critical DSL gap must be addressed before anything else: the existing `all`/`any`/`none` combinators cannot express "4 of 6 signals agree." This requires a new `vote` condition type in the DSL registry.

The primary risks are: (1) complexity accumulation — the DSL makes deep condition trees trivially composable and Optuna will exploit them, (2) the autoresearch loop going off the rails without explicit immutability guardrails on the evaluation layer, (3) treating Nunchi's Sharpe 21.4 as a realistic target — it is not validated out-of-sample across 103 sequential experiments on a single 9-month window, expect Sharpe 1.0–3.0 on properly validated strategies, and (4) regime classification being net negative if deployed without an outperformance gate — Nunchi's regime_mm benchmark scored -0.322 vs simple momentum +2.724 on the same data. Regime conditioning must be treated as an optional enhancement that proves its value before production integration.

## Key Findings

### Recommended Stack

See [STACK.md](STACK.md) for full analysis. Poseidon's existing stack covers all v2.0 requirements. The autoresearch loop, voting strategy, composite scoring, and git audit trail are all architecture patterns implemented in pure Python with existing dependencies.

**Core technologies (all already present):**
- **Optuna + PostgreSQL via RDBStorage** — parameter search and experiment tracking; one-line refactor from in-memory to persistent storage (current BayesianOptimizer loses results on restart); enables resumable studies and cross-trial comparison
- **BayesianOptimizer (Optuna TPE)** — 10–100x more efficient than GridSearch for high-dimensional spaces; already works; just needs RDBStorage wiring
- **WalkForwardAnalyzer** — validation gate with WFE >= 50%; unchanged for v2.0
- **XGBoostRegimeModel** — 3-class volatility classifier (low/medium/high) with 19 regime features; needs hysteresis filter layer added
- **RuleStrategy DSL** — existing `all`/`any`/`none` combinators cover sub-signal logic; needs one new condition type: `vote` for K-of-N semantics
- **FeatureEngine** — already computes RSI, EMA, MACD, Bollinger, ATR; needs EMA(7), RSI(8), MACD(14,23,9), BB(7) added to DEFAULT_FEATURES for Nunchi config

**New dependencies (minimal — 2 total):**
- `hmmlearn >= 0.3.3` — optional GPU extra; HMM for regime label generation at training time only; fallback is pure XGBoost which already works
- `optuna-dashboard >= 0.20` — dev dependency; Docker sidecar with zero config against existing PostgreSQL

**Explicit exclusions:** MLflow, W&B, Ray, LangChain, backtrader, scikit-learn VotingClassifier (wrong abstraction for rule-based signals), Jinja2 explicit (already a transitive FastAPI dep).

### Expected Features

See [FEATURES.md](FEATURES.md) for full dependency graph and MVP ordering.

**Must have (table stakes for v2.0 to deliver value):**
- VotingStrategy class wrapping N RuleStrategy sub-signals with K-of-N majority vote
- Nunchi 6-signal starting configuration: momentum x2 (12h + 6h returns), EMA(7/26) crossover, RSI(8) above/below 50, MACD(14,23,9) histogram sign, Bollinger squeeze (7-period width percentile)
- Configurable vote threshold: `min_votes` parameter, default 4/6; per-regime overrides
- Composite scoring function: `sharpe * sqrt(min(trades/50, 1.0)) - dd_penalty - turnover_penalty`; hard cutoffs at <10 trades, >50% drawdown, >50% capital loss returning -999
- DSL `vote` condition type: `{"vote": {"conditions": [...], "min_votes": 4}}` in CONDITION_REGISTRY
- Optuna persistent studies: RDBStorage migration in BayesianOptimizer (one-line change)
- Exit pipeline: ATR trailing stop + RSI mean-reversion exit + signal flip, priority-ordered; ATR multiplier in optimization space

**Should have (v2.0 differentiators):**
- AutoResearch experiment loop: Celery task, program.md template generation, git commit per experiment, ablation mode
- Hysteresis regime filter: 3 consecutive same-regime bars before switching (from Nunchi regime_mm.py)
- Multi-objective Optuna: optimize Sharpe AND max_drawdown simultaneously (built into Optuna >= 3.0, trivial change)
- optuna-dashboard Docker sidecar deployment
- Optuna MedianPruner: early-stop unpromising trials (30–50% compute savings)
- Regime-to-strategy mapping: dict from regime label to min_votes and position size multiplier

**Explicit anti-features (do NOT build):**
- ML direction prediction (confirmed dead end — this is the entire reason for the v2.0 pivot)
- Pyramiding or variable position sizing (Nunchi: removing variable sizing = +1.7, the single largest gain)
- Multi-timeframe confirmation (Nunchi: "net harmful," removed in experiments 65-66)
- Funding rate overlay (Nunchi: +0.7 when removed)
- LLM-generated Python strategy code (mutable layer must be JSON, never executable code)
- Cross-asset correlation strategies (defer to v3.0)

**Defer to v2.1:** HMM-enhanced regime labeling, soft voting with confidence weighting, per-regime independent parameter optimization, adaptive search modes (explore/exploit/combine/ablation).

### Architecture Approach

See [ARCHITECTURE.md](ARCHITECTURE.md) for full component design and data flow. The three-layer autoresearch pattern is the organizing principle for all of v2.0. The fixed evaluation layer must be completely immutable during any autoresearch run — this is the most critical architectural constraint, shared independently by both Karpathy ("DO NOT modify prepare.py") and Nunchi ("Only strategy.py may be modified"). Poseidon's advantage is that the variable layer is Pydantic-validated JSON, so the agent cannot accidentally break imports, introduce syntax errors, or drift the search space beyond defined bounds.

**Major components:**
1. **VotingStrategy** — new `BaseStrategy` subclass; wraps N `RuleStrategy` sub-signals; K-of-N majority vote; confidence = votes / total sub-signals
2. **ExperimentRunner** — autoresearch orchestrator (Celery task); calls VotingStrategyFactory, BacktestRunner, Optuna, and git subprocess; no business logic
3. **VotingStrategyFactory** — stateless; creates VotingStrategy instances from Optuna trial param dicts or raw parameter dicts
4. **CompositeScorer** — pure function; Nunchi-inspired formula with hard cutoffs, trade count factor, drawdown and turnover penalties
5. **RegimeMapper** — maps XGBoostRegimeModel output (with hysteresis filter) to per-regime VotingStrategy configurations
6. **ProgramGenerator** — renders program.md template with current best params, regime state, and experiment history for the AI agent

**Key data flows:** Single experiment cycle: ProgramGenerator renders program.md → AI agent mutates RuleConfig JSON → VotingStrategyFactory creates strategy → BacktestRunner evaluates → CompositeScorer scores → WalkForwardAnalyzer validates (WFE >= 50%) → Optuna records trial → git commit if improvement. Live signal generation: Celery Beat hourly → FeatureEngine → XGBoostRegimeModel + hysteresis → RegimeMapper selects config → VotingStrategy.evaluate() → RiskEngine → Redis Streams → Thalassa.

### Critical Pitfalls

See [PITFALLS.md](PITFALLS.md) for full analysis with experiment-level evidence from Nunchi's 103-run log.

1. **Complexity accumulation disguised as improvement** (Critical, Phase 1 and Phase 4) — Every added feature creates tunable parameters that Optuna exploits on historical noise. Nunchi's evidence is unambiguous: removing 7 features = +52% score improvement. Prevention: enforce parameter budget (max 15–20 tunable params), add mandatory ablation check every 10 experiments, start with Nunchi's final architecture as a ceiling not a floor.

2. **Autoresearch loop without guardrails** (Critical, Phase 3) — Without explicit immutability enforcement, the AI agent can modify scoring formulas, reduce WFE thresholds, or create DSL conditions overfit to specific date ranges. Prevention: read-only enforcement on all Python source; only RuleConfig JSON is mutable; DSL depth limit lowered to 4 for agent-generated configs; explicit parameter bounds on every tunable field; program.md is an immutable specification.

3. **Nunchi's Sharpe 21.4 is overfit** (Critical, all phases) — 103 sequential experiments on a single 9-month dataset with no held-out test set. The reported metrics are in-sample with respect to the experimental design process. Prevention: copy the architecture (voting, ATR stops, exit priority ordering), do NOT copy the parameters; expect Sharpe 1.0–3.0 on properly validated strategies; reserve a true holdout (last 20% of data) before the first experiment runs.

4. **Regime classification is net negative without outperformance gate** (Critical, Phase 4) — Nunchi's regime_mm benchmark scored -0.322 vs simple momentum +2.724. Regime conditioning must demonstrably outperform the static voting baseline on OOS data or it does not ship. Prevention: start with rule-based regime (realized vol thresholds) before ML; require 3-bar hysteresis; default to static config when classifier confidence < 0.6; per-market regime models.

5. **DSL gap: no K-of-N semantics** (Critical, Phase 1) — The existing `all`/`any`/`none` combinators cannot express "4 of 6 signals agree." Encoding this as nested combinators enumerating C(6,4)=15 combinations is unmaintainable and breaks under parameter search. Prevention: add a `vote` condition type to CONDITION_REGISTRY with `{"vote": {"conditions": [...], "min_votes": N}}` semantics. This is a clean extension that preserves DSL architecture.

## Implications for Roadmap

The feature dependency graph forces a specific phase ordering. VotingStrategy must exist before parameter search is meaningful. Composite scoring must be locked before any Optuna study runs (scoring formula cannot change after experiments start). Holdout data must be reserved before the first experiment touches OOS windows. Regime classification is explicitly optional and must prove value against a no-regime baseline before integration.

### Phase 1: Voting Strategy Foundation

**Rationale:** All subsequent phases depend on VotingStrategy existing and being validated. This is the core strategic pivot. Nothing else can be built or optimized without it. The DSL `vote` extension must come first because the autoresearch loop generates DSL configs.

**Delivers:** VotingStrategy class, Nunchi 6-signal configuration validated via BacktestRunner and WalkForwardAnalyzer on crypto data, composite scoring function in backtest/metrics.py, DSL `vote` condition type in CONDITION_REGISTRY, exit pipeline (ATR trailing stop + RSI exit + signal flip, priority-ordered), EMA(7)/RSI(8)/MACD(14,23,9)/BB(7) added to FeatureEngine DEFAULT_FEATURES.

**Addresses (from FEATURES.md):** VotingStrategy class, Nunchi 6-signal config, configurable vote threshold, composite scoring, walk-forward integration, DSL vote extension, exit pipeline.

**Avoids (from PITFALLS.md):** DSL K-of-N gap (Pitfall 5), exit logic afterthought (Pitfall 8), feature column mismatches (Pitfall 12), parameter transfer from Nunchi (Pitfall 11 — use architecture, not parameters).

**Research flag:** Standard patterns. VotingStrategy is straightforward Python composition of existing components. Skip research-phase for this phase.

### Phase 2: Optuna Persistence + Experiment Infrastructure

**Rationale:** Persistent Optuna storage must be in place before any parameter search accumulates results worth keeping. This is a low-effort, high-leverage change. Establish the holdout data split and scoring formula here — these cannot change after experiments begin.

**Delivers:** BayesianOptimizer refactored to RDBStorage. Optuna `optuna` PostgreSQL schema isolated from Alembic. optuna-dashboard deployed as Docker sidecar. VotingStrategyFactory (parameterized creation from Optuna trial dicts). Composite score integrated as Optuna objective. Holdout data split documented and committed (last 20% reserved, never touched by optimization).

**Uses (from STACK.md):** Optuna RDBStorage (PostgreSQL, existing), optuna-dashboard Docker image (new dev dep).

**Implements:** VotingStrategyFactory, ExperimentRunner skeleton, holdout protocol.

**Avoids (from PITFALLS.md):** Data snooping via repeated optimization (Pitfall 3) — reserve holdout before ANY experiments run. Optuna table collision with Alembic (Pitfall 9) — use separate `optuna` PostgreSQL schema.

**Research flag:** Standard patterns. Optuna RDBStorage is well-documented. Skip research-phase for this phase.

### Phase 3: AutoResearch Loop

**Rationale:** The highest-value differentiator. Only meaningful after Phase 1 validation produces a working no-regime baseline and Phase 2 persistence infrastructure is ready. The three-layer pattern requires careful guardrail design — getting this wrong means all subsequent experiments are unreliable.

**Delivers:** ExperimentRunner as Celery task. program.md template generation (rendered with current best params, regime state, last 10 experiment results). Git subprocess for atomic experiment commits. Read-only enforcement on Python source (only RuleConfig JSON is agent-writable). DSL depth limit lowered to 4 for agent configs. Explicit parameter bounds per field in search space definition. Ablation mode: every 10 experiments, attempt removal of each non-core signal. Automated daily summary (best candidate, WFE score, parameter stability). Kill switch feature flag.

**Uses (from STACK.md):** Celery + Redis (existing), git subprocess (stdlib), Optuna RDBStorage (Phase 2).

**Implements:** Full autoresearch three-layer pattern. Fixed layer locked. Variable layer = RuleConfig JSON only. Guide layer = rendered program.md.

**Avoids (from PITFALLS.md):** Agent modifying fixed layers (Pitfall 7). Loop without guardrails (Pitfall 2). Survivor bias (Pitfall 10) — log all trials including rejects; periodic re-evaluation against original baseline every 20 experiments. Single developer overload (Pitfall 13) — automated monitoring, not manual checking.

**Research flag:** Needs research before implementation. Review Karpathy's program.md and Nunchi's program.md verbatim before writing Poseidon's version. The guardrail design (what the agent can and cannot modify, how parameter bounds are specified) has nuances not fully resolved in this research.

### Phase 4: Regime Classification (Optional, Gated)

**Rationale:** Research shows regime classification is net negative without proper gating. Build last, after the static VotingStrategy has proven real results. Structured as an optional overlay that must outperform the no-regime baseline on OOS data before being deployed.

**Delivers:** Rule-based regime baseline (realized vol thresholds) validated first. RegimeMapper with 3-bar hysteresis filter. Per-regime min_votes and position multiplier configuration. Outperformance gate: regime-conditional system must beat static VotingStrategy on OOS data or is not deployed. Per-market regime models (crypto and stock volatility structures differ). Default-to-static fallback when classifier confidence < 0.6. hmmlearn integration only if XGBoost regime is insufficient and justified.

**Uses (from STACK.md):** XGBoostRegimeModel (existing), hmmlearn (optional).

**Implements:** RegimeMapper, hysteresis filter, rule-based vs ML regime comparison, per-market model training.

**Avoids (from PITFALLS.md):** Regime classifier as fragility bottleneck (Pitfall 4). Multi-market parameter transfer (Pitfall 6) — per-market isolation from the beginning. Complexity accumulation (Pitfall 1) — regime conditioning is only added if it proves value.

**Research flag:** Needs research before implementation. Per-market regime model calibration (crypto vs TW stocks vs US stocks) and the choice between rule-based and ML regime definitions are non-trivial. Consider Combinatorial Purged Cross-Validation for regime model validation.

### Phase Ordering Rationale

- VotingStrategy before parameter search: you cannot optimize what you have not built and validated.
- Composite scoring locked in Phase 1: the scoring formula cannot change after experiments start (it is the fixed evaluation layer).
- Holdout reserved in Phase 2 before any experiments run: OOS data is compromised the moment optimization observes it; this cannot be undone retroactively.
- AutoResearch (Phase 3) before regime conditioning (Phase 4): the static no-regime baseline must exist and be validated before regime conditioning has anything meaningful to beat.
- Ship incrementally (single developer): each phase deployed and running in production before the next begins. Kill switch on every phase.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (AutoResearch Loop):** Guardrail design — how to structure program.md constraints for JSON-config-based autoresearch (different from Karpathy/Nunchi's code-based approach), what DSL parameter bounds to enforce per market/timeframe, how to detect agent drift. Review Karpathy program.md and Nunchi program.md verbatim before writing Poseidon's version.
- **Phase 4 (Regime Classification):** Per-market calibration — crypto vs TW stocks have structurally different volatility profiles. Verify that Poseidon's 19 REGIME_FEATURES are appropriate for non-crypto markets. Determine whether rule-based or ML regime is sufficient first.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Voting Strategy):** VotingStrategy composition is well-understood Python. DSL extension follows existing CONDITION_REGISTRY pattern. FeatureEngine additions are additive with no breaking changes.
- **Phase 2 (Optuna Persistence):** RDBStorage is the official Optuna pattern with complete documentation. Docker sidecar for optuna-dashboard is a standard deployment.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Poseidon pyproject.toml directly verified. Only 2 new packages, both optional. Existing stack confirmed against all v2.0 feature requirements. |
| Features | HIGH | Feature set derived from Nunchi's 103-experiment empirical record. Anti-features are evidence-backed with specific experiment numbers, not opinion. |
| Architecture | HIGH | Three-layer pattern confirmed independently in Karpathy and Nunchi. VotingStrategy is a direct code-level translation from Nunchi strategy.py. JSON variable layer is a structural improvement over both reference implementations. |
| Pitfalls | HIGH | All critical pitfalls grounded in specific experiment numbers from Nunchi STRATEGIES.md or Karpathy's explicit constraints. Not speculative — directly observed in reference projects. |

**Overall confidence:** HIGH

### Gaps to Address

- **Per-market parameter bounds:** Nunchi's parameters (RSI=8, EMA=7/26, momentum threshold=0.012, ATR=5.5x) are hourly crypto-specific. Poseidon supports TW stocks, TW futures, US stocks, and crypto. Parameter search bounds for non-crypto markets must be defined empirically before Phase 4. Start crypto-only, then expand.

- **Exit pipeline design:** Poseidon's RuleStrategy DSL has no concept of trailing stops that track peak price per position. ATR trailing stops require bar-by-bar state tracking in BacktestRunner. This is a BacktestRunner concern, not a DSL concern. The scope of BacktestRunner changes needed must be defined before Phase 1 implementation begins.

- **Holdout period definition:** The exact data split point and which OHLCV datasets are in scope must be documented and committed before Phase 2 begins. This decision is irreversible — once any experiment observes an OOS window, that window is compromised for true holdout testing.

- **Optuna schema isolation:** Optuna RDBStorage auto-creates tables that bypass Alembic. The `optuna` PostgreSQL schema approach must be verified against Poseidon's existing Alembic migration setup before committing to it. Test in CI.

- **Realistic Sharpe expectations per market:** Sharpe 1.0–3.0 on properly validated strategies is a reasonable target. Per-market minimum acceptable thresholds (what constitutes a deployable strategy) must be defined before the autoresearch loop has meaningful stop criteria. Otherwise the loop runs indefinitely.

- **hmmlearn Python 3.13 compatibility:** Poseidon targets Python >= 3.12. hmmlearn 0.3.3 has Python 3.12 wheels. Python 3.13 support needs verification. Has a clean fallback: pure XGBoost regime (already works).

## Sources

### Primary (HIGH confidence)
- [Nunchi auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading) — strategy.py (6-signal voting, MIN_VOTES=4, BB squeeze), STRATEGIES.md (103-experiment evolution log with specific scores), prepare.py (composite scoring formula with hard cutoffs), regime_mm.py (4-regime hysteresis pattern), program.md (autoresearch constraints)
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) — three-layer pattern (prepare.py/train.py/program.md), guardrail design, immutability enforcement, simplicity criterion, results.tsv format
- Poseidon v1.0 codebase — pyproject.toml, backtest/optimizer.py, backtest/walk_forward.py, backtest/metrics.py, ml/implementations/xgboost_regime.py, strategies/dsl/, strategies/rule_strategy.py, strategies/base.py, data/feature_engine.py

### Secondary (MEDIUM confidence)
- [n-autoresearch](https://github.com/iii-hq/n-autoresearch) — adaptive search modes (explore/exploit/combine/ablation), structured experiment tracking with KV store
- [Optuna RDB Storage docs](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html) — PostgreSQL backend configuration, study persistence, distributed optimization
- [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) — real-time visualization, PostgreSQL support, Docker deployment
- [Regime-switching XGBoost (2025)](https://arxiv.org/html/2510.03236v1) — XGBoost+HMM hybrid for regime classification

### Tertiary (LOW confidence — validate during implementation)
- [GT-Score MDPI 2025](https://www.mdpi.com/1911-8074/19/1/60) — multiple-testing correction for walk-forward strategies; theoretical backing for trial count limits
- [Statistical Jump Model arXiv](https://arxiv.org/html/2402.05272v2) — jump penalty at regime transitions; informs hysteresis filter design
- [Volatility Regime Detection Dozen Diamonds](https://www.dozendiamonds.com/volatility-regime-shifting/) — transition period risk characterization
- [Walk-Forward Optimization Guide AlgoTrading101](https://algotrading101.com/learn/walk-forward-optimization/) — IS/OOS snooping risk documentation

---
*Research completed: 2026-03-25*
*Ready for roadmap: yes*
