# Phase 6: Backtest Engine - Research

**Researched:** 2026-03-21
**Domain:** Backtesting engine (event-driven simulation, cost models, walk-forward analysis, parameter optimization)
**Confidence:** HIGH

## Summary

Phase 6 builds the backtest engine for Poseidon. The central design constraint is pipeline reuse: the backtest runner must invoke the exact same `FeatureEngine.compute_from_df()` + `Strategy.evaluate()` + `RiskEngine.evaluate()` code path as live prediction, ensuring no train-serve skew. The engine simulates trades bar-by-bar over historical OHLCV data, applying market-specific fee schedules and configurable slippage via a virtual portfolio simulator. Walk-forward analysis validates strategy robustness by rolling train/validate/test windows and computing Walk-Forward Efficiency (WFE). Parameter optimization is handled via grid search (for small parameter spaces) and Optuna-based Bayesian optimization (TPE sampler for larger spaces). All results -- backtest metadata, individual trades, and equity curves -- are stored in separate queryable PostgreSQL tables (not JSONB blobs) to enable cross-backtest comparison queries.

The codebase already provides all upstream components: `FeatureEngine` with `compute_from_df()` for in-memory feature computation (no DB round-trip during replay), `BaseStrategy`/`ModelStrategy`/`RuleStrategy` with a unified `evaluate(features) -> list[Signal]` interface, `RiskEngine.evaluate(signal, portfolio)` for risk checks, `VirtualPortfolio` for position tracking, and `Signal` Pydantic schema with status tracking. The backtest module (`src/poseidon/backtest/__init__.py`) exists as an empty placeholder ready for implementation. The next Alembic migration will be 005.

**Primary recommendation:** Build an event-driven (bar-by-bar) backtest runner that iterates over historical rows, calling the strategy's `evaluate()` on expanding-window feature DataFrames, piping resulting signals through risk checks, and recording fills in a `BacktestPortfolio` (a backtest-specific subclass/wrapper around the virtual portfolio concept that tracks PnL, fees, and slippage). Use Optuna 4.8 for Bayesian optimization with TPE sampler. Store results in three new tables: `backtests` (metadata + JSONB metrics), `backtest_trades` (one row per trade), and `backtest_equity` (one row per bar). Create Alembic migration 005.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BT-01 | Backtest engine shares exact same FeatureEngine + Strategy + Risk pipeline as live prediction | BacktestRunner calls `FeatureEngine.compute_from_df()`, `strategy.evaluate(features_slice)`, `risk_engine.evaluate(signal, portfolio)` -- identical code paths, no backtest-only logic |
| BT-02 | Virtual portfolio simulator with configurable fees and slippage per market (TW stock tax, crypto maker/taker, etc.) | `CostModel` dataclass with per-market fee schedules; `BacktestPortfolio` applies fees and slippage on each simulated fill; design spec provides exact fee rates |
| BT-03 | Walk-forward analysis with WFE >= 50% pass criteria and minimum 30 trades per OOS segment | `WalkForwardAnalyzer` with configurable rolling windows (default: 252/63/63/63 trading days); WFE = annualized OOS return / annualized IS return; flag strategies with WFE < 50% or trades < 30 per OOS window |
| BT-04 | Parameter optimization via Grid Search and Bayesian Optimization | `GridSearchOptimizer` for exhaustive search over small param grids; `BayesianOptimizer` wrapping Optuna 4.8 TPE sampler for larger spaces; both return ranked parameter sets with metrics |
| BT-05 | Backtest trades and equity curves stored in separate queryable tables (not JSONB blobs) | Three new tables: `backtests`, `backtest_trades`, `backtest_equity` with proper column types, FKs, and indexes; Alembic migration 005 |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 3.0.1 (installed) | DataFrame manipulation for bar-by-bar replay, metrics computation | Already used throughout project; natural fit for time-series iteration |
| numpy | 2.4.3 (installed) | Numerical computations (returns, drawdown, Sharpe) | Already installed; required for vectorized metric calculations |
| sqlalchemy | 2.0.48 (installed) | ORM for backtest result tables | Already used for all persistence in the project |
| alembic | 1.18 (installed) | Migration 005 for new backtest tables | Already used; next migration number is 005 |
| optuna | 4.8.0 | Bayesian optimization (TPE sampler) for parameter search | Industry standard; lightweight (only pulls colorlog+tqdm as new deps); already depends on SQLAlchemy and alembic which are installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydantic | (via fastapi, installed) | Config validation for backtest params, cost models | Validate backtest request inputs |
| itertools | (stdlib) | `product()` for grid search parameter combinations | Grid search implementation |
| dataclasses | (stdlib) | CostModel, BacktestConfig, TradeRecord value objects | Lightweight structured data without ORM overhead |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Optuna TPE | scikit-optimize (skopt) | skopt is abandoned/unmaintained since 2021; Optuna is actively maintained with 4.8 release in March 2026 |
| Optuna TPE | Custom random search | Misses the point -- Bayesian search converges faster on large spaces |
| Custom bar-by-bar engine | vectorbt / backtesting.py | External frameworks impose their own data model and signal format; Poseidon's pipeline-reuse requirement (BT-01) prohibits external backtest logic |
| Per-trade PnL tracking | vectorized returns | Bar-by-bar is required for proper risk engine integration and realistic fill simulation |

**Installation:**
```bash
pip install "optuna>=4.8"
```

Add to `pyproject.toml` dependencies:
```toml
"optuna>=4.8",
```

**Version verification:** Optuna 4.8.0 confirmed available via pip (dry-run verified 2026-03-21). Only new transitive dependencies are `colorlog` and `tqdm`.

## Architecture Patterns

### Recommended Project Structure
```
src/poseidon/backtest/
    __init__.py            # Public API exports
    runner.py              # BacktestRunner -- bar-by-bar event loop
    portfolio.py           # BacktestPortfolio -- tracks trades, PnL, fees, equity
    cost_model.py          # CostModel dataclass + COST_MODELS registry
    metrics.py             # compute_metrics() -- Sharpe, drawdown, win rate, etc.
    walk_forward.py        # WalkForwardAnalyzer -- rolling window orchestration
    optimizer.py           # GridSearchOptimizer + BayesianOptimizer
    schemas.py             # Pydantic schemas for backtest config/request/response
    repository.py          # BacktestRepository -- DB persistence for results
src/poseidon/models/
    backtest.py            # ORM: BacktestRecord, BacktestTradeRecord, BacktestEquityRecord
alembic/versions/
    005_create_backtest_tables.py
tests/
    test_backtest.py       # Unit tests for runner, portfolio, cost model
    test_walk_forward.py   # Walk-forward analysis tests
    test_optimizer.py      # Grid search + Bayesian optimization tests
```

### Pattern 1: Bar-by-Bar Event Loop with Pipeline Reuse
**What:** The backtest runner iterates over historical bars one at a time, building an expanding window of feature data, calling `strategy.evaluate()` on the accumulated features (the strategy only looks at the last row anyway, but gets full history for indicator computation), then piping signals through the risk engine.
**When to use:** Always -- this is the core backtest loop.
**Example:**
```python
class BacktestRunner:
    def __init__(
        self,
        strategy: BaseStrategy,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
    ):
        self.strategy = strategy
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.portfolio = BacktestPortfolio(initial_capital, cost_model)

    def run(self, ohlcv: pd.DataFrame) -> BacktestResult:
        """Run backtest over historical OHLCV data.

        1. Compute features once on full OHLCV (same FeatureEngine code path)
        2. Iterate bar by bar from warmup period onward
        3. For each bar: strategy.evaluate(features[:i+1]) -> signals
        4. For each signal: risk_engine.evaluate(signal, portfolio) -> pass/reject
        5. If passed: portfolio.execute_fill(signal, current_bar)
        6. Record equity curve point
        """
        features = self.feature_engine.compute_from_df(ohlcv)
        # ... bar-by-bar iteration
```

### Pattern 2: CostModel as a Per-Market Configuration
**What:** Encapsulate fee schedules and slippage rules per market as immutable dataclasses. A registry maps market names to cost models.
**When to use:** Every fill in the backtest portfolio.
**Example:**
```python
@dataclass(frozen=True)
class CostModel:
    """Market-specific transaction cost model."""
    market: str
    buy_commission_rate: float    # e.g., 0.001425 for TW stock
    sell_commission_rate: float   # e.g., 0.001425 for TW stock
    tax_rate: float              # e.g., 0.003 for TW stock sell tax
    slippage_pct: float          # e.g., 0.0005 for crypto
    slippage_ticks: float        # e.g., 1.0 for TW stock (1 tick)
    description: str = ""

COST_MODELS: dict[str, CostModel] = {
    "tw_stock": CostModel(
        market="tw_stock",
        buy_commission_rate=0.001425,
        sell_commission_rate=0.001425,
        tax_rate=0.003,       # 0.3% sell tax (stock)
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW stock: 0.1425% commission + 0.3% sell tax + 1 tick slippage",
    ),
    "tw_stock_etf": CostModel(
        market="tw_stock_etf",
        buy_commission_rate=0.001425,
        sell_commission_rate=0.001425,
        tax_rate=0.001,       # 0.1% sell tax (ETF)
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW ETF: 0.1425% commission + 0.1% sell tax + 1 tick slippage",
    ),
    "tw_stock_daytrade": CostModel(
        market="tw_stock_daytrade",
        buy_commission_rate=0.001425,
        sell_commission_rate=0.001425,
        tax_rate=0.0015,      # 0.15% sell tax (day trade)
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW day trade: 0.1425% commission + 0.15% sell tax + 1 tick slippage",
    ),
    "tw_futures": CostModel(
        market="tw_futures",
        buy_commission_rate=0.0,  # flat fee handled separately
        sell_commission_rate=0.0,
        tax_rate=0.00002,     # futures tax is negligible
        slippage_pct=0.0,
        slippage_ticks=1.0,   # 1 point
        description="TW futures: ~$50/contract round trip + 1 point slippage",
    ),
    "us_stock": CostModel(
        market="us_stock",
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.0,
        slippage_pct=0.0,
        slippage_ticks=0.01,  # $0.01
        description="US stock: commission-free + $0.01 slippage",
    ),
    "crypto_spot": CostModel(
        market="crypto_spot",
        buy_commission_rate=0.001,   # 0.1% maker
        sell_commission_rate=0.001,  # 0.1% taker
        tax_rate=0.0,
        slippage_pct=0.0005,         # 0.05%
        slippage_ticks=0.0,
        description="Crypto spot: 0.1% maker/taker + 0.05% slippage",
    ),
    "crypto_perp": CostModel(
        market="crypto_perp",
        buy_commission_rate=0.0002,   # 0.02% maker
        sell_commission_rate=0.0005,  # 0.05% taker
        tax_rate=0.0,
        slippage_pct=0.0005,          # 0.05%
        slippage_ticks=0.0,
        description="Crypto perp: 0.02% maker / 0.05% taker + 0.05% slippage",
    ),
}
```

### Pattern 3: Walk-Forward Rolling Windows
**What:** Split historical data into overlapping train/validate/test windows, optimize on train, validate (optional), then test on out-of-sample data. Aggregate OOS results and compute WFE.
**When to use:** BT-03 -- walk-forward analysis.
**Example:**
```python
@dataclass
class WalkForwardConfig:
    train_days: int = 252       # ~1 year
    validate_days: int = 63     # ~3 months (optional, for model strategies)
    test_days: int = 63         # ~3 months OOS
    step_days: int = 63         # roll forward by 1 quarter
    min_trades_per_oos: int = 30
    min_wfe: float = 0.50       # 50%

class WalkForwardAnalyzer:
    def analyze(
        self,
        strategy: BaseStrategy,
        ohlcv: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        """Run walk-forward analysis with rolling windows.

        For each window:
        1. Train/optimize on IS data (for model strategies: retrain model)
        2. Test on OOS data using BacktestRunner
        3. Record IS and OOS metrics per window
        After all windows:
        4. Compute WFE = annualized_oos_return / annualized_is_return
        5. Flag if WFE < min_wfe or trades < min_trades_per_oos
        """
```

### Pattern 4: Optuna-Based Bayesian Optimization
**What:** Wrap Optuna's `study.optimize()` with TPE sampler around the backtest runner. Each trial suggests parameter values, runs a backtest, returns a metric (e.g., Sharpe ratio).
**When to use:** BT-04 -- when parameter space is too large for grid search.
**Example:**
```python
import optuna

class BayesianOptimizer:
    def optimize(
        self,
        strategy_factory,  # Callable that creates strategy from params
        ohlcv: pd.DataFrame,
        param_space: dict,      # {param_name: (low, high, type)}
        n_trials: int = 100,
        metric: str = "sharpe",
    ) -> list[dict]:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, (low, high, ptype) in param_space.items():
                if ptype == "int":
                    params[name] = trial.suggest_int(name, low, high)
                elif ptype == "float":
                    params[name] = trial.suggest_float(name, low, high)
            strategy = strategy_factory(params)
            result = BacktestRunner(...).run(ohlcv)
            return result.metrics[metric]

        study.optimize(objective, n_trials=n_trials)
        return self._ranked_results(study)
```

### Anti-Patterns to Avoid
- **Separate backtest-only logic:** Never duplicate feature computation, strategy evaluation, or risk checks for backtest. Always call the same code path.
- **Look-ahead bias:** The bar-by-bar loop MUST only provide `features[:i+1]` to the strategy (data up to and including the current bar). Never pass future data.
- **Storing trades as JSONB:** The design spec explicitly requires separate queryable tables. Do not store trade arrays in a JSONB column on the backtests table.
- **Ignoring warmup period:** Feature indicators (SMA-60, RSI-14, etc.) need warmup rows. The runner must skip signals during the warmup period to avoid NaN-based signals.
- **Equity tracking at trade level only:** Equity curve must be recorded per-bar (not per-trade) to properly compute drawdown and visualize the curve.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bayesian parameter optimization | Custom BO with GP surrogate | Optuna 4.8 TPE sampler | Optuna handles trial pruning, parameter suggestion, result storage; building a GP surrogate is a research project |
| Sharpe ratio calculation | Manual loop | Vectorized `(mean(returns) / std(returns)) * sqrt(252)` | Standard formula; avoid off-by-one errors in annualization |
| Max drawdown | Iterative tracking | `(cummax - equity) / cummax` vectorized with pandas | Vectorized pandas is correct and fast; hand-rolled loops are error-prone |
| Grid search combinatorics | Nested for-loops | `itertools.product()` over param ranges | Cleaner, handles arbitrary dimensions, avoids nested loop bugs |

**Key insight:** The backtest engine's complexity lies in the event loop correctness (no look-ahead, proper warmup, accurate cost application), not in fancy algorithms. Use standard formulas for metrics and Optuna for optimization -- the custom work is in the runner and portfolio simulator.

## Common Pitfalls

### Pitfall 1: Look-Ahead Bias
**What goes wrong:** Strategy receives future data during backtesting, inflating performance metrics.
**Why it happens:** Passing the full feature DataFrame to `strategy.evaluate()` instead of slicing to current bar.
**How to avoid:** The runner must pass `features.iloc[:i+1]` (or equivalently, slice the DataFrame up to and including the current index). The strategy's `evaluate()` method already looks at the last row, but it needs the full historical context for indicator computation.
**Warning signs:** Unrealistically high Sharpe ratios (>5), near-100% win rates, suspiciously good results.

### Pitfall 2: Feature Warmup Period
**What goes wrong:** Signals generated during the first N bars have NaN feature values (SMA-60 needs 60 bars of data).
**Why it happens:** Not skipping the initial warmup bars where indicator values are undefined.
**How to avoid:** Determine the maximum lookback period from the feature specs (e.g., SMA-60 requires 60 bars). Skip signal generation for rows where critical features are NaN. The runner should start signal evaluation only after `warmup_bars = max(lookback_periods)` rows.
**Warning signs:** NaN values in feature columns during early bars, unexpected signals in the first few trading days.

### Pitfall 3: Fee and Tax Calculation Errors
**What goes wrong:** TW stock sell tax is applied to both buys and sells, or ETF tax rate is used for regular stocks.
**Why it happens:** Not separating buy-side and sell-side cost components.
**How to avoid:** CostModel explicitly separates `buy_commission_rate`, `sell_commission_rate`, and `tax_rate` (tax is always on the sell side for TW stocks). Day-trade tax rate (0.15%) requires knowing if a position was opened and closed same-day -- the portfolio must track intraday round-trips.
**Warning signs:** Backtest PnL significantly differs from manual calculation on a few sample trades.

### Pitfall 4: Walk-Forward Window Alignment
**What goes wrong:** Windows overlap incorrectly or leave gaps, resulting in data leakage or untested periods.
**Why it happens:** Off-by-one errors in date/index slicing when constructing rolling windows.
**How to avoid:** Use exclusive-end semantics consistently: `train = data[start:start+train_days]`, `test = data[start+train_days:start+train_days+test_days]`. Validate that windows are contiguous and non-overlapping for the test segments.
**Warning signs:** Test data appearing in training data, gaps in equity curve, mismatch between sum of OOS periods and total data length.

### Pitfall 5: Optuna Study Overhead
**What goes wrong:** Bayesian optimization is slow because each trial runs a full backtest.
**Why it happens:** Not pruning unpromising trials early; running too many trials on large datasets.
**How to avoid:** Use Optuna's `MedianPruner` to stop trials that are clearly underperforming. Start with a smaller time range for initial exploration, then validate top candidates on the full range. Set reasonable `n_trials` limits (50-200 for most strategies).
**Warning signs:** Optimization taking hours for a simple strategy, all trials returning similar metrics.

### Pitfall 6: Equity Curve Granularity
**What goes wrong:** Equity curve only has entries when trades occur, making drawdown calculations inaccurate.
**Why it happens:** Only recording equity on trade events, not on every bar.
**How to avoid:** Record equity at every bar (mark-to-market with current close price). This captures drawdowns between trades and produces smooth equity curves for visualization.
**Warning signs:** Jagged equity curves, drawdown values that don't match visual inspection.

## Code Examples

### Backtest Result Metrics Computation
```python
# Source: Standard quantitative finance formulas
def compute_metrics(equity_series: pd.Series, trades: list[TradeRecord]) -> dict:
    """Compute standard backtest performance metrics."""
    returns = equity_series.pct_change().dropna()
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    n_years = len(equity_series) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe ratio (risk-free rate = 0 for simplicity)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    cummax = equity_series.cummax()
    drawdown = (cummax - equity_series) / cummax
    max_drawdown = drawdown.max()

    # Calmar ratio
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0

    # Trade-level metrics
    if trades:
        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trade_count": len(trades),
    }
```

### WFE Calculation
```python
# Source: Design spec + industry standard formula
def compute_wfe(is_ann_return: float, oos_ann_return: float) -> float:
    """Walk-Forward Efficiency = OOS annualized return / IS annualized return.

    WFE >= 50% suggests the strategy is not overfitted.
    Returns 0.0 if IS return is zero or negative.
    """
    if is_ann_return <= 0:
        return 0.0
    return oos_ann_return / is_ann_return
```

### DB Schema for Backtest Results
```python
# Source: Design spec backtest_trades and backtest_equity table definitions
class BacktestRecord(Base):
    __tablename__ = "backtests"

    id = mapped_column(UUID(as_uuid=True), primary_key=True,
                       server_default=func.gen_random_uuid())
    strategy_id = mapped_column(UUID(as_uuid=True), nullable=True)
    strategy_type = mapped_column(String(16), nullable=False)  # "model" or "rule"
    symbol = mapped_column(String(32), nullable=False)
    market = mapped_column(String(32), nullable=False)
    interval = mapped_column(String(8), nullable=False, server_default="1d")
    config = mapped_column(JSONB, nullable=False)       # backtest params
    metrics = mapped_column(JSONB, nullable=True)        # performance metrics
    walk_forward = mapped_column(JSONB, nullable=True)   # per-window metrics
    status = mapped_column(String(16), nullable=False, server_default="running")
    error_message = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True),
                               server_default=func.now())
    completed_at = mapped_column(DateTime(timezone=True), nullable=True)

class BacktestTradeRecord(Base):
    __tablename__ = "backtest_trades"

    id = mapped_column(UUID(as_uuid=True), primary_key=True,
                       server_default=func.gen_random_uuid())
    backtest_id = mapped_column(UUID(as_uuid=True),
                                ForeignKey("backtests.id"), nullable=False)
    symbol = mapped_column(String(32), nullable=False)
    action = mapped_column(String(16), nullable=False)
    entry_time = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time = mapped_column(DateTime(timezone=True), nullable=True)
    entry_price = mapped_column(Numeric, nullable=False)
    exit_price = mapped_column(Numeric, nullable=True)
    quantity = mapped_column(Numeric, nullable=False)
    pnl = mapped_column(Numeric, nullable=True)
    fees = mapped_column(Numeric, nullable=False, server_default="0")
    metadata_ = mapped_column("metadata", JSONB, nullable=False,
                              server_default="{}")

class BacktestEquityRecord(Base):
    __tablename__ = "backtest_equity"

    backtest_id = mapped_column(UUID(as_uuid=True),
                                ForeignKey("backtests.id"), primary_key=True)
    time = mapped_column(DateTime(timezone=True), primary_key=True)
    equity = mapped_column(Numeric, nullable=False)
    drawdown = mapped_column(Numeric, nullable=False, server_default="0")
```

### BacktestPortfolio Fill Execution
```python
# Source: Project-specific design combining VirtualPortfolio pattern with cost model
class BacktestPortfolio:
    def execute_fill(
        self,
        signal: Signal,
        bar: pd.Series,  # Current OHLCV bar
    ) -> TradeRecord | None:
        """Execute a simulated fill with fees and slippage.

        Args:
            signal: Passed signal from risk engine.
            bar: Current OHLCV bar (time, open, high, low, close, volume).

        Returns:
            TradeRecord for the fill, or None if fill cannot be executed.
        """
        price = float(bar["close"])

        # Apply slippage
        if self.cost_model.slippage_pct > 0:
            if signal.action == SignalAction.LONG:
                price *= (1 + self.cost_model.slippage_pct)
            elif signal.action in (SignalAction.SHORT, SignalAction.CLOSE):
                price *= (1 - self.cost_model.slippage_pct)
        if self.cost_model.slippage_ticks > 0:
            # For tick-based slippage, need tick_size config per market
            pass

        # Apply fees
        trade_value = price * quantity
        if signal.action == SignalAction.LONG:
            fees = trade_value * self.cost_model.buy_commission_rate
        else:
            fees = (trade_value * self.cost_model.sell_commission_rate
                    + trade_value * self.cost_model.tax_rate)

        # Update equity and positions
        self.cash -= (trade_value + fees)
        # ... record trade
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vectorized backtesting (zipline-style) | Event-driven bar-by-bar with risk integration | 2020+ | Event-driven allows realistic risk engine integration and complex position management |
| Manual grid search loops | Optuna TPE Bayesian optimization | Optuna 1.0 (2019), mature at 4.8 (2026) | 10-50x faster convergence on large parameter spaces |
| scikit-optimize for Bayesian search | Optuna | skopt abandoned ~2021 | Optuna is actively maintained, better API, pruning support |
| JSONB blob for backtest results | Separate queryable tables | N/A (design choice) | Enables SQL-level cross-backtest comparison without application-layer parsing |

**Deprecated/outdated:**
- **scikit-optimize (skopt):** Last release 0.9.0 in October 2021, effectively abandoned. Use Optuna instead.
- **zipline:** Original Quantopian zipline is archived. Community fork exists but imposes its own data model which conflicts with Poseidon's pipeline-reuse requirement.
- **backtrader:** Still maintained but heavyweight; forces its own event system, incompatible with shared pipeline requirement (BT-01).

## Open Questions

1. **Day trade detection for TW stock tax**
   - What we know: TW day-trade tax is 0.15% (vs 0.3% regular). The cost model needs to know if a position was opened and closed on the same day.
   - What's unclear: Should day-trade detection be automatic (portfolio tracks entry date and compares to exit date) or configured as a separate market type?
   - Recommendation: Use separate cost model keys (`tw_stock` vs `tw_stock_daytrade`). The strategy or API caller specifies which cost model to use. Automatic detection adds complexity for minimal value.

2. **Walk-forward retraining for model strategies**
   - What we know: Walk-forward analysis requires retraining the model on each new train window. For rule strategies, parameter optimization replaces retraining.
   - What's unclear: Should the walk-forward analyzer call `model.train()` directly, or should it create new model versions?
   - Recommendation: For backtest walk-forward, train in-memory without creating DB model versions. Walk-forward is an evaluation tool, not a production training pipeline. Keep it lightweight.

3. **Tick-based slippage implementation**
   - What we know: TW stock uses tick-based slippage (1 tick), but tick sizes vary by price range (e.g., TWD 0.01 for <10, TWD 0.05 for 10-50, etc.).
   - What's unclear: Should we implement the full TWSE tick size table, or use a simplified percentage approximation?
   - Recommendation: Start with percentage-based slippage approximation for all markets. Add tick-size table as a future enhancement if backtesting accuracy requires it. The design spec already shows both `slippage_pct` and `slippage_ticks` fields, so the data model supports both.

4. **Backtest runner parallelization for optimization**
   - What we know: Parameter optimization (especially grid search) runs many independent backtests. Celery CPU workers exist.
   - What's unclear: Should optimization trials run as Celery tasks, or in-process with Optuna's built-in parallelism?
   - Recommendation: Start with in-process sequential execution (Optuna single-threaded). The optimization itself will be triggered as a single Celery task. Add Celery-parallel trials only if performance is insufficient.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x (installed) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/test_backtest.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BT-01 | Runner uses same FeatureEngine + Strategy.evaluate() + RiskEngine code path | unit | `python -m pytest tests/test_backtest.py::test_pipeline_reuse -x` | Wave 0 |
| BT-01 | No backtest-only logic divergence | unit | `python -m pytest tests/test_backtest.py::test_no_backtest_only_logic -x` | Wave 0 |
| BT-02 | TW stock fee schedule (commission + 0.3% tax) applied correctly | unit | `python -m pytest tests/test_backtest.py::test_tw_stock_cost_model -x` | Wave 0 |
| BT-02 | Crypto maker/taker fees applied correctly | unit | `python -m pytest tests/test_backtest.py::test_crypto_cost_model -x` | Wave 0 |
| BT-02 | Slippage applied in correct direction | unit | `python -m pytest tests/test_backtest.py::test_slippage_direction -x` | Wave 0 |
| BT-03 | Walk-forward windows are contiguous and non-overlapping for OOS | unit | `python -m pytest tests/test_walk_forward.py::test_window_generation -x` | Wave 0 |
| BT-03 | WFE calculated correctly and flags strategies below 50% | unit | `python -m pytest tests/test_walk_forward.py::test_wfe_calculation -x` | Wave 0 |
| BT-03 | Strategies with < 30 trades per OOS flagged | unit | `python -m pytest tests/test_walk_forward.py::test_min_trades_flag -x` | Wave 0 |
| BT-04 | Grid search produces all parameter combinations | unit | `python -m pytest tests/test_optimizer.py::test_grid_search -x` | Wave 0 |
| BT-04 | Bayesian optimization returns ranked results | unit | `python -m pytest tests/test_optimizer.py::test_bayesian_optimization -x` | Wave 0 |
| BT-05 | BacktestRecord, BacktestTradeRecord, BacktestEquityRecord persist correctly | unit | `python -m pytest tests/test_backtest.py::test_db_persistence -x` | Wave 0 |
| BT-05 | Cross-backtest comparison query works | unit | `python -m pytest tests/test_backtest.py::test_cross_backtest_query -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_backtest.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_backtest.py` -- covers BT-01, BT-02, BT-05 (runner, portfolio, cost model, DB persistence)
- [ ] `tests/test_walk_forward.py` -- covers BT-03 (walk-forward analysis, WFE)
- [ ] `tests/test_optimizer.py` -- covers BT-04 (grid search, Bayesian optimization)

*(Existing test infrastructure covers framework setup. `conftest.py` exists with test settings.)*

## Sources

### Primary (HIGH confidence)
- Project design spec: `/Users/jordanchen/Workspace/Projects/aquarium/docs/poseidon-design.md` -- backtest engine section with cost models, walk-forward config, DB schema
- Project codebase: `src/poseidon/` -- FeatureEngine, BaseStrategy, RiskEngine, VirtualPortfolio, Signal schema, SQLAlchemy models
- Optuna 4.8.0 -- verified via `pip install --dry-run`, released 2026-03-16

### Secondary (MEDIUM confidence)
- [Optuna official site](https://optuna.org/) -- TPE sampler, study API, pruning
- [QuantInsti walk-forward optimization guide](https://blog.quantinsti.com/walk-forward-optimization-introduction/) -- WFE formula verification
- [PyQuantNews walk-forward analysis](https://www.pyquantnews.com/free-python-resources/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis) -- implementation patterns

### Tertiary (LOW confidence)
- None -- all findings verified against primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries either already installed or verified via pip; Optuna is the only new dependency
- Architecture: HIGH -- design spec provides detailed schemas and flow; codebase provides all upstream interfaces to integrate with
- Pitfalls: HIGH -- standard backtesting pitfalls are well-documented in quantitative finance literature; verified against design spec requirements

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable domain, no fast-moving dependencies)
