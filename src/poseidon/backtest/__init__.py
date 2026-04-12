"""Backtest engine: portfolio simulation, cost models, metrics, runner, walk-forward, and optimization.

Public API:
    BacktestPortfolio    - Portfolio simulator with fee/slippage
    CostModel            - Market-specific cost configuration
    COST_MODELS          - Registry of all market cost models
    get_cost_model       - Lookup cost model by market name
    TradeRecord          - Completed trade data
    compute_metrics      - Performance metrics from equity + trades
    BacktestRunner       - Bar-by-bar event loop with pipeline reuse
    BacktestConfig       - Pydantic config schema for backtest runs
    BacktestResult       - Pydantic result schema for backtest output
    BacktestRepository   - DB persistence for backtest results
    WalkForwardConfig    - Walk-forward analysis configuration
    WalkForwardResult    - Walk-forward analysis result
    WalkForwardAnalyzer  - Walk-forward analysis engine
    compute_wfe          - Walk-Forward Efficiency computation
    GridSearchOptimizer  - Exhaustive parameter search
    BayesianOptimizer    - Optuna TPE Bayesian optimization
    OptimizationTrial    - Single optimization trial result
    VotingStrategyFactory - Create VotingStrategy from config or Optuna trial
    PARAM_BOUNDS         - Searchable parameter bounds for optimization
    ParameterSearchPipeline - Holdout + Optuna search + WFE gate + experiment logging
    SearchConfig         - Configuration for parameter search runs
    SearchResult         - Result of parameter search runs
"""

from poseidon.backtest.cost_model import COST_MODELS, CostModel, get_cost_model
from poseidon.backtest.pending_orders import FillEvent, FillModel, PendingOrderBook
from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.backtest.holdout import HoldoutConfig, HoldoutViolationError
from poseidon.backtest.metrics import compute_composite_score, compute_metrics
from poseidon.backtest.optimizer import (
    BayesianOptimizer,
    GridSearchOptimizer,
    OptimizationTrial,
)
from poseidon.backtest.param_search import (
    ParameterSearchPipeline,
    SearchConfig,
    SearchResult,
)
from poseidon.backtest.portfolio import (
    BacktestPortfolio,
    SizingConfig,
    SizingMode,
    TradeRecord,
)
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.backtest.voting_strategy_factory import (
    PARAM_BOUNDS,
    VotingStrategyFactory,
)
from poseidon.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    compute_wfe,
)

__all__ = [
    "ExperimentTracker",
    "FillEvent",
    "FillModel",
    "HoldoutConfig",
    "HoldoutViolationError",
    "PendingOrderBook",
    "BacktestPortfolio",
    "SizingConfig",
    "SizingMode",
    "CostModel",
    "COST_MODELS",
    "get_cost_model",
    "TradeRecord",
    "compute_composite_score",
    "compute_metrics",
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "BacktestRepository",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardAnalyzer",
    "compute_wfe",
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "OptimizationTrial",
    "VotingStrategyFactory",
    "PARAM_BOUNDS",
    "ParameterSearchPipeline",
    "SearchConfig",
    "SearchResult",
]
