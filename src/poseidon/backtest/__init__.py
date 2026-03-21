"""Backtest engine: portfolio simulation, cost models, metrics, and walk-forward analysis.

Public API:
    BacktestPortfolio    - Portfolio simulator with fee/slippage
    CostModel            - Market-specific cost configuration
    COST_MODELS          - Registry of all market cost models
    get_cost_model       - Lookup cost model by market name
    TradeRecord          - Completed trade data
    compute_metrics      - Performance metrics from equity + trades
    WalkForwardConfig    - Walk-forward analysis configuration
    WalkForwardResult    - Walk-forward analysis result
    WalkForwardAnalyzer  - Walk-forward analysis engine
    compute_wfe          - Walk-Forward Efficiency computation
"""

from poseidon.backtest.cost_model import COST_MODELS, CostModel, get_cost_model
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import BacktestPortfolio, TradeRecord
from poseidon.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    compute_wfe,
)

__all__ = [
    "BacktestPortfolio",
    "CostModel",
    "COST_MODELS",
    "get_cost_model",
    "TradeRecord",
    "compute_metrics",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardAnalyzer",
    "compute_wfe",
]
