"""Dual-mode backtest comparison utility (LIMIT-06, D-04/D-05/D-06).

Runs the same strategy with both optimistic and pessimistic fill models,
producing side-by-side metrics for viability assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from poseidon.backtest.pending_orders import FillModel
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestResult
from poseidon.strategies.base import BaseStrategy


@dataclass
class DualModeResult:
    """Side-by-side comparison of optimistic vs pessimistic fill results (D-05)."""

    optimistic_result: BacktestResult
    pessimistic_result: BacktestResult
    delta_metrics: dict = field(default_factory=dict)
    is_viable: bool = False  # pessimistic Sharpe > 0 (D-06)


def run_dual_mode_comparison(
    strategy_factory: Callable[[], BaseStrategy],
    ohlcv: pd.DataFrame,
    feature_engine: Any,
    risk_engine: Any,
    cost_model: Any,
    initial_capital: float = 1_000_000.0,
    sizing_config: Any = None,
    max_pending_bars: int = 5,
    feature_specs: list[tuple[str, dict]] | None = None,
    include_funding: bool = False,
    **runner_kwargs: Any,
) -> DualModeResult:
    """Run same strategy config with both fill models and compare (D-04).

    Args:
        strategy_factory: Callable that returns a fresh strategy instance.
                          Must be called twice (once per fill model).
        ohlcv: OHLCV DataFrame with datetime index.
        feature_engine: FeatureEngine instance.
        risk_engine: RiskEngine instance.
        cost_model: CostModel instance.
        initial_capital: Starting capital.
        sizing_config: SizingConfig instance.
        max_pending_bars: Pending order timeout.
        feature_specs: Feature specifications (None = strategy default).
        include_funding: Whether to include funding rate costs.
        **runner_kwargs: Additional BacktestRunner kwargs.

    Returns:
        DualModeResult with both results and viability flag.
    """
    results: dict[FillModel, BacktestResult] = {}
    for fill_model in (FillModel.OPTIMISTIC, FillModel.PESSIMISTIC):
        strategy = strategy_factory()
        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=initial_capital,
            sizing_config=sizing_config,
            fill_model=fill_model,
            max_pending_bars=max_pending_bars,
            **runner_kwargs,
        )
        run_kwargs: dict[str, Any] = {}
        if feature_specs is not None:
            run_kwargs["feature_specs"] = feature_specs
        result = runner.run(ohlcv, **run_kwargs)
        results[fill_model] = result

    opt_result = results[FillModel.OPTIMISTIC]
    pess_result = results[FillModel.PESSIMISTIC]

    # Compute delta metrics (D-05)
    delta_keys = ["sharpe_ratio", "max_drawdown", "win_rate", "total_pnl"]
    delta: dict[str, float] = {}
    for key in delta_keys:
        opt_val = float(opt_result.metrics.get(key, 0.0) or 0.0)
        pess_val = float(pess_result.metrics.get(key, 0.0) or 0.0)
        delta[key] = opt_val - pess_val

    # Viability gate (D-06): pessimistic Sharpe > 0
    pess_sharpe = float(pess_result.metrics.get("sharpe_ratio", 0.0) or 0.0)
    is_viable = pess_sharpe > 0

    return DualModeResult(
        optimistic_result=opt_result,
        pessimistic_result=pess_result,
        delta_metrics=delta,
        is_viable=is_viable,
    )


def validate_cross_symbol(
    results: list[dict],
) -> tuple[bool, dict]:
    """Cross-symbol validation gate (SWEEP-06, D-17).

    Each result dict must contain: symbol, pessimistic_sharpe, wfe.
    Returns (all_passed, per_symbol_report) where:
    - all_passed: True only if EVERY symbol has pessimistic Sharpe > 0 AND WFE >= 50%
    - per_symbol_report: {symbol: {sharpe, wfe, passed}}
    """
    report: dict[str, dict] = {}
    all_passed = True
    for r in results:
        symbol = r["symbol"]
        sharpe = r.get("pessimistic_sharpe", 0.0)
        wfe = r.get("wfe", 0.0)
        passed = sharpe > 0 and wfe >= 0.50
        report[symbol] = {"sharpe": sharpe, "wfe": wfe, "passed": passed}
        if not passed:
            all_passed = False
    return all_passed, report
