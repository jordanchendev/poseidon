"""Walk-forward analysis engine for strategy robustness validation.

Validates strategies by running rolling train/test windows and computing
Walk-Forward Efficiency (WFE). Strategies with WFE < 50% or insufficient
trades per OOS segment are flagged.

Public API:
    WalkForwardConfig   - Configuration for rolling window parameters
    WindowResult        - Per-window IS/OOS metrics
    WalkForwardResult   - Aggregate result with WFE and flags
    WalkForwardAnalyzer - Main analyzer that runs walk-forward
    compute_wfe         - WFE = OOS annualized return / IS annualized return
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.portfolio import SizingConfig
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis rolling windows.

    Defaults correspond to standard quarterly walk-forward:
    - 1 year in-sample training
    - 3 months validation (reserved for model strategies)
    - 3 months out-of-sample testing
    - Roll forward by 1 quarter
    """

    train_days: int = 252
    validate_days: int = 63
    test_days: int = 63
    step_days: int = 63
    min_trades_per_oos: int = 5
    min_wfe: float = 0.50


@dataclass
class WindowResult:
    """Per-window in-sample and out-of-sample metrics."""

    window_index: int
    is_metrics: dict
    oos_metrics: dict
    is_trade_count: int
    oos_trade_count: int


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward analysis result.

    Attributes:
        wfe: Walk-Forward Efficiency ratio.
        passed: True if WFE >= threshold and all windows meet trade minimums.
        flags: List of flag strings identifying issues.
        per_window: Per-window IS/OOS metrics.
        aggregate_oos_metrics: Combined OOS metrics across all windows.
        config: Configuration used for this analysis.
    """

    wfe: float
    passed: bool
    flags: list[str]
    per_window: list[WindowResult]
    aggregate_oos_metrics: dict
    config: WalkForwardConfig


def compute_wfe(is_ann_return: float, oos_ann_return: float) -> float:
    """Compute Walk-Forward Efficiency.

    WFE = OOS annualized return / IS annualized return.
    Returns 0.0 if IS return is zero or negative (no meaningful ratio).

    Args:
        is_ann_return: Annualized in-sample return.
        oos_ann_return: Annualized out-of-sample return.

    Returns:
        WFE ratio. Values >= 0.50 suggest the strategy is not overfitted.
    """
    if is_ann_return <= 0:
        return 0.0
    return oos_ann_return / is_ann_return


class WalkForwardAnalyzer:
    """Walk-forward analysis engine.

    Validates strategy robustness by running rolling train/test windows,
    computing per-window metrics, and aggregating Walk-Forward Efficiency.

    Uses BacktestRunner for each IS and OOS segment to ensure pipeline
    reuse (same FeatureEngine + Strategy + RiskEngine code path).
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
    ) -> None:
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()

    def generate_windows(
        self,
        data_length: int,
        config: WalkForwardConfig,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Generate rolling train/test window index ranges.

        Each window is a tuple of ((train_start, train_end), (test_start, test_end))
        using exclusive-end semantics (consistent with Python slicing).

        Windows step forward by config.step_days. The first window starts at
        index 0 with train=[0, train_days) and test=[train_days, train_days+test_days).
        Subsequent windows shift by step_days. Generation stops when test_end
        would exceed data_length.

        Note: validate_days is reserved for model retraining (future use).
        For rule strategies, train and test are directly adjacent.

        Args:
            data_length: Total number of bars in the OHLCV data.
            config: Walk-forward configuration with window sizes.

        Returns:
            List of ((train_start, train_end), (test_start, test_end)) tuples.
        """
        if config.step_days < config.test_days:
            logger.warning(
                "step_days (%d) < test_days (%d): OOS windows will overlap. "
                "Set step_days >= test_days for non-overlapping OOS segments.",
                config.step_days,
                config.test_days,
            )

        windows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        offset = 0

        while True:
            train_start = offset
            train_end = train_start + config.train_days
            test_start = train_end
            test_end = test_start + config.test_days

            if test_end > data_length:
                break

            windows.append(((train_start, train_end), (test_start, test_end)))
            offset += config.step_days

        return windows

    def analyze(
        self,
        strategy: BaseStrategy,
        ohlcv: pd.DataFrame,
        config: WalkForwardConfig | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward analysis on a strategy.

        1. Generate rolling windows from the OHLCV data length.
        2. For each window, run backtest on IS and OOS slices.
        3. Compute per-window metrics.
        4. Aggregate IS and OOS annualized returns.
        5. Compute WFE and build flags.

        Args:
            strategy: Strategy to evaluate.
            ohlcv: Historical OHLCV DataFrame.
            config: Walk-forward configuration. Uses defaults if None.

        Returns:
            WalkForwardResult with WFE, flags, and per-window metrics.
        """
        if config is None:
            config = WalkForwardConfig()

        windows = self.generate_windows(len(ohlcv), config)

        per_window: list[WindowResult] = []
        is_ann_returns: list[float] = []
        oos_ann_returns: list[float] = []

        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(
            windows
        ):
            is_ohlcv = ohlcv.iloc[train_start:train_end].reset_index(drop=True)
            oos_ohlcv = ohlcv.iloc[test_start:test_end].reset_index(drop=True)

            # Run backtest on IS data
            is_runner = BacktestRunner(
                strategy=strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
            )
            is_result = is_runner.run(is_ohlcv)

            # Run backtest on OOS data
            oos_runner = BacktestRunner(
                strategy=strategy,
                feature_engine=self.feature_engine,
                risk_engine=self.risk_engine,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
                sizing_config=self.sizing_config,
            )
            oos_result = oos_runner.run(oos_ohlcv)

            is_metrics = is_result.metrics
            oos_metrics = oos_result.metrics

            is_ann_returns.append(is_metrics.get("annualized_return", 0.0))
            oos_ann_returns.append(oos_metrics.get("annualized_return", 0.0))

            per_window.append(
                WindowResult(
                    window_index=i,
                    is_metrics=is_metrics,
                    oos_metrics=oos_metrics,
                    is_trade_count=is_metrics.get("trade_count", 0),
                    oos_trade_count=oos_metrics.get("trade_count", 0),
                )
            )

            logger.info(
                "Window %d: IS ann_return=%.4f, OOS ann_return=%.4f, "
                "IS trades=%d, OOS trades=%d",
                i,
                is_metrics.get("annualized_return", 0.0),
                oos_metrics.get("annualized_return", 0.0),
                is_metrics.get("trade_count", 0),
                oos_metrics.get("trade_count", 0),
            )

        # Aggregate annualized returns (average across windows)
        if is_ann_returns:
            agg_is_ann = sum(is_ann_returns) / len(is_ann_returns)
        else:
            agg_is_ann = 0.0

        if oos_ann_returns:
            agg_oos_ann = sum(oos_ann_returns) / len(oos_ann_returns)
        else:
            agg_oos_ann = 0.0

        wfe = compute_wfe(agg_is_ann, agg_oos_ann)

        # Build flags
        flags: list[str] = []
        if wfe < config.min_wfe:
            flags.append("wfe_below_threshold")

        for w in per_window:
            if w.oos_trade_count < config.min_trades_per_oos:
                flags.append(f"insufficient_trades_window_{w.window_index}")

        # Determine pass/fail
        has_insufficient_trades = any("insufficient_trades" in f for f in flags)
        passed = wfe >= config.min_wfe and not has_insufficient_trades

        # Aggregate OOS metrics (average across windows)
        aggregate_oos_metrics: dict = {}
        if per_window:
            all_keys = per_window[0].oos_metrics.keys()
            for key in all_keys:
                values = [w.oos_metrics.get(key, 0.0) for w in per_window]
                aggregate_oos_metrics[key] = sum(values) / len(values)

        logger.info(
            "Walk-forward complete: WFE=%.4f, passed=%s, flags=%s",
            wfe,
            passed,
            flags,
        )

        return WalkForwardResult(
            wfe=wfe,
            passed=passed,
            flags=flags,
            per_window=per_window,
            aggregate_oos_metrics=aggregate_oos_metrics,
            config=config,
        )
