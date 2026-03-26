"""BacktestRunner -- bar-by-bar event loop with pipeline reuse.

Calls the exact same FeatureEngine + Strategy + RiskEngine pipeline
as live prediction. No backtest-only logic for those concerns.

The only backtest-specific logic is:
1. Bar-by-bar iteration with expanding-window feature slicing
2. Warmup period detection (skip bars where features are NaN)
3. Portfolio fill simulation via BacktestPortfolio
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd

import numpy as np

from poseidon.autoresearch.guard import autoresearch_guard
from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import SignalAction, SignalStatus
from poseidon.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class _PortfolioAdapter:
    """Thin adapter making BacktestPortfolio duck-type compatible with VirtualPortfolio.

    RiskEngine.evaluate() expects a VirtualPortfolio interface with:
    - open_position_count (property)
    - total_exposure() -> float
    - get_position(market, symbol) -> PositionEntry | None

    BacktestPortfolio has a different internal structure (dict of dicts instead
    of PositionEntry objects). This adapter bridges the gap so the SAME risk
    engine code path runs without modification.
    """

    def __init__(self, portfolio: BacktestPortfolio) -> None:
        self._portfolio = portfolio

    @property
    def open_position_count(self) -> int:
        return len(self._portfolio.positions)

    def total_exposure(self) -> float:
        return sum(
            pos["quantity"] * pos["entry_price"]
            for pos in self._portfolio.positions.values()
        ) / self._portfolio.initial_capital if self._portfolio.positions else 0.0

    def get_position(self, market: str, symbol: str):  # noqa: ANN201
        key = f"{market}:{symbol}"
        pos = self._portfolio.positions.get(key)
        if pos is None:
            return None
        # Return a duck-typed object with the fields VirtualPortfolio.PositionEntry has
        return _FakePositionEntry(
            symbol=symbol,
            market=market,
            instrument="spot",
            side=pos["side"],
            quantity_pct=pos["quantity"] * pos["entry_price"] / self._portfolio.initial_capital,
            entry_time=pos["entry_time"],
        )


class _FakePositionEntry:
    """Duck-typed PositionEntry for the adapter."""

    def __init__(self, symbol: str, market: str, instrument: str,
                 side: str, quantity_pct: float, entry_time) -> None:  # noqa: ANN001
        self.symbol = symbol
        self.market = market
        self.instrument = instrument
        self.side = side
        self.quantity_pct = quantity_pct
        self.entry_time = entry_time


@autoresearch_guard
class BacktestRunner:
    """Bar-by-bar backtest engine that reuses the live prediction pipeline.

    Pipeline calls (identical to live):
    1. FeatureEngine.compute_from_df(ohlcv) -- feature computation
    2. strategy.evaluate(features_slice) -- signal generation
    3. risk_engine.evaluate(signal, portfolio) -- risk checks

    Backtest-only logic:
    - Bar-by-bar iteration with expanding window
    - Warmup period skipping
    - Portfolio fill simulation
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        feature_engine: FeatureEngine,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()

    def run(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> BacktestResult:
        """Run backtest over historical OHLCV data.

        1. Compute features ONCE on full OHLCV (same FeatureEngine code path)
        2. Determine warmup period (bars where features have NaN)
        3. Iterate bar by bar from warmup onward
        4. For each bar: strategy.evaluate(features[:i+1]) -> signals
        5. For each signal: risk_engine.evaluate(signal, portfolio)
        6. If passed: portfolio.execute_fill(signal, current_bar)
        7. Record equity curve point at each bar

        Args:
            ohlcv: Historical OHLCV DataFrame.
            feature_specs: Optional feature specs override.

        Returns:
            BacktestResult with metrics, trades, and equity curve length.
        """
        try:
            return self._run_loop(ohlcv, feature_specs)
        except Exception as exc:
            logger.exception("Backtest failed: %s", exc)
            config = BacktestConfig(
                strategy_type=self.strategy.strategy_type.value,
                symbol=self.strategy.symbol,
                market=self.strategy.market,
                interval=self.strategy.interval,
                initial_capital=self.initial_capital,
            )
            return BacktestResult(
                config=config,
                metrics={},
                trade_count=0,
                equity_curve_length=0,
                status="failed",
                error_message=str(exc),
            )

    def _run_loop(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> BacktestResult:
        """Core event loop implementation."""
        # Step 1: Compute features ONCE -- same code path as live prediction (BT-01)
        features = self.feature_engine.compute_from_df(ohlcv, feature_specs)

        # Step 2: Determine warmup period
        # Feature columns are those not in the original OHLCV
        ohlcv_cols = {"time", "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in features.columns if c not in ohlcv_cols]
        warmup_end = self._find_warmup_end(features, feature_cols)

        # Step 3: Create portfolio
        portfolio = BacktestPortfolio(
            self.initial_capital, self.cost_model, self.sizing_config,
        )
        adapter = _PortfolioAdapter(portfolio)

        # Step 4: Bar-by-bar loop
        n_bars = len(features)
        logger.info(
            "Backtest: %d bars, warmup=%d, tradeable=%d",
            n_bars, warmup_end, n_bars - warmup_end,
        )

        for i in range(n_bars):
            bar = features.iloc[i]

            # Skip warmup bars (features still have NaN)
            if i < warmup_end:
                portfolio.record_equity_point(bar["time"], float(bar["close"]))
                continue

            # Step 4a: Expanding window slice -- no look-ahead (BT-01)
            features_slice = features.iloc[:i + 1]

            # Step 4b: Strategy evaluate -- same code path as live (BT-01)
            signals = self.strategy.evaluate(features_slice)

            # Step 4c: Compute sizing, risk check, and fill for each signal
            for signal in signals:
                if signal.action == SignalAction.HOLD:
                    continue

                # Compute position size based on SizingConfig (before risk check)
                signal.quantity_pct = self._compute_sizing(
                    signal, features_slice,
                )

                # Same risk engine code path as live (BT-01)
                signal = self.risk_engine.evaluate(signal, adapter)

                if signal.status == SignalStatus.PASSED:
                    portfolio.execute_fill(signal, bar)

            # Step 4d: Record equity point at every bar
            portfolio.record_equity_point(bar["time"], float(bar["close"]))

        # Step 5: Compute metrics
        if portfolio.equity_curve:
            equity_series = pd.Series(
                [eq for _, eq, _ in portfolio.equity_curve],
                index=[t for t, _, _ in portfolio.equity_curve],
            )
        else:
            equity_series = pd.Series([self.initial_capital])

        metrics = compute_metrics(equity_series, portfolio.trades)

        # Step 6: Build result
        config = BacktestConfig(
            strategy_type=self.strategy.strategy_type.value,
            symbol=self.strategy.symbol,
            market=self.strategy.market,
            interval=self.strategy.interval,
            initial_capital=self.initial_capital,
        )

        trades_dicts = [
            {
                "symbol": t.symbol,
                "action": t.action,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "fees": t.fees,
                "pnl": t.pnl,
            }
            for t in portfolio.trades
        ]

        return BacktestResult(
            config=config,
            metrics=metrics,
            trade_count=len(portfolio.trades),
            equity_curve_length=len(portfolio.equity_curve),
            status="completed",
            trades=trades_dicts,
        )

    def _compute_sizing(
        self,
        signal,  # noqa: ANN001
        features_slice: pd.DataFrame,
    ) -> float:
        """Compute position size as quantity_pct based on SizingConfig.

        Called before risk checks so that RiskEngine sees the correct
        quantity_pct for leverage/exposure calculations.

        Returns:
            quantity_pct value to set on the signal.
        """
        if signal.action == SignalAction.CLOSE:
            return 1.0

        cfg = self.sizing_config

        if cfg.mode == SizingMode.FIXED_PCT:
            return signal.quantity_pct or cfg.quantity_pct

        if cfg.mode == SizingMode.FIXED_NOTIONAL:
            # Portfolio._sizing_base() returns initial_capital for this mode,
            # so notional_pct * initial_capital / price gives fixed notional.
            return cfg.notional_pct

        if cfg.mode == SizingMode.VOL_TARGET:
            closes = features_slice["close"].tail(cfg.vol_lookback + 1)
            if len(closes) < 3:
                return cfg.notional_pct  # fallback

            returns = closes.pct_change().dropna()
            realized_vol = float(returns.std()) * np.sqrt(252)
            if realized_vol <= 0:
                return cfg.notional_pct  # fallback

            raw_pct = cfg.target_vol / realized_vol
            return min(raw_pct, cfg.max_position_pct)

        return 0.1  # unreachable fallback

    @staticmethod
    def _find_warmup_end(features: pd.DataFrame, feature_cols: list[str]) -> int:
        """Find the first row index where no feature columns have NaN.

        Returns 0 if there are no feature columns or no NaN values.
        """
        if not feature_cols:
            return 0

        # Find first row where ALL feature columns are non-NaN
        non_nan_mask = features[feature_cols].notna().all(axis=1)
        non_nan_indices = non_nan_mask[non_nan_mask].index

        if len(non_nan_indices) == 0:
            # All bars have NaN features -- entire dataset is warmup
            return len(features)

        # Return the positional index of the first non-NaN row
        return features.index.get_loc(non_nan_indices[0])
