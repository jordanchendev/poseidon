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
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

import numpy as np

from poseidon.autoresearch.guard import autoresearch_guard
from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.pending_orders import FillModel, PendingOrderBook
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.data.feature_engine import FeatureEngine, _is_nonprice_spec
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import OrderType, SignalAction, SignalStatus
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


@autoresearch_guard(mutable_attrs=frozenset({
    "_funding_costs_total",
    "_funding_costs_by_trade",
    "_last_funding_settlement",
}))
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
        fill_model: FillModel | None = None,
        max_pending_bars: int = 24,
        include_funding: bool = False,
        funding_rates: pd.DataFrame | None = None,
    ) -> None:
        self.strategy = strategy
        self.feature_engine = feature_engine
        self.risk_engine = risk_engine
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()
        self._fill_model = fill_model or FillModel.OPTIMISTIC
        self._max_pending_bars = max_pending_bars
        self._include_funding = include_funding
        self._funding_rates = funding_rates
        self._funding_costs_total = 0.0
        self._funding_costs_by_trade: list[float] = []
        self._last_funding_settlement: pd.Timestamp | None = None

        # Capability enforcement (Phase 34 - COMP-05)
        from poseidon.capabilities.validation import (
            validate_backtest_components,
            warn_bias_risks,
        )

        validate_backtest_components([self.strategy])
        warn_bias_risks([self.strategy])

    def validate_model_bias(
        self,
        model_version_id: UUID | None,
        backtest_start: datetime | None,
        db_session: Session | None = None,
    ) -> tuple[datetime | None, UUID | None]:
        """Validate no look-ahead bias when using a ModelVersion (per D-07, PRED-04).

        Checks that model_version.train_end < backtest_start_date.
        Returns (active_model_timestamp, model_version_id) for audit trail.
        Raises ValueError if look-ahead bias detected.

        Args:
            model_version_id: UUID of the ModelVersion to validate.
            backtest_start: Start date of the backtest.
            db_session: SQLAlchemy session for DB lookup. Required if model_version_id is set.

        Returns:
            Tuple of (active_model_timestamp, model_version_id).
        """
        if model_version_id is None:
            return None, None

        if db_session is None:
            raise ValueError(
                "db_session is required when model_version_id is specified"
            )

        from poseidon.models.model_version import ModelVersion

        mv = db_session.query(ModelVersion).filter(
            ModelVersion.id == model_version_id
        ).first()
        if mv is None:
            raise ValueError(
                f"ModelVersion {model_version_id} not found"
            )

        active_model_timestamp = datetime.now(timezone.utc)

        if mv.train_end is not None and backtest_start is not None:
            # Normalize both to aware datetimes for comparison
            train_end = mv.train_end
            bt_start = backtest_start
            if train_end.tzinfo is not None and (bt_start.tzinfo is None):
                bt_start = bt_start.replace(tzinfo=timezone.utc)
            elif (train_end.tzinfo is None) and bt_start.tzinfo is not None:
                train_end = train_end.replace(tzinfo=timezone.utc)

            if train_end >= bt_start:
                raise ValueError(
                    f"Look-ahead bias detected: ModelVersion {model_version_id} "
                    f"train_end={mv.train_end.isoformat()} >= backtest_start={backtest_start.isoformat()}. "
                    f"The model's training period must end before the backtest start date."
                )

        logger.info(
            "Model bias check passed: model_version_id=%s, train_end=%s, backtest_start=%s",
            model_version_id, mv.train_end, backtest_start,
        )

        return active_model_timestamp, model_version_id

    def _load_prediction_data(
        self,
        model_version_id: UUID | None,
        symbol: str,
        db_session: Session | None = None,
    ) -> pd.DataFrame | None:
        """Load prediction cache for a model version, filtered to a single symbol.

        Reads the "test" segment Parquet (primary backtest segment), falls back
        to checking all available segments. Returns a single-instrument DataFrame
        with DatetimeIndex and "prediction" column, or None if unavailable.

        Args:
            model_version_id: UUID of the model version.
            symbol: Symbol to filter predictions for.
            db_session: SQLAlchemy session for ModelVersion lookup.

        Returns:
            DataFrame with DatetimeIndex and "prediction" column, or None.
        """
        if model_version_id is None or db_session is None:
            return None

        from poseidon.ml.artifacts import get_predictions_path
        from poseidon.models.model_version import ModelVersion

        mv = db_session.query(ModelVersion).filter(
            ModelVersion.id == model_version_id
        ).first()
        if mv is None or mv.artifact_path is None:
            logger.warning(
                "ModelVersion %s not found or has no artifact_path", model_version_id,
            )
            return None

        # Try segments in order: test first (standard backtest), then valid, then train
        segments_to_try = (
            mv.params.get("prediction_segments", ["test"]) if mv.params else ["test"]
        )
        # Prefer test segment for backtesting
        ordered: list[str] = []
        for pref in ("test", "valid", "train"):
            if pref in segments_to_try:
                ordered.append(pref)
        if not ordered:
            ordered = list(segments_to_try)

        all_dfs: list[pd.DataFrame] = []
        for segment in ordered:
            pred_path = get_predictions_path(mv.artifact_path, segment)
            if pred_path.exists():
                try:
                    seg_df = pd.read_parquet(str(pred_path))
                    all_dfs.append(seg_df)
                    logger.info(
                        "Loaded %s predictions from %s (%d rows)",
                        segment, pred_path, len(seg_df),
                    )
                except Exception as exc:
                    logger.warning("Failed to read predictions %s: %s", pred_path, exc)

        if not all_dfs:
            logger.warning(
                "No prediction Parquet files found for ModelVersion %s",
                model_version_id,
            )
            return None

        # Concatenate all segments (they don't overlap in date ranges)
        combined = pd.concat(all_dfs)
        # Remove duplicates keeping last (in case segments overlap at boundaries)
        combined = combined[~combined.index.duplicated(keep="last")]

        # Filter to single instrument
        # Qlib MultiIndex is (datetime, instrument). Extract this symbol's predictions.
        if isinstance(combined.index, pd.MultiIndex):
            instruments = combined.index.get_level_values("instrument").unique()
            # Find matching instrument: try exact match, then suffix match
            # Poseidon symbols: "2330", "BTCUSDT"
            # Qlib instruments: "SH600519", "BTCUSDT", "2330.TW" (varies by handler)
            matched_instrument = None
            for inst in instruments:
                if inst == symbol or inst.endswith(symbol) or symbol in inst:
                    matched_instrument = inst
                    break
            if matched_instrument is None:
                logger.warning(
                    "Symbol %s not found in prediction instruments: %s",
                    symbol, list(instruments)[:10],
                )
                return None

            # Extract single instrument, collapse to DatetimeIndex
            instrument_df = combined.loc[(slice(None), matched_instrument), :]
            instrument_df = instrument_df.droplevel("instrument")
        else:
            # Already a flat DatetimeIndex (single instrument predictions)
            instrument_df = combined

        instrument_df = instrument_df.sort_index()
        return instrument_df

    def _resolve_model_version_id(
        self,
        config_model_version_id: UUID | None,
    ) -> UUID | None:
        """Resolve model_version_id from strategy sub-signals or backtest config.

        Per D-13: Scans strategy's sub_signals for type=='ml_prediction',
        extracts model_version_id from the FIRST such condition found.

        Per D-14: Sub-signal model_version_id takes precedence over
        config-level model_version_id (Phase 44 path).
        """
        # Phase 45 path: scan sub-signals for ml_prediction condition
        all_signals: list[dict] = []
        if hasattr(self.strategy, "_sub_signals"):
            all_signals.extend(self.strategy._sub_signals)
        if hasattr(self.strategy, "_bear_sub_signals"):
            all_signals.extend(self.strategy._bear_sub_signals)

        for cond in all_signals:
            if cond.get("type") == "ml_prediction":
                sub_mv_id = cond.get("model_version_id")
                if sub_mv_id is not None:
                    # Ensure UUID type for DB compatibility
                    if not isinstance(sub_mv_id, UUID):
                        sub_mv_id = UUID(str(sub_mv_id))
                    logger.info(
                        "Resolved model_version_id=%s from ml_prediction sub-signal (Phase 45 path)",
                        sub_mv_id,
                    )
                    return sub_mv_id

        # Fallback: Phase 44 path (config-level model_version_id)
        return config_model_version_id

    @property
    def portfolio(self) -> BacktestPortfolio | None:
        """Access the portfolio after run() completes. Returns None if run() hasn't been called."""
        return getattr(self, "_ar_last_portfolio", None)

    def _evaluate_sl_tp(
        self, portfolio: BacktestPortfolio, bar: pd.Series, bar_index: int,
    ) -> None:
        """Evaluate intra-bar SL/TP exits on open positions (Phase A of bar loop).

        For each open position with SL/TP prices set, checks if the bar's
        high/low triggers either exit condition:

        Long positions:
        - SL triggers if bar_low <= stop_loss_price
        - TP triggers if bar_high >= take_profit_price

        Short positions:
        - SL triggers if bar_high >= stop_loss_price
        - TP triggers if bar_low <= take_profit_price

        When both SL and TP trigger in the same bar, SL takes priority (D-12).

        Args:
            portfolio: The BacktestPortfolio to evaluate.
            bar: Current OHLCV bar with 'high' and 'low' columns.
            bar_index: Current bar index (unused but available for logging).
        """
        # Collect exits first, then execute (cannot modify dict while iterating)
        exits: list[tuple[str, float, str]] = []  # (key, exit_price, exit_type)

        for key, pos in portfolio.positions.items():
            sl_price = pos.get("stop_loss_price")
            tp_price = pos.get("take_profit_price")

            if sl_price is None and tp_price is None:
                continue

            bar_low = float(bar["low"])
            bar_high = float(bar["high"])

            sl_triggered = False
            tp_triggered = False

            if pos["side"] == "long":
                if sl_price is not None and bar_low <= sl_price:
                    sl_triggered = True
                if tp_price is not None and bar_high >= tp_price:
                    tp_triggered = True
            else:  # short
                if sl_price is not None and bar_high >= sl_price:
                    sl_triggered = True
                if tp_price is not None and bar_low <= tp_price:
                    tp_triggered = True

            # D-12: SL takes priority when both trigger in same bar
            if sl_triggered:
                exits.append((key, sl_price, "sl"))
            elif tp_triggered:
                exits.append((key, tp_price, "tp"))

        # Execute all SL/TP exits
        for key, exit_price, exit_type in exits:
            portfolio.execute_sl_tp_exit(key, exit_price, exit_type, bar)

    def _crosses_8h_boundary(self, bar_time: pd.Timestamp) -> bool:
        """Check if bar_time crosses an 8h funding settlement boundary.

        Settlement times: 00:00, 08:00, 16:00 UTC.
        Uses _last_funding_settlement to ensure each settlement is applied exactly once.
        """
        # Normalize to UTC
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize("UTC")

        # Compute the 8h slot for this bar
        hour_slot = (bar_time.hour // 8) * 8
        current_settlement = bar_time.normalize() + pd.Timedelta(hours=hour_slot)

        if self._last_funding_settlement is None:
            self._last_funding_settlement = current_settlement
            return False  # Don't apply on first bar

        if current_settlement > self._last_funding_settlement:
            self._last_funding_settlement = current_settlement
            return True
        return False

    def _get_funding_rate(self, bar_time: pd.Timestamp) -> float | None:
        """Look up funding rate for the given settlement time.

        Searches funding_rates DataFrame for the closest rate <= bar_time.
        Returns None if no rate found.
        """
        if self._funding_rates is None or self._funding_rates.empty:
            return None
        # Ensure bar_time has timezone for comparison
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize("UTC")

        # Build a tz-aware index for comparison
        idx = self._funding_rates.index
        if idx.tzinfo is None:
            idx = idx.tz_localize("UTC")

        mask = idx <= bar_time
        if not mask.any():
            return None
        # Get the last matching position index
        matching_positions = np.where(mask)[0]
        last_pos = matching_positions[-1]
        return float(self._funding_rates.iloc[last_pos]["funding_rate"])

    def run(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
        model_version_id: UUID | None = None,
        backtest_start: datetime | None = None,
        db_session: Session | None = None,
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
            model_version_id: Optional ModelVersion UUID for ML prediction look-ahead bias check.
            backtest_start: Backtest start date for bias validation.
            db_session: SQLAlchemy session for ModelVersion lookup.

        Returns:
            BacktestResult with metrics, trades, and equity curve length.
        """
        try:
            return self._run_loop(
                ohlcv, feature_specs, model_version_id, backtest_start, db_session,
            )
        except Exception as exc:
            logger.exception("Backtest failed: %s", exc)
            config = BacktestConfig(
                strategy_type=self.strategy.strategy_type.value,
                symbol=self.strategy.symbol,
                market=self.strategy.market,
                interval=self.strategy.interval,
                initial_capital=self.initial_capital,
                model_version_id=model_version_id,
            )
            return BacktestResult(
                config=config,
                metrics={},
                trade_count=0,
                equity_curve_length=0,
                status="failed",
                error_message=str(exc),
                model_version_id=model_version_id,
            )

    def _run_loop(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
        model_version_id: UUID | None = None,
        backtest_start: datetime | None = None,
        db_session: Session | None = None,
    ) -> BacktestResult:
        """Core event loop implementation."""
        # Reset funding state for this run
        self._funding_costs_total = 0.0
        self._funding_costs_by_trade = []
        self._last_funding_settlement = None

        # Step 0a: Resolve model_version_id from sub-signals or config (Phase 45)
        resolved_mv_id = self._resolve_model_version_id(model_version_id)

        # Step 0b: Look-ahead bias validation (per D-07, PRED-04)
        active_model_timestamp, validated_mv_id = self.validate_model_bias(
            resolved_mv_id, backtest_start, db_session,
        )

        # Step 1: Compute features ONCE -- same code path as live prediction (BT-01)
        # If no feature_specs given, ask the strategy for its required specs
        if feature_specs is None and hasattr(self.strategy, "get_feature_specs"):
            feature_specs = self.strategy.get_feature_specs()

        # Check if we need prediction data injection (Phase 44/45 - ML vote)
        extra_nonprice_data = None
        if resolved_mv_id is not None and feature_specs is not None:
            # Check if any spec references qlib_prediction
            has_prediction_spec = any(
                name == "qlib_prediction" for name, _ in feature_specs
            )
            if has_prediction_spec:
                pred_df = self._load_prediction_data(
                    resolved_mv_id, self.strategy.symbol, db_session,
                )
                if pred_df is not None:
                    extra_nonprice_data = {"prediction_data": pred_df}
                    logger.info(
                        "Injecting prediction_data (%d rows) for model_version_id=%s",
                        len(pred_df), resolved_mv_id,
                    )

        # Use compute_with_companions when non-price features are present
        has_nonprice = feature_specs is not None and any(
            _is_nonprice_spec(name) for name, _ in feature_specs
        )
        if has_nonprice:
            features = self.feature_engine.compute_with_companions(
                ohlcv,
                self.strategy.symbol,
                self.strategy.market,
                self.strategy.interval,
                feature_specs=feature_specs,
                db_session=db_session,
                extra_nonprice_data=extra_nonprice_data,
            )
        else:
            features = self.feature_engine.compute_from_df(ohlcv, feature_specs)

        # Step 2: Determine warmup period
        # Only consider columns NEWLY computed by compute_from_df for warmup,
        # not pre-existing R2 columns that were already on the input ohlcv.
        pre_existing_cols = set(ohlcv.columns)
        feature_cols = [c for c in features.columns if c not in pre_existing_cols]
        warmup_end = self._find_warmup_end(features, feature_cols)

        # Step 3: Create portfolio
        portfolio = BacktestPortfolio(
            self.initial_capital, self.cost_model, self.sizing_config,
        )
        # The autoresearch guard explicitly allows internal `_ar_*` state.
        self._ar_last_portfolio = portfolio
        adapter = _PortfolioAdapter(portfolio)

        # Step 4: Bar-by-bar four-phase loop (Phase 50 refactor)
        # Phase A: SL/TP evaluation on open positions
        # Phase B: Check pending limit order fills
        # Phase C: Strategy evaluate + signal routing (market vs limit)
        # Phase D: Record equity + expire timed-out orders
        #
        # For existing market-order backtests (no pending orders, no SL/TP):
        # Phase A is a no-op (no positions have SL/TP), Phase B returns empty
        # list, Phase C is identical to the original loop, Phase D adds a
        # no-op expire call before equity recording. Results are IDENTICAL.
        n_bars = len(features)
        logger.info(
            "Backtest: %d bars, warmup=%d, tradeable=%d",
            n_bars, warmup_end, n_bars - warmup_end,
        )

        # Create PendingOrderBook for this run
        pending_book = PendingOrderBook(fill_model=self._fill_model)

        for i in range(n_bars):
            bar = features.iloc[i]

            # Skip warmup bars (features still have NaN)
            if i < warmup_end:
                portfolio.record_equity_point(bar.name, float(bar["close"]))
                continue

            # Phase A: SL/TP evaluation on open positions (D-02)
            self._evaluate_sl_tp(portfolio, bar, i)

            # Phase B: Check pending limit order fills
            fill_events = pending_book.check_fills(bar, i)
            for fe in fill_events:
                portfolio.execute_limit_fill(fe, bar)

            # Phase C: Strategy evaluate + signal routing (D-03)
            features_slice = features.iloc[:i + 1]
            signals = self.strategy.evaluate(features_slice)

            for signal in signals:
                if signal.action == SignalAction.HOLD:
                    # Check for trailing stop SL updates (ADV-01, D-10)
                    if signal.metadata:
                        updated_sl = signal.metadata.get("updated_stop_loss")
                        if updated_sl is not None:
                            key = f"{self.strategy.market}:{self.strategy.symbol}"
                            if key in portfolio.positions:
                                portfolio.positions[key]["stop_loss_price"] = updated_sl
                    continue

                # Compute position size based on SizingConfig (before risk check)
                signal.quantity_pct = self._compute_sizing(
                    signal, features_slice,
                )

                # Same risk engine code path as live (BT-01)
                signal = self.risk_engine.evaluate(signal, adapter)

                if signal.status == SignalStatus.PASSED:
                    # Route by order type: limit -> pending book, market -> immediate fill
                    if signal.order_type == OrderType.LIMIT:
                        pending_book.submit(signal, i)
                    else:
                        portfolio.execute_fill(signal, bar)

            # Phase D: Record equity + expire timed-out orders
            pending_book.expire_orders(i, self._max_pending_bars)

            # Funding rate settlement (ADV-02, D-12)
            if self._include_funding and self._funding_rates is not None:
                bar_time = pd.Timestamp(bar.name)
                if self._crosses_8h_boundary(bar_time):
                    for _key, pos in portfolio.positions.items():
                        rate = self._get_funding_rate(bar_time)
                        if rate is not None:
                            mark_price = float(bar["close"])
                            qty = pos["quantity"]
                            # Long pays positive rate, short pays negative rate
                            if pos["side"] == "long":
                                cost = abs(qty) * mark_price * rate
                            else:
                                cost = -(abs(qty) * mark_price * rate)
                            portfolio.cash -= cost
                            self._funding_costs_total += abs(cost)
                            self._funding_costs_by_trade.append(cost)

            portfolio.record_equity_point(bar.name, float(bar["close"]))

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

        # Compute PnL with/without funding (D-13)
        total_pnl = metrics.get("total_pnl", 0.0)
        pnl_with_funding = total_pnl - self._funding_costs_total if self._include_funding else None
        pnl_without_funding = total_pnl if self._include_funding else None

        return BacktestResult(
            config=config,
            metrics=metrics,
            trade_count=len(portfolio.trades),
            equity_curve_length=len(portfolio.equity_curve),
            status="completed",
            trades=trades_dicts,
            active_model_timestamp=active_model_timestamp,
            model_version_id=validated_mv_id,
            funding_costs_total=self._funding_costs_total,
            funding_costs_by_trade=self._funding_costs_by_trade,
            pnl_with_funding=pnl_with_funding,
            pnl_without_funding=pnl_without_funding,
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

        if cfg.mode == SizingMode.FIXED_RISK:
            # D-18: entry price = order_price for limit, bar close for market
            entry_price = (
                signal.order_price
                if signal.order_price is not None
                else float(features_slice.iloc[-1]["close"])
            )
            sl_price = signal.stop_loss_price

            if sl_price is None:
                logger.warning(
                    "FIXED_RISK without stop_loss_price, falling back to FIXED_NOTIONAL"
                )
                return cfg.notional_pct

            distance = abs(entry_price - sl_price)
            if distance < 1e-10:
                logger.warning(
                    "FIXED_RISK: SL distance near zero (%.10f), falling back to FIXED_NOTIONAL",
                    distance,
                )
                return cfg.notional_pct

            # D-17: qty = (equity * risk_pct) / distance
            # self._ar_last_portfolio is set in _run_loop (line ~526) BEFORE bar iteration.
            # It holds the BacktestPortfolio instance, giving access to .equity during sizing.
            # The autoresearch_guard decorator explicitly allows _ar_* prefixed attributes.
            equity = self._ar_last_portfolio.equity
            risk_amount = equity * cfg.risk_pct
            raw_qty = risk_amount / distance
            raw_notional_pct = (raw_qty * entry_price) / self.initial_capital

            # Cap at max_notional_pct
            return min(raw_notional_pct, cfg.max_notional_pct)

        return 0.1  # unreachable fallback

    @staticmethod
    def _find_warmup_end(features: pd.DataFrame, feature_cols: list[str]) -> int:
        """Find the first row index where no feature columns have NaN.

        Returns 0 if there are no feature columns or no NaN values.
        Columns that are entirely NaN (e.g., optional companion data not
        available) are excluded from the warmup check — they represent
        graceful degradation, not incomplete warmup.
        """
        if not feature_cols:
            return 0

        # Exclude columns that are entirely NaN (optional companion data)
        active_cols = [c for c in feature_cols if features[c].notna().any()]
        if not active_cols:
            return 0

        # Find first row where ALL active feature columns are non-NaN
        non_nan_mask = features[active_cols].notna().all(axis=1)
        non_nan_indices = non_nan_mask[non_nan_mask].index

        if len(non_nan_indices) == 0:
            # All bars have NaN features -- entire dataset is warmup
            return len(features)

        # Return the positional index of the first non-NaN row
        return features.index.get_loc(non_nan_indices[0])
