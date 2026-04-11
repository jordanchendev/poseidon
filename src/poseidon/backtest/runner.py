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
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.data.feature_engine import FeatureEngine, _is_nonprice_spec
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

    @property
    def portfolio(self) -> BacktestPortfolio | None:
        """Access the portfolio after run() completes. Returns None if run() hasn't been called."""
        return getattr(self, "_ar_last_portfolio", None)

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
        # Step 0: Look-ahead bias validation (per D-07, PRED-04)
        active_model_timestamp, validated_mv_id = self.validate_model_bias(
            model_version_id, backtest_start, db_session,
        )

        # Step 1: Compute features ONCE -- same code path as live prediction (BT-01)
        # If no feature_specs given, ask the strategy for its required specs
        if feature_specs is None and hasattr(self.strategy, "get_feature_specs"):
            feature_specs = self.strategy.get_feature_specs()

        # Check if we need prediction data injection (Phase 44 - ML vote)
        extra_nonprice_data = None
        if model_version_id is not None and feature_specs is not None:
            # Check if any spec references qlib_prediction
            has_prediction_spec = any(
                name == "qlib_prediction" for name, _ in feature_specs
            )
            if has_prediction_spec:
                pred_df = self._load_prediction_data(
                    model_version_id, self.strategy.symbol, db_session,
                )
                if pred_df is not None:
                    extra_nonprice_data = {"prediction_data": pred_df}
                    logger.info(
                        "Injecting prediction_data (%d rows) for model_version_id=%s",
                        len(pred_df), model_version_id,
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
                portfolio.record_equity_point(bar.name, float(bar["close"]))
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

        return BacktestResult(
            config=config,
            metrics=metrics,
            trade_count=len(portfolio.trades),
            equity_curve_length=len(portfolio.equity_curve),
            status="completed",
            trades=trades_dicts,
            active_model_timestamp=active_model_timestamp,
            model_version_id=validated_mv_id,
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
