"""Backtest, dual-mode, and optimization tasks, plus the market-data fetch trigger."""

import logging
import uuid
from datetime import UTC, datetime

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory
from poseidon.backtest.optimizer import BayesianOptimizer, GridSearchOptimizer
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig
from poseidon.core.database import db_session
from poseidon.data.feature_engine import FeatureEngine
from poseidon.models.backtest import BacktestRecord
from poseidon.models.strategy import StrategyRecord
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.rule_strategy_v2 import build_rule_strategy
from poseidon.strategies.voting_strategy import VotingStrategy
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _build_portfolio_strategy(record, repo):
    """Build a PortfolioStrategy instance from StrategyRecord config."""
    from poseidon.strategies.portfolio.crypto_trend import CryptoTrendConfig
    from poseidon.strategies.portfolio.fundamental_selection import (
        FundamentalSelectionConfig,
    )
    from poseidon.strategies.portfolio.prediction_ranking import (
        PredictionRankingConfig,
    )
    from poseidon.strategies.portfolio.registry import get_portfolio_strategy
    from poseidon.strategies.portfolio.schemas import RevenueBreakoutConfig

    portfolio_cfg = dict(record.config or {})
    strategy_name = portfolio_cfg.get("strategy")
    if not strategy_name:
        raise ValueError("portfolio_strategy config must include 'strategy'")

    portfolio_cfg.setdefault("market", record.market)
    portfolio_cfg.setdefault("symbols", [record.symbol] if record.symbol else [])

    config_models = {
        "fundamental_selection": FundamentalSelectionConfig,
        "revenue_breakout": RevenueBreakoutConfig,
        "crypto_trend": CryptoTrendConfig,
        "prediction_ranking": PredictionRankingConfig,
    }
    if strategy_name not in config_models:
        raise ValueError(f"Unknown portfolio strategy: {strategy_name!r}")

    strategy_cls = get_portfolio_strategy(strategy_name)
    strategy_cfg = config_models[strategy_name](**portfolio_cfg)

    try:
        return strategy_cls(strategy_cfg, repo=repo)
    except TypeError:
        return strategy_cls(strategy_cfg)


def _serialize_portfolio_trades(trades: list[dict]):
    """Convert portfolio backtester trade dicts to repository TradeRecord objects."""
    from poseidon.backtest.portfolio import TradeRecord

    result: list[TradeRecord] = []
    for trade in trades:
        trade_dt = datetime.fromisoformat(trade["date"]).replace(tzinfo=UTC)
        metadata = {
            key: value for key, value in trade.items() if key not in {"symbol", "action", "date", "price", "shares"}
        }
        result.append(
            TradeRecord(
                symbol=trade["symbol"],
                action=trade["action"],
                entry_time=trade_dt,
                entry_price=float(trade["price"]),
                quantity=float(trade.get("shares", 0.0)),
                fees=0.0,
                metadata=metadata,
            )
        )
    return result


def _serialize_portfolio_equity_curve(equity_curve: list[tuple]):
    """Convert portfolio equity points to repository-compatible tuples."""
    result: list[tuple[datetime, float, float]] = []
    peak = 0.0

    for point_in_time, equity in equity_curve:
        if isinstance(point_in_time, datetime):
            ts = point_in_time if point_in_time.tzinfo else point_in_time.replace(tzinfo=UTC)
        else:
            ts = datetime(
                point_in_time.year,
                point_in_time.month,
                point_in_time.day,
                tzinfo=UTC,
            )

        equity_value = float(equity)
        peak = max(peak, equity_value)
        drawdown = 0.0 if peak <= 0 else (peak - equity_value) / peak
        result.append((ts, equity_value, drawdown))

    return result


@celery_app.task(name="poseidon.workers.cpu_tasks.fetch_market_data")
def fetch_market_data(market: str, interval: str, symbol: str | None = None) -> dict:
    """Deprecated — data fetching moved to Thalassa.

    Task name kept so any stale Beat schedule entry does not crash the worker.
    """
    logger.warning("fetch_market_data: removed — data fetching is now handled by Thalassa")
    return {"skipped": "removed", "market": market, "interval": interval}


@celery_app.task(
    name="poseidon.workers.cpu_tasks.run_backtest_task",
    bind=True,
    max_retries=0,
)
def run_backtest_task(
    self,
    strategy_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 1_000_000.0,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
    fill_model: str | None = None,
    include_funding: bool = False,
) -> dict:
    """Run a backtest for an existing strategy.

    Loads the strategy from the DB, fetches OHLCV data, runs BacktestRunner,
    and persists results to BacktestRecord.

    Args:
        strategy_id: UUID string of the strategy to backtest.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).
        initial_capital: Starting capital for the backtest.
        sizing_mode: Position sizing mode (fixed_pct, fixed_notional, vol_target, fixed_risk).
        sizing_params: Additional sizing parameters dict.
        fill_model: Fill model for limit orders ('optimistic' or 'pessimistic').
        include_funding: Whether to include funding rate settlement costs.

    Returns:
        Dict with backtest_id, status, and trade_count.
    """
    backtest_id = uuid.uuid4()
    with db_session() as session:
        from poseidon.data.remote_repository import RemoteDataRepository

        repo = RemoteDataRepository.from_settings()
        try:
            # Load strategy record from DB
            sid = uuid.UUID(strategy_id)
            record = session.get(StrategyRecord, sid)
            if not record:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Reconstruct strategy object
            if record.strategy_type == "liquidity_sweep":
                strategy = LiquiditySweepStrategyFactory.from_config(record.config, strategy_id=record.id)
            elif record.strategy_type == "rule":
                strategy = build_rule_strategy(record.config, strategy_id=record.id)
            elif record.strategy_type == "voting":
                strategy = VotingStrategy(config=record.config, strategy_id=record.id)
            elif record.strategy_type == "model":
                # Route to GPU worker which has model dependencies
                from poseidon.workers.gpu_tasks import run_model_backtest

                result = run_model_backtest.delay(
                    strategy_id,
                    start_date,
                    end_date,
                    initial_capital,
                    sizing_mode,
                    sizing_params,
                )
                logger.info("Routed model backtest to GPU worker: %s", result.id)
                return {"backtest_id": result.id, "status": "routed_to_gpu", "trade_count": 0}
            elif record.strategy_type == "portfolio_strategy":
                from poseidon.backtest.portfolio_backtester import PortfolioBacktester
                from poseidon.backtest.schemas import BacktestResult

                parsed_start = datetime.fromisoformat(start_date) if start_date else None
                parsed_end = datetime.fromisoformat(end_date) if end_date else None

                strategy = _build_portfolio_strategy(record, repo)
                symbols = list(
                    dict.fromkeys((record.config or {}).get("symbols") or ([record.symbol] if record.symbol else []))
                )
                if not symbols:
                    raise ValueError("portfolio_strategy config must include at least one symbol")

                ohlcv_dict: dict[str, pd.DataFrame] = {}
                for symbol in symbols:
                    symbol_ohlcv = repo.read_ohlcv(
                        symbol,
                        record.market,
                        record.interval,
                        start=parsed_start,
                        end=parsed_end,
                    )
                    if not symbol_ohlcv.empty:
                        ohlcv_dict[symbol] = symbol_ohlcv

                if not ohlcv_dict:
                    raise ValueError(f"No OHLCV data for portfolio strategy {record.id}")

                if parsed_start is None:
                    parsed_start = min(pd.Timestamp(df.index.min()).to_pydatetime() for df in ohlcv_dict.values())
                if parsed_end is None:
                    parsed_end = max(pd.Timestamp(df.index.max()).to_pydatetime() for df in ohlcv_dict.values())

                backtester = PortfolioBacktester(
                    cost_model=COST_MODELS[record.market],
                    initial_capital=initial_capital,
                )
                portfolio_result = backtester.run(
                    strategy=strategy,
                    ohlcv_dict=ohlcv_dict,
                    start_date=parsed_start.date(),
                    end_date=parsed_end.date(),
                )

                trade_records = _serialize_portfolio_trades(portfolio_result.trades)
                equity_curve = _serialize_portfolio_equity_curve(portfolio_result.equity_curve)
                bt_config = BacktestConfig(
                    strategy_type=record.strategy_type,
                    symbol=record.symbol,
                    market=record.market,
                    interval=record.interval,
                    initial_capital=initial_capital,
                    start_date=parsed_start,
                    end_date=parsed_end,
                    strategy_params=record.config,
                    sizing_mode=sizing_mode,
                    sizing_params=sizing_params or {},
                )
                result_model = BacktestResult(
                    config=bt_config,
                    metrics=portfolio_result.metrics,
                    trade_count=len(trade_records),
                    equity_curve_length=len(equity_curve),
                    status=portfolio_result.status,
                    error_message=portfolio_result.error_message,
                    trades=portfolio_result.trades,
                )

                repo = BacktestRepository(session)
                repo.save_result(
                    config=bt_config,
                    result=result_model,
                    trades=trade_records,
                    equity_curve=equity_curve,
                    strategy_id=sid,
                    completed_at=datetime.now(UTC),
                )
                session.commit()

                return {
                    "backtest_id": str(result_model.backtest_id),
                    "status": portfolio_result.status,
                    "trade_count": len(trade_records),
                }
            else:
                raise ValueError(f"Unknown strategy_type: {record.strategy_type!r}")

            # Parse date filters
            parsed_start = datetime.fromisoformat(start_date) if start_date else None
            parsed_end = datetime.fromisoformat(end_date) if end_date else None

            # Load OHLCV data
            ohlcv_df = repo.read_ohlcv(
                record.symbol,
                record.market,
                record.interval,
                start=parsed_start,
                end=parsed_end,
            )
            if ohlcv_df.empty:
                raise ValueError(f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}")

            # Build pipeline components
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            cost_model = COST_MODELS[record.market]
            sizing_cfg = SizingConfig(
                mode=SizingMode(sizing_mode),
                **(sizing_params or {}),
            )

            # Resolve fill model enum
            from poseidon.backtest.pending_orders import FillModel

            fill_model_enum = FillModel(fill_model) if fill_model else None

            # Load funding rates if requested and market is crypto_perp
            funding_df = None
            if include_funding and record.market == "crypto_perp":
                from poseidon.models.funding_rate import FundingRateRecord

                rows = (
                    session.query(FundingRateRecord)
                    .filter(FundingRateRecord.symbol == record.symbol)
                    .order_by(FundingRateRecord.time)
                    .all()
                )
                if rows:
                    funding_df = pd.DataFrame([{"datetime": r.time, "funding_rate": r.funding_rate} for r in rows])
                    funding_df.index = pd.to_datetime(funding_df["datetime"], utc=True)
                    funding_df = funding_df.drop(columns=["datetime"])

            # Run backtest
            runner = BacktestRunner(
                strategy=strategy,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
                initial_capital=initial_capital,
                sizing_config=sizing_cfg,
                fill_model=fill_model_enum,
                include_funding=include_funding,
                funding_rates=funding_df,
            )
            result = runner.run(ohlcv_df)
            backtest_id = result.backtest_id

            # Build BacktestConfig for repository persistence
            bt_config = BacktestConfig(
                strategy_type=record.strategy_type,
                symbol=record.symbol,
                market=record.market,
                interval=record.interval,
                initial_capital=initial_capital,
                start_date=parsed_start,
                end_date=parsed_end,
                strategy_params=record.config,
            )

            # Persist via BacktestRepository (trades + equity curve included)
            repo = BacktestRepository(session)
            repo.save_result(
                config=bt_config,
                result=result,
                trades=runner.portfolio.trades if runner.portfolio else [],
                equity_curve=runner.portfolio.equity_curve if runner.portfolio else [],
                strategy_id=sid,
                completed_at=datetime.now(UTC),
            )
            session.commit()

            logger.info(
                "Backtest completed: %s (strategy=%s, trades=%d)",
                backtest_id,
                strategy_id,
                result.trade_count,
            )
            return {
                "backtest_id": str(backtest_id),
                "status": result.status,
                "trade_count": result.trade_count,
            }

        except Exception as exc:
            # Persist error state
            try:
                error_record = BacktestRecord(
                    id=backtest_id,
                    strategy_id=uuid.UUID(strategy_id) if strategy_id else None,
                    strategy_type="unknown",
                    symbol="",
                    market="",
                    interval="1d",
                    config={},
                    status="failed",
                    error_message=str(exc),
                )
                session.rollback()
                session.add(error_record)
                session.commit()
            except Exception:
                logger.exception("Failed to persist error backtest record")
            logger.exception("Backtest task failed for strategy %s", strategy_id)
            raise


@celery_app.task(
    name="poseidon.workers.cpu_tasks.run_dual_mode_task",
    bind=True,
    max_retries=0,
)
def run_dual_mode_task(
    self,
    strategy_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 1_000_000.0,
    include_funding: bool = False,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
) -> dict:
    """Run dual-mode fill comparison for a strategy.

    Runs the same strategy with both OPTIMISTIC and PESSIMISTIC fill models
    via run_dual_mode_comparison(), returns serialized DualModeResult.

    Args:
        strategy_id: UUID string of the strategy to compare.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).
        initial_capital: Starting capital for the backtest.
        include_funding: Whether to include funding rate settlement costs.
        sizing_mode: Position sizing mode.
        sizing_params: Additional sizing parameters dict.

    Returns:
        Dict with optimistic_metrics, pessimistic_metrics, delta_metrics, is_viable.
    """
    from poseidon.backtest.comparison import run_dual_mode_comparison

    with db_session() as session:
        from poseidon.data.remote_repository import RemoteDataRepository

        repo = RemoteDataRepository.from_settings()
        try:
            sid = uuid.UUID(strategy_id)
            record = session.get(StrategyRecord, sid)
            if not record:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Build strategy factory based on strategy_type
            if record.strategy_type == "voting":
                strategy_factory = lambda: VotingStrategy(config=record.config, strategy_id=record.id)  # noqa: E731
            elif record.strategy_type == "liquidity_sweep":
                strategy_factory = lambda: LiquiditySweepStrategyFactory.from_config(  # noqa: E731
                    record.config, strategy_id=record.id
                )
            elif record.strategy_type == "rule":
                strategy_factory = lambda: build_rule_strategy(record.config, strategy_id=record.id)  # noqa: E731
            elif record.strategy_type == "model":
                raise ValueError("Dual-mode comparison not supported for model strategies")
            else:
                raise ValueError(f"Unknown strategy_type: {record.strategy_type!r}")

            # Parse date filters
            parsed_start = datetime.fromisoformat(start_date) if start_date else None
            parsed_end = datetime.fromisoformat(end_date) if end_date else None

            # Load OHLCV data
            ohlcv_df = repo.read_ohlcv(
                record.symbol,
                record.market,
                record.interval,
                start=parsed_start,
                end=parsed_end,
            )
            if ohlcv_df.empty:
                raise ValueError(f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}")

            # Build pipeline components
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            cost_model = COST_MODELS[record.market]
            sizing_cfg = SizingConfig(
                mode=SizingMode(sizing_mode),
                **(sizing_params or {}),
            )

            # Load funding rates if requested
            funding_df = None
            if include_funding and record.market == "crypto_perp":
                from poseidon.models.funding_rate import FundingRateRecord

                rows = (
                    session.query(FundingRateRecord)
                    .filter(FundingRateRecord.symbol == record.symbol)
                    .order_by(FundingRateRecord.time)
                    .all()
                )
                if rows:
                    funding_df = pd.DataFrame([{"datetime": r.time, "funding_rate": r.funding_rate} for r in rows])
                    funding_df.index = pd.to_datetime(funding_df["datetime"], utc=True)
                    funding_df = funding_df.drop(columns=["datetime"])

            # Run dual-mode comparison
            dual_result = run_dual_mode_comparison(
                strategy_factory=strategy_factory,
                ohlcv=ohlcv_df,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
                initial_capital=initial_capital,
                sizing_config=sizing_cfg,
                include_funding=include_funding,
                db_session=session,
                funding_rates=funding_df,
            )

            logger.info(
                "Dual-mode comparison completed: strategy=%s, is_viable=%s",
                strategy_id,
                dual_result.is_viable,
            )
            return {
                "optimistic_metrics": dual_result.optimistic_result.metrics,
                "pessimistic_metrics": dual_result.pessimistic_result.metrics,
                "delta_metrics": dual_result.delta_metrics,
                "is_viable": dual_result.is_viable,
            }

        except Exception:
            logger.exception("Dual-mode task failed for strategy %s", strategy_id)
            raise


@celery_app.task(
    name="poseidon.workers.cpu_tasks.run_optimization_task",
    bind=True,
    max_retries=0,
)
def run_optimization_task(
    self,
    strategy_id: str,
    param_grid: dict,
    method: str = "grid",
    n_trials: int = 50,
    target_metric: str = "sharpe_ratio",
    start_date: str | None = None,
    end_date: str | None = None,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
) -> dict:
    """Run parameter optimization for an existing strategy.

    Loads the strategy template from the DB, runs grid or Bayesian optimization
    over the param_grid, and persists the best result to BacktestRecord.

    Args:
        strategy_id: UUID string of the strategy to optimize.
        param_grid: Parameter grid/space to search.
        method: Optimization method ("grid" or "bayesian").
        n_trials: Number of trials for Bayesian optimization.
        target_metric: Metric to maximize (default: sharpe_ratio).
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).

    Returns:
        Dict with trial count, best params, and best metric value.
    """
    with db_session() as session:
        from poseidon.data.remote_repository import RemoteDataRepository

        repo = RemoteDataRepository.from_settings()
        try:
            # Load strategy record from DB
            sid = uuid.UUID(strategy_id)
            record = session.get(StrategyRecord, sid)
            if not record:
                raise ValueError(f"Strategy {strategy_id} not found")

            if record.strategy_type != "rule":
                raise NotImplementedError(f"Optimization for strategy_type={record.strategy_type!r} not yet supported")

            # Parse date filters
            parsed_start = datetime.fromisoformat(start_date) if start_date else None
            parsed_end = datetime.fromisoformat(end_date) if end_date else None

            # Load OHLCV data
            ohlcv_df = repo.read_ohlcv(
                record.symbol,
                record.market,
                record.interval,
                start=parsed_start,
                end=parsed_end,
            )
            if ohlcv_df.empty:
                raise ValueError(f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}")

            # Build pipeline components
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            cost_model = COST_MODELS[record.market]
            sizing_cfg = SizingConfig(
                mode=SizingMode(sizing_mode),
                **(sizing_params or {}),
            )

            # Strategy factory: merges param_grid values into the base config
            base_config = dict(record.config) if record.config else {}

            def strategy_factory(params: dict) -> RuleStrategy:
                merged = {**base_config, **params}
                return RuleStrategy(config=merged, strategy_id=record.id)

            # Run optimization
            if method == "grid":
                optimizer = GridSearchOptimizer(
                    feature_engine=feature_engine,
                    risk_engine=risk_engine,
                    cost_model=cost_model,
                    initial_capital=1_000_000.0,
                    sizing_config=sizing_cfg,
                )
                trials = optimizer.optimize(
                    strategy_factory=strategy_factory,
                    ohlcv=ohlcv_df,
                    param_grid=param_grid,
                    metric=target_metric,
                )
            elif method == "bayesian":
                optimizer = BayesianOptimizer(
                    feature_engine=feature_engine,
                    risk_engine=risk_engine,
                    cost_model=cost_model,
                    initial_capital=1_000_000.0,
                    sizing_config=sizing_cfg,
                )
                # Convert param_grid to Bayesian param_space format
                # Expected input: {"param": [low, high, "int"|"float"]}
                param_space = {}
                for key, spec in param_grid.items():
                    if isinstance(spec, (list, tuple)) and len(spec) == 3:
                        param_space[key] = (spec[0], spec[1], spec[2])
                    else:
                        raise ValueError(f"Bayesian param_grid[{key!r}] must be [low, high, type], got {spec!r}")
                trials = optimizer.optimize(
                    strategy_factory=strategy_factory,
                    ohlcv=ohlcv_df,
                    param_space=param_space,
                    n_trials=n_trials,
                    metric=target_metric,
                )
            else:
                raise ValueError(f"Unknown optimization method: {method!r}. Use 'grid' or 'bayesian'.")

            # Persist best result as a BacktestRecord with walk_forward metadata
            best = trials[0] if trials else None
            if best:
                bt_record = BacktestRecord(
                    id=uuid.uuid4(),
                    strategy_id=sid,
                    strategy_type=record.strategy_type,
                    symbol=record.symbol,
                    market=record.market,
                    interval=record.interval,
                    config={
                        "optimization_method": method,
                        "param_grid": param_grid,
                        "target_metric": target_metric,
                        "strategy_params": record.config,
                    },
                    metrics=best.metrics,
                    walk_forward={
                        "best_params": best.params,
                        "best_metric_value": best.metric_value,
                        "total_trials": len(trials),
                    },
                    status="completed",
                    completed_at=datetime.now(UTC),
                )
                session.add(bt_record)
                session.commit()

            logger.info(
                "Optimization completed: strategy=%s method=%s trials=%d best_metric=%.4f",
                strategy_id,
                method,
                len(trials),
                best.metric_value if best else 0.0,
            )
            return {
                "trials": len(trials),
                "best_params": best.params if best else {},
                "best_metric": best.metric_value if best else 0.0,
            }

        except Exception:
            logger.exception("Optimization task failed for strategy %s", strategy_id)
            raise
