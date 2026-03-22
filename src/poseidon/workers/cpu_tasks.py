"""CPU worker Celery tasks for data fetching, backfill, backtest, and optimization."""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.optimizer import BayesianOptimizer, GridSearchOptimizer
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.fetchers import get_fetcher
from poseidon.data.storage import (
    get_or_create_backfill_progress,
    read_ohlcv,
    update_backfill_progress,
    upsert_ohlcv,
)
from poseidon.data.symbols import get_market_config, get_symbols_for_market, load_symbols
from poseidon.models.backtest import BacktestRecord
from poseidon.models.base import SessionLocal
from poseidon.models.strategy import StrategyRecord
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Backfill target: 5 years of historical data
BACKFILL_YEARS = 5

# Batch sizes for backfill pagination (per provider)
BATCH_DAYS = {
    "tw_stock": 365,       # FinMind: 1 year per request
    "tw_futures": 365,     # FinMind: 1 year per request
    "us_stock": 1825,      # yfinance: all 5 years in one call
    "crypto_spot": 41,     # CCXT 1h: ~41 days per 1000 candles
}
BATCH_DAYS_DAILY = {
    "crypto_spot": 900,    # CCXT 1d: ~2.7 years per 1000 candles
}
BATCH_DAYS_5M = {
    "crypto_spot": 3,      # CCXT 5m: ~3.47 days per 1000 candles
}


@celery_app.task(name="poseidon.workers.cpu_tasks.fetch_market_data")
def fetch_market_data(market: str, interval: str, symbol: str | None = None) -> dict:
    """Fetch latest data for all symbols (or a specific symbol) in a market.

    This is the task called by Celery Beat on schedule.

    Args:
        market: Market name (e.g., "crypto_spot", "tw_stock").
        interval: Candle interval ("1d" or "1h").
        symbol: Optional symbol ID to fetch. If None, fetches all symbols in the market.
    """
    config = load_symbols()
    symbols = get_symbols_for_market(market, config)
    market_cfg = get_market_config(market, config)

    # Filter to specific symbol if requested
    if symbol and symbols:
        symbols = [s for s in symbols if s.id == symbol or s.ccxt_symbol == symbol]

    # If a specific symbol was requested but not found in config, create an ad-hoc entry
    if not symbols and symbol:
        from poseidon.data.symbols import SymbolInfo
        symbols = [SymbolInfo(id=symbol, name=symbol)]
        logger.info("Symbol %s not in config, fetching ad-hoc for market %s", symbol, market)
    elif not symbols:
        logger.warning("No symbols configured for market: %s", market)
        return {"market": market, "interval": interval, "fetched": 0}

    fetcher = get_fetcher(market)
    instrument = market_cfg.instrument if market_cfg else "spot"

    # Determine date range: fetch last 7 days to catch any missed data
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    fetched_count = 0
    session = SessionLocal()
    try:
        for sym_info in symbols:
            try:
                # Use ccxt_symbol if available (crypto), otherwise use id
                fetch_symbol = sym_info.ccxt_symbol or sym_info.id
                df = fetcher.fetch_ohlcv(fetch_symbol, interval, start_date, end_date)
                if not df.empty:
                    count = upsert_ohlcv(session, df, sym_info.id, market, instrument, interval)
                    fetched_count += count
                    logger.info("Fetched %d rows for %s/%s/%s", count, market, sym_info.id, interval)
                else:
                    logger.info("No new data for %s/%s/%s", market, sym_info.id, interval)
            except Exception as exc:
                session.rollback()
                logger.error("Failed to fetch %s/%s/%s: %s", market, sym_info.id, interval, exc)
    finally:
        session.close()

    logger.info("Completed fetch for %s/%s: %d total rows", market, interval, fetched_count)
    return {"market": market, "interval": interval, "fetched": fetched_count}


@celery_app.task(
    name="poseidon.workers.cpu_tasks.backfill_symbol",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def backfill_symbol(self, symbol: str, market: str, interval: str) -> dict:
    """Backfill historical data for a single symbol. Resumable via backfill_progress table.

    Args:
        symbol: Symbol ID (e.g., "2330", "BTCUSDT")
        market: Market name (e.g., "tw_stock", "crypto_spot")
        interval: Candle interval ("1d" or "1h")

    Returns:
        Dict with backfill status.
    """
    session = SessionLocal()
    try:
        target_start = datetime.now(timezone.utc) - timedelta(days=BACKFILL_YEARS * 365)
        progress = get_or_create_backfill_progress(session, symbol, market, interval, target_start)

        if progress.status == "completed":
            logger.info("Backfill already completed for %s/%s/%s", market, symbol, interval)
            return {"symbol": symbol, "market": market, "interval": interval, "status": "already_completed"}

        update_backfill_progress(session, progress, status="in_progress")

        # Determine the fetcher and instrument
        config = load_symbols()
        market_cfg = get_market_config(market, config)
        fetcher = get_fetcher(market)
        instrument = market_cfg.instrument if market_cfg else "spot"

        # Determine the ccxt_symbol for crypto
        symbols_list = get_symbols_for_market(market, config)
        fetch_symbol = symbol
        for s in symbols_list:
            if s.id == symbol and s.ccxt_symbol:
                fetch_symbol = s.ccxt_symbol
                break

        # Resume from last checkpoint or start from target
        start_dt = progress.last_fetched_date or progress.target_start_date
        end_dt = datetime.now(timezone.utc)

        # Choose batch size based on market and interval
        if interval == "1d" and market in BATCH_DAYS_DAILY:
            batch_days = BATCH_DAYS_DAILY[market]
        elif interval == "5m" and market in BATCH_DAYS_5M:
            batch_days = BATCH_DAYS_5M[market]
        else:
            batch_days = BATCH_DAYS.get(market, 365)

        try:
            # Fetch in batches
            current_start = start_dt
            total_rows = 0
            while current_start < end_dt:
                batch_end = min(current_start + timedelta(days=batch_days), end_dt)
                start_str = current_start.strftime("%Y-%m-%d")
                end_str = batch_end.strftime("%Y-%m-%d")

                logger.info("Backfill %s/%s/%s: %s to %s", market, symbol, interval, start_str, end_str)
                df = fetcher.fetch_ohlcv(fetch_symbol, interval, start_str, end_str)

                if not df.empty:
                    count = upsert_ohlcv(session, df, symbol, market, instrument, interval)
                    total_rows += count
                    # Update checkpoint after each batch
                    update_backfill_progress(session, progress, status="in_progress", last_fetched_date=batch_end)

                current_start = batch_end + timedelta(days=1)

            update_backfill_progress(session, progress, status="completed", last_fetched_date=end_dt)
            logger.info("Backfill completed for %s/%s/%s: %d total rows", market, symbol, interval, total_rows)
            return {"symbol": symbol, "market": market, "interval": interval, "status": "completed", "rows": total_rows}

        except Exception as exc:
            update_backfill_progress(session, progress, status="failed", error_message=str(exc))
            logger.error("Backfill failed for %s/%s/%s: %s", market, symbol, interval, exc)
            raise self.retry(exc=exc)

    finally:
        session.close()


@celery_app.task(name="poseidon.workers.cpu_tasks.trigger_backfill")
def trigger_backfill(market: str | None = None, symbol: str | None = None) -> dict:
    """Trigger backfill for all symbols (or a specific market/symbol).

    Dispatches individual backfill_symbol tasks for each symbol/interval combo.
    If a specific symbol is requested but not in config, it is still backfilled
    with the default interval ("1d").

    Args:
        market: Optional market to limit backfill to.
        symbol: Optional symbol ID to limit backfill to.
    """
    config = load_symbols()
    dispatched = 0

    markets_to_process = [market] if market else list(config.markets.keys())

    for m in markets_to_process:
        market_cfg = config.markets.get(m)
        if not market_cfg:
            # Market not in config but symbol explicitly requested — backfill with default interval
            if symbol and market:
                backfill_symbol.delay(symbol, m, "1d")
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/1d (not in config)", m, symbol)
            continue
        configured_ids = {sym.id for sym in market_cfg.symbols}
        for sym in market_cfg.symbols:
            if symbol and sym.id != symbol:
                continue
            for interval in market_cfg.intervals:
                backfill_symbol.delay(sym.id, m, interval)
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/%s", m, sym.id, interval)

        # Symbol requested but not in this market's config — backfill anyway
        if symbol and symbol not in configured_ids:
            for interval in market_cfg.intervals:
                backfill_symbol.delay(symbol, m, interval)
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/%s (not in config)", m, symbol, interval)

    logger.info("Dispatched %d backfill tasks", dispatched)
    return {"dispatched": dispatched}


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
) -> dict:
    """Run a backtest for an existing strategy.

    Loads the strategy from the DB, fetches OHLCV data, runs BacktestRunner,
    and persists results to BacktestRecord.

    Args:
        strategy_id: UUID string of the strategy to backtest.
        start_date: ISO date string for backtest start (optional).
        end_date: ISO date string for backtest end (optional).
        initial_capital: Starting capital for the backtest.

    Returns:
        Dict with backtest_id, status, and trade_count.
    """
    session = SessionLocal()
    backtest_id = uuid.uuid4()
    try:
        # Load strategy record from DB
        sid = uuid.UUID(strategy_id)
        record = session.get(StrategyRecord, sid)
        if not record:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Reconstruct strategy object
        if record.strategy_type == "rule":
            strategy = RuleStrategy(config=record.config, strategy_id=record.id)
        elif record.strategy_type == "model":
            # Route to GPU worker which has model dependencies
            from poseidon.workers.gpu_tasks import run_model_backtest
            result = run_model_backtest.delay(
                strategy_id, start_date, end_date, initial_capital,
                sizing_mode, sizing_params,
            )
            logger.info("Routed model backtest to GPU worker: %s", result.id)
            return {"backtest_id": result.id, "status": "routed_to_gpu", "trade_count": 0}
        else:
            raise ValueError(f"Unknown strategy_type: {record.strategy_type!r}")

        # Parse date filters
        parsed_start = datetime.fromisoformat(start_date) if start_date else None
        parsed_end = datetime.fromisoformat(end_date) if end_date else None

        # Load OHLCV data
        ohlcv_df = read_ohlcv(
            session, record.symbol, record.market, record.interval,
            start=parsed_start, end=parsed_end,
        )
        if ohlcv_df.empty:
            raise ValueError(
                f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}"
            )

        # Build pipeline components
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine()
        cost_model = COST_MODELS[record.market]
        sizing_cfg = SizingConfig(
            mode=SizingMode(sizing_mode),
            **(sizing_params or {}),
        )

        # Run backtest
        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=initial_capital,
            sizing_config=sizing_cfg,
        )
        result = runner.run(ohlcv_df)
        backtest_id = result.backtest_id

        # Build config dict for persistence
        config_dict = {
            "strategy_type": record.strategy_type,
            "symbol": record.symbol,
            "market": record.market,
            "interval": record.interval,
            "initial_capital": initial_capital,
            "start_date": start_date,
            "end_date": end_date,
            "strategy_params": record.config,
        }

        # Persist to BacktestRecord directly (the repository expects TradeRecord
        # objects, but we only have trade dicts from BacktestResult; persist
        # the main record with metrics for the API to query)
        bt_record = BacktestRecord(
            id=backtest_id,
            strategy_id=sid,
            strategy_type=record.strategy_type,
            symbol=record.symbol,
            market=record.market,
            interval=record.interval,
            config=config_dict,
            metrics=result.metrics,
            status=result.status,
            error_message=result.error_message,
            completed_at=datetime.now(timezone.utc),
        )
        session.add(bt_record)
        session.commit()

        logger.info(
            "Backtest completed: %s (strategy=%s, trades=%d)",
            backtest_id, strategy_id, result.trade_count,
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

    finally:
        session.close()


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
    session = SessionLocal()
    try:
        # Load strategy record from DB
        sid = uuid.UUID(strategy_id)
        record = session.get(StrategyRecord, sid)
        if not record:
            raise ValueError(f"Strategy {strategy_id} not found")

        if record.strategy_type != "rule":
            raise NotImplementedError(
                f"Optimization for strategy_type={record.strategy_type!r} not yet supported"
            )

        # Parse date filters
        parsed_start = datetime.fromisoformat(start_date) if start_date else None
        parsed_end = datetime.fromisoformat(end_date) if end_date else None

        # Load OHLCV data
        ohlcv_df = read_ohlcv(
            session, record.symbol, record.market, record.interval,
            start=parsed_start, end=parsed_end,
        )
        if ohlcv_df.empty:
            raise ValueError(
                f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}"
            )

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
                    raise ValueError(
                        f"Bayesian param_grid[{key!r}] must be [low, high, type], "
                        f"got {spec!r}"
                    )
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
                completed_at=datetime.now(timezone.utc),
            )
            session.add(bt_record)
            session.commit()

        logger.info(
            "Optimization completed: strategy=%s method=%s trials=%d best_metric=%.4f",
            strategy_id, method, len(trials),
            best.metric_value if best else 0.0,
        )
        return {
            "trials": len(trials),
            "best_params": best.params if best else {},
            "best_metric": best.metric_value if best else 0.0,
        }

    except Exception as exc:
        logger.exception("Optimization task failed for strategy %s", strategy_id)
        raise

    finally:
        session.close()
