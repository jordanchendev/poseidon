"""GPU worker Celery tasks for model training and model-based backtesting.

Tasks in this module are routed to the ``gpu`` queue via the
``poseidon.workers.gpu_tasks.*`` routing rule in ``celery_app.py``.
"""

from __future__ import annotations

import logging
import traceback
import uuid as uuid_mod
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np

from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="poseidon.workers.gpu_tasks.train_model",
    bind=True,
    max_retries=0,
)
def train_model(
    self,
    model_name: str,
    version_id: str,
    symbol: str,
    market: str,
    interval: str = "1d",
    params: dict | None = None,
    feature_list: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Train a model version end-to-end.

    Orchestrates the full pipeline:
    1. Load OHLCV data from the database.
    2. Compute features via :class:`FeatureEngine`.
    3. Instantiate the model class from the registry.
    4. Train the model on feature data.
    5. Save model artifacts to the version directory.
    6. Transition the model version to ``ready`` (or ``failed`` on error).

    Returns:
        dict with ``version_id`` and ``status`` on success.
    """
    # Defer heavy imports so the module can be imported without side-effects
    # during testing / import-time checks.
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.storage import read_ohlcv
    from poseidon.ml.manager import ModelManager
    from poseidon.ml.registry import get_model
    from poseidon.models.base import SessionLocal

    session = SessionLocal()
    manager = None
    try:
        manager = ModelManager(session)
        vid = UUID(version_id)

        mv = manager.get_version(vid)
        if mv is None:
            raise ValueError(f"Model version not found: {version_id}")

        logger.info(
            "Starting training for %s v%d (id=%s) — symbol=%s market=%s",
            mv.name,
            mv.version,
            version_id,
            symbol,
            market,
        )

        # 1. Load OHLCV data
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        ohlcv_df = read_ohlcv(session, symbol, market, interval, start=start_dt, end=end_dt)

        if ohlcv_df.empty:
            raise ValueError(
                f"No OHLCV data found for {symbol}/{market}/{interval}"
            )

        # 2. Compute features
        engine = FeatureEngine()
        feature_specs = None
        if feature_list:
            # Convert simple feature names to (name, {}) tuples if needed
            feature_specs = [(f, {}) for f in feature_list]
        elif params and params.get("label_mode") == "regime":
            # Auto-select regime-optimized features when no explicit list given
            from poseidon.data.feature_engine import REGIME_FEATURES
            feature_specs = REGIME_FEATURES
        features_df = engine.compute_from_df(ohlcv_df, feature_specs=feature_specs)

        # 3. Instantiate model
        model_cls = get_model(model_name)
        model_instance = model_cls()
        effective_params = params or model_instance.get_default_params()

        # 4. Build classification targets.
        if "close" not in ohlcv_df.columns:
            raise ValueError("OHLCV data missing 'close' column for target construction")

        label_mode = effective_params.pop("label_mode", "direction")
        test_ratio = effective_params.pop("test_ratio", 0.2)

        if label_mode == "direction":
            # Direction labels: 0=hold, 1=long, 2=short
            threshold = effective_params.pop("label_threshold", 0.005)
            future_returns = ohlcv_df["close"].pct_change().shift(-1).dropna()
            targets = future_returns.map(
                lambda r: 1 if r > threshold else (2 if r < -threshold else 0)
            )
        elif label_mode == "regime":
            # Volatility regime labels: 0=low_vol, 1=medium_vol, 2=high_vol
            regime_horizon = effective_params.pop("regime_horizon", 10)
            log_returns = np.log(ohlcv_df["close"] / ohlcv_df["close"].shift(1))
            # Forward-looking realized vol: std of next N log returns
            forward_vol = (
                log_returns.shift(-1)
                .rolling(window=regime_horizon)
                .std()
                .shift(-(regime_horizon - 1))
            )
            forward_vol = forward_vol.dropna()

            # Classify into 3 regimes using percentile thresholds
            low_pct = effective_params.pop("regime_low_percentile", 33)
            high_pct = effective_params.pop("regime_high_percentile", 67)
            low_threshold = forward_vol.quantile(low_pct / 100)
            high_threshold = forward_vol.quantile(high_pct / 100)

            targets = forward_vol.map(
                lambda v: 0 if v <= low_threshold else (2 if v >= high_threshold else 1)
            )
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}. Use 'direction' or 'regime'.")
        common_idx = features_df.index.intersection(targets.index)
        features_df = features_df.loc[common_idx]
        targets = targets.loc[common_idx]

        # Time-based train/test split — train on earlier data, validate on later
        split_idx = int(len(features_df) * (1 - test_ratio))
        train_features = features_df.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        test_features = features_df.iloc[split_idx:]
        test_targets = targets.iloc[split_idx:]

        if len(train_features) < 50:
            raise ValueError(f"Not enough training data: {len(train_features)} rows (need >= 50)")

        metrics = model_instance.train(train_features, train_targets, effective_params)

        # Out-of-sample validation
        if len(test_features) > 0:
            oos_metrics = model_instance.validate(test_features, test_targets)
            metrics["train_accuracy"] = metrics.pop("accuracy", 0)
            metrics["train_samples"] = metrics.pop("n_samples", 0)
            metrics["oos_accuracy"] = oos_metrics.get("accuracy", 0)
            metrics["oos_samples"] = oos_metrics.get("n_samples", 0)
            metrics["split_ratio"] = f"{split_idx}/{len(features_df)}"

        # 5. Save artifacts
        artifact_path = mv.artifact_path
        if artifact_path:
            model_instance.save(Path(artifact_path))

        # 6. Transition to ready
        manager.transition(vid, "ready", metrics=metrics)
        logger.info("Training complete for %s v%d — status=ready", mv.name, mv.version)

        return {"version_id": version_id, "status": "ready"}

    except Exception:
        logger.exception("Training failed for version %s", version_id)
        # Best-effort transition to failed (guard against manager not yet assigned)
        if manager is not None:
            try:
                manager.transition(
                    UUID(version_id),
                    "failed",
                    error_message=traceback.format_exc()[-500:],
                )
            except Exception:
                logger.exception("Failed to transition version %s to 'failed'", version_id)
        raise
    finally:
        session.close()


@celery_app.task(
    name="poseidon.workers.gpu_tasks.run_model_backtest",
    bind=True,
    max_retries=0,
)
def run_model_backtest(
    self,
    strategy_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 1_000_000.0,
    sizing_mode: str = "fixed_notional",
    sizing_params: dict | None = None,
) -> dict:
    """Run a backtest for a model-based strategy on the GPU worker.

    Loads the model from artifacts, runs predictions on historical data,
    and executes the backtest via BacktestRunner.
    """
    from poseidon.backtest.cost_model import COST_MODELS
    from poseidon.backtest.runner import BacktestRunner
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.storage import read_ohlcv
    from poseidon.ml.manager import ModelManager
    from poseidon.ml.registry import get_model
    from poseidon.models.backtest import BacktestRecord
    from poseidon.models.base import SessionLocal
    from poseidon.models.strategy import StrategyRecord
    from poseidon.risk.engine import RiskEngine
    from poseidon.strategies.model_strategy import ModelStrategy

    session = SessionLocal()
    backtest_id = uuid_mod.uuid4()
    try:
        sid = UUID(strategy_id)
        record = session.get(StrategyRecord, sid)
        if not record:
            raise ValueError(f"Strategy {strategy_id} not found")

        if record.strategy_type != "model":
            raise ValueError(f"Strategy {strategy_id} is not a model strategy")

        # Load model version
        manager = ModelManager(session)
        mv = manager.get_version(record.model_version_id)
        if mv is None or mv.status != "ready":
            raise ValueError(f"Model version {record.model_version_id} not ready")

        # Load model from artifacts
        model_cls = get_model(mv.name)
        if mv.artifact_path:
            model_instance = model_cls.load(Path(mv.artifact_path))
        else:
            raise ValueError(f"Model version {mv.id} has no artifact path")

        # Build strategy
        strategy = ModelStrategy(
            name=record.name,
            model=model_instance,
            symbol=record.symbol,
            market=record.market,
            interval=record.interval,
            strategy_id=record.id,
        )

        # Parse dates and load data
        parsed_start = datetime.fromisoformat(start_date) if start_date else None
        parsed_end = datetime.fromisoformat(end_date) if end_date else None
        ohlcv_df = read_ohlcv(
            session, record.symbol, record.market, record.interval,
            start=parsed_start, end=parsed_end,
        )
        if ohlcv_df.empty:
            raise ValueError(f"No OHLCV data for {record.symbol}/{record.market}/{record.interval}")

        # Run backtest
        from poseidon.backtest.portfolio import SizingConfig, SizingMode

        feature_engine = FeatureEngine()
        risk_engine = RiskEngine()
        cost_model = COST_MODELS[record.market]
        sizing_cfg = SizingConfig(
            mode=SizingMode(sizing_mode),
            **(sizing_params or {}),
        )

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=initial_capital,
            sizing_config=sizing_cfg,
        )
        result = runner.run(ohlcv_df)

        # Persist result
        bt_record = BacktestRecord(
            id=backtest_id,
            strategy_id=sid,
            strategy_type=record.strategy_type,
            symbol=record.symbol,
            market=record.market,
            interval=record.interval,
            config={
                "strategy_type": "model",
                "model_version_id": str(record.model_version_id),
                "model_name": mv.name,
                "model_version": mv.version,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
            },
            metrics=result.metrics,
            status=result.status,
            error_message=result.error_message,
            completed_at=datetime.now(),
        )
        session.add(bt_record)
        session.commit()

        logger.info(
            "Model backtest completed: %s (strategy=%s, trades=%d)",
            backtest_id, strategy_id, result.metrics.get("trade_count", 0),
        )
        return {
            "backtest_id": str(backtest_id),
            "status": "completed",
            "trade_count": result.metrics.get("trade_count", 0),
        }

    except Exception:
        logger.exception("Model backtest failed for strategy %s", strategy_id)
        session.rollback()
        try:
            failed_record = BacktestRecord(
                id=backtest_id,
                strategy_id=UUID(strategy_id),
                status="failed",
                error_message=traceback.format_exc()[-500:],
            )
            session.add(failed_record)
            session.commit()
        except Exception:
            logger.exception("Failed to save failed backtest record")
        raise
    finally:
        session.close()
