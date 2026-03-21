"""GPU worker Celery tasks for model training.

Tasks in this module are routed to the ``gpu`` queue via the
``poseidon.workers.gpu_tasks.*`` routing rule in ``celery_app.py``.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path
from uuid import UUID

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
        features_df = engine.compute_from_df(ohlcv_df, feature_specs=feature_specs)

        # 3. Instantiate model
        model_cls = get_model(model_name)
        model_instance = model_cls()
        effective_params = params or model_instance.get_default_params()

        # 4. Build classification targets from future returns.
        # Convert continuous returns to 3-class labels: 0=hold, 1=long, 2=short
        # using a threshold (default 0.5% move).
        if "close" not in ohlcv_df.columns:
            raise ValueError("OHLCV data missing 'close' column for target construction")

        threshold = effective_params.pop("label_threshold", 0.005)
        future_returns = ohlcv_df["close"].pct_change().shift(-1).dropna()
        # Classify: long if return > threshold, short if < -threshold, else hold
        targets = future_returns.map(
            lambda r: 1 if r > threshold else (2 if r < -threshold else 0)
        )
        common_idx = features_df.index.intersection(targets.index)
        features_df = features_df.loc[common_idx]
        targets = targets.loc[common_idx]

        metrics = model_instance.train(features_df, targets, effective_params)

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
