"""Qlib training tasks -- runs in cp312 qlib-research container only.

All qlib/mlflow imports are deferred to inside the task body to prevent
ImportError in cp313 containers that discover this module via celery_app.
Per Phase 41 D-12: separate module from cpu_tasks/gpu_tasks.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone

from poseidon.models.base import SessionLocal
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_cancelled(session, run_id: str) -> bool:
    """Re-read the TrainingRun row to check for cooperative cancel.

    A fresh read is required because the ORM session cache could otherwise
    hold a stale ``status`` from before the API cancel committed.
    Same pattern as backfill_tasks._job_cancelled.
    """
    from poseidon.models.training_run import TrainingRun

    run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).one()
    session.refresh(run)
    return run.status == "cancelled"


@celery_app.task(
    name="poseidon.workers.qlib_tasks.qlib_train",
    queue="qlib_queue",
    bind=True,
    max_retries=0,
)
def qlib_train(self, run_id: str) -> dict:
    """Execute a Qlib model training run with full lifecycle management.

    Transitions TrainingRun through pending -> running -> succeeded/failed.
    Only succeeded runs produce a ModelVersion via QlibModelExporter.export().
    Cooperative cancel checks training_runs.status between Qlib segments.

    All qlib/mlflow imports are inside the function body so cp313 containers
    that auto-discover this module via celery_app.conf.imports do not crash.
    """
    session = SessionLocal()
    try:
        from poseidon.models.training_run import TrainingRun

        # 1. Load the training run row
        run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).first()
        if run is None:
            logger.warning("Training run %s not found, skipping", run_id)
            return {"run_id": run_id, "status": "not_found"}
        if run.status != "pending":
            logger.info("Training run %s status is %s, not pending -- skipping", run_id, run.status)
            return {"run_id": run_id, "status": run.status, "noop": True}

        # 2. Transition to running
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        session.commit()

        # 3. Import Qlib/MLflow inside task body (Research Pitfall 2)
        import importlib
        import os
        from pathlib import Path

        import mlflow
        import pandas as pd
        import qlib
        from qlib.config import REG_CN
        from qlib.data.dataset import DatasetH
        from poseidon.qlib.allowlist import resolve_model
        from poseidon.qlib.data_handler import PoseidonDataHandler
        from poseidon.qlib.dataset_builder import DatasetBuilder
        from poseidon.qlib.model_exporter import QlibModelExporter

        # 4. Initialize Qlib (Research Pitfall 5)
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

        # 5. Set MLflow tracking URI from environment + clean stale state
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        # End any stale MLflow run left from a previous failed task
        mlflow.end_run()

        # 6. Build dataset via DatasetBuilder -> PoseidonDataHandler -> DataHandlerLP -> DatasetH
        segments = run.segments  # dict from JSONB: {"train": ["start", "end"], ...}

        # Compute the full date range spanning all segments
        all_dates: list[str] = []
        for _seg_name, seg_dates in segments.items():
            all_dates.extend([seg_dates[0], seg_dates[1]])
        overall_start = min(all_dates)
        overall_end = max(all_dates)

        # Step 6b: Create DatasetBuilder (reads OHLCV from TimescaleDB)
        ds_builder = DatasetBuilder(session=session, market=run.market, interval=run.interval)

        # Step 6c: Create PoseidonDataHandler (calls ds_builder.build() internally)
        handler = PoseidonDataHandler(
            dataset_builder=ds_builder,
            symbols=run.symbols,
            start=datetime.fromisoformat(overall_start),
            end=datetime.fromisoformat(overall_end),
        )

        # Step 6d: Convert to Qlib-native DataHandlerLP
        qlib_handler = handler.to_qlib_handler()

        # Step 6e: Build Qlib DatasetH with time-based segments
        qlib_segments = {
            seg_name: (seg_dates[0], seg_dates[1])
            for seg_name, seg_dates in segments.items()
        }
        dataset = DatasetH(handler=qlib_handler, segments=qlib_segments)

        # CHECK CANCEL after dataset build
        if _run_cancelled(session, run_id):
            run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).one()
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            return {"run_id": run_id, "status": "cancelled"}

        # 7. Train model using the resolved model class
        model_fqn = resolve_model(run.model_class)
        module_path, class_name = model_fqn.rsplit(".", 1)
        model_module = importlib.import_module(module_path)
        model_cls = getattr(model_module, class_name)

        trained_model = model_cls(**run.model_params)
        trained_model.fit(dataset)

        # CHECK CANCEL after training
        if _run_cancelled(session, run_id):
            run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).one()
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            return {"run_id": run_id, "status": "cancelled"}

        # 8. Generate predictions on test segment
        test_predictions = trained_model.predict(dataset, segment="test")

        # 9. Record with MLflow directly (bypass Qlib Recorder to avoid state conflicts)
        mlflow_run_id = None
        mlflow.end_run()  # ensure no stale run
        mlflow.set_experiment(f"phase41_{run.run_id}")
        with mlflow.start_run() as mlflow_active:
            mlflow_run_id = mlflow_active.info.run_id
            # Log model params
            mlflow.log_params(run.model_params or {})
            mlflow.log_param("handler_class", run.handler_class)
            mlflow.log_param("model_class", run.model_class)
            mlflow.log_param("market", run.market)
            # Compute and log IC/ICIR if predictions available
            if test_predictions is not None:
                try:
                    pred_series = test_predictions if isinstance(test_predictions, pd.Series) else test_predictions.iloc[:, 0]
                    label_series = dataset.prepare("test", col_set="label").iloc[:, 0]
                    # Align indices
                    common_idx = pred_series.index.intersection(label_series.index)
                    if len(common_idx) > 0:
                        p = pred_series.loc[common_idx]
                        l = label_series.loc[common_idx]
                        ic = p.corr(l)
                        # Group IC by date for ICIR
                        ic_by_date = p.groupby(level="datetime").corrwith(l.groupby(level="datetime")) if hasattr(p.groupby(level="datetime"), 'corrwith') else None
                        if ic_by_date is not None and len(ic_by_date) > 1:
                            icir = ic_by_date.mean() / ic_by_date.std()
                        else:
                            icir = 0.0
                        mlflow.log_metric("IC", float(ic) if pd.notna(ic) else 0.0)
                        mlflow.log_metric("ICIR", float(icir) if pd.notna(icir) else 0.0)
                except Exception as metric_exc:
                    logger.warning("Failed to compute IC/ICIR: %s", metric_exc)

        # 10. Prepare predictions DataFrame for Parquet
        pred_df = None
        if test_predictions is not None:
            if isinstance(test_predictions, pd.Series):
                pred_df = test_predictions.to_frame(name="prediction")
            else:
                pred_df = test_predictions

        # 11. Scrape MLflow metrics (D-16)
        scraped_metrics: dict = {}
        if mlflow_run_id:
            mlflow_run_data = mlflow.get_run(mlflow_run_id)
            scraped_metrics = dict(mlflow_run_data.data.metrics)
            # Reload run from DB to avoid stale ORM state
            run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).one()
            run.metrics = scraped_metrics
            run.mlflow_run_id = mlflow_run_id
            session.commit()

        # 12. Export to ModelVersion via QlibModelExporter (D-06)
        exporter = QlibModelExporter(session)
        model_version = exporter.export(
            model=trained_model,
            model_name=f"{run.handler_class}_{run.model_class}",
            model_class=run.model_class,
            model_params=run.model_params,
            feature_list=list(handler.get_data().columns),
            train_start=datetime.fromisoformat(segments["train"][0]),
            train_end=datetime.fromisoformat(segments["train"][1]),
            valid_start=datetime.fromisoformat(segments["valid"][0]) if "valid" in segments else None,
            valid_end=datetime.fromisoformat(segments["valid"][1]) if "valid" in segments else None,
            test_start=datetime.fromisoformat(segments["test"][0]) if "test" in segments else None,
            test_end=datetime.fromisoformat(segments["test"][1]) if "test" in segments else None,
            metrics=scraped_metrics or None,
        )
        run.model_version_id = model_version.id

        # 13. Save predictions Parquet to the ModelVersion's artifact_path
        if pred_df is not None and model_version.artifact_path:
            pred_path = Path(model_version.artifact_path) / "predictions_test.parquet"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_parquet(str(pred_path))
            logger.info("Saved test predictions to %s (%d rows)", pred_path, len(pred_df))

        # 14. Mark succeeded
        run.status = "succeeded"
        run.finished_at = datetime.now(timezone.utc)
        session.commit()

        logger.info(
            "Training run %s succeeded: model_version_id=%s",
            run_id,
            model_version.id,
        )
        return {
            "run_id": run_id,
            "status": "succeeded",
            "model_version_id": str(model_version.id),
        }

    except Exception as exc:
        logger.exception("Training run %s failed: %s", run_id, exc)
        session.rollback()
        from poseidon.models.training_run import TrainingRun

        run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).first()
        if run is not None:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
        # Do NOT re-raise -- per Research Pitfall 6, solo pool crash on unhandled exception
        return {"run_id": run_id, "status": "failed", "error": str(exc)}
    finally:
        session.close()
