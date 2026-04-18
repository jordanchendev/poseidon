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

from poseidon.core.database import db_session
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
    queue="poseidon_qlib",
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
    with db_session() as session:
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

            # 3. Import Qlib inside task body (Research Pitfall 2)
            import importlib
            from pathlib import Path

            import pandas as pd
            import qlib
            from qlib.config import REG_CN
            from qlib.data.dataset import DatasetH
            from poseidon.qlib.allowlist import resolve_model
            from poseidon.qlib.data_handler import PoseidonDataHandler
            from poseidon.qlib.dataset_builder import DatasetBuilder
            from poseidon.qlib.model_exporter import QlibModelExporter
            from poseidon.ml.artifacts import get_predictions_path

            # 4. Initialize Qlib (Research Pitfall 5) — guard for solo pool reuse
            try:
                qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            except Exception:
                logger.debug("Qlib already initialized, skipping re-init")

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

            # Step 6b2: Resolve feature_specs for expanded features (Phase 42)
            from poseidon.data.feature_engine import get_r2_specs, is_nonprice_spec

            expand_features = run.model_params.get("expand_features", True)
            feature_specs = None
            if expand_features:
                all_r2 = get_r2_specs(run.symbols[0] if run.symbols else "", run.market)
                # Only pass nonprice features to DatasetBuilder -- TA features are
                # already handled by Qlib Alpha158/360 expressions via DataHandlerLP.
                feature_specs = [(name, params) for name, params in all_r2 if is_nonprice_spec(name)]
                if feature_specs:
                    logger.info("Expanded features: %d nonprice specs for market=%s", len(feature_specs), run.market)

            # Step 6c: Create PoseidonDataHandler (calls ds_builder.build() internally)
            handler = PoseidonDataHandler(
                dataset_builder=ds_builder,
                symbols=run.symbols,
                start=datetime.fromisoformat(overall_start),
                end=datetime.fromisoformat(overall_end),
                feature_specs=feature_specs,
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

            # 8. Generate predictions on ALL segments (per Phase 43 D-01: train, valid, test)
            all_predictions: dict[str, pd.DataFrame | pd.Series | None] = {}
            for seg_name in segments.keys():
                try:
                    seg_pred = trained_model.predict(dataset, segment=seg_name)
                    all_predictions[seg_name] = seg_pred
                    logger.info(
                        "Generated predictions for segment '%s': %d rows",
                        seg_name,
                        len(seg_pred) if seg_pred is not None else 0,
                    )
                except Exception as seg_exc:
                    logger.warning("Failed to predict segment '%s': %s", seg_name, seg_exc)
                    all_predictions[seg_name] = None

                # CHECK CANCEL between segment predictions (per _run_cancelled pattern)
                if _run_cancelled(session, run_id):
                    run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).one()
                    run.finished_at = datetime.now(timezone.utc)
                    session.commit()
                    return {"run_id": run_id, "status": "cancelled"}

            # Use test predictions for metrics (backward compatible)
            test_predictions = all_predictions.get("test")

            # 9. Compute metrics directly (IC/ICIR) — no MLflow dependency
            # MLflow PostgreSQL backend conflicts with Poseidon's `experiments` table.
            # Metrics are computed in-process and stored directly in training_runs.metrics.
            computed_metrics: dict = {}
            if test_predictions is not None:
                try:
                    pred_series = test_predictions if isinstance(test_predictions, pd.Series) else test_predictions.iloc[:, 0]
                    label_series = dataset.prepare("test", col_set="label").iloc[:, 0]
                    common_idx = pred_series.index.intersection(label_series.index)
                    if len(common_idx) > 0:
                        p = pred_series.loc[common_idx]
                        la = label_series.loc[common_idx]
                        ic = float(p.corr(la)) if pd.notna(p.corr(la)) else 0.0
                        # Per-date IC for ICIR calculation
                        daily_ic = p.groupby(level="datetime").apply(
                            lambda x: x.corr(la.loc[x.index]) if len(x) > 1 else 0.0
                        )
                        icir = float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) > 1 and daily_ic.std() > 0 else 0.0
                        computed_metrics = {"IC": ic, "ICIR": icir}
                except Exception as metric_exc:
                    logger.warning("Failed to compute IC/ICIR: %s", metric_exc)

            # 10. Store metrics in training_runs row
            run = session.query(TrainingRun).filter_by(run_id=uuid.UUID(run_id)).one()
            if computed_metrics:
                run.metrics = computed_metrics
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
                metrics=computed_metrics or None,
            )
            run.model_version_id = model_version.id

            # 13. Save predictions Parquet for ALL segments (per Phase 43 D-02)
            saved_segments: list[str] = []
            if model_version.artifact_path:
                Path(model_version.artifact_path).mkdir(parents=True, exist_ok=True)
                for seg_name, seg_pred in all_predictions.items():
                    if seg_pred is None:
                        continue
                    # Normalize to DataFrame with "prediction" column
                    if isinstance(seg_pred, pd.Series):
                        seg_df = seg_pred.to_frame(name="prediction")
                    else:
                        seg_df = seg_pred
                    pred_path = get_predictions_path(model_version.artifact_path, seg_name)
                    seg_df.to_parquet(str(pred_path))
                    saved_segments.append(seg_name)
                    logger.info(
                        "Saved %s predictions to %s (%d rows)",
                        seg_name,
                        pred_path,
                        len(seg_df),
                    )

            # Record which segments were generated in ModelVersion params JSONB
            if saved_segments:
                if model_version.params is None:
                    model_version.params = {}
                # SQLAlchemy JSONB mutation detection requires reassignment
                updated_params = dict(model_version.params)
                updated_params["prediction_segments"] = saved_segments
                model_version.params = updated_params
                session.commit()

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


@celery_app.task(
    name="poseidon.workers.qlib_tasks.factor_shapley_analysis",
    queue="poseidon_qlib",
    bind=True,
    max_retries=0,
)
def factor_shapley_analysis(self, run_id: str):
    """Compute SHAP feature importance for a trained model."""
    with db_session() as session:
        run = None
        try:
            from poseidon.models.factor_analysis_run import FactorAnalysisRun
            from poseidon.research.shapley_analysis import run_shapley_analysis

            run = session.query(FactorAnalysisRun).filter_by(id=uuid.UUID(run_id)).one()
            run.status = "running"
            run.error = None
            session.commit()

            config = run.config_json
            run.results_json = run_shapley_analysis(
                model_version_id=config["model_version_id"],
                max_samples=config.get("max_samples", 500),
            )
            run.status = "succeeded"
            session.commit()
            return {"run_id": run_id, "status": "succeeded"}
        except Exception as exc:
            logger.exception("factor_shapley_analysis failed for run_id=%s", run_id)
            if run is not None:
                session.rollback()
                from poseidon.models.factor_analysis_run import FactorAnalysisRun

                run = session.query(FactorAnalysisRun).filter_by(id=uuid.UUID(run_id)).first()
                if run is not None:
                    run.status = "failed"
                    run.error = str(exc)[:2000]
                    session.commit()
            return {"run_id": run_id, "status": "failed", "error": str(exc)}
