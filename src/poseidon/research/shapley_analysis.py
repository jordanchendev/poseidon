"""Shapley value analysis using TreeSHAP for tree-based models.

Per D-05: Use shap library with TreeSHAP.
Per D-07: Input is a trained ModelVersion ID.
Per D-08: Output is per-feature mean absolute SHAP value.

NOTE: shap is only available in qlib-research container (D-06).
This module must only be imported inside qlib_tasks.py task body.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def compute_shapley_values(
    model, feature_matrix: pd.DataFrame, max_samples: int | None = 500
) -> dict:
    """Compute global mean absolute SHAP values for a tree model."""
    import shap

    if feature_matrix.empty:
        return {"features": {}, "num_samples": 0}

    working = feature_matrix
    if max_samples is not None and len(working) > max_samples:
        working = working.sample(n=max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(working)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    mean_abs_shap = np.abs(values).mean(axis=0)
    return {
        "features": dict(zip(working.columns, mean_abs_shap.tolist(), strict=False)),
        "num_samples": len(working),
    }


def run_shapley_analysis(
    model_version_id: str, max_samples: int | None = 500
) -> dict:
    """Load a trained model and compute SHAP feature importances."""
    import qlib
    from qlib.config import REG_CN
    from qlib.data.dataset import DatasetH

    from poseidon.data.feature_engine import _is_nonprice_spec, get_r2_specs
    from poseidon.ml.artifacts import get_predictions_path
    from poseidon.models.base import SessionLocal
    from poseidon.models.model_version import ModelVersion
    from poseidon.models.training_run import TrainingRun
    from poseidon.qlib.data_handler import PoseidonDataHandler
    from poseidon.qlib.dataset_builder import DatasetBuilder

    session = SessionLocal()
    try:
        model_version = (
            session.query(ModelVersion)
            .filter(ModelVersion.id == uuid.UUID(model_version_id))
            .one()
        )
        training_run = (
            session.query(TrainingRun)
            .filter(TrainingRun.model_version_id == model_version.id)
            .order_by(TrainingRun.created_at.desc())
            .first()
        )
        if training_run is None:
            raise ValueError(
                f"No TrainingRun found for ModelVersion {model_version_id}"
            )
        if not model_version.artifact_path:
            raise ValueError(
                f"ModelVersion {model_version_id} has no artifact_path"
            )

        model_path = Path(model_version.artifact_path) / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        with model_path.open("rb") as fh:
            model = pickle.load(fh)

        try:
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
        except Exception:
            pass

        all_dates: list[str] = []
        for segment_name, segment_dates in training_run.segments.items():
            if segment_name == "all":
                continue
            all_dates.extend([segment_dates[0], segment_dates[1]])
        if not all_dates:
            raise ValueError("TrainingRun has no segments to rebuild feature matrix")

        feature_specs = None
        if training_run.model_params.get("expand_features", True):
            all_r2 = get_r2_specs(
                training_run.symbols[0] if training_run.symbols else "",
                training_run.market,
            )
            feature_specs = [
                (name, params) for name, params in all_r2 if _is_nonprice_spec(name)
            ]

        dataset_builder = DatasetBuilder(
            session=session,
            market=training_run.market,
            interval=training_run.interval,
        )
        handler = PoseidonDataHandler(
            dataset_builder=dataset_builder,
            symbols=training_run.symbols,
            start=pd.Timestamp(min(all_dates)).to_pydatetime(),
            end=pd.Timestamp(max(all_dates)).to_pydatetime(),
            feature_specs=feature_specs,
        )
        dataset = DatasetH(
            handler=handler.to_qlib_handler(),
            segments={
                name: (segment[0], segment[1])
                for name, segment in training_run.segments.items()
            },
        )

        available_segments = (
            model_version.params.get("prediction_segments", ["test"])
            if model_version.params
            else ["test"]
        )
        feature_frames: list[pd.DataFrame] = []
        for segment in available_segments:
            pred_path = get_predictions_path(model_version.artifact_path, segment)
            if not pred_path.exists():
                continue
            segment_features = dataset.prepare(segment, col_set="feature")
            if isinstance(segment_features.columns, pd.MultiIndex):
                segment_features.columns = [
                    str(column[-1]) for column in segment_features.columns.to_list()
                ]
            feature_frames.append(segment_features)

        if not feature_frames:
            raise ValueError(
                f"No prepared feature data found for ModelVersion {model_version_id}"
            )

        feature_matrix = pd.concat(feature_frames)
        feature_matrix = feature_matrix[~feature_matrix.index.duplicated(keep="last")]
        results = compute_shapley_values(model, feature_matrix, max_samples)
        results["model_version_id"] = model_version_id
        results["market"] = training_run.market
        return results
    finally:
        session.close()
