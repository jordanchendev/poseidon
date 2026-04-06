"""QlibModelExporter -- exports trained Qlib models to Poseidon's model registry.

Bridges the gap between Qlib's research workflow and Poseidon's model versioning
system. A trained Qlib model is pickled to the artifact directory and registered
as a ModelVersion with status='trained'.

Promotion to 'active' for live deployment is NOT done here (per D-14).
The researcher must manually call ModelManager.transition(version_id, "active")
after evaluation.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class QlibModelExporter:
    """Exports trained Qlib models to Poseidon's model registry.

    Capability metadata (per D-21):
        supports_backtest = True
        supports_live = False
        bias_risk = []
        stateful = False
    """

    # Capability metadata
    name: str = "qlib_model_exporter"
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    def __init__(self, session: Session) -> None:
        """Initialize QlibModelExporter.

        Args:
            session: SQLAlchemy session for ModelManager operations.
        """
        from poseidon.ml.manager import ModelManager

        self._session = session
        self._manager = ModelManager(session)

    def export(
        self,
        model: Any,
        model_name: str,
        model_class: str,
        model_params: dict[str, Any],
        feature_list: list[str],
        train_start: datetime,
        train_end: datetime,
        valid_start: datetime | None = None,
        valid_end: datetime | None = None,
        test_start: datetime | None = None,
        test_end: datetime | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Any:
        """Export a trained Qlib model to Poseidon's model registry.

        Creates a ModelVersion record, pickles the model to the artifact directory,
        saves metadata, and transitions the version to 'trained' status.

        Args:
            model: The trained Qlib model object (e.g., LGBModel instance).
            model_name: Name for the model in Poseidon's registry (e.g., "qlib_lgbm_tw_stock").
            model_class: Qlib model class name (e.g., "LGBModel", "ALSTMModel").
            model_params: Qlib model hyperparameters dict.
            feature_list: List of feature names used for training.
            train_start: Start of training period.
            train_end: End of training period.
            valid_start: Start of validation period (optional).
            valid_end: End of validation period (optional).
            test_start: Start of test period (optional).
            test_end: End of test period (optional).
            metrics: Evaluation metrics dict (optional).

        Returns:
            ModelVersion: The created and transitioned ModelVersion record.
        """
        from poseidon.ml import artifacts

        # Build params JSONB with Qlib bridge metadata (per D-15)
        version_params: dict[str, Any] = {
            "qlib_model_class": model_class,
            "qlib_model_params": model_params,
            "feature_list": feature_list,
            "train_period": {"start": str(train_start), "end": str(train_end)},
            "source": "qlib_bridge",
        }
        if valid_start is not None:
            version_params["valid_period"] = {
                "start": str(valid_start),
                "end": str(valid_end),
            }
        if test_start is not None:
            version_params["test_period"] = {
                "start": str(test_start),
                "end": str(test_end),
            }

        # 1. Create ModelVersion record (status='training')
        mv = self._manager.create_version(
            name=model_name,
            params=version_params,
            feature_list=feature_list,
            train_start=train_start,
            train_end=train_end,
        )

        # 2. Save model artifact as pickle
        version_dir = artifacts.ensure_version_dir(model_name, mv.version)
        model_path = version_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata alongside the model
        metadata: dict[str, Any] = {
            "model_class": model_class,
            "model_params": model_params,
            "feature_list": feature_list,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "source": "qlib_bridge",
        }
        if valid_start is not None:
            metadata["valid_start"] = str(valid_start)
            metadata["valid_end"] = str(valid_end)
        if test_start is not None:
            metadata["test_start"] = str(test_start)
            metadata["test_end"] = str(test_end)
        if metrics is not None:
            metadata["metrics"] = metrics

        artifacts.save_metadata(model_name, mv.version, metadata)

        # 3. Transition to 'trained' status
        mv = self._manager.transition(mv.id, "trained", metrics=metrics)

        logger.info(
            "Exported Qlib model: %s v%d (id=%s)",
            model_name,
            mv.version,
            mv.id,
        )

        return mv
