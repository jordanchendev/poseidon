"""XGBoost model implementation for Poseidon.

Uses XGBClassifier with 3-class output (long/short/hold).
Falls back to a stub if xgboost is not installed.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.ml.base import BaseModel
from poseidon.ml.registry import register_model

logger = logging.getLogger(__name__)

try:
    import joblib
    import xgboost as xgb

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


# Label mapping
LABEL_MAP = {0: "hold", 1: "long", 2: "short"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Default feature set (matching FeatureEngine defaults)
DEFAULT_FEATURES = [
    "sma_5", "sma_10", "sma_20", "sma_60",
    "ema_12", "ema_26",
    "rsi_14",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper_20", "bb_middle_20", "bb_lower_20",
    "atr_14",
    "return_1d", "log_return_1d",
    "std_vol_20",
]


@register_model
class XGBoostModel(BaseModel):
    """XGBoost 3-class classifier (long/short/hold)."""

    name = "xgboost"
    description = "XGBoost gradient boosting classifier for directional prediction"

    def __init__(self):
        self._model = None
        self._feature_list = list(DEFAULT_FEATURES)

    def train(self, features: pd.DataFrame, targets: pd.Series, params: dict) -> dict:
        if not _HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed. Install with: pip install xgboost joblib")

        merged_params = {**self.get_default_params(), **params}
        feature_cols = [c for c in self._feature_list if c in features.columns]
        self._feature_list = feature_cols

        X = features[feature_cols].dropna()
        y = targets.loc[X.index]

        self._model = xgb.XGBClassifier(**merged_params)
        self._model.fit(X, y)

        preds = self._model.predict(X)
        accuracy = float(np.mean(preds == y))
        return {"accuracy": accuracy, "n_samples": len(X), "n_features": len(feature_cols)}

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        feature_cols = [c for c in self._feature_list if c in features.columns]
        X = features[feature_cols].dropna()

        # Get class probabilities
        proba = self._model.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)

        predictions = [LABEL_MAP.get(i, "hold") for i in pred_indices]
        confidences = np.max(proba, axis=1)

        # Build result for clean rows, then reindex to original to fill NaN rows
        result = pd.DataFrame(
            {"prediction": predictions, "confidence": confidences},
            index=X.index,
        )
        return result.reindex(features.index, fill_value="hold")

    def validate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        feature_cols = [c for c in self._feature_list if c in features.columns]
        X = features[feature_cols].dropna()
        y = targets.loc[X.index]

        preds = self._model.predict(X)
        accuracy = float(np.mean(preds == y))

        result = {"accuracy": accuracy, "n_samples": len(X)}

        # Per-class accuracy
        for label_idx, label_name in LABEL_MAP.items():
            mask = y == label_idx
            if mask.sum() > 0:
                result[f"accuracy_{label_name}"] = float(np.mean(preds[mask] == label_idx))

        return result

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, path / "model.pkl")
        (path / "features.json").write_text(json.dumps(self._feature_list))
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        if not _HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed")

        path = Path(path)
        instance = cls()
        instance._model = joblib.load(path / "model.pkl")
        features_file = path / "features.json"
        if features_file.exists():
            instance._feature_list = json.loads(features_file.read_text())
        return instance

    def get_default_params(self) -> dict:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
        }

    def get_feature_list(self) -> list[str]:
        return list(self._feature_list)
