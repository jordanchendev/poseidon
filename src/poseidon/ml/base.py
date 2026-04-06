"""BaseModel ABC — interface for all ML models in Poseidon.

Every model (XGBoost, LSTM, RL, etc.) implements these 7 methods.
predict() returns a raw DataFrame, NOT Signal objects — the Strategy
layer handles conversion.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for ML models.

    Subclasses must define:
        name: str -- unique model type identifier (e.g., "xgboost_tw_stock")
        description: str -- human-readable description
    """

    name: str = ""
    description: str = ""

    # Capability metadata (Phase 34)
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series, params: dict) -> dict:
        """Train the model on feature data.

        Args:
            features: Wide DataFrame from FeatureEngine.
            targets: Target series (e.g., future returns).
            params: Hyperparameters for training.

        Returns:
            Metrics dict (e.g., {"sharpe": 1.2, "win_rate": 0.55}).
        """
        ...

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from feature data.

        Args:
            features: Wide DataFrame from FeatureEngine.

        Returns:
            DataFrame with columns: prediction (long/short/hold), confidence (0.0~1.0).
        """
        ...

    @abstractmethod
    def validate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        """Validate model on held-out data.

        Args:
            features: Validation feature DataFrame.
            targets: Validation target series.

        Returns:
            Metrics dict for walk-forward analysis.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model artifacts to directory.

        Args:
            path: Directory to save artifacts (model.pkl, scaler.pkl, metadata.json, etc.).
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from saved artifacts.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            Loaded model instance.
        """
        ...

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return default hyperparameters for this model type."""
        ...

    @abstractmethod
    def get_feature_list(self) -> list[str]:
        """Return list of feature names this model expects."""
        ...
