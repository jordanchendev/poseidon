"""Base class and registry for feature computations."""

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

# Module-level feature registry
_registry: dict[str, type["BaseFeature"]] = {}


def register_feature(cls):
    """Decorator to register a feature class in the global registry."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Feature class {cls.__name__} must define a 'name' attribute")
    _registry[cls.name] = cls
    return cls


def get_feature(name: str) -> type["BaseFeature"]:
    """Look up a registered feature by name."""
    if name not in _registry:
        raise KeyError(f"Unknown feature: '{name}'. Available: {sorted(_registry.keys())}")
    return _registry[name]


def list_features() -> list[str]:
    """List all registered feature names."""
    return sorted(_registry.keys())


class BaseFeature(ABC):
    """Abstract base class for all feature computations.

    Each feature class computes one type of indicator from OHLCV data.
    Features are registered via the @register_feature decorator and
    discovered by the FeatureEngine through the registry.

    Subclasses must define:
        name: str -- unique identifier (e.g., "sma", "rsi")
        description: str -- human-readable description

    The compute() method receives a DataFrame with columns:
        time, open, high, low, close, volume
    and returns either:
        - pd.Series for single-column features (e.g., SMA)
        - pd.DataFrame for multi-column features (e.g., MACD, Bollinger)
    """

    name: str = ""
    description: str = ""

    # Capability metadata (Phase 34)
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame, **params) -> pd.Series | pd.DataFrame:
        """Compute feature values from OHLCV data.

        Args:
            ohlcv: DataFrame with columns [time, open, high, low, close, volume].
            **params: Indicator-specific parameters (e.g., period=20).

        Returns:
            Series with a single feature column, or DataFrame with multiple columns.
            Column names should follow the convention: {name}_{param} (e.g., sma_20).
            Returns empty Series/DataFrame if input is empty.
        """
        ...

    def _validate(self, ohlcv: pd.DataFrame, min_rows: int = 1) -> bool:
        """Check that OHLCV has sufficient data. Returns False if empty or too short."""
        if ohlcv.empty or len(ohlcv) < min_rows:
            return False
        return True
