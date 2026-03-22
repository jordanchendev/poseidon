"""Feature computation framework.

Register new features by creating a class that extends BaseFeature
and decorating it with @register_feature.
"""

from poseidon.data.features.base import BaseFeature, get_feature, list_features, register_feature

# Import all feature modules to trigger registration
from poseidon.data.features import returns, technical, volatility, volume

__all__ = ["BaseFeature", "register_feature", "get_feature", "list_features"]
