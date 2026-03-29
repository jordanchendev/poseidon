"""Feature computation framework.

Register new features by creating a class that extends BaseFeature
and decorating it with @register_feature.
"""

from poseidon.data.features.base import BaseFeature, get_feature, list_features, register_feature

# Import all feature modules to trigger registration
from poseidon.data.features import cross_asset, fundamentals, funding_rate, hmm_regime, institutional, intermarket, macro, regime, returns, technical, trade_structure, volatility, volume, volume_profile

__all__ = ["BaseFeature", "register_feature", "get_feature", "list_features"]
