"""FeatureEngine package -- backward-compatible re-exports.

All existing ``from poseidon.data.feature_engine import X`` imports continue
working unchanged via these re-exports.
"""

from poseidon.data.feature_engine.computer import FeatureComputer
from poseidon.data.feature_engine.orchestrator import FeatureOrchestrator
from poseidon.data.feature_engine.specs import (
    CROSS_ASSET_PAIRS,
    DEFAULT_FEATURES,
    EXPANDED_FEATURES,
    EXPANDED_FEATURES_R2,
    FUNDAMENTAL_NAMES,
    FUNDING_NAMES,
    INSTITUTIONAL_PREFIXES,
    MACRO_PREFIX,
    MARGIN_NAMES,
    OI_NAMES,
    PREDICTION_NAMES,
    REGIME_FEATURES,
    TRADE_STRUCTURE_NAMES,
    get_cross_asset_specs,
    get_r2_specs,
    is_nonprice_spec,
    nonprice_data_key,
)

# Backward compatibility: FeatureEngine is an alias for FeatureOrchestrator (D-03)
FeatureEngine = FeatureOrchestrator

__all__ = [
    "FeatureEngine",
    "FeatureComputer",
    "FeatureOrchestrator",
    "is_nonprice_spec",
    "nonprice_data_key",
    "get_r2_specs",
    "get_cross_asset_specs",
    "DEFAULT_FEATURES",
    "EXPANDED_FEATURES",
    "EXPANDED_FEATURES_R2",
    "REGIME_FEATURES",
    "CROSS_ASSET_PAIRS",
    "INSTITUTIONAL_PREFIXES",
    "FUNDAMENTAL_NAMES",
    "TRADE_STRUCTURE_NAMES",
    "FUNDING_NAMES",
    "MARGIN_NAMES",
    "OI_NAMES",
    "PREDICTION_NAMES",
    "MACRO_PREFIX",
]
