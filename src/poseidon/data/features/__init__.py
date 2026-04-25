"""Feature computation framework.

Register new features by creating a class that extends BaseFeature
and decorating it with @register_feature.
"""

# Import all feature modules to trigger registration
from poseidon.data.features import (
    cascade,  # noqa: F401
    cross_asset,  # noqa: F401
    cvd,  # noqa: F401
    fundamentals,  # noqa: F401
    funding_rate,  # noqa: F401
    hmm_regime,  # noqa: F401
    institutional,  # noqa: F401
    intermarket,  # noqa: F401
    macro,  # noqa: F401
    margin,  # noqa: F401
    model_prediction,  # noqa: F401
    monthly_revenue,  # noqa: F401
    ofi,  # noqa: F401
    open_interest,  # noqa: F401
    price_momentum,  # noqa: F401
    quality_factor,  # noqa: F401
    regime,  # noqa: F401
    returns,  # noqa: F401
    swing,  # noqa: F401
    technical,  # noqa: F401
    trade_structure,  # noqa: F401
    trend,  # noqa: F401
    valuation,  # noqa: F401
    volatility,  # noqa: F401
    volume,  # noqa: F401
    volume_profile,  # noqa: F401
    vpin,  # noqa: F401
    wick,  # noqa: F401
)
from poseidon.data.features.base import BaseFeature, get_feature, list_features, register_feature

__all__ = ["BaseFeature", "get_feature", "list_features", "register_feature"]
