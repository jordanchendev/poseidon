"""Feature computation framework.

Register new features by creating a class that extends BaseFeature
and decorating it with @register_feature.
"""

from poseidon.data.features.base import BaseFeature, get_feature, list_features, register_feature

# Import all feature modules to trigger registration
from poseidon.data.features import cross_asset  # noqa: F401
from poseidon.data.features import fundamentals  # noqa: F401
from poseidon.data.features import funding_rate  # noqa: F401
from poseidon.data.features import hmm_regime  # noqa: F401
from poseidon.data.features import institutional  # noqa: F401
from poseidon.data.features import intermarket  # noqa: F401
from poseidon.data.features import macro  # noqa: F401
from poseidon.data.features import margin  # noqa: F401
from poseidon.data.features import model_prediction  # noqa: F401
from poseidon.data.features import open_interest  # noqa: F401
from poseidon.data.features import regime  # noqa: F401
from poseidon.data.features import returns  # noqa: F401
from poseidon.data.features import swing  # noqa: F401
from poseidon.data.features import technical  # noqa: F401
from poseidon.data.features import trade_structure  # noqa: F401
from poseidon.data.features import trend  # noqa: F401
from poseidon.data.features import volatility  # noqa: F401
from poseidon.data.features import volume  # noqa: F401
from poseidon.data.features import volume_profile  # noqa: F401
from poseidon.data.features import wick  # noqa: F401

__all__ = ["BaseFeature", "register_feature", "get_feature", "list_features"]
