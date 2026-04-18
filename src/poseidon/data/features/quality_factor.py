"""Quality factor features -- pre-computed Z-scores from Thalassa.

Profitability, growth, and safety Z-scores are computed cross-sectionally
in Thalassa and delivered as quarterly/monthly data. Feature classes only
forward-fill to daily frequency.

Feature classes added in Phase 66 Wave 2.
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature
