"""Monthly revenue momentum features.

CumulativeRevenueGrowth and RevenueAccelerationMonths computed from
Thalassa monthly revenue data.

Feature classes added in Phase 66 Wave 2.
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature
