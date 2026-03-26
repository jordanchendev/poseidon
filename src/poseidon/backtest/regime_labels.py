"""Percentile-based regime label generator.

Converts realized_vol_20 into 3-class regime labels:
  0 = low_vol, 1 = medium_vol, 2 = high_vol

Thresholds are computed from training data percentiles and returned
for persistence and inference reuse.
"""

import numpy as np
import pandas as pd


def generate_regime_labels(
    features: pd.DataFrame,
    low_pct: float = 33.0,
    high_pct: float = 66.0,
) -> tuple[pd.Series, dict]:
    """Generate 3-class regime labels from realized_vol_20.

    Args:
        features: DataFrame containing a ``realized_vol_20`` column.
        low_pct: Percentile threshold for low/medium boundary (default 33).
        high_pct: Percentile threshold for medium/high boundary (default 66).

    Returns:
        Tuple of (labels Series with int dtype, thresholds dict).
    """
    vol = features["realized_vol_20"].dropna()
    low_threshold = float(np.percentile(vol, low_pct))
    high_threshold = float(np.percentile(vol, high_pct))

    labels = pd.Series(1, index=vol.index, dtype=int)
    labels[vol < low_threshold] = 0
    labels[vol > high_threshold] = 2

    thresholds = {
        "low_pct": float(low_pct),
        "high_pct": float(high_pct),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }
    return labels, thresholds
