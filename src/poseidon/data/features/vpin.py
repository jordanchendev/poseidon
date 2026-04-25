"""VPIN (Volume-synchronized Probability of Informed Trading) feature.

Computes VPIN using volume buckets with BVC classification.  Bucket size
adapts to the median single-bar volume over a trailing window (D-17).
Output is a probability in the [0, 1] range (D-18).

Algorithm
---------
1. Classify each bar's volume into buy/sell using BVC (``classify_volume``).
2. Compute adaptive bucket size: ``median(volume[i-lookback:i])`` per bar.
3. Iterate over bars, accumulating buy/sell volume into the current bucket.
   When the cumulative volume fills the bucket, compute the bucket's VPIN
   contribution: ``|bucket_buy - bucket_sell| / bucket_total``.  Start a new
   bucket with the remainder volume.
4. VPIN at bar *i* = mean of the last *n_buckets* completed bucket
   contributions.

References
----------
- Easley, Lopez de Prado & O'Hara (2012) -- original VPIN on tick data.
- Adapted for OHLCV bars per CONTEXT.md D-16, D-17, D-18.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class VPIN(BaseFeature):
    """Volume-Synchronized Probability of Informed Trading.

    Measures order-flow toxicity by computing the imbalance between buy
    and sell volume across volume-synchronized buckets.  High VPIN
    indicates informed trading activity.
    """

    name = "vpin"
    description = "VPIN toxicity metric (0-1 range, volume-bucketed)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 50,
        n_buckets: int = 50,
        **kwargs,
    ) -> pd.Series:
        col = f"vpin_{n_buckets}"

        if not self._validate(ohlcv, min_rows=lookback + n_buckets):
            return pd.Series(dtype=float, name=col)

        from poseidon.data.features.bvc import classify_volume

        bvc = classify_volume(ohlcv)
        buy_vol = bvc["buy_volume"].values
        sell_vol = bvc["sell_volume"].values
        volume = ohlcv["volume"].values

        # Adaptive bucket size: rolling median of bar volume (D-17)
        median_vol = pd.Series(volume).rolling(window=lookback, min_periods=lookback).median().values

        n = len(ohlcv)
        result = np.full(n, np.nan)

        # Iterative bucket filling
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_remaining = 0.0  # how much volume the current bucket still needs
        bucket_vpins: list[float] = []  # completed bucket contributions

        for i in range(n):
            bs = median_vol[i]
            # Skip warmup period where median is NaN
            if np.isnan(bs):
                continue

            # Enforce minimum bucket size to avoid infinite micro-buckets (T-76-08)
            bs = max(bs, 1.0)

            # If this is the first bar with a valid bucket size, start a new bucket
            if bucket_remaining <= 0.0:
                bucket_remaining = bs
                bucket_buy = 0.0
                bucket_sell = 0.0

            bar_buy = buy_vol[i]
            bar_sell = sell_vol[i]
            bar_total = bar_buy + bar_sell  # == volume[i]

            if np.isnan(bar_total) or bar_total <= 0.0:
                continue

            # Proportionally split buy/sell within this bar
            buy_frac = bar_buy / bar_total if bar_total > 0 else 0.5
            sell_frac = 1.0 - buy_frac

            remaining_bar = bar_total

            while remaining_bar > 1e-12:
                fill = min(remaining_bar, bucket_remaining)

                bucket_buy += fill * buy_frac
                bucket_sell += fill * sell_frac
                bucket_remaining -= fill
                remaining_bar -= fill

                if bucket_remaining <= 1e-12:
                    # Bucket complete
                    bucket_total = bucket_buy + bucket_sell
                    vpin_contrib = abs(bucket_buy - bucket_sell) / bucket_total if bucket_total > 0 else 0.0
                    bucket_vpins.append(vpin_contrib)

                    # Start new bucket with current bar's adaptive bucket size
                    bucket_remaining = max(median_vol[i], 1.0)
                    bucket_buy = 0.0
                    bucket_sell = 0.0

            # Compute VPIN as rolling mean of last n_buckets
            if len(bucket_vpins) >= n_buckets:
                result[i] = np.mean(bucket_vpins[-n_buckets:])

        out = pd.Series(result, index=ohlcv.index, name=col)
        # Clamp to [0, 1] for safety
        out = out.clip(lower=0.0, upper=1.0)
        return out
