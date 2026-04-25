"""Cascade Composite feature -- boolean composition of OI/wick/volume features.

Detects confluence of multiple micro-structure signals as a single composite
indicator. Fires when N of M conditions are met (configurable threshold).

Conditions checked (D-12):
1. oi_extreme: |oiwap_distance_168| > 5.0
2. oi_divergence: |oi_price_divergence_24| > 10.0
3. wick_extreme: wick_ratio_upper > 0.4 OR wick_ratio_lower > 0.4
4. vol_spike: volume_ratio_20 > 2.0

Direction from oiwap_distance sign (D-13):
- Negative distance (below cost basis) -> bullish reversal (1.0)
- Positive distance (above cost basis) -> bearish reversal (-1.0)
- NaN or zero -> no direction (0.0)

When oi_data is None, only wick+volume conditions are available (2 of 4).
OI-based conditions are treated as False (not met).

References: CONTEXT.md D-10, D-11, D-12, D-13.
Pitfall 2: Re-computes component features inline -- no cross-feature dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


@register_feature
class CascadeComposite(BaseFeature):
    """Cascade Composite -- boolean composition of OI/wick/volume features.

    Re-computes component features inline to avoid cross-feature dependency.
    Fires when N of M conditions are met (configurable threshold).
    """

    name = "cascade"
    description = "Cascade detection composite signal"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        oi_data: pd.DataFrame | None = None,
        threshold: int = 2,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute Cascade Composite signal.

        Args:
            ohlcv: Primary OHLCV DataFrame.
            oi_data: Optional DataFrame with 'open_interest' column.
            threshold: Minimum number of conditions to fire signal (default 2).

        Returns:
            DataFrame with columns:
            - cascade_signal: 1.0 when N conditions met, 0.0 otherwise
            - cascade_direction: direction * signal (bullish 1.0 / bearish -1.0 / none 0.0)
        """
        out_cols = ["cascade_signal", "cascade_direction"]

        if not self._validate(ohlcv, min_rows=170):
            return pd.DataFrame(dtype=float, columns=out_cols)

        # ── Re-compute component features inline (Pitfall 2) ──────────

        from poseidon.data.features.open_interest import OIBuildup, OICostBasis
        from poseidon.data.features.volume import VolumeRatio
        from poseidon.data.features.wick import WickRatio

        # OI features (need oi_data companion)
        oiwap_result = OICostBasis().compute(ohlcv, oi_data=oi_data, period=168)
        oi_div_result = OIBuildup().compute(ohlcv, oi_data=oi_data, period=24)

        # Wick features
        wick_result = WickRatio().compute(ohlcv)

        # Volume features
        vol_result = VolumeRatio().compute(ohlcv, period=20)

        # ── Build conditions DataFrame (D-12) ─────────────────────────

        conditions = pd.DataFrame(index=ohlcv.index, dtype=float)

        # Condition 1: OI extreme -- |oiwap_distance_168| > 5.0
        if isinstance(oiwap_result, pd.DataFrame) and "oiwap_distance_168" in oiwap_result.columns:
            conditions["oi_extreme"] = oiwap_result["oiwap_distance_168"].abs() > 5.0
        else:
            conditions["oi_extreme"] = False

        # Condition 2: OI divergence -- |oi_price_divergence_24| > 10.0
        if isinstance(oi_div_result, pd.DataFrame) and "oi_price_divergence_24" in oi_div_result.columns:
            conditions["oi_divergence"] = oi_div_result["oi_price_divergence_24"].abs() > 10.0
        else:
            conditions["oi_divergence"] = False

        # Condition 3: Wick extreme -- upper > 0.4 OR lower > 0.4
        if isinstance(wick_result, pd.DataFrame) and "wick_ratio_upper" in wick_result.columns:
            conditions["wick_extreme"] = (wick_result["wick_ratio_upper"] > 0.4) | (
                wick_result["wick_ratio_lower"] > 0.4
            )
        else:
            conditions["wick_extreme"] = False

        # Condition 4: Volume spike -- volume_ratio_20 > 2.0
        if isinstance(vol_result, pd.Series):
            conditions["vol_spike"] = vol_result > 2.0
        else:
            conditions["vol_spike"] = False

        # ── NaN handling: NaN conditions treated as False ─────────────
        conditions = conditions.fillna(False).astype(float)

        # ── N-of-M threshold logic (D-11) ─────────────────────────────
        n_conditions_met = conditions.sum(axis=1)
        signal = (n_conditions_met >= threshold).astype(float)

        # ── Direction from oiwap_distance sign (D-13) ─────────────────
        if isinstance(oiwap_result, pd.DataFrame) and "oiwap_distance_168" in oiwap_result.columns:
            oiwap_dist = oiwap_result["oiwap_distance_168"]
            direction = pd.Series(
                np.where(
                    oiwap_dist < 0,
                    1.0,  # below cost basis -> bullish reversal
                    np.where(oiwap_dist > 0, -1.0, 0.0),  # above -> bearish
                ),
                index=ohlcv.index,
            )
            # NaN oiwap_distance -> no direction
            direction = direction.where(oiwap_dist.notna(), 0.0)
        else:
            # No OI data -> no direction information
            direction = pd.Series(0.0, index=ohlcv.index)

        # Direction is only meaningful when signal fires
        cascade_direction = signal * direction

        return pd.DataFrame(
            {
                "cascade_signal": signal,
                "cascade_direction": cascade_direction,
            },
            index=ohlcv.index,
        )
