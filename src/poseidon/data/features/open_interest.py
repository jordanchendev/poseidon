"""Open interest features for liquidity sweep detection.

OIChange: rate of change + z-score of OI (H-A sweep detection).
OIBuildup: cumulative OI accumulation vs price movement (H-B buildup detection).

Data is injected via the ``oi_data`` kwarg by FeatureEngine.compute_with_companions().
"""

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature


def _align_oi_to_index(
    oi_series: pd.Series, target_index: pd.Index, method: str = "ffill"
) -> pd.Series:
    """Align OI series to OHLCV index via forward-fill.

    Binance ``fetchOpenInterestHistory`` returns timestamps representing
    the START of the measurement period (period-start convention).  For
    example, a 5-minute OI snapshot stamped ``10:00`` reflects OI measured
    during [10:00, 10:05).

    Forward-fill (``method="ffill"``) guarantees that at any bar time *T*,
    the OI value used is the most recent snapshot with timestamp <= T.
    This eliminates look-ahead bias regardless of whether the exchange
    uses period-start or period-end conventions, because:

    - Period-start T: OI was measured at or after T, so using it at bar T
      is conservative (data is available by T).
    - Period-end T: OI was measured during [T-interval, T], fully
      available by T.

    In either case, ``reindex(..., method="ffill")`` never pulls a future
    value forward -- it only carries past/current values into gaps.

    Handles timezone mismatch: OI may have different tz-awareness than
    OHLCV.  Normalises before reindex to prevent silent all-NaN results.
    """
    if isinstance(oi_series.index, pd.DatetimeIndex) and isinstance(
        target_index, pd.DatetimeIndex
    ):
        if oi_series.index.tz is None and target_index.tz is not None:
            oi_series = oi_series.copy()
            oi_series.index = oi_series.index.tz_localize("UTC")
        elif oi_series.index.tz is not None and target_index.tz is None:
            oi_series = oi_series.copy()
            oi_series.index = oi_series.index.tz_localize(None)
    return oi_series.reindex(target_index, method=method)


def _nan_df(index: pd.Index, columns: list[str]) -> pd.DataFrame:
    """Return a NaN-filled DataFrame with the given index and columns."""
    return pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)


@register_feature
class OIChange(BaseFeature):
    """Open interest change rate and z-score.

    Detects rapid OI shifts that may precede liquidation cascades (H-A).
    Computed from companion ``oi_data`` DataFrame injected by FeatureEngine.
    """

    name = "oi_change"
    description = "Open interest change rate and z-score"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        oi_data: pd.DataFrame | None = None,
        period: int = 20,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute OI change rate and z-score.

        Args:
            ohlcv: Primary OHLCV DataFrame.
            oi_data: DataFrame with 'open_interest' column, indexed by time.
            period: Rolling window for z-score computation.

        Returns:
            DataFrame with columns: oi_change_pct, oi_change_zscore_{period}.
        """
        col_pct = "oi_change_pct"
        col_zscore = f"oi_change_zscore_{period}"
        out_cols = [col_pct, col_zscore]

        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float, columns=out_cols)

        if oi_data is None or (hasattr(oi_data, "empty") and oi_data.empty):
            return _nan_df(ohlcv.index, out_cols)

        if "open_interest" not in oi_data.columns:
            return _nan_df(ohlcv.index, out_cols)

        oi = _align_oi_to_index(oi_data["open_interest"], ohlcv.index)

        # Percentage change
        oi_change_pct = (oi - oi.shift(1)) / oi.shift(1) * 100

        # Z-score: how abnormal is this change relative to recent history
        rolling_mean = oi_change_pct.rolling(period).mean()
        rolling_std = oi_change_pct.rolling(period).std()
        oi_change_zscore = (oi_change_pct - rolling_mean) / rolling_std

        result = pd.DataFrame(
            {col_pct: oi_change_pct, col_zscore: oi_change_zscore},
            index=ohlcv.index,
        )
        return result


@register_feature
class OIBuildup(BaseFeature):
    """OI accumulation detection -- rising OI + converging price = H-B signal.

    Positive divergence (OI rising but price flat) indicates hidden position
    buildup before a large move. Core signal for the Liquidity Sweep Reversal
    Strategy.
    """

    name = "oi_buildup"
    description = "OI accumulation detection -- rising OI + converging price = H-B signal"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        oi_data: pd.DataFrame | None = None,
        period: int = 24,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute OI buildup and price-OI divergence.

        Args:
            ohlcv: Primary OHLCV DataFrame (must have 'close' column).
            oi_data: DataFrame with 'open_interest' column, indexed by time.
            period: Lookback window for cumulative change.

        Returns:
            DataFrame with columns: oi_buildup_{period}, oi_price_divergence_{period}.
        """
        col_buildup = f"oi_buildup_{period}"
        col_divergence = f"oi_price_divergence_{period}"
        out_cols = [col_buildup, col_divergence]

        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float, columns=out_cols)

        if oi_data is None or (hasattr(oi_data, "empty") and oi_data.empty):
            return _nan_df(ohlcv.index, out_cols)

        if "open_interest" not in oi_data.columns:
            return _nan_df(ohlcv.index, out_cols)

        oi = _align_oi_to_index(oi_data["open_interest"], ohlcv.index)
        close = ohlcv["close"]

        # Cumulative percentage change over period
        oi_cumulative_pct = (oi - oi.shift(period)) / oi.shift(period) * 100
        price_change_pct = (close - close.shift(period)) / close.shift(period) * 100

        # Divergence: positive = OI rising more than price moving
        # This indicates hidden position buildup
        oi_price_divergence = oi_cumulative_pct - abs(price_change_pct)

        result = pd.DataFrame(
            {col_buildup: oi_cumulative_pct, col_divergence: oi_price_divergence},
            index=ohlcv.index,
        )
        return result


@register_feature
class OICostBasis(BaseFeature):
    """OI-weighted average price (OIWAP) -- cost-basis estimation.

    Estimates the aggregate cost basis of open positions by computing
    an OI-increment-weighted average price.  When OI increases (new
    positions opened), the close price at that bar is treated as the
    entry price.  OI decreases (closures / liquidations) do NOT update
    the cost basis (D-06).

    Outputs
    -------
    oiwap_{period}
        Rolling OI-weighted average entry price.
    oiwap_distance_{period}
        Current close relative to OIWAP as a percentage.
        Positive = price above aggregate cost basis (positions in profit).
        Negative = price below aggregate cost basis (positions underwater).

    Data is injected via the ``oi_data`` kwarg by
    FeatureEngine.compute_with_companions().

    References: CONTEXT.md D-04, D-05, D-06, D-07, D-08.
    """

    name = "oi_cost_basis"
    description = "OI-weighted average price (cost-basis estimation)"

    def compute(
        self,
        ohlcv: pd.DataFrame,
        oi_data: pd.DataFrame | None = None,
        period: int = 168,
        **kwargs,
    ) -> pd.DataFrame:
        col_oiwap = f"oiwap_{period}"
        col_distance = f"oiwap_distance_{period}"
        out_cols = [col_oiwap, col_distance]

        if not self._validate(ohlcv):
            return pd.DataFrame(dtype=float, columns=out_cols)

        if oi_data is None or (hasattr(oi_data, "empty") and oi_data.empty):
            return _nan_df(ohlcv.index, out_cols)

        if "open_interest" not in oi_data.columns:
            return _nan_df(ohlcv.index, out_cols)

        oi = _align_oi_to_index(oi_data["open_interest"], ohlcv.index)
        close = ohlcv["close"]

        # OI change per bar
        delta_oi = oi.diff()
        # Only OI increases contribute -- new positions opened (D-06)
        oi_increase = delta_oi.clip(lower=0)

        # Weighted price: close * max(delta_oi, 0)
        weighted_price = close * oi_increase

        # Rolling OIWAP over lookback window (D-07)
        rolling_weighted_sum = weighted_price.rolling(period, min_periods=1).sum()
        rolling_oi_sum = oi_increase.rolling(period, min_periods=1).sum()

        # Division: 0/0 -> NaN naturally (correct: no OI increases = undefined cost basis)
        oiwap = rolling_weighted_sum / rolling_oi_sum

        # Distance: percentage difference from estimated cost basis (D-08)
        oiwap_distance = (close - oiwap) / oiwap * 100

        return pd.DataFrame(
            {col_oiwap: oiwap, col_distance: oiwap_distance},
            index=ohlcv.index,
        )
