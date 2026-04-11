"""Model prediction feature for ML vote integration.

Reads prediction cache Parquet data and exposes ML model confidence as the
``qlib_prediction`` computed feature column. The prediction data is injected
via the ``prediction_data`` kwarg by FeatureEngine.compute_with_companions()
-- this feature class does NOT load Parquet files or access the database.

The actual data loading and injection is handled by BacktestRunner (Phase 44
Plan 02) which pre-loads prediction Parquet and passes it through
FeatureEngine's ``extra_nonprice_data`` mechanism.
"""

import logging

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature

logger = logging.getLogger(__name__)


def _align_predictions_to_index(
    pred_series: pd.Series, target_index: pd.Index, method: str = "ffill"
) -> pd.Series:
    """Align prediction series to OHLCV index with forward-fill.

    Handles timezone mismatch: prediction data may have different tz
    awareness than the OHLCV index.
    """
    if isinstance(pred_series.index, pd.DatetimeIndex) and isinstance(
        target_index, pd.DatetimeIndex
    ):
        if pred_series.index.tz is None and target_index.tz is not None:
            pred_series = pred_series.copy()
            pred_series.index = pred_series.index.tz_localize("UTC")
        elif pred_series.index.tz is not None and target_index.tz is None:
            pred_series = pred_series.copy()
            pred_series.index = pred_series.index.tz_localize(None)
    return pred_series.reindex(target_index, method=method)


def _nan_series(index: pd.Index, name: str) -> pd.Series:
    """Return a NaN-filled Series with the given index and name."""
    return pd.Series(np.nan, index=index, name=name, dtype=float)


@register_feature
class ModelPredictionFeature(BaseFeature):
    """Qlib model prediction confidence as a computed feature.

    Exposes the ML model's prediction score from the prediction cache
    as a feature column named ``qlib_prediction``. This allows
    VotingStrategy to use ML confidence as a formal sub-signal vote
    via DSL conditions (e.g., ``feature_above``).

    The prediction data must come from a model trained *before* the
    backtest period to avoid look-ahead bias. This is enforced by the
    prediction cache's bias protection (Phase 43).
    """

    name = "qlib_prediction"
    description = "Qlib model prediction confidence from prediction cache"
    supports_backtest = True
    supports_live = False
    bias_risk = ["look_ahead"]
    stateful = False

    def compute(
        self,
        ohlcv: pd.DataFrame,
        prediction_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.Series:
        """Compute prediction feature from injected prediction data.

        Args:
            ohlcv: Primary OHLCV DataFrame.
            prediction_data: DataFrame with 'prediction' column. May have
                a MultiIndex (datetime, instrument) or a simple DatetimeIndex.
                Pre-filtered to a single instrument by the caller.

        Returns:
            Series named 'qlib_prediction' aligned to ohlcv.index.
            Returns NaN-filled series when prediction_data is unavailable.
        """
        col_name = "qlib_prediction"

        if not self._validate(ohlcv):
            return pd.Series(dtype=float, name=col_name)

        if prediction_data is None or (
            hasattr(prediction_data, "empty") and prediction_data.empty
        ):
            return _nan_series(ohlcv.index, col_name)

        if "prediction" not in prediction_data.columns:
            logger.warning(
                "prediction_data missing 'prediction' column; available: %s",
                list(prediction_data.columns),
            )
            return _nan_series(ohlcv.index, col_name)

        pred = prediction_data["prediction"]

        # Handle MultiIndex (datetime, instrument) from Qlib format
        if isinstance(pred.index, pd.MultiIndex):
            # Caller should have pre-filtered to single instrument,
            # but handle degenerate case: take mean per datetime
            pred = pred.groupby(level=0).mean()

        result = _align_predictions_to_index(pred, ohlcv.index)
        result.name = col_name
        return result
