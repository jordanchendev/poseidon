"""HMM-based regime detection features using Gaussian Hidden Markov Model.

Produces regime labels sorted by ascending volatility (0=low vol, 2=high vol),
transition probabilities, and regime duration counts.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from poseidon.data.features.base import BaseFeature, register_feature

logger = logging.getLogger(__name__)


@register_feature
class HMMRegime(BaseFeature):
    """HMM-based volatility regime detection.

    Fits a Gaussian HMM on (log_returns, rolling_volatility) to identify
    volatility regimes. Regimes are sorted by ascending volatility to ensure
    deterministic label assignment (label 0 = low vol, label 2 = high vol).
    """

    name = "hmm_regime"
    description = "HMM volatility regime (label, transition prob, duration)"
    bias_risk = ["future_data_leakage"]
    stateful = True

    def compute(
        self,
        ohlcv: pd.DataFrame,
        n_regimes: int = 3,
        vol_period: int = 20,
        random_state: int = 42,
        **kwargs,
    ) -> pd.DataFrame:
        min_rows = vol_period + 30
        empty_df = pd.DataFrame(
            {
                "hmm_regime_label": pd.Series(dtype=float),
                "hmm_regime_transition_prob": pd.Series(dtype=float),
                "hmm_regime_duration": pd.Series(dtype=float),
            }
        )
        if not self._validate(ohlcv, min_rows=min_rows):
            return empty_df

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("hmmlearn or sklearn not installed; returning NaN DataFrame")
            return self._nan_result(ohlcv)

        # Compute input features: log returns and rolling volatility
        log_returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        rolling_vol = log_returns.rolling(window=vol_period).std()

        # Stack into 2D feature matrix
        features = pd.DataFrame({"returns": log_returns, "vol": rolling_vol}).dropna()
        if len(features) < min_rows:
            return self._nan_result(ohlcv)

        X = features.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=random_state,
                )
                model.fit(X_scaled)

            # Sort regimes by ascending mean of volatility dimension (column 1)
            vol_means = model.means_[:, 1]
            sort_order = np.argsort(vol_means)

            # Remap model parameters by sorted order
            model.means_ = model.means_[sort_order]
            model.covars_ = model.covars_[sort_order][:, sort_order]
            model.transmat_ = model.transmat_[sort_order][:, sort_order]
            model.startprob_ = model.startprob_[sort_order]

            # Predict regime labels (now sorted: 0=low vol, 2=high vol)
            raw_labels = model.predict(X_scaled)

            # Build label remap: old_label -> new_label
            remap = np.empty(n_regimes, dtype=int)
            for new_idx, old_idx in enumerate(sort_order):
                remap[old_idx] = new_idx
            labels = remap[raw_labels]

            # Transition probability: max probability of transitioning from current regime
            transition_prob = np.array([model.transmat_[label].max() for label in labels])

            # Regime duration: count consecutive bars in same regime
            duration = self._compute_duration(labels)

        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning("HMM fit failed: %s; returning NaN DataFrame", e)
            return self._nan_result(ohlcv)

        # Align results back to original index
        result = pd.DataFrame(
            {
                "hmm_regime_label": np.nan,
                "hmm_regime_transition_prob": np.nan,
                "hmm_regime_duration": np.nan,
            },
            index=ohlcv.index,
        )
        result.loc[features.index, "hmm_regime_label"] = labels.astype(float)
        result.loc[features.index, "hmm_regime_transition_prob"] = transition_prob
        result.loc[features.index, "hmm_regime_duration"] = duration.astype(float)

        return result

    @staticmethod
    def _compute_duration(labels: np.ndarray) -> np.ndarray:
        """Count consecutive bars in the same regime."""
        n = len(labels)
        duration = np.ones(n, dtype=int)
        for i in range(1, n):
            if labels[i] == labels[i - 1]:
                duration[i] = duration[i - 1] + 1
            else:
                duration[i] = 1
        return duration

    @staticmethod
    def _nan_result(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return all-NaN DataFrame matching ohlcv index."""
        return pd.DataFrame(
            {
                "hmm_regime_label": np.nan,
                "hmm_regime_transition_prob": np.nan,
                "hmm_regime_duration": np.nan,
            },
            index=ohlcv.index,
        )
