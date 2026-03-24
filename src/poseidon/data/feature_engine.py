"""FeatureEngine — single entry point for feature computation.

Used by training, prediction, and backtesting paths.
Loads OHLCV from database, computes requested features, returns wide DataFrame.
"""

import logging
from datetime import datetime

import pandas as pd

from poseidon.data.features.base import get_feature
from poseidon.data.storage import read_ohlcv
from poseidon.models.base import SessionLocal

logger = logging.getLogger(__name__)

# Standard feature set with default parameters.
# Each entry is (feature_name, params_dict).
DEFAULT_FEATURES: list[tuple[str, dict]] = [
    ("sma", {"period": 5}),
    ("sma", {"period": 10}),
    ("sma", {"period": 20}),
    ("sma", {"period": 60}),
    ("ema", {"period": 12}),
    ("ema", {"period": 26}),
    ("rsi", {"period": 14}),
    ("macd", {}),
    ("bollinger", {"period": 20}),
    ("atr", {"period": 14}),
    ("returns", {}),
    ("std_vol", {"period": 20}),
    ("volume_sma", {"period": 20}),
    ("volume_ratio", {"period": 20}),
    ("obv", {}),
]


REGIME_FEATURES: list[tuple[str, dict]] = [
    # Existing TA features relevant to volatility prediction
    ("rsi", {"period": 14}),
    ("atr", {"period": 14}),
    ("bollinger", {"period": 20}),
    ("returns", {}),
    ("std_vol", {"period": 20}),
    ("std_vol", {"period": 5}),
    ("std_vol", {"period": 10}),
    # Regime-specific features
    ("vol_ratio", {"short_period": 5, "long_period": 20}),
    ("realized_vol", {"period": 5}),
    ("realized_vol", {"period": 10}),
    ("realized_vol", {"period": 20}),
    ("vol_of_vol", {"vol_period": 20, "meta_period": 10}),
    ("return_autocorr", {"period": 20}),
    ("volume_ratio", {"period": 20}),
    ("parkinson_vol", {"period": 20}),
    ("garman_klass_vol", {"period": 20}),
]


class FeatureEngine:
    """Compute features from OHLCV data.

    This is the single computation entry point. Training, prediction, and
    backtesting all use the same FeatureEngine to avoid train-serve skew.

    Two modes of operation:
    - compute(): loads OHLCV from DB, computes features, returns wide DataFrame
    - compute_from_df(): computes features from an already-loaded DataFrame
    """

    def compute(
        self,
        symbol: str,
        market: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV from database and compute features.

        Args:
            symbol: Symbol identifier (e.g., "2330", "AAPL", "BTCUSDT")
            market: Market name (e.g., "tw_stock", "us_stock", "crypto_spot")
            interval: Candle interval ("1d", "1h")
            start: Start datetime (inclusive). None for all available data.
            end: End datetime (inclusive). None for up to latest.
            feature_specs: List of (feature_name, params_dict). None uses DEFAULT_FEATURES.

        Returns:
            Wide DataFrame with columns: time, open, high, low, close, volume, + feature columns.
            Empty DataFrame if no OHLCV data found.
        """
        session = SessionLocal()
        try:
            ohlcv = read_ohlcv(session, symbol, market, interval, start, end)
        finally:
            session.close()

        if ohlcv.empty:
            logger.warning("No OHLCV data for %s/%s/%s", market, symbol, interval)
            return ohlcv

        return self.compute_from_df(ohlcv, feature_specs)

    def compute_from_df(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> pd.DataFrame:
        """Compute features from an already-loaded OHLCV DataFrame.

        This is the core computation method. compute() delegates to this
        after loading data. Backtesting uses this directly when OHLCV
        is already in memory.

        Args:
            ohlcv: DataFrame with columns [time, open, high, low, close, volume].
            feature_specs: List of (feature_name, params_dict). None uses DEFAULT_FEATURES.

        Returns:
            Wide DataFrame with original OHLCV columns + computed feature columns.
        """
        if ohlcv.empty:
            return ohlcv

        specs = feature_specs if feature_specs is not None else DEFAULT_FEATURES
        result = ohlcv.copy()

        for feature_name, params in specs:
            feature_cls = get_feature(feature_name)
            feature = feature_cls()
            computed = feature.compute(ohlcv, **params)

            if isinstance(computed, pd.Series):
                result[computed.name] = computed
            elif isinstance(computed, pd.DataFrame):
                for col in computed.columns:
                    result[col] = computed[col]

        logger.info(
            "Computed %d feature specs -> %d columns (from %d OHLCV rows)",
            len(specs),
            len(result.columns) - len(ohlcv.columns),
            len(ohlcv),
        )
        return result
