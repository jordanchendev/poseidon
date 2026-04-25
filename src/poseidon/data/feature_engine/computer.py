"""FeatureComputer -- pure feature computation, zero I/O.

Extracted from the monolithic FeatureEngine class. This module must NOT
import any I/O library (repository, database, requests, ccxt, yfinance,
finlab, sqlalchemy) or cross-module dependency.

Only allowed imports: logging, pandas, poseidon.data.features.base.get_feature,
and poseidon.data.feature_engine.specs.
"""

from __future__ import annotations

import logging

import pandas as pd

from poseidon.data.feature_engine.specs import (
    DEFAULT_FEATURES,
    is_nonprice_spec,
    nonprice_data_key,
)
from poseidon.data.features.base import get_feature

logger = logging.getLogger(__name__)


class FeatureComputer:
    """Stateless feature computation engine. No database, no network, no side effects."""

    def compute_from_df(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> pd.DataFrame:
        """Compute features from an already-loaded OHLCV DataFrame.

        This is the core computation method. Backtesting uses this directly
        when OHLCV is already in memory.

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

            self._merge_computed(result, computed)

        logger.info(
            "Computed %d feature specs -> %d columns (from %d OHLCV rows)",
            len(specs),
            len(result.columns) - len(ohlcv.columns),
            len(ohlcv),
        )
        return result

    def compute_with_data(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]],
        companion_cache: dict[tuple[str, str], pd.DataFrame] | None = None,
        nonprice_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Compute features with all required data pre-loaded.

        Classifies specs into single/cross/nonprice, computes each category.
        All data is passed in -- never loaded internally.

        Args:
            ohlcv: Primary symbol OHLCV DataFrame.
            feature_specs: Feature specifications.
            companion_cache: Pre-loaded companion OHLCV keyed by (market, symbol).
            nonprice_data: Pre-loaded non-price data keyed by data type.

        Returns:
            Wide DataFrame with OHLCV + all computed feature columns.
        """
        if ohlcv.empty:
            return ohlcv

        companion_cache = companion_cache or {}
        nonprice_data = nonprice_data or {}

        # Separate single-symbol, cross-asset, and non-price specs
        single_specs: list[tuple[str, dict]] = []
        cross_specs: list[tuple[str, dict]] = []
        nonprice_specs: list[tuple[str, dict]] = []
        for name, params in feature_specs:
            if "companion_symbol" in params:
                cross_specs.append((name, params))
            elif is_nonprice_spec(name):
                nonprice_specs.append((name, params))
            else:
                single_specs.append((name, params))

        # Compute single-symbol features first
        result = self.compute_from_df(ohlcv, single_specs) if single_specs else ohlcv.copy()

        # --- Cross-asset features ---
        for feature_name, params in cross_specs:
            comp_symbol = params.get("companion_symbol", "")
            comp_market = params.get("companion_market", "")
            cache_key = (comp_market, comp_symbol)
            companion_df = companion_cache.get(cache_key, pd.DataFrame())

            feature_cls = get_feature(feature_name)
            feature = feature_cls()
            computed = feature.compute(
                ohlcv,
                companion_data=companion_df if not companion_df.empty else None,
                **params,
            )
            self._merge_computed(result, computed)

        # --- Non-price data features ---
        for feature_name, params in nonprice_specs:
            data_key = nonprice_data_key(feature_name)
            extra_kwargs = {data_key: nonprice_data.get(data_key)}

            feature_cls = get_feature(feature_name)
            feature = feature_cls()
            computed = feature.compute(ohlcv, **params, **extra_kwargs)
            self._merge_computed(result, computed)

        logger.info(
            "Computed %d single + %d cross-asset + %d non-price specs -> %d total columns",
            len(single_specs),
            len(cross_specs),
            len(nonprice_specs),
            len(result.columns) - len(ohlcv.columns),
        )
        return result

    @staticmethod
    def _merge_computed(result: pd.DataFrame, computed: pd.Series | pd.DataFrame) -> None:
        """Merge a computed Series or DataFrame into the result DataFrame."""
        if isinstance(computed, pd.Series):
            result[computed.name] = computed
        elif isinstance(computed, pd.DataFrame):
            for col in computed.columns:
                result[col] = computed[col]
