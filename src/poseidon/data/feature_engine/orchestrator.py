"""FeatureOrchestrator -- data loading + computation coordination.

Drop-in replacement for the old monolithic FeatureEngine class. Wraps
FeatureComputer for pure computation and DataRepository for data loading.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from poseidon.core.database import SessionLocal, db_session
from poseidon.data.feature_engine.computer import FeatureComputer
from poseidon.data.feature_engine.specs import (
    DEFAULT_FEATURES,
    FUNDING_NAMES,
    FUNDAMENTAL_NAMES,
    INSTITUTIONAL_PREFIXES,
    MACRO_PREFIX,
    MARGIN_NAMES,
    OI_NAMES,
    PREDICTION_NAMES,
    TRADE_STRUCTURE_NAMES,
    is_nonprice_spec,
    nonprice_data_key,
)
from poseidon.data.repository import DataRepository

logger = logging.getLogger(__name__)


class FeatureOrchestrator:
    """Coordinates data loading and feature computation.

    Drop-in replacement for FeatureEngine. Two modes of operation:
    - compute(): loads OHLCV from DB, computes features, returns wide DataFrame
    - compute_from_df(): computes features from an already-loaded DataFrame
    - compute_with_companions(): handles cross-asset + non-price features with DB loading
    """

    def __init__(self, repo: DataRepository | None = None) -> None:
        self._repo = repo
        self._computer = FeatureComputer()

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
        with db_session() as session:
            repo = DataRepository(session)
            ohlcv = repo.read_ohlcv(symbol, market, interval, start, end)

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

        Direct delegation to FeatureComputer.compute_from_df().

        Args:
            ohlcv: DataFrame with columns [time, open, high, low, close, volume].
            feature_specs: List of (feature_name, params_dict). None uses DEFAULT_FEATURES.

        Returns:
            Wide DataFrame with original OHLCV columns + computed feature columns.
        """
        return self._computer.compute_from_df(ohlcv, feature_specs)

    def compute_with_companions(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        market: str,
        interval: str,
        feature_specs: list[tuple[str, dict]] | None = None,
        db_session=None,
        extra_nonprice_data: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Compute features including cross-asset and non-price data features.

        Three categories of feature specs:
        1. Single-symbol: standard TA features computed from OHLCV only.
        2. Cross-asset: specs with ``companion_symbol`` in params; loads
           companion OHLCV from DB.
        3. Non-price: specs whose name maps to external data (institutional
           flow, fundamentals, trade structure, funding rates, macro indices).
           Data is loaded lazily via DataRepository and injected as kwargs.

        Args:
            ohlcv: Primary symbol OHLCV DataFrame.
            symbol: Primary symbol identifier.
            market: Primary market name.
            interval: Candle interval.
            feature_specs: Feature specifications. None uses DEFAULT_FEATURES.
            db_session: SQLAlchemy session for loading companion data. If None, creates one.
            extra_nonprice_data: Externally-injected non-price data to merge
                into the loader results. Used by BacktestRunner to inject
                prediction cache data (e.g., ``{"prediction_data": df}``).

        Returns:
            Wide DataFrame with OHLCV + all computed feature columns.
        """
        if ohlcv.empty:
            return ohlcv

        specs = feature_specs if feature_specs is not None else DEFAULT_FEATURES

        # Separate single-symbol, cross-asset, and non-price specs
        single_specs: list[tuple[str, dict]] = []
        cross_specs: list[tuple[str, dict]] = []
        nonprice_specs: list[tuple[str, dict]] = []
        for name, params in specs:
            if "companion_symbol" in params:
                cross_specs.append((name, params))
            elif is_nonprice_spec(name):
                nonprice_specs.append((name, params))
            else:
                single_specs.append((name, params))

        # Compute single-symbol features first
        result = self._computer.compute_from_df(ohlcv, single_specs) if single_specs else ohlcv.copy()

        has_external = bool(cross_specs) or bool(nonprice_specs)
        if not has_external:
            return result

        # Load companion data and compute cross-asset features
        close_session = False
        if db_session is None:
            db_session = SessionLocal()
            close_session = True

        repo = DataRepository(db_session)

        try:
            # --- Cross-asset features ---
            companion_cache: dict[tuple[str, str], pd.DataFrame] = {}

            for feature_name, params in cross_specs:
                comp_symbol = params.get("companion_symbol", "")
                comp_market = params.get("companion_market", market)
                cache_key = (comp_market, comp_symbol)

                if cache_key not in companion_cache:
                    companion_cache[cache_key] = repo.read_ohlcv(
                        comp_symbol, comp_market, interval,
                    )

                companion_df = companion_cache[cache_key]
                feature_cls = get_feature(feature_name)
                feature = feature_cls()
                computed = feature.compute(
                    ohlcv,
                    companion_data=companion_df if not companion_df.empty else None,
                    **params,
                )

                if isinstance(computed, pd.Series):
                    result[computed.name] = computed
                elif isinstance(computed, pd.DataFrame):
                    for col in computed.columns:
                        result[col] = computed[col]

            # --- Non-price data features ---
            if nonprice_specs:
                nonprice_data = self._load_nonprice_data(
                    nonprice_specs, symbol, repo,
                )
                # Merge externally-injected non-price data (e.g., prediction_data
                # from BacktestRunner). External data takes precedence over loader.
                if extra_nonprice_data:
                    nonprice_data.update(extra_nonprice_data)
                for feature_name, params in nonprice_specs:
                    data_key = nonprice_data_key(feature_name)
                    extra_kwargs = {data_key: nonprice_data.get(data_key)}

                    feature_cls = get_feature(feature_name)
                    feature = feature_cls()
                    computed = feature.compute(ohlcv, **params, **extra_kwargs)

                    if isinstance(computed, pd.Series):
                        result[computed.name] = computed
                    elif isinstance(computed, pd.DataFrame):
                        for col in computed.columns:
                            result[col] = computed[col]
        finally:
            if close_session:
                db_session.close()

        logger.info(
            "Computed %d single + %d cross-asset + %d non-price specs -> %d total columns",
            len(single_specs),
            len(cross_specs),
            len(nonprice_specs),
            len(result.columns) - len(ohlcv.columns),
        )
        return result

    @staticmethod
    def _load_nonprice_data(
        nonprice_specs: list[tuple[str, dict]],
        symbol: str,
        repo: DataRepository | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Lazily load non-price data required by the given specs via DataRepository.

        Only fetches data when the corresponding feature names are present
        in *nonprice_specs*. All data access goes through DataRepository.

        Args:
            nonprice_specs: Feature specs requiring non-price data.
            symbol: Symbol identifier.
            repo: DataRepository instance. If None, creates one from SessionLocal
                (backward compat for callers not yet migrated).
        """
        close_repo_session = False
        if repo is None:
            _session = SessionLocal()
            repo = DataRepository(_session)
            close_repo_session = True

        try:
            nonprice_data: dict[str, pd.DataFrame] = {}

            needs_institutional = any(
                n.startswith(INSTITUTIONAL_PREFIXES) for n, _ in nonprice_specs
            )
            needs_fundamental = any(n in FUNDAMENTAL_NAMES for n, _ in nonprice_specs)
            needs_trade_structure = any(n in TRADE_STRUCTURE_NAMES for n, _ in nonprice_specs)
            needs_funding = any(n in FUNDING_NAMES for n, _ in nonprice_specs)
            needs_margin = any(n in MARGIN_NAMES for n, _ in nonprice_specs)
            needs_oi = any(n in OI_NAMES for n, _ in nonprice_specs)
            needs_macro = any(n.startswith(MACRO_PREFIX) for n, _ in nonprice_specs)

            if needs_institutional:
                nonprice_data["institutional_data"] = repo.read_institutional_flow(symbol)
            if needs_fundamental:
                nonprice_data["fundamental_data"] = repo.read_fundamentals_df(symbol)
            if needs_trade_structure:
                nonprice_data["trade_structure_data"] = repo.read_trade_structure(symbol)
            if needs_margin:
                nonprice_data["margin_data"] = repo.read_margin_transactions(symbol)

            if needs_funding:
                nonprice_data["funding_data"] = repo.read_funding_rates(symbol)

            if needs_oi:
                nonprice_data["oi_data"] = repo.read_open_interest(symbol)

            if needs_macro:
                nonprice_data["macro_data"] = repo.read_macro()

            return nonprice_data
        finally:
            if close_repo_session:
                _session.close()


# Need get_feature for cross-asset computation in compute_with_companions
from poseidon.data.features.base import get_feature  # noqa: E402
