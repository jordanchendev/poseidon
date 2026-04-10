"""FinLab data loader for institutional flow, fundamentals, and trade structure."""

import logging

import pandas as pd

from poseidon.data.fetchers.finlab_fetcher import _finlab_lock

logger = logging.getLogger(__name__)


class FinLabDataLoader:
    """Load non-price datasets from FinLab for TW stock symbols.

    Provides institutional investor flow, fundamental ratios, and trade
    structure data. Uses the same thread lock as FinLabFetcher to safely
    call ``data.set_market()`` / ``data.get()``.
    """

    def __init__(self, token: str | None = None):
        self._token = token
        self._initialized = False
        self._cache: dict[str, pd.DataFrame] = {}

    def _ensure_login(self) -> None:
        """Lazy login to FinLab using settings or provided token."""
        if self._initialized:
            return
        import finlab

        from poseidon.core.config import settings

        token = self._token or settings.finlab_api_token
        if token:
            finlab.login(token)
        self._initialized = True

    def get_dataset(self, dataset_key: str) -> pd.DataFrame:
        """Fetch a wide DataFrame from FinLab with caching.

        Returns an empty DataFrame if the dataset is unavailable.
        """
        if dataset_key in self._cache:
            return self._cache[dataset_key]

        from finlab import data

        self._ensure_login()

        try:
            with _finlab_lock:
                data.set_market("tw")
                df = data.get(dataset_key)
        except Exception:
            logger.exception("Failed to load FinLab dataset: %s", dataset_key)
            return pd.DataFrame()

        if df is not None and not df.empty:
            self._cache[dataset_key] = df
            return df

        return pd.DataFrame()

    def get_institutional_flow(self, symbol: str) -> pd.DataFrame:
        """Load institutional investor buy/sell data for a TW stock symbol.

        Returns DataFrame with columns: foreign, trust, dealer, dealer_hedge
        indexed by date. Returns empty DataFrame if symbol not found.
        """
        datasets = {
            "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)": "foreign",
            "institutional_investors_trading_summary:投信買賣超股數": "trust",
            "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)": "dealer",
            "institutional_investors_trading_summary:自營商買賣超股數(避險)": "dealer_hedge",
        }
        return self._extract_columns(symbol, datasets)

    def get_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Load fundamental ratios for a TW stock symbol.

        Returns DataFrame with columns: pe_ratio, pb_ratio, monthly_rev, prev_year_rev, roe, roa
        indexed by date. Returns empty DataFrame if symbol not found.
        """
        datasets = {
            "price_earning_ratio:本益比": "pe_ratio",
            "price_earning_ratio:股價淨值比": "pb_ratio",
            "monthly_revenue:當月營收": "monthly_rev",
            "monthly_revenue:去年當月營收": "prev_year_rev",
            "fundamental_features:ROE稅後": "roe",
            "fundamental_features:ROA稅後息前": "roa",
        }
        return self._extract_columns(symbol, datasets)

    def get_trade_structure(self, symbol: str) -> pd.DataFrame:
        """Load trade structure data for a TW stock symbol.

        Returns DataFrame with columns: trade_value, trade_count
        indexed by date. Returns empty DataFrame if symbol not found.
        """
        datasets = {
            "price:成交金額": "trade_value",
            "price:成交筆數": "trade_count",
        }
        return self._extract_columns(symbol, datasets)

    def _extract_columns(
        self, symbol: str, datasets: dict[str, str]
    ) -> pd.DataFrame:
        """Extract a symbol column from multiple wide DataFrames and combine.

        Args:
            symbol: Stock code (e.g. "2330").
            datasets: Mapping of FinLab dataset key -> output column name.

        Returns:
            DataFrame indexed by date with the requested columns.
        """
        series_map: dict[str, pd.Series] = {}

        for dataset_key, col_name in datasets.items():
            wide_df = self.get_dataset(dataset_key)
            if wide_df.empty or symbol not in wide_df.columns:
                logger.debug(
                    "Symbol %s not found in dataset %s", symbol, dataset_key
                )
                return pd.DataFrame()
            series_map[col_name] = wide_df[symbol]

        if not series_map:
            return pd.DataFrame()

        result = pd.DataFrame(series_map)
        result.index.name = "date"
        return result
