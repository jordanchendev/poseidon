"""Macro index loader using yfinance for VIX, DXY, TNX, TWDUSD."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MacroIndexLoader:
    """Load macro economic indices from yfinance as a single DataFrame.

    Fetches VIX, DXY, US 10Y Treasury yield, and TWD/USD exchange rate.
    Forward-fills NaN gaps caused by different market calendars.
    """

    MACRO_TICKERS: dict[str, str] = {
        "macro_vix": "^VIX",
        "macro_dxy": "DX-Y.NYB",
        "macro_tnx": "^TNX",
        "macro_twdusd": "TWDUSD=X",
    }

    def __init__(self):
        self._cache: pd.DataFrame | None = None

    def get_macro_data(self, start: str = "2020-01-01") -> pd.DataFrame:
        """Fetch macro indices and return as a single DataFrame.

        Args:
            start: Start date "YYYY-MM-DD".

        Returns:
            DataFrame indexed by date with columns:
            macro_vix, macro_dxy, macro_tnx, macro_twdusd.
            Returns empty DataFrame on error.
        """
        if self._cache is not None:
            return self._cache

        import yfinance as yf

        tickers = list(self.MACRO_TICKERS.values())

        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                interval="1d",
                auto_adjust=True,
                threads=False,
                progress=False,
            )
        except Exception:
            logger.exception("Failed to download macro data from yfinance")
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning("No macro data returned from yfinance")
            return pd.DataFrame()

        # Build inverse map: ticker symbol -> clean feature name
        ticker_to_name = {v: k for k, v in self.MACRO_TICKERS.items()}

        result_cols: dict[str, pd.Series] = {}

        for ticker, feature_name in ticker_to_name.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    # MultiIndex: (Price, Ticker) format from yfinance >= 0.2.x
                    # Try "Close" price level
                    if ("Close", ticker) in raw.columns:
                        result_cols[feature_name] = raw[("Close", ticker)]
                    elif ticker in raw.columns.get_level_values(1):
                        # Fallback: try to extract from second level
                        close_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
                        if ticker in close_df.columns:
                            result_cols[feature_name] = close_df[ticker]
                else:
                    # Single ticker or flat columns
                    if "Close" in raw.columns:
                        result_cols[feature_name] = raw["Close"]
                    elif ticker in raw.columns:
                        result_cols[feature_name] = raw[ticker]
            except Exception:
                logger.warning("Failed to extract %s (%s) from yfinance data", feature_name, ticker)

        if not result_cols:
            logger.warning("No macro columns extracted from yfinance response")
            return pd.DataFrame()

        result = pd.DataFrame(result_cols)
        result.index.name = "date"

        # Forward-fill NaN from different market calendars
        result = result.ffill()

        self._cache = result
        return result
