"""Data fetcher factory and registry."""

import logging

from poseidon.data.fetchers.base import BaseFetcher
from poseidon.data.fetchers.ccxt_fetcher import CCXTFetcher
from poseidon.data.fetchers.finmind import FinMindFetcher
from poseidon.data.fetchers.yfinance_fetcher import YFinanceFetcher

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency on finlab at import time
_FinLabFetcher = None


def _get_finlab_fetcher_class():
    global _FinLabFetcher
    if _FinLabFetcher is None:
        from poseidon.data.fetchers.finlab_fetcher import FinLabFetcher
        _FinLabFetcher = FinLabFetcher
    return _FinLabFetcher


# Fetcher instances are reusable (they hold no per-request state beyond config)
_fetcher_cache: dict[str, BaseFetcher] = {}


def get_fetcher(market: str, backend: str | None = None) -> BaseFetcher:
    """Return the appropriate fetcher for a given market.

    Args:
        market: One of "tw_stock", "tw_futures", "us_stock", "crypto_spot"
        backend: Optional override: "finlab", "finmind", "yfinance", "ccxt".
                 If None, uses FinLab as default for stocks/futures, CCXT for crypto.

    Returns:
        BaseFetcher instance for the market.

    Raises:
        ValueError: If market is not supported.
    """
    cache_key = f"{market}:{backend or 'default'}"
    if cache_key in _fetcher_cache:
        return _fetcher_cache[cache_key]

    fetcher: BaseFetcher

    # Explicit backend override
    if backend == "finmind":
        fetcher = FinMindFetcher(market=market)
    elif backend == "yfinance":
        fetcher = YFinanceFetcher()
    elif backend == "ccxt":
        fetcher = CCXTFetcher()
    elif backend == "finlab":
        FinLabCls = _get_finlab_fetcher_class()
        fetcher = FinLabCls(market=market)
    # Default routing: FinLab for stocks/futures, CCXT for crypto
    elif market in ("tw_stock", "tw_futures", "us_stock"):
        try:
            FinLabCls = _get_finlab_fetcher_class()
            fetcher = FinLabCls(market=market)
        except Exception:
            # Fallback to legacy fetchers
            logger.warning("FinLab unavailable for %s, falling back to legacy fetcher", market)
            if market == "us_stock":
                fetcher = YFinanceFetcher()
            else:
                fetcher = FinMindFetcher(market=market)
    elif market == "crypto_spot":
        fetcher = CCXTFetcher()
    elif market == "crypto_perp":
        from poseidon.data.fetchers.perp_fetcher import PerpFetcher
        fetcher = PerpFetcher()
    else:
        raise ValueError(f"Unsupported market: {market}. Supported: tw_stock, tw_futures, us_stock, crypto_spot, crypto_perp")

    _fetcher_cache[cache_key] = fetcher
    return fetcher


__all__ = ["BaseFetcher", "FinMindFetcher", "YFinanceFetcher", "CCXTFetcher", "get_fetcher"]
