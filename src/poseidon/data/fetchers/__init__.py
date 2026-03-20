"""Data fetcher factory and registry."""

from poseidon.data.fetchers.base import BaseFetcher
from poseidon.data.fetchers.ccxt_fetcher import CCXTFetcher
from poseidon.data.fetchers.finmind import FinMindFetcher
from poseidon.data.fetchers.yfinance_fetcher import YFinanceFetcher

# Fetcher instances are reusable (they hold no per-request state beyond config)
_fetcher_cache: dict[str, BaseFetcher] = {}


def get_fetcher(market: str) -> BaseFetcher:
    """Return the appropriate fetcher for a given market.

    Args:
        market: One of "tw_stock", "tw_futures", "us_stock", "crypto_spot"

    Returns:
        BaseFetcher instance for the market.

    Raises:
        ValueError: If market is not supported.
    """
    if market in _fetcher_cache:
        return _fetcher_cache[market]

    if market == "tw_stock":
        fetcher = FinMindFetcher(market="tw_stock")
    elif market == "tw_futures":
        fetcher = FinMindFetcher(market="tw_futures")
    elif market == "us_stock":
        fetcher = YFinanceFetcher()
    elif market == "crypto_spot":
        fetcher = CCXTFetcher()
    else:
        raise ValueError(f"Unsupported market: {market}. Supported: tw_stock, tw_futures, us_stock, crypto_spot")

    _fetcher_cache[market] = fetcher
    return fetcher


__all__ = ["BaseFetcher", "FinMindFetcher", "YFinanceFetcher", "CCXTFetcher", "get_fetcher"]
