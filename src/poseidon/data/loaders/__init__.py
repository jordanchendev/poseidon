"""Non-price data loaders for institutional flow, funding rates, OI, and macro indices."""

from poseidon.data.loaders.finlab_loader import FinLabDataLoader
from poseidon.data.loaders.funding_loader import FundingRateLoader
from poseidon.data.loaders.macro_loader import MacroIndexLoader
from poseidon.data.loaders.oi_loader import OpenInterestLoader

__all__ = ["FinLabDataLoader", "FundingRateLoader", "MacroIndexLoader", "OpenInterestLoader"]
