"""Non-price data loaders for institutional flow, funding rates, and macro indices."""

from poseidon.data.loaders.finlab_loader import FinLabDataLoader
from poseidon.data.loaders.funding_loader import FundingRateLoader
from poseidon.data.loaders.macro_loader import MacroIndexLoader

__all__ = ["FinLabDataLoader", "FundingRateLoader", "MacroIndexLoader"]
