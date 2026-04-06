"""Abstract base classes for universe sources and filters."""

from abc import ABC, abstractmethod

from poseidon.data.symbols import SymbolInfo


class UniverseSource(ABC):
    """Abstract base class for universe symbol sources.

    Each source resolves a list of candidate symbols for a given market.
    Subclasses must define ``name`` and implement ``resolve()``.
    """

    name: str = ""
    description: str = ""

    # Phase 34 capability metadata
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    @abstractmethod
    def resolve(self, market: str) -> list[SymbolInfo]:
        """Return candidate symbols for the given market."""
        ...


class UniverseFilter(ABC):
    """Abstract base class for universe filters.

    Each filter takes a list of symbols and returns a (possibly smaller)
    list of symbols that pass the filter criteria.
    Subclasses must define ``name`` and implement ``filter()``.
    """

    name: str = ""
    description: str = ""

    # Phase 34 capability metadata
    supports_backtest: bool = True
    supports_live: bool = True
    bias_risk: list[str] = []
    stateful: bool = False

    @abstractmethod
    def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
        """Filter symbols, returning only those that pass criteria."""
        ...
