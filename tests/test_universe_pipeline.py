"""Tests for UniversePipeline with filter chaining and position-aware retention."""

from unittest.mock import MagicMock, patch

import pytest

from poseidon.data.symbols import SymbolInfo
from poseidon.universe.base import UniverseFilter, UniverseSource
from poseidon.universe.pipeline import UniversePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSource(UniverseSource):
    """A fake source returning a fixed list of symbols."""
    name = "fake"

    def __init__(self, symbols: list[SymbolInfo]):
        self._symbols = symbols

    def resolve(self, market: str) -> list[SymbolInfo]:
        return list(self._symbols)


class RemoveFilter(UniverseFilter):
    """A filter that removes symbols by id."""
    name = "remove"

    def __init__(self, remove_ids: set[str]):
        self._remove_ids = remove_ids

    def filter(self, symbols: list[SymbolInfo], market: str) -> list[SymbolInfo]:
        return [s for s in symbols if s.id not in self._remove_ids]


def _symbols() -> list[SymbolInfo]:
    return [
        SymbolInfo(id="BTC", name="Bitcoin", ccxt_symbol="BTC/USDT"),
        SymbolInfo(id="ETH", name="Ethereum", ccxt_symbol="ETH/USDT"),
        SymbolInfo(id="SOL", name="Solana", ccxt_symbol="SOL/USDT"),
        SymbolInfo(id="DOGE", name="Dogecoin", ccxt_symbol="DOGE/USDT"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUniversePipeline:
    def test_run_calls_source_and_applies_filters(self):
        """Pipeline resolves source, then applies each filter in order."""
        syms = _symbols()
        source = FakeSource(syms)
        f1 = RemoveFilter({"DOGE"})
        f2 = RemoveFilter({"SOL"})

        pipeline = UniversePipeline(source=source, filters=[f1, f2])
        result = pipeline.run("crypto_spot")
        ids = [s.id for s in result]
        assert ids == ["BTC", "ETH"]

    def test_position_aware_retention(self):
        """Symbols with open positions are retained even after filtering."""
        syms = _symbols()
        source = FakeSource(syms)
        # Filter removes ETH and SOL
        f1 = RemoveFilter({"ETH", "SOL"})

        pipeline = UniversePipeline(source=source, filters=[f1])

        # Mock DB session: ETH has open position
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [MagicMock(symbol="ETH")]

        result = pipeline.run("crypto_spot", db_session=mock_session)
        ids = [s.id for s in result]
        assert "BTC" in ids  # passed filter
        assert "DOGE" in ids  # passed filter
        assert "ETH" in ids  # retained due to open position
        assert "SOL" not in ids  # filtered out, no open position

    def test_empty_source_returns_empty(self):
        """Pipeline with empty source returns empty list without crashing."""
        source = FakeSource([])
        pipeline = UniversePipeline(source=source, filters=[RemoveFilter({"X"})])
        result = pipeline.run("crypto_spot")
        assert result == []

    def test_no_filters_returns_source_as_is(self):
        """Pipeline with no filters returns source resolve output unchanged."""
        syms = _symbols()
        source = FakeSource(syms)
        pipeline = UniversePipeline(source=source, filters=[])
        result = pipeline.run("crypto_spot")
        assert [s.id for s in result] == ["BTC", "ETH", "SOL", "DOGE"]

    def test_no_db_session_skips_retention(self):
        """Without db_session, position-aware retention is skipped."""
        syms = _symbols()
        source = FakeSource(syms)
        f1 = RemoveFilter({"ETH"})
        pipeline = UniversePipeline(source=source, filters=[f1])
        result = pipeline.run("crypto_spot")  # no db_session
        ids = [s.id for s in result]
        assert "ETH" not in ids
