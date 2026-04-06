"""Tests for universe filters: VolumeFilter, ListingAgeFilter, BlacklistFilter."""

from unittest.mock import MagicMock, patch

import pytest

from poseidon.data.symbols import SymbolInfo
from poseidon.universe.filters import BlacklistFilter, ListingAgeFilter, VolumeFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_symbols() -> list[SymbolInfo]:
    """Create a deterministic set of test symbols."""
    return [
        SymbolInfo(id="AAPL", name="Apple"),
        SymbolInfo(id="GOOG", name="Google"),
        SymbolInfo(id="MSFT", name="Microsoft"),
        SymbolInfo(id="TSLA", name="Tesla"),
    ]


# ---------------------------------------------------------------------------
# BlacklistFilter
# ---------------------------------------------------------------------------

class TestBlacklistFilter:
    def test_removes_blacklisted_symbols(self):
        symbols = _make_symbols()
        f = BlacklistFilter(blacklist={"GOOG", "TSLA"})
        result = f.filter(symbols, "us_stock")
        ids = [s.id for s in result]
        assert ids == ["AAPL", "MSFT"]

    def test_empty_blacklist_keeps_all(self):
        symbols = _make_symbols()
        f = BlacklistFilter()
        result = f.filter(symbols, "us_stock")
        assert len(result) == 4

    def test_blacklist_not_in_symbols_no_effect(self):
        symbols = _make_symbols()
        f = BlacklistFilter(blacklist={"NVDA"})
        result = f.filter(symbols, "us_stock")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# VolumeFilter
# ---------------------------------------------------------------------------

class TestVolumeFilter:
    def test_removes_symbols_below_min_volume(self):
        """VolumeFilter should remove symbols whose avg volume < min_volume."""
        symbols = _make_symbols()
        f = VolumeFilter(min_volume=500_000, lookback_days=30)

        # Mock _get_avg_volumes to return controlled data
        # AAPL: 1M (keep), GOOG: 200K (remove), MSFT: 600K (keep), TSLA: None (keep, conservative)
        with patch.object(f, "_get_avg_volumes", return_value={
            "AAPL": 1_000_000,
            "GOOG": 200_000,
            "MSFT": 600_000,
            # TSLA missing -> no data -> keep (conservative)
        }):
            result = f.filter(symbols, "us_stock")
        ids = [s.id for s in result]
        assert "AAPL" in ids
        assert "MSFT" in ids
        assert "TSLA" in ids  # no data -> conservative keep
        assert "GOOG" not in ids  # below threshold

    def test_no_volume_data_keeps_all(self):
        """When no volume data at all, keep all symbols (conservative)."""
        symbols = _make_symbols()
        f = VolumeFilter(min_volume=500_000)
        with patch.object(f, "_get_avg_volumes", return_value={}):
            result = f.filter(symbols, "us_stock")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# ListingAgeFilter
# ---------------------------------------------------------------------------

class TestListingAgeFilter:
    def test_removes_recently_listed_symbols(self):
        """ListingAgeFilter should remove symbols with < min_days of data."""
        symbols = _make_symbols()
        f = ListingAgeFilter(min_days=90)

        # AAPL: 365 days (keep), GOOG: 30 days (remove), MSFT: 120 days (keep), TSLA: None (keep)
        with patch.object(f, "_get_listing_ages", return_value={
            "AAPL": 365,
            "GOOG": 30,
            "MSFT": 120,
            # TSLA missing -> no data -> keep (conservative)
        }):
            result = f.filter(symbols, "us_stock")
        ids = [s.id for s in result]
        assert "AAPL" in ids
        assert "MSFT" in ids
        assert "TSLA" in ids  # conservative
        assert "GOOG" not in ids  # too recent

    def test_no_listing_data_keeps_all(self):
        symbols = _make_symbols()
        f = ListingAgeFilter(min_days=90)
        with patch.object(f, "_get_listing_ages", return_value={}):
            result = f.filter(symbols, "us_stock")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Filter chaining (determinism)
# ---------------------------------------------------------------------------

class TestFilterChaining:
    def test_chain_volume_and_blacklist_deterministic(self):
        """Chaining VolumeFilter + BlacklistFilter produces deterministic result."""
        symbols = _make_symbols()
        vol_filter = VolumeFilter(min_volume=500_000)
        bl_filter = BlacklistFilter(blacklist={"MSFT"})

        with patch.object(vol_filter, "_get_avg_volumes", return_value={
            "AAPL": 1_000_000,
            "GOOG": 200_000,
            "MSFT": 600_000,
            "TSLA": 800_000,
        }):
            after_vol = vol_filter.filter(symbols, "us_stock")
        after_bl = bl_filter.filter(after_vol, "us_stock")

        ids = [s.id for s in after_bl]
        assert ids == ["AAPL", "TSLA"]

    def test_same_input_same_output(self):
        """Same inputs always produce same outputs (determinism check)."""
        symbols = _make_symbols()
        f = BlacklistFilter(blacklist={"GOOG"})
        r1 = f.filter(symbols, "us_stock")
        r2 = f.filter(symbols, "us_stock")
        assert [s.id for s in r1] == [s.id for s in r2]
