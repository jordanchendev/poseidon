"""Tests for MarketCapWeightedAllocator (Phase 71 D-16)."""

import pytest

from poseidon.strategies.portfolio.fundamental_selection import MarketCapWeightedAllocator


class TestMarketCapWeightedAllocator:
    """Unit tests for market-cap proportional weight allocation with position caps."""

    def test_proportional_weights_sum_to_one(self):
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.50)
        symbols = ["A", "B", "C"]
        mv = {"A": 100.0, "B": 200.0, "C": 300.0}
        weights = allocator.allocate(symbols, mv)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # B should have ~2x weight of A
        assert abs(weights["B"] / weights["A"] - 2.0) < 0.01
        # C should have ~3x weight of A
        assert abs(weights["C"] / weights["A"] - 3.0) < 0.01

    def test_position_limit_clamp(self):
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.20)
        # One dominant symbol
        symbols = ["TSMC", "B", "C", "D", "E"]
        mv = {"TSMC": 10000.0, "B": 100.0, "C": 100.0, "D": 100.0, "E": 100.0}
        weights = allocator.allocate(symbols, mv)
        # TSMC should be clamped at 20%
        assert weights["TSMC"] <= 0.20 + 1e-9
        # All weights <= position_limit_pct
        assert all(w <= 0.20 + 1e-9 for w in weights.values())
        # Weights still sum to ~1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_missing_mv_fallback_to_equal_weight(self):
        """D-08: Symbols without market_value get equal weight fallback."""
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.50)
        symbols = ["A", "B", "C"]
        mv = {"A": 300.0}  # B and C have no market_value
        weights = allocator.allocate(symbols, mv)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # B and C should get equal weight (1/3 each as fallback)
        assert "B" in weights
        assert "C" in weights

    def test_all_missing_mv_equal_weight(self):
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.50)
        symbols = ["A", "B", "C"]
        mv = {}  # No market values at all
        weights = allocator.allocate(symbols, mv)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # Should be ~equal weight for all
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 1e-6

    def test_empty_symbols_returns_empty(self):
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.20)
        weights = allocator.allocate([], {})
        assert weights == {}

    def test_single_symbol(self):
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.20)
        weights = allocator.allocate(["A"], {"A": 1000.0})
        # Single symbol gets clamped to position_limit_pct
        assert weights["A"] <= 0.20 + 1e-9

    def test_iterative_clamp_convergence(self):
        """Pitfall 3: Redistribution can push unclamped positions over limit."""
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.15)
        # 3 large + 7 small: multiple rounds of clamping needed
        symbols = [f"S{i}" for i in range(10)]
        mv = {f"S{i}": 1000.0 if i < 3 else 10.0 for i in range(10)}
        weights = allocator.allocate(symbols, mv)
        assert all(w <= 0.15 + 1e-9 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_negative_market_value_treated_as_missing(self):
        allocator = MarketCapWeightedAllocator(position_limit_pct=0.50)
        symbols = ["A", "B"]
        mv = {"A": -100.0, "B": 200.0}
        weights = allocator.allocate(symbols, mv)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
