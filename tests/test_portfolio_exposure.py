"""Tests for VirtualPortfolio exposure tracking methods."""

from datetime import datetime, timezone

import numpy as np
import pytest


def _make_portfolio():
    """Create a VirtualPortfolio with test positions, bypassing heavy imports."""
    from poseidon.risk.portfolio import PositionEntry, VirtualPortfolio

    p = VirtualPortfolio()
    now = datetime.now(tz=timezone.utc)
    p.positions["tw_stock:2330"] = PositionEntry(
        symbol="2330",
        market="tw_stock",
        instrument="spot",
        side="long",
        quantity_pct=0.08,
        entry_time=now,
    )
    p.positions["us_stock:AAPL"] = PositionEntry(
        symbol="AAPL",
        market="us_stock",
        instrument="spot",
        side="long",
        quantity_pct=0.08,
        entry_time=now,
    )
    p.positions["crypto_spot:BTC"] = PositionEntry(
        symbol="BTC",
        market="crypto_spot",
        instrument="spot",
        side="long",
        quantity_pct=0.04,
        entry_time=now,
    )
    return p


class TestExposureByMarket:
    """Test exposure_by_market method."""

    def test_returns_dict(self):
        p = _make_portfolio()
        result = p.exposure_by_market()
        assert isinstance(result, dict)

    def test_correct_market_exposure(self):
        p = _make_portfolio()
        result = p.exposure_by_market()
        assert result == {"tw_stock": 0.08, "us_stock": 0.08, "crypto_spot": 0.04}

    def test_empty_portfolio(self):
        from poseidon.risk.portfolio import VirtualPortfolio

        p = VirtualPortfolio()
        assert p.exposure_by_market() == {}

    def test_same_market_sums(self):
        """Two positions in same market should sum."""
        from poseidon.risk.portfolio import PositionEntry, VirtualPortfolio

        p = VirtualPortfolio()
        now = datetime.now(tz=timezone.utc)
        p.positions["tw_stock:2330"] = PositionEntry(
            symbol="2330", market="tw_stock", instrument="spot",
            side="long", quantity_pct=0.08, entry_time=now,
        )
        p.positions["tw_stock:2317"] = PositionEntry(
            symbol="2317", market="tw_stock", instrument="spot",
            side="long", quantity_pct=0.06, entry_time=now,
        )
        result = p.exposure_by_market()
        assert result == {"tw_stock": pytest.approx(0.14)}


class TestExposureByAsset:
    """Test exposure_by_asset method."""

    def test_returns_dict(self):
        p = _make_portfolio()
        result = p.exposure_by_asset()
        assert isinstance(result, dict)

    def test_correct_asset_exposure(self):
        p = _make_portfolio()
        result = p.exposure_by_asset()
        assert result == {
            "tw_stock:2330": 0.08,
            "us_stock:AAPL": 0.08,
            "crypto_spot:BTC": 0.04,
        }

    def test_empty_portfolio(self):
        from poseidon.risk.portfolio import VirtualPortfolio

        p = VirtualPortfolio()
        assert p.exposure_by_asset() == {}


class TestPositionSymbols:
    """Test position_symbols method."""

    def test_returns_sorted_keys(self):
        p = _make_portfolio()
        result = p.position_symbols()
        assert result == ["crypto_spot:BTC", "tw_stock:2330", "us_stock:AAPL"]

    def test_empty_portfolio(self):
        from poseidon.risk.portfolio import VirtualPortfolio

        p = VirtualPortfolio()
        assert p.position_symbols() == []


class TestWeights:
    """Test weights method."""

    def test_returns_numpy_array(self):
        p = _make_portfolio()
        result = p.weights()
        assert isinstance(result, np.ndarray)

    def test_weights_sum_to_one(self):
        p = _make_portfolio()
        w = p.weights()
        assert w.sum() == pytest.approx(1.0)

    def test_weights_order_matches_symbols(self):
        """Weights should be in position_symbols() order."""
        p = _make_portfolio()
        symbols = p.position_symbols()
        w = p.weights()
        # crypto_spot:BTC=0.04, tw_stock:2330=0.08, us_stock:AAPL=0.08
        # total=0.20, normalized: 0.2, 0.4, 0.4
        assert len(w) == len(symbols)
        np.testing.assert_allclose(w, [0.2, 0.4, 0.4])

    def test_empty_portfolio_returns_empty_array(self):
        from poseidon.risk.portfolio import VirtualPortfolio

        p = VirtualPortfolio()
        w = p.weights()
        assert isinstance(w, np.ndarray)
        assert len(w) == 0


class TestTotalExposureWithNewMethods:
    """Verify total_exposure still works correctly with the 3-position portfolio."""

    def test_total_exposure(self):
        p = _make_portfolio()
        assert p.total_exposure() == pytest.approx(0.20)
