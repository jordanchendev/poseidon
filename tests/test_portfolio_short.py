"""Tests for short position equity valuation in BacktestPortfolio.record_equity_point()."""

from datetime import datetime

import pytest

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig


@pytest.fixture
def zero_cost_model():
    """CostModel with zero fees/slippage for clean equity tests."""
    return CostModel(
        market="crypto_spot",
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.0,
        slippage_pct=0.0,
        slippage_ticks=0,
    )


@pytest.fixture
def portfolio(zero_cost_model):
    """BacktestPortfolio with 100k capital and zero costs."""
    return BacktestPortfolio(
        initial_capital=100_000.0,
        cost_model=zero_cost_model,
        sizing_config=SizingConfig(),
    )


class TestShortPositionEquity:
    """Tests for correct short position valuation in record_equity_point()."""

    def test_short_position_profitable(self, portfolio):
        """Short at 100, price drops to 90 -> position_value = (100-90)*qty = +10*qty."""
        # Manually add a short position
        portfolio.positions["crypto_spot:BTCUSDT"] = {
            "side": "short",
            "quantity": 10,
            "entry_price": 100.0,
            "entry_time": datetime(2025, 1, 1),
        }
        # Cash after short: received 100*10 = 1000 proceeds
        portfolio.cash = 100_000.0 + 1000.0  # 101000

        now = datetime(2025, 1, 2)
        portfolio.record_equity_point(now, current_price=90.0)

        # position_value = (entry_price - current_price) * qty = (100-90)*10 = 100
        # equity = cash + position_value = 101000 + 100 = 101100
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(101_100.0)

    def test_short_position_losing(self, portfolio):
        """Short at 100, price rises to 110 -> position_value = (100-110)*qty = -10*qty."""
        portfolio.positions["crypto_spot:BTCUSDT"] = {
            "side": "short",
            "quantity": 10,
            "entry_price": 100.0,
            "entry_time": datetime(2025, 1, 1),
        }
        portfolio.cash = 100_000.0 + 1000.0  # 101000

        now = datetime(2025, 1, 2)
        portfolio.record_equity_point(now, current_price=110.0)

        # position_value = (100-110)*10 = -100
        # equity = 101000 + (-100) = 100900
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(100_900.0)

    def test_long_position_unchanged(self, portfolio):
        """Long position equity calculation unchanged: quantity * current_price."""
        portfolio.positions["crypto_spot:ETHUSDT"] = {
            "side": "long",
            "quantity": 10,
            "entry_price": 100.0,
            "entry_time": datetime(2025, 1, 1),
        }
        portfolio.cash = 99_000.0  # paid 1000 for the position

        now = datetime(2025, 1, 2)
        portfolio.record_equity_point(now, current_price=120.0)

        # position_value = 10 * 120 = 1200
        # equity = 99000 + 1200 = 100200
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(100_200.0)

    def test_mixed_long_short_positions(self, portfolio):
        """Mixed long+short positions: each valued correctly."""
        portfolio.positions["crypto_spot:BTCUSDT"] = {
            "side": "short",
            "quantity": 5,
            "entry_price": 200.0,
            "entry_time": datetime(2025, 1, 1),
        }
        portfolio.positions["crypto_spot:ETHUSDT"] = {
            "side": "long",
            "quantity": 10,
            "entry_price": 100.0,
            "entry_time": datetime(2025, 1, 1),
        }
        portfolio.cash = 100_000.0

        now = datetime(2025, 1, 2)
        # Using current_price=150 for both (simplified single-price model)
        portfolio.record_equity_point(now, current_price=150.0)

        # short BTC: (200-150)*5 = 250
        # long ETH: 10*150 = 1500
        # equity = 100000 + 250 + 1500 = 101750
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(101_750.0)
