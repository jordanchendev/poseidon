"""Tests for short position equity valuation in BacktestPortfolio.record_equity_point()."""

from datetime import datetime
from uuid import uuid4

import pandas as pd
import pytest

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction


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

        # The proceeds are already in cash, so the short is a buy-back liability:
        # position_value = -current_price * qty = -900; equity = 101000 - 900 = 100100
        # (= 100000 + unrealized (100-90)*10)
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(100_100.0)

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

        # position_value = -110*10 = -1100; equity = 101000 - 1100 = 99900
        # (= 100000 + unrealized (100-110)*10)
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(99_900.0)

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
        # Short proceeds 200*5 = 1000 in, long cost 10*100 = 1000 out
        portfolio.cash = 100_000.0

        now = datetime(2025, 1, 2)
        # Using current_price=150 for both (simplified single-price model)
        portfolio.record_equity_point(now, current_price=150.0)

        # short BTC: -150*5 = -750 (unrealized +250)
        # long ETH: 10*150 = 1500 (unrealized +500)
        # equity = 100000 - 750 + 1500 = 100750
        _, equity, _ = portfolio.equity_curve[-1]
        assert equity == pytest.approx(100_750.0)


class TestFundedShortRoundTrip:
    """A fully funded short marks to market as initial capital + unrealized PnL - fees, with no jump."""

    @pytest.mark.parametrize("exit_price", [90.0, 110.0])
    def test_equity_tracks_unrealized_and_ends_at_trade_pnl(self, exit_price):
        cost = CostModel(
            market="us_stock",
            buy_commission_rate=0.001,
            sell_commission_rate=0.001,
            tax_rate=0.0,
            slippage_pct=0.0,
            slippage_ticks=0,
        )
        p = BacktestPortfolio(100_000.0, cost, SizingConfig())

        def fill(action, price, day):
            t = datetime(2025, 1, day)
            signal = Signal(
                id=uuid4(),
                symbol="AAPL",
                market="us_stock",
                instrument=InstrumentType.SPOT,
                action=action,
                confidence=0.8,
                quantity_pct=0.5,
                signal_time=t,
            )
            return p.execute_fill(signal, pd.Series({"close": price}, name=t))

        t_open = fill(SignalAction.SHORT, 100.0, 1)
        p.record_equity_point(datetime(2025, 1, 1), 100.0)
        assert t_open.quantity == 500
        assert p.equity_curve[-1][1] == pytest.approx(100_000.0 - t_open.fees)
        assert p.equity == pytest.approx(100_000.0 - t_open.fees)

        p.record_equity_point(datetime(2025, 1, 2), exit_price)
        marked = p.equity_curve[-1][1]
        assert marked == pytest.approx(100_000.0 + (100.0 - exit_price) * 500 - t_open.fees)

        t_close = fill(SignalAction.CLOSE, exit_price, 2)
        p.record_equity_point(datetime(2025, 1, 2), exit_price)
        exit_fees = t_close.fees - t_open.fees
        assert marked - p.equity_curve[-1][1] == pytest.approx(exit_fees)
        assert p.equity_curve[-1][1] - 100_000.0 == pytest.approx(t_close.pnl)
        # The drawdown never sees a phantom peak from the short's notional.
        assert max(eq for _, eq, _ in p.equity_curve) <= 100_000.0 + max(0.0, (100.0 - exit_price) * 500)
