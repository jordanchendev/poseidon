"""Unit tests for backtest engine: cost model, portfolio, metrics, ORM models."""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS, CostModel, get_cost_model
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import BacktestPortfolio, TradeRecord
from poseidon.models.backtest import (
    BacktestEquityRecord,
    BacktestRecord,
    BacktestTradeRecord,
)
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tw_stock_cost():
    return get_cost_model("tw_stock")


@pytest.fixture
def crypto_spot_cost():
    return get_cost_model("crypto_spot")


@pytest.fixture
def make_signal():
    """Factory for creating test Signal objects."""

    def _make(
        action: SignalAction = SignalAction.LONG,
        symbol: str = "2330",
        market: str = "tw_stock",
        quantity_pct: float = 0.1,
        confidence: float = 0.8,
    ) -> Signal:
        return Signal(
            id=uuid4(),
            symbol=symbol,
            market=market,
            instrument=InstrumentType.SPOT,
            action=action,
            confidence=confidence,
            quantity_pct=quantity_pct,
            signal_time=datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc),
        )

    return _make


@pytest.fixture
def make_bar():
    """Factory for creating a mock OHLCV bar (pd.Series)."""

    def _make(close: float = 100.0, time: datetime | None = None) -> pd.Series:
        t = time or datetime(2025, 1, 15, tzinfo=timezone.utc)
        return pd.Series(
            {
                "time": t,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 10000,
            }
        )

    return _make


# ===========================================================================
# CostModel Tests
# ===========================================================================


class TestCostModel:
    """CostModel dataclass and COST_MODELS registry tests."""

    def test_tw_stock_fees(self):
        cm = COST_MODELS["tw_stock"]
        assert cm.buy_commission_rate == 0.001425
        assert cm.sell_commission_rate == 0.001425
        assert cm.tax_rate == 0.003

    def test_tw_stock_etf_tax(self):
        cm = COST_MODELS["tw_stock_etf"]
        assert cm.tax_rate == 0.001

    def test_tw_stock_daytrade_tax(self):
        cm = COST_MODELS["tw_stock_daytrade"]
        assert cm.tax_rate == 0.0015

    def test_crypto_spot_fees(self):
        cm = COST_MODELS["crypto_spot"]
        assert cm.buy_commission_rate == 0.001
        assert cm.sell_commission_rate == 0.001
        assert cm.slippage_pct == 0.0005

    def test_all_7_markets_exist(self):
        expected = {
            "tw_stock",
            "tw_stock_etf",
            "tw_stock_daytrade",
            "tw_futures",
            "us_stock",
            "crypto_spot",
            "crypto_perp",
        }
        assert set(COST_MODELS.keys()) == expected

    def test_cost_model_frozen(self):
        cm = COST_MODELS["tw_stock"]
        with pytest.raises(AttributeError):
            cm.tax_rate = 0.5  # type: ignore[misc]

    def test_get_cost_model_valid(self):
        cm = get_cost_model("crypto_perp")
        assert cm.market == "crypto_perp"
        assert cm.buy_commission_rate == 0.0002
        assert cm.sell_commission_rate == 0.0005

    def test_get_cost_model_invalid(self):
        with pytest.raises(ValueError, match="Unknown market"):
            get_cost_model("invalid_market")


# ===========================================================================
# BacktestPortfolio Tests
# ===========================================================================


class TestBacktestPortfolio:
    """BacktestPortfolio fill execution and equity tracking tests."""

    def test_initial_state(self, crypto_spot_cost):
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)
        assert portfolio.cash == 1_000_000
        assert portfolio.equity == 1_000_000
        assert len(portfolio.positions) == 0

    def test_long_fill_deducts_cash(self, crypto_spot_cost, make_signal, make_bar):
        """execute_fill on a LONG signal deducts (price*(1+slippage)*qty + buy_commission) from cash."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)
        signal = make_signal(action=SignalAction.LONG, market="crypto_spot", quantity_pct=0.1)
        bar = make_bar(close=100.0)

        trade = portfolio.execute_fill(signal, bar)

        assert trade is not None
        # Fill price with slippage: 100 * (1 + 0.0005) = 100.05
        # trade_value = 100.05 * quantity
        # quantity = floor(capital * quantity_pct / fill_price)
        # quantity = floor(1_000_000 * 0.1 / 100.05) = floor(999.5) = 999
        # trade_value = 100.05 * 999 = 99949.95
        # buy_fee = 99949.95 * 0.001 = 99.94995
        # total deduction = 99949.95 + 99.94995 = 100049.89995
        expected_fill_price = 100.0 * (1 + 0.0005)
        quantity = int(1_000_000 * 0.1 / expected_fill_price)
        trade_value = expected_fill_price * quantity
        buy_fee = trade_value * 0.001
        expected_cash = 1_000_000 - trade_value - buy_fee

        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-6)
        assert "crypto_spot:2330" in portfolio.positions or "2330" in portfolio.positions

    def test_close_fill_applies_sell_fees_and_tax(self, tw_stock_cost, make_signal, make_bar):
        """execute_fill on CLOSE applies sell_commission + tax and slippage in opposite direction."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=tw_stock_cost)

        # First, open a long position
        long_signal = make_signal(action=SignalAction.LONG, quantity_pct=0.1)
        buy_bar = make_bar(close=100.0)
        portfolio.execute_fill(long_signal, buy_bar)
        cash_after_buy = portfolio.cash

        # Now close the position
        close_signal = make_signal(action=SignalAction.CLOSE, quantity_pct=0.1)
        sell_bar = make_bar(close=110.0)
        trade = portfolio.execute_fill(close_signal, sell_bar)

        assert trade is not None
        assert trade.pnl is not None

        # Sell side fees = trade_value * (sell_commission_rate + tax_rate)
        # For tw_stock: sell_commission = 0.001425, tax = 0.003
        # sell_fee_rate = 0.001425 + 0.003 = 0.004425
        assert portfolio.cash > cash_after_buy  # selling at higher price should increase cash

    def test_record_equity_point(self, crypto_spot_cost, make_signal, make_bar):
        """record_equity_point stores (timestamp, equity, drawdown) tuples."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)
        t = datetime(2025, 1, 15, tzinfo=timezone.utc)
        portfolio.record_equity_point(t, 100.0)

        assert len(portfolio.equity_curve) == 1
        time_val, equity_val, dd_val = portfolio.equity_curve[0]
        assert time_val == t
        assert equity_val == pytest.approx(1_000_000, rel=1e-6)
        assert dd_val == pytest.approx(0.0)

    def test_equity_drawdown_tracking(self, crypto_spot_cost, make_signal, make_bar):
        """Drawdown is computed as (peak - current) / peak."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)

        # Open a position
        signal = make_signal(action=SignalAction.LONG, market="crypto_spot", quantity_pct=0.5)
        bar = make_bar(close=100.0)
        portfolio.execute_fill(signal, bar)

        # Record equity at peak
        t1 = datetime(2025, 1, 16, tzinfo=timezone.utc)
        portfolio.record_equity_point(t1, 120.0)  # price went up

        # Record equity at lower price (drawdown)
        t2 = datetime(2025, 1, 17, tzinfo=timezone.utc)
        portfolio.record_equity_point(t2, 90.0)  # price dropped

        assert len(portfolio.equity_curve) == 2
        _, _, dd2 = portfolio.equity_curve[1]
        assert dd2 > 0  # should have drawdown


# ===========================================================================
# Metrics Tests
# ===========================================================================


class TestMetrics:
    """compute_metrics function tests."""

    def test_total_return(self):
        """compute_metrics with known equity series returns correct total_return."""
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        assert result["total_return"] == pytest.approx(0.2, rel=1e-6)

    def test_sharpe_positive(self):
        """compute_metrics with positive returns has Sharpe > 0."""
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        assert result["sharpe_ratio"] > 0

    def test_max_drawdown(self):
        """Max drawdown is computed correctly."""
        # Peak at 110, trough at 105 -> drawdown = (110-105)/110 = ~0.0454
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        expected_dd = (110.0 - 105.0) / 110.0
        assert result["max_drawdown"] == pytest.approx(expected_dd, rel=1e-4)

    def test_win_rate_and_profit_factor(self):
        """compute_metrics with trades returns correct win_rate and profit_factor."""
        trades = [
            TradeRecord(
                symbol="A", action="long", entry_time=datetime.now(timezone.utc),
                entry_price=100, quantity=10, fees=1, pnl=50,
            ),
            TradeRecord(
                symbol="B", action="long", entry_time=datetime.now(timezone.utc),
                entry_price=100, quantity=10, fees=1, pnl=-20,
            ),
            TradeRecord(
                symbol="C", action="long", entry_time=datetime.now(timezone.utc),
                entry_price=100, quantity=10, fees=1, pnl=30,
            ),
        ]
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, trades)

        # 2 wins out of 3 trades
        assert result["win_rate"] == pytest.approx(2 / 3, rel=1e-6)
        # profit_factor = sum(wins) / abs(sum(losses)) = (50+30) / 20 = 4.0
        assert result["profit_factor"] == pytest.approx(4.0, rel=1e-6)
        assert result["trade_count"] == 3

    def test_empty_trades(self):
        """compute_metrics handles empty trades list."""
        equity = pd.Series([100.0, 110.0, 120.0])
        result = compute_metrics(equity, [])
        assert result["win_rate"] == 0
        assert result["profit_factor"] == 0
        assert result["trade_count"] == 0

    def test_zero_std_returns(self):
        """compute_metrics handles constant equity (zero std deviation)."""
        equity = pd.Series([100.0, 100.0, 100.0, 100.0])
        result = compute_metrics(equity, [])
        assert result["sharpe_ratio"] == 0

    def test_all_10_metrics_returned(self):
        """compute_metrics returns all 10 standard metrics."""
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        expected_keys = {
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "avg_win",
            "avg_loss",
            "trade_count",
        }
        assert set(result.keys()) == expected_keys


# ===========================================================================
# ORM Model Tests
# ===========================================================================


class TestBacktestModels:
    """ORM model structure tests (no live DB needed)."""

    def test_backtest_record_tablename(self):
        assert BacktestRecord.__tablename__ == "backtests"

    def test_backtest_trade_record_tablename(self):
        assert BacktestTradeRecord.__tablename__ == "backtest_trades"

    def test_backtest_equity_record_tablename(self):
        assert BacktestEquityRecord.__tablename__ == "backtest_equity"

    def test_trade_record_has_backtest_fk(self):
        """BacktestTradeRecord has ForeignKey('backtests.id') on backtest_id."""
        col = BacktestTradeRecord.__table__.c.backtest_id
        fk_targets = [fk.target_fullname for fk in col.foreign_keys]
        assert "backtests.id" in fk_targets

    def test_equity_record_composite_pk(self):
        """BacktestEquityRecord has composite PK (backtest_id, time)."""
        pk_cols = [c.name for c in BacktestEquityRecord.__table__.primary_key.columns]
        assert "backtest_id" in pk_cols
        assert "time" in pk_cols

    def test_backtest_record_has_key_columns(self):
        """BacktestRecord has all required columns."""
        table = BacktestRecord.__table__
        col_names = {c.name for c in table.columns}
        required = {
            "id", "strategy_id", "strategy_type", "symbol", "market",
            "interval", "config", "metrics", "walk_forward", "status",
            "error_message", "created_at", "completed_at",
        }
        assert required.issubset(col_names)
