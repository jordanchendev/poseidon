"""Unit tests for futures contract support in BacktestPortfolio.

Tests cover:
- Futures PnL with point_value multiplier (long/short/loss)
- Tick-based slippage using explicit tick_size
- Mark-to-market equity with point_value
- SL/TP exit PnL with point_value
- Backward compatibility for stock and crypto markets
- CostModel registry entries for tw_futures and tw_futures_mtx
"""

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import BacktestPortfolio
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tw_futures_cost():
    return COST_MODELS["tw_futures"]


@pytest.fixture
def tw_futures_mtx_cost():
    return COST_MODELS["tw_futures_mtx"]


@pytest.fixture
def tw_stock_cost():
    return COST_MODELS["tw_stock"]


@pytest.fixture
def crypto_perp_cost():
    return COST_MODELS["crypto_perp"]


def make_futures_signal(
    action: SignalAction,
    symbol: str = "TX",
    market: str = "tw_futures",
    quantity_pct: float = 0.025,
    signal_time: datetime | None = None,
) -> Signal:
    """Factory for creating futures test Signal objects."""
    return Signal(
        id=uuid4(),
        symbol=symbol,
        market=market,
        instrument=InstrumentType.FUTURES,
        action=action,
        confidence=0.8,
        quantity_pct=quantity_pct,
        signal_time=signal_time or datetime(2024, 1, 1, tzinfo=UTC),
    )


def make_bar(close: float, time: datetime | None = None) -> pd.Series:
    """Factory for creating a mock OHLCV bar."""
    t = time or datetime(2024, 1, 1, tzinfo=UTC)
    return pd.Series(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 50000,
        },
        name=t,
    )


# ---------------------------------------------------------------------------
# Test: CostModel registry entries
# ---------------------------------------------------------------------------


class TestCostModelRegistry:
    """Tests for CostModel registry entries."""

    def test_cost_model_tw_futures_exists(self):
        """COST_MODELS['tw_futures'] has point_value=200.0."""
        cm = COST_MODELS["tw_futures"]
        assert cm.point_value == 200.0
        assert cm.tick_size == 1.0
        assert cm.market == "tw_futures"

    def test_cost_model_tw_futures_mtx_exists(self):
        """COST_MODELS['tw_futures_mtx'] has point_value=50.0."""
        cm = COST_MODELS["tw_futures_mtx"]
        assert cm.point_value == 50.0
        assert cm.tick_size == 1.0
        assert cm.market == "tw_futures_mtx"


# ---------------------------------------------------------------------------
# Test: Futures PnL
# ---------------------------------------------------------------------------


class TestFuturesPnL:
    """Tests for futures PnL calculation with point_value multiplier."""

    def test_futures_pnl_long_profit(self, tw_futures_cost):
        """TX long entry at 22000, close at 22100. PnL ~ 100 * 200 = 20000 TWD."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None
        assert t_open.quantity >= 1

        # Close long with +100 points
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(22100.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        # ~100 points * 200 TWD/point * 1 contract - fees ~ 19000+
        assert t_close.pnl > 19000, f"Expected PnL > 19000, got {t_close.pnl}"

    def test_futures_pnl_short_profit(self, tw_futures_cost):
        """TX short entry at 22000, close at 21900. PnL ~ 100 * 200 = 20000 TWD."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost)

        # Open short
        sig_open = make_futures_signal(SignalAction.SHORT)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close short with -100 points (profit for short)
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(21900.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        assert t_close.pnl > 19000, f"Expected PnL > 19000, got {t_close.pnl}"

    def test_futures_pnl_long_loss(self, tw_futures_cost):
        """TX long entry at 22000, close at 21900. PnL ~ -100 * 200 = -20000 TWD."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close long with -100 points (loss)
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(21900.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        assert t_close.pnl < -19000, f"Expected PnL < -19000, got {t_close.pnl}"


# ---------------------------------------------------------------------------
# Test: Slippage with tick_size
# ---------------------------------------------------------------------------


class TestFuturesSlippage:
    """Tests for tick-based slippage using explicit tick_size."""

    def test_futures_slippage_tick_size(self, tw_futures_cost):
        """Verify _apply_slippage with tick_size=1.0 gives exactly 1 point slippage."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost)
        price = 22000.0

        # Buy slippage: price + 1 tick (= 1 point for TX)
        buy_fill = p._apply_slippage(price, is_buy=True)
        assert buy_fill == 22001.0, f"Expected 22001.0, got {buy_fill}"

        # Sell slippage: price - 1 tick
        sell_fill = p._apply_slippage(price, is_buy=False)
        assert sell_fill == 21999.0, f"Expected 21999.0, got {sell_fill}"

    def test_stock_slippage_uses_approximation(self, tw_stock_cost):
        """Stock slippage still uses price*0.001 approximation (tick_size=0.0)."""
        p = BacktestPortfolio(1_000_000, tw_stock_cost)
        price = 100.0

        buy_fill = p._apply_slippage(price, is_buy=True)
        # tick_value = 100 * 0.001 = 0.1, slippage = 1 * 0.1 = 0.1
        assert buy_fill == pytest.approx(100.1, abs=0.001)


# ---------------------------------------------------------------------------
# Test: Equity mark-to-market
# ---------------------------------------------------------------------------


class TestFuturesEquity:
    """Tests for mark-to-market equity calculation with point_value."""

    def test_futures_equity_mark_to_market(self, tw_futures_cost):
        """Open TX long at 22000, record_equity_point at 22100.
        Unrealized gain should reflect point_value=200."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None
        qty = t_open.quantity
        entry_price = t_open.entry_price

        # Record equity at +100 points
        t1 = datetime(2024, 1, 2, tzinfo=UTC)
        p.record_equity_point(t1, 22100.0)

        assert len(p.equity_curve) == 1
        _, equity_at_22100, _ = p.equity_curve[0]

        # Unrealized gain = (22100 - entry_price) * qty * 200
        # This should be significantly more than (22100 - entry_price) * qty
        price_diff = 22100.0 - entry_price
        unrealized_with_pv = price_diff * qty * 200.0
        unrealized_without_pv = price_diff * qty

        # The equity difference should be consistent with point_value=200
        # equity = cash + entry_price*qty + unrealized
        expected_equity = p.cash + entry_price * qty + unrealized_with_pv
        assert equity_at_22100 == pytest.approx(expected_equity, rel=1e-6)

        # Verify it's NOT the non-point-value result
        wrong_equity = p.cash + entry_price * qty + unrealized_without_pv
        assert equity_at_22100 != pytest.approx(wrong_equity, rel=1e-2)


# ---------------------------------------------------------------------------
# Test: SL exit with point_value
# ---------------------------------------------------------------------------


class TestFuturesSLExit:
    """Tests for SL/TP exit PnL with point_value."""

    def test_futures_sl_exit_pnl(self, tw_futures_cost):
        """Open TX long, trigger SL via execute_sl_tp_exit. PnL uses point_value."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # SL exit at 21900 (100 points below entry + slippage)
        sl_price = 21900.0
        bar_sl = make_bar(21900.0, datetime(2024, 1, 2, tzinfo=UTC))
        key = "tw_futures:TX"
        t_sl = p.execute_sl_tp_exit(key, sl_price, "sl", bar_sl)

        assert t_sl is not None
        assert t_sl.pnl is not None
        # PnL = (21900 - entry_price) * qty * 200 - fees
        # entry_price = 22001 (with buy slippage), so diff ~ -101 points
        # PnL ~ -101 * 200 = -20200 minus fees
        assert t_sl.pnl < -19000, f"Expected SL PnL < -19000, got {t_sl.pnl}"


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Tests for backward compatibility with stock and crypto markets."""

    def test_backward_compat_stock_pnl(self, tw_stock_cost):
        """Stock PnL unchanged (point_value=1.0 default)."""
        p = BacktestPortfolio(1_000_000, tw_stock_cost)

        # Open stock long
        sig_open = Signal(
            symbol="2330",
            market="tw_stock",
            instrument=InstrumentType.SPOT,
            action=SignalAction.LONG,
            confidence=0.8,
            quantity_pct=0.1,
            signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        bar_open = make_bar(600.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None
        entry_price = t_open.entry_price
        qty = t_open.quantity

        # Close stock at 610
        sig_close = Signal(
            symbol="2330",
            market="tw_stock",
            instrument=InstrumentType.SPOT,
            action=SignalAction.CLOSE,
            confidence=0.8,
            quantity_pct=1.0,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(610.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        # Stock PnL = (exit - entry) * qty * 1.0 - fees
        # With point_value=1.0, this should be a normal stock PnL
        exit_price = t_close.exit_price
        raw_pnl = (exit_price - entry_price) * qty
        # PnL should be close to raw_pnl minus fees
        assert t_close.pnl < raw_pnl, "PnL should be less than raw due to fees"
        assert t_close.pnl > 0, "10-point stock gain should be profitable"

    def test_backward_compat_crypto_pnl(self, crypto_perp_cost):
        """Crypto perp PnL unchanged (point_value=1.0 default)."""
        p = BacktestPortfolio(100_000, crypto_perp_cost)

        # Open crypto long
        sig_open = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            instrument=InstrumentType.PERPETUAL,
            action=SignalAction.LONG,
            confidence=0.8,
            quantity_pct=0.1,
            signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        bar_open = make_bar(40000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close crypto at 41000 (+2.5%)
        sig_close = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            instrument=InstrumentType.PERPETUAL,
            action=SignalAction.CLOSE,
            confidence=0.8,
            quantity_pct=1.0,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(41000.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        assert t_close.pnl > 0, "1000-point crypto gain should be profitable"


# ---------------------------------------------------------------------------
# Test: MTX point_value
# ---------------------------------------------------------------------------


class TestMTXPointValue:
    """Tests for MTX (Mini TAIEX) futures with point_value=50."""

    def test_mtx_point_value(self, tw_futures_mtx_cost):
        """MTX entry/exit with point_value=50. PnL = diff * 50."""
        p = BacktestPortfolio(500_000, tw_futures_mtx_cost)

        # Open MTX long (quantity_pct=0.05 so 500K*0.05/22001 ~= 1 contract)
        sig_open = make_futures_signal(
            SignalAction.LONG,
            symbol="MTX",
            market="tw_futures_mtx",
            quantity_pct=0.05,
        )
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close at +100 points
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            symbol="MTX",
            market="tw_futures_mtx",
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(22100.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        # PnL = ~(100 - 2 slippage) * 50 * qty - fees
        # With point_value=50 (vs TX 200), PnL should be ~1/4 of TX equivalent
        assert t_close.pnl > 4000, f"Expected MTX PnL > 4000, got {t_close.pnl}"
        assert t_close.pnl < 6000, f"Expected MTX PnL < 6000, got {t_close.pnl}"
