"""Tests for PendingOrderBook lifecycle (submit/fill/expire) and fill models.

Phase 50-01 Task 1: Unit tests for pending order management.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from poseidon.backtest.pending_orders import (
    FillEvent,
    FillModel,
    PendingOrderBook,
    PendingOrder,
)
from poseidon.signals.schemas import OrderType, Signal, SignalAction


def _make_signal(
    action: SignalAction = SignalAction.LONG,
    order_price: float = 50000.0,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    confidence: float = 0.8,
    quantity_pct: float = 0.1,
) -> Signal:
    """Create a minimal limit-order Signal for testing."""
    return Signal(
        symbol="BTCUSDT",
        market="crypto_perp",
        action=action,
        confidence=confidence,
        order_type=OrderType.LIMIT,
        order_price=order_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        quantity_pct=quantity_pct,
    )


def _make_bar(
    high: float = 51000.0,
    low: float = 49000.0,
    close: float = 50500.0,
    open_: float = 50000.0,
    volume: float = 1000.0,
) -> pd.Series:
    """Create a synthetic OHLCV bar for testing."""
    return pd.Series(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        name=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


class TestPendingOrderBookSubmit:
    """Tests for PendingOrderBook.submit()."""

    def test_submit_adds_order(self):
        """submit() adds a PendingOrder to the book."""
        book = PendingOrderBook()
        signal = _make_signal()
        book.submit(signal, bar_index=0)
        assert book.pending_count == 1

    def test_submit_deep_copies_signal(self):
        """submit() stores a deep copy of the signal (mutation safety)."""
        book = PendingOrderBook()
        signal = _make_signal()
        book.submit(signal, bar_index=0)
        # Mutate original signal
        signal.confidence = 0.1
        # Book's copy should be unaffected
        assert book._orders[0].signal.confidence == 0.8

    def test_pending_count_property(self):
        """pending_count returns the number of pending orders."""
        book = PendingOrderBook()
        assert book.pending_count == 0
        book.submit(_make_signal(), bar_index=0)
        assert book.pending_count == 1
        book.submit(_make_signal(action=SignalAction.SHORT), bar_index=1)
        assert book.pending_count == 2


class TestOptimisticFills:
    """Tests for OPTIMISTIC fill model."""

    def test_optimistic_buy_fill(self):
        """OPTIMISTIC buy-limit fills when bar_low <= order_price."""
        book = PendingOrderBook(fill_model=FillModel.OPTIMISTIC)
        signal = _make_signal(order_price=50000.0)
        book.submit(signal, bar_index=0)
        # bar_low=49000 <= 50000 -> should fill
        bar = _make_bar(low=49000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 1
        assert fills[0].fill_price == 50000.0
        assert book.pending_count == 0

    def test_optimistic_buy_no_fill(self):
        """OPTIMISTIC buy-limit does NOT fill when bar_low > order_price."""
        book = PendingOrderBook(fill_model=FillModel.OPTIMISTIC)
        signal = _make_signal(order_price=48000.0)
        book.submit(signal, bar_index=0)
        # bar_low=49000 > 48000 -> no fill... wait, 49000 > 48000 means price didn't
        # reach down to 48000. But we want NO fill when low > order_price.
        # Actually: low=49000 > order_price=48000 means the low touched below 48000? No.
        # low=49000 means the minimum price was 49000.
        # order_price=48000 means we want to buy at 48000.
        # 49000 > 48000 -> price never went as low as 48000 -> no fill. Correct.
        bar = _make_bar(low=49000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 0
        assert book.pending_count == 1

    def test_optimistic_buy_touch_fills(self):
        """OPTIMISTIC buy-limit fills when bar_low == order_price (touch = fill)."""
        book = PendingOrderBook(fill_model=FillModel.OPTIMISTIC)
        signal = _make_signal(order_price=49000.0)
        book.submit(signal, bar_index=0)
        # bar_low=49000 == 49000 -> optimistic fills on touch
        bar = _make_bar(low=49000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 1

    def test_optimistic_sell_fill(self):
        """OPTIMISTIC sell-limit fills when bar_high >= order_price."""
        book = PendingOrderBook(fill_model=FillModel.OPTIMISTIC)
        signal = _make_signal(action=SignalAction.SHORT, order_price=51000.0)
        book.submit(signal, bar_index=0)
        # bar_high=52000 >= 51000 -> should fill
        bar = _make_bar(high=52000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 1
        assert fills[0].fill_price == 51000.0


class TestPessimisticFills:
    """Tests for PESSIMISTIC fill model."""

    def test_pessimistic_buy_fill(self):
        """PESSIMISTIC buy-limit fills when bar_low < order_price (strict penetration)."""
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        signal = _make_signal(order_price=50000.0)
        book.submit(signal, bar_index=0)
        # bar_low=49000 < 50000 -> strict penetration, should fill
        bar = _make_bar(low=49000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 1

    def test_pessimistic_buy_touch_no_fill(self):
        """PESSIMISTIC buy-limit does NOT fill when bar_low == order_price (touch only)."""
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        signal = _make_signal(order_price=49000.0)
        book.submit(signal, bar_index=0)
        # bar_low=49000 == 49000 -> touch only, pessimistic requires strict penetration
        bar = _make_bar(low=49000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 0
        assert book.pending_count == 1

    def test_pessimistic_sell_fill(self):
        """PESSIMISTIC sell-limit fills when bar_high > order_price (strict)."""
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        signal = _make_signal(action=SignalAction.SHORT, order_price=51000.0)
        book.submit(signal, bar_index=0)
        # bar_high=52000 > 51000 -> strict penetration, should fill
        bar = _make_bar(high=52000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 1

    def test_pessimistic_sell_touch_no_fill(self):
        """PESSIMISTIC sell-limit does NOT fill when bar_high == order_price."""
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        signal = _make_signal(action=SignalAction.SHORT, order_price=52000.0)
        book.submit(signal, bar_index=0)
        # bar_high=52000 == 52000 -> touch only
        bar = _make_bar(high=52000.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 0
        assert book.pending_count == 1


class TestExpireOrders:
    """Tests for PendingOrderBook.expire_orders()."""

    def test_expire_orders_removes_timed_out(self):
        """expire_orders() removes orders past max_bars."""
        book = PendingOrderBook()
        book.submit(_make_signal(), bar_index=0)
        expired = book.expire_orders(current_bar_index=5, max_bars=5)
        assert len(expired) == 1
        assert book.pending_count == 0

    def test_expire_orders_keeps_valid(self):
        """expire_orders() keeps orders within timeout window."""
        book = PendingOrderBook()
        book.submit(_make_signal(), bar_index=0)
        expired = book.expire_orders(current_bar_index=4, max_bars=5)
        assert len(expired) == 0
        assert book.pending_count == 1

    def test_expire_orders_boundary(self):
        """expire_orders() removes at exactly max_bars boundary."""
        book = PendingOrderBook()
        book.submit(_make_signal(), bar_index=3)
        # current=8, submit=3, diff=5, max_bars=5 -> 5 >= 5 -> expired
        expired = book.expire_orders(current_bar_index=8, max_bars=5)
        assert len(expired) == 1


class TestFillEvent:
    """Tests for FillEvent dataclass."""

    def test_fill_event_is_maker_default(self):
        """FillEvent.is_maker defaults to True."""
        signal = _make_signal()
        fe = FillEvent(signal=signal, fill_price=50000.0, fill_bar_index=1)
        assert fe.is_maker is True

    def test_fill_price_equals_order_price(self):
        """FillEvent.fill_price equals order.order_price (not bar close)."""
        book = PendingOrderBook()
        signal = _make_signal(order_price=49500.0)
        book.submit(signal, bar_index=0)
        bar = _make_bar(low=49000.0, close=50500.0)
        fills = book.check_fills(bar, current_bar_index=1)
        assert len(fills) == 1
        assert fills[0].fill_price == 49500.0
        assert fills[0].fill_price != 50500.0  # not bar close


# ---------------------------------------------------------------------------
# Portfolio extension tests (Task 2)
# ---------------------------------------------------------------------------

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.portfolio import BacktestPortfolio, TradeRecord


def _make_portfolio(initial_capital: float = 100_000.0) -> BacktestPortfolio:
    """Create a BacktestPortfolio with crypto_perp cost model for testing."""
    cost_model = get_cost_model("crypto_perp")
    return BacktestPortfolio(initial_capital=initial_capital, cost_model=cost_model)


class TestExecuteLimitFill:
    """Tests for BacktestPortfolio.execute_limit_fill()."""

    def test_execute_limit_fill_long(self):
        """execute_limit_fill opens a long position with correct fill price."""
        portfolio = _make_portfolio()
        signal = _make_signal(
            order_price=49500.0,
            stop_loss_price=48000.0,
            take_profit_price=52000.0,
        )
        fe = FillEvent(signal=signal, fill_price=49500.0, fill_bar_index=1)
        bar = _make_bar(close=50000.0)

        trade = portfolio.execute_limit_fill(fe, bar)

        assert trade is not None
        assert trade.entry_price == 49500.0  # fill price, not bar close
        assert trade.action == "long"
        key = "crypto_perp:BTCUSDT"
        assert key in portfolio.positions
        assert portfolio.positions[key]["entry_price"] == 49500.0

    def test_execute_limit_fill_short(self):
        """execute_limit_fill opens a short position with correct fill price."""
        portfolio = _make_portfolio()
        signal = _make_signal(
            action=SignalAction.SHORT,
            order_price=51000.0,
        )
        fe = FillEvent(signal=signal, fill_price=51000.0, fill_bar_index=1)
        bar = _make_bar(close=50000.0)

        trade = portfolio.execute_limit_fill(fe, bar)

        assert trade is not None
        assert trade.entry_price == 51000.0
        assert trade.action == "short"

    def test_execute_limit_fill_no_slippage(self):
        """execute_limit_fill uses fill_price directly without slippage."""
        portfolio = _make_portfolio()
        signal = _make_signal(order_price=49500.0)
        fe = FillEvent(signal=signal, fill_price=49500.0, fill_bar_index=1)
        bar = _make_bar(close=50000.0)

        trade = portfolio.execute_limit_fill(fe, bar)

        # The entry price should be exactly the order price (no slippage applied)
        assert trade.entry_price == 49500.0
        # Verify slippage was NOT applied: crypto_perp has slippage_pct=0.0005
        # If slippage was applied, price would be 49500 * 1.0005 = 49524.75
        assert trade.entry_price != 49500.0 * 1.0005

    def test_maker_fee_rate(self):
        """Limit fill uses maker fee rate (buy_commission_rate = 0.0002), not taker."""
        portfolio = _make_portfolio()
        signal = _make_signal(order_price=50000.0, quantity_pct=0.1)
        fe = FillEvent(signal=signal, fill_price=50000.0, fill_bar_index=1)
        bar = _make_bar()

        trade = portfolio.execute_limit_fill(fe, bar)

        assert trade is not None
        # Expected fee: trade_value * 0.0002 (maker rate)
        # trade_value = 50000 * (100000 * 0.1 / 50000) = 50000 * 0.2 = 10000
        # fee = 10000 * 0.0002 = 2.0
        expected_fee = trade.quantity * 50000.0 * 0.0002
        assert trade.fees == pytest.approx(expected_fee, abs=1e-6)
        # Verify it's NOT the taker rate (0.0005)
        taker_fee = trade.quantity * 50000.0 * 0.0005
        assert trade.fees != pytest.approx(taker_fee, abs=1e-6)


class TestExecuteSlTpExit:
    """Tests for BacktestPortfolio.execute_sl_tp_exit()."""

    def test_execute_sl_tp_exit_long_sl(self):
        """execute_sl_tp_exit closes long position at SL trigger price."""
        portfolio = _make_portfolio()
        # Open a long position first
        signal = _make_signal(
            order_price=50000.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
        )
        fe = FillEvent(signal=signal, fill_price=50000.0, fill_bar_index=0)
        bar = _make_bar()
        portfolio.execute_limit_fill(fe, bar)

        key = "crypto_perp:BTCUSDT"
        assert key in portfolio.positions

        # SL exit at 48000
        exit_bar = _make_bar(low=47500.0, close=47800.0)
        trade = portfolio.execute_sl_tp_exit(key, exit_price=48000.0, exit_type="sl", bar=exit_bar)

        assert trade is not None
        assert trade.exit_price == 48000.0  # trigger price, not bar close
        assert trade.exit_price != 47800.0  # not bar close
        assert trade.action == "close_sl"
        assert trade.pnl is not None
        assert trade.pnl < 0  # SL exit = loss
        assert key not in portfolio.positions

    def test_execute_sl_tp_exit_long_tp(self):
        """execute_sl_tp_exit closes long position at TP trigger price."""
        portfolio = _make_portfolio()
        signal = _make_signal(
            order_price=50000.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
        )
        fe = FillEvent(signal=signal, fill_price=50000.0, fill_bar_index=0)
        bar = _make_bar()
        portfolio.execute_limit_fill(fe, bar)

        key = "crypto_perp:BTCUSDT"

        # TP exit at 55000
        exit_bar = _make_bar(high=55500.0, close=55200.0)
        trade = portfolio.execute_sl_tp_exit(key, exit_price=55000.0, exit_type="tp", bar=exit_bar)

        assert trade is not None
        assert trade.exit_price == 55000.0  # trigger price
        assert trade.action == "close_tp"
        assert trade.pnl is not None
        assert trade.pnl > 0  # TP exit = profit

    def test_sl_priority_over_tp(self):
        """When both SL and TP trigger, SL takes priority (D-12)."""
        # This tests the execute_sl_tp_exit behavior directly
        # The SL priority logic lives in _evaluate_sl_tp (runner.py, Task 3)
        # Here we just verify that calling execute_sl_tp_exit with "sl"
        # closes the position and records the correct exit type
        portfolio = _make_portfolio()
        signal = _make_signal(
            order_price=50000.0,
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
        )
        fe = FillEvent(signal=signal, fill_price=50000.0, fill_bar_index=0)
        bar = _make_bar()
        portfolio.execute_limit_fill(fe, bar)

        key = "crypto_perp:BTCUSDT"

        # Call with SL first (runner would choose SL when both trigger)
        exit_bar = _make_bar(low=47000.0, high=56000.0)
        trade = portfolio.execute_sl_tp_exit(key, exit_price=48000.0, exit_type="sl", bar=exit_bar)

        assert trade is not None
        assert trade.action == "close_sl"
        assert trade.exit_price == 48000.0
        # Position should be gone -- TP cannot trigger
        assert key not in portfolio.positions


class TestPositionSlTpStorage:
    """Tests for SL/TP field storage on position dict."""

    def test_position_stores_sl_tp(self):
        """Opening a position via execute_fill stores SL/TP from signal."""
        portfolio = _make_portfolio()
        signal = _make_signal(
            stop_loss_price=48000.0,
            take_profit_price=55000.0,
        )
        # Use execute_fill (market order path) with a bar
        signal.order_type = None  # market order
        bar = _make_bar(close=50000.0)
        trade = portfolio.execute_fill(signal, bar)

        assert trade is not None
        key = "crypto_perp:BTCUSDT"
        pos = portfolio.positions[key]
        assert pos["stop_loss_price"] == 48000.0
        assert pos["take_profit_price"] == 55000.0

    def test_position_stores_none_sl_tp(self):
        """Position stores None when SL/TP not set on signal."""
        portfolio = _make_portfolio()
        signal = _make_signal()  # no SL/TP
        signal.order_type = None  # market order
        bar = _make_bar(close=50000.0)
        trade = portfolio.execute_fill(signal, bar)

        assert trade is not None
        key = "crypto_perp:BTCUSDT"
        pos = portfolio.positions[key]
        assert pos["stop_loss_price"] is None
        assert pos["take_profit_price"] is None
