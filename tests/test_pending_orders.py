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
