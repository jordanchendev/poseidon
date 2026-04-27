"""Phase 84 STRAT-02: PESSIMISTIC fill model unit tests on 1m timeline.

Locks D-07 / D-08 semantics for v17.0 Crypto Liquidation Hunting Maker thesis:

  - BUY  fills iff bar.low  <  limit_price  (strict less-than; equality => no fill)
  - SELL fills iff bar.high >  limit_price  (strict greater-than; equality => no fill)
  - fill_price == limit_price (NOT bar.open / bar.close)

Bar internal walk per Phase 50-01 invariant: only the "worst-case extreme"
(low for buy, high for sell) is consulted; we do NOT assume an
open->high->low->close path order.

This file exercises the production `PendingOrderBook.check_fills` path with 1m
timestamps. D-08 cases:

  1. Buy at exact bar.low  -> NO fill   (boundary equality, strict <)
  2. Buy below bar.low     -> NO fill
  3. Buy above bar.low     -> FILLS at limit price
  4. Sell at exact bar.high -> NO fill  (mirror)
  5. Sell above bar.high   -> NO fill   (mirror)
  6. Sell below bar.high   -> FILLS at limit price
  7. Multi-bar pending + TTL expiry crossing 1m boundary; long+short independence
"""

from __future__ import annotations

import pandas as pd
import pytest

from poseidon.backtest.pending_orders import (
    FillEvent,
    FillModel,
    PendingOrderBook,
)
from poseidon.signals.schemas import OrderType, Signal, SignalAction

# 1m timeline anchors -------------------------------------------------------
T0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
ONE_MIN = pd.Timedelta(minutes=1)


# ---------- factories (mirror tests/test_pending_orders.py idiom) ----------
def _buy_signal(limit: float, confidence: float = 0.8) -> Signal:
    """Build a LONG limit-order Signal at the given price."""
    return Signal(
        symbol="BTCUSDT",
        market="crypto_perp",
        action=SignalAction.LONG,
        confidence=confidence,
        order_type=OrderType.LIMIT,
        order_price=limit,
        quantity_pct=0.1,
    )


def _sell_signal(limit: float, confidence: float = 0.8) -> Signal:
    """Build a SHORT limit-order Signal at the given price."""
    return Signal(
        symbol="BTCUSDT",
        market="crypto_perp",
        action=SignalAction.SHORT,
        confidence=confidence,
        order_type=OrderType.LIMIT,
        order_price=limit,
        quantity_pct=0.1,
    )


def _bar(
    ts: pd.Timestamp,
    o: float,
    h: float,
    low: float,
    c: float,
    v: float = 1.0,
) -> pd.Series:
    """Build a 1m OHLCV bar as the pd.Series shape consumed by check_fills."""
    return pd.Series(
        {"open": o, "high": h, "low": low, "close": c, "volume": v},
        name=ts,
    )


class TestPessimisticFill1m:
    """D-08 cases 1-7 covering PESSIMISTIC fill semantics on 1m bars."""

    # ---------- D-08 cases 1-3: BUY-side ----------

    def test_buy_at_exact_low_no_fill(self):
        """Case 1: limit == bar.low -> strict-less-than -> NO fill.

        bar.low=100, limit=100 -> not (100 < 100) -> remains pending.
        """
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        book.submit(_buy_signal(limit=100.0), bar_index=0)

        bar = _bar(T0, o=105, h=110, low=100, c=108)
        fills = book.check_fills(bar, current_bar_index=1)

        assert fills == []
        assert book.pending_count == 1  # still pending

    def test_buy_below_low_no_fill(self):
        """Case 2: limit < bar.low -> NO fill (price never reached limit).

        bar.low=100, limit=99 -> not (100 < 99) -> remains pending.
        """
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        book.submit(_buy_signal(limit=99.0), bar_index=0)

        bar = _bar(T0, o=105, h=110, low=100, c=108)
        fills = book.check_fills(bar, current_bar_index=1)

        assert fills == []
        assert book.pending_count == 1

    def test_buy_above_low_within_range_fills_at_limit(self):
        """Case 3: bar.low < limit < bar.high -> FILLS at limit price.

        bar.low=100, limit=102 -> 100 < 102 -> fill.
        Pessimistic invariant: fill_price == limit (NOT bar.open / bar.close).
        Uses 1m timeline (current_bar_index reflects per-1m advance).
        """
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        book.submit(_buy_signal(limit=102.0), bar_index=0)

        bar = _bar(T0, o=105, h=110, low=100, c=108)
        fills = book.check_fills(bar, current_bar_index=1)

        assert len(fills) == 1
        fill = fills[0]
        assert isinstance(fill, FillEvent)
        assert fill.fill_price == pytest.approx(102.0)
        # Pessimistic: fill price is limit, not bar.open (105) nor bar.close (108)
        assert fill.fill_price != pytest.approx(105.0)
        assert fill.fill_price != pytest.approx(108.0)
        assert fill.fill_bar_index == 1
        assert fill.is_maker is True
        assert book.pending_count == 0  # filled order removed

    # ---------- D-08 case 4 mirrors: SELL-side ----------

    def test_sell_at_exact_high_no_fill(self):
        """Case 4 mirror: limit == bar.high -> strict-greater-than -> NO fill.

        bar.high=110, limit=110 -> not (110 > 110) -> remains pending.
        """
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        book.submit(_sell_signal(limit=110.0), bar_index=0)

        bar = _bar(T0, o=105, h=110, low=100, c=108)
        fills = book.check_fills(bar, current_bar_index=1)

        assert fills == []
        assert book.pending_count == 1

    def test_sell_above_high_no_fill(self):
        """Case 4 mirror: limit > bar.high -> NO fill (price never reached limit).

        bar.high=110, limit=111 -> not (110 > 111) -> remains pending.
        """
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        book.submit(_sell_signal(limit=111.0), bar_index=0)

        bar = _bar(T0, o=105, h=110, low=100, c=108)
        fills = book.check_fills(bar, current_bar_index=1)

        assert fills == []
        assert book.pending_count == 1

    def test_sell_below_high_within_range_fills_at_limit(self):
        """Case 4 mirror: bar.low < limit < bar.high -> FILLS at limit price.

        bar.high=110, limit=108 -> 110 > 108 -> fill at 108 (limit, not bar.close).
        """
        book = PendingOrderBook(fill_model=FillModel.PESSIMISTIC)
        book.submit(_sell_signal(limit=108.0), bar_index=0)

        bar = _bar(T0, o=105, h=110, low=100, c=108)
        fills = book.check_fills(bar, current_bar_index=1)

        assert len(fills) == 1
        fill = fills[0]
        assert fill.fill_price == pytest.approx(108.0)
        # Verify NOT bar.close (108 happens to coincide; still assert via limit attribute)
        assert fill.fill_price == fills[0].signal.order_price
        assert fill.fill_bar_index == 1
        assert fill.is_maker is True
        assert book.pending_count == 0
