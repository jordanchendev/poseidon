"""PendingOrderBook -- limit order lifecycle management for backtesting.

Tracks unfilled limit orders across bars, evaluates fill conditions using
optimistic or pessimistic fill models, and expires timed-out orders.

Phase 50-01: Core pending order infrastructure for limit order backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from poseidon.signals.schemas import Signal, SignalAction


class FillModel(str, Enum):
    """Fill model for evaluating pending order fills.

    OPTIMISTIC: Touch fills (bar_low <= buy_price or bar_high >= sell_price).
    PESSIMISTIC: Strict penetration required (< for buy, > for sell).
    """

    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"


@dataclass
class PendingOrder:
    """A pending limit order awaiting fill.

    Attributes:
        signal: Deep copy of the original Signal (mutation-safe).
        submit_bar_index: Bar index when the order was submitted.
        order_price: Target fill price from the signal.
        side: "buy" for LONG, "sell" for SHORT/CLOSE.
    """

    signal: Signal
    submit_bar_index: int
    order_price: float
    side: str


@dataclass
class FillEvent:
    """Event emitted when a pending order is filled.

    Attributes:
        signal: The filled order's signal.
        fill_price: Price at which the order was filled (= order_price, not bar close).
        fill_bar_index: Bar index when the fill occurred.
        is_maker: Always True for limit order fills (maker fee applies).
    """

    signal: Signal
    fill_price: float
    fill_bar_index: int
    is_maker: bool = True


class PendingOrderBook:
    """Manages pending limit orders across bars in a backtest.

    Provides submit/check_fills/expire_orders lifecycle management
    with configurable fill models (optimistic vs pessimistic).
    """

    def __init__(self, fill_model: FillModel = FillModel.OPTIMISTIC) -> None:
        self._orders: list[PendingOrder] = []
        self._fill_model = fill_model

    def submit(self, signal: Signal, bar_index: int) -> None:
        """Submit a new pending limit order.

        Stores a deep copy of the signal to prevent external mutation
        from affecting the pending order state.

        Args:
            signal: Trading signal with order_type=LIMIT and order_price set.
            bar_index: Current bar index when the order is submitted.
        """
        side = "buy" if signal.action == SignalAction.LONG else "sell"
        order = PendingOrder(
            signal=signal.model_copy(deep=True),
            submit_bar_index=bar_index,
            order_price=signal.order_price,
            side=side,
        )
        self._orders.append(order)

    def check_fills(self, bar: pd.Series, current_bar_index: int) -> list[FillEvent]:
        """Check which pending orders are filled by the current bar.

        Filled orders are removed from the book and returned as FillEvents.
        Fill price is the order's limit price (not bar close).

        Args:
            bar: Current OHLCV bar with 'high' and 'low' columns.
            current_bar_index: Current bar index for the FillEvent record.

        Returns:
            List of FillEvents for orders that were filled this bar.
        """
        filled: list[FillEvent] = []
        remaining: list[PendingOrder] = []

        for order in self._orders:
            if self._is_filled(order, bar):
                filled.append(
                    FillEvent(
                        signal=order.signal,
                        fill_price=order.order_price,
                        fill_bar_index=current_bar_index,
                    )
                )
            else:
                remaining.append(order)

        self._orders = remaining
        return filled

    def _is_filled(self, order: PendingOrder, bar: pd.Series) -> bool:
        """Check if an order's fill condition is met by the bar.

        Buy orders: price must reach DOWN to order_price (check bar low).
        Sell orders: price must reach UP to order_price (check bar high).

        Optimistic: touch fills (<=, >=).
        Pessimistic: strict penetration required (<, >).
        """
        if order.side == "buy":
            bar_low = float(bar["low"])
            if self._fill_model == FillModel.OPTIMISTIC:
                return bar_low <= order.order_price
            else:
                return bar_low < order.order_price
        else:
            bar_high = float(bar["high"])
            if self._fill_model == FillModel.OPTIMISTIC:
                return bar_high >= order.order_price
            else:
                return bar_high > order.order_price

    def expire_orders(
        self, current_bar_index: int, max_bars: int
    ) -> list[PendingOrder]:
        """Remove and return orders that have exceeded the maximum pending bars.

        Args:
            current_bar_index: Current bar index.
            max_bars: Maximum number of bars an order can remain pending.

        Returns:
            List of expired PendingOrders that were removed.
        """
        expired: list[PendingOrder] = []
        remaining: list[PendingOrder] = []

        for order in self._orders:
            if current_bar_index - order.submit_bar_index >= max_bars:
                expired.append(order)
            else:
                remaining.append(order)

        self._orders = remaining
        return expired

    @property
    def pending_count(self) -> int:
        """Number of currently pending orders."""
        return len(self._orders)
