"""Shioaji broker adapter -- wraps Yuanta/SinoPac Shioaji SDK.

Shioaji is lazily imported inside methods so that this module can be
imported on machines without the shioaji package (e.g. Mac dev).
"""

from __future__ import annotations

from datetime import datetime, timezone

from poseidon.broker.base import BrokerAdapter
from poseidon.orders.schemas import Fill, Order
from poseidon.orders.state_machine import OrderStatus

# Map Shioaji status strings to OrderStatus.
SHIOAJI_STATUS_MAP: dict[str, OrderStatus] = {
    "PendingSubmit": OrderStatus.PENDING,
    "PreSubmitted": OrderStatus.PENDING,
    "Submitted": OrderStatus.SUBMITTED,
    "Failed": OrderStatus.REJECTED,
    "Cancelled": OrderStatus.CANCELLED,
    "Filled": OrderStatus.FILLED,
    "Filling": OrderStatus.PARTIALLY_FILLED,
    "PartFilled": OrderStatus.PARTIALLY_FILLED,
}


class ShioajiBrokerAdapter(BrokerAdapter):
    """Wraps Shioaji SDK for live trading with Yuanta/SinoPac.

    Shioaji is imported lazily inside methods so the module loads
    cleanly on systems without the SDK installed.
    """

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key
        self._api = None

    def login(self) -> bool:
        """Authenticate with Shioaji. Returns True on success."""
        import shioaji as sj

        self._api = sj.Shioaji()
        self._api.login(api_key=self._api_key, secret_key=self._secret_key)
        return True

    def place_order(self, order: Order) -> str:
        """Submit order via Shioaji SDK.

        Quantity is divided by 1000 because Shioaji uses lots
        (1 lot = 1000 shares for TW stocks).
        """
        import shioaji as sj  # noqa: F811 -- lazy import

        if self._api is None:
            raise RuntimeError("Broker not logged in. Call login() first.")

        # Default to TSE exchange
        exchange = "TSE"
        contract = self._api.Contracts.Stocks[exchange][f"{exchange}{order.symbol}"]

        sj_order = self._api.Order(
            price=order.price or 0,
            quantity=order.quantity // 1000,
            action=sj.Action.Buy if order.action == "buy" else sj.Action.Sell,
            price_type=sj.StockPriceType.MKT if order.order_type == "market" else sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
            order_lot=sj.TFTStockOrderLot.Common,
        )
        trade = self._api.place_order(contract=contract, order=sj_order)
        return str(trade.status.id) if hasattr(trade.status, "id") else str(trade.order.id)

    def query_fills(self, broker_order_id: str) -> list[Fill]:
        """Query fills from Shioaji trade list."""
        if self._api is None:
            return []

        trades = self._api.list_trades()
        fills: list[Fill] = []
        for trade in trades:
            trade_id = str(trade.status.id) if hasattr(trade.status, "id") else str(trade.order.id)
            if trade_id == broker_order_id:
                for deal in getattr(trade, "deals", []):
                    fills.append(
                        Fill(
                            order_id=broker_order_id,
                            fill_price=float(deal.price),
                            fill_quantity=int(deal.quantity) * 1000,
                            fill_time=datetime.now(timezone.utc),
                            broker_fill_id=str(getattr(deal, "seq", "")),
                        )
                    )
                break
        return fills

    def query_positions(self) -> list[dict]:
        """Query current broker positions from Shioaji."""
        if self._api is None:
            return []
        positions = self._api.list_positions(self._api.stock_account)
        return [pos.__dict__ for pos in positions]

    def logout(self) -> None:
        """Disconnect from Shioaji."""
        if self._api is not None:
            self._api.logout()
            self._api = None
