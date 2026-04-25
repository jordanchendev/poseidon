"""Shioaji broker adapter -- wraps Yuanta/SinoPac Shioaji SDK.

Shioaji is lazily imported inside methods so that this module can be
imported on machines without the shioaji package (e.g. Mac dev).
"""

from __future__ import annotations

from datetime import UTC, datetime

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

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        simulation: bool = True,
        ca_cert_path: str = "",
        ca_password: str = "",
        person_id: str = "",
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._simulation = simulation
        self._ca_cert_path = ca_cert_path
        self._ca_password = ca_password
        self._person_id = person_id
        self._api = None

    def login(self) -> bool:
        """Authenticate with Shioaji. Returns True on success."""
        import shioaji as sj

        self._api = sj.Shioaji(simulation=self._simulation)
        self._api.login(api_key=self._api_key, secret_key=self._secret_key)
        if self._ca_cert_path and self._ca_password and self._person_id:
            self._api.activate_ca(
                ca_path=self._ca_cert_path,
                ca_passwd=self._ca_password,
                person_id=self._person_id,
            )
        return True

    def place_order(self, order: Order) -> str:
        """Submit order via Shioaji SDK.

        Quantity is divided by 1000 because Shioaji uses lots
        (1 lot = 1000 shares for TW stocks).
        """
        import shioaji as sj

        if self._api is None:
            raise RuntimeError("Broker not logged in. Call login() first.")

        contract = self._resolve_stock_contract(order.symbol)

        sj_order = self._api.Order(
            price=order.price or 0,
            quantity=order.quantity // 1000,
            action=sj.constant.Action.Buy if order.action == "buy" else sj.constant.Action.Sell,
            price_type=(
                sj.constant.StockPriceType.MKT if order.order_type == "market" else sj.constant.StockPriceType.LMT
            ),
            order_type=sj.constant.OrderType.ROD,
            order_lot=sj.constant.StockOrderLot.Common,
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
                            fill_time=datetime.now(UTC),
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

    def _resolve_stock_contract(self, symbol: str):
        stocks = self._api.Contracts.Stocks

        try:
            return stocks[symbol]
        except Exception:
            pass

        for exchange in ("TSE", "OTC", "OES"):
            market_contracts = getattr(stocks, exchange, None)
            if market_contracts is None:
                continue
            try:
                return market_contracts[symbol]
            except Exception:
                continue

        raise KeyError(f"Could not resolve Shioaji stock contract for symbol: {symbol}")
