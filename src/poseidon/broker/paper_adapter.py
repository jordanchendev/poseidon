"""Paper trading broker adapter -- fills at the latest remote close price."""

import uuid
from datetime import UTC, datetime

from poseidon.broker.base import BrokerAdapter
from poseidon.orders.schemas import Fill, Order


class PaperBrokerAdapter(BrokerAdapter):
    """Simulates order fills using historical OHLCV close prices.

    Never imports shioaji. Fills are instantaneous at the latest
    available close price from the database.
    """

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._fills: dict[str, list[Fill]] = {}

    def login(self) -> bool:
        """No-op for paper trading."""
        return True

    def place_order(self, order: Order) -> str:
        """Fill order at the latest close price from Thalassa.

        Raises ValueError if no price data exists for the symbol.
        """
        from poseidon.data.remote_repository import RemoteDataRepository

        repo = RemoteDataRepository.from_settings()
        latest = repo.read_ohlcv(order.symbol, order.market, "1d")
        if latest.empty:
            raise ValueError(f"No price data for {order.symbol}")

        broker_order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        fill = Fill(
            order_id=order.id,
            fill_price=float(latest["close"].iloc[-1]),
            fill_quantity=order.quantity,
            fill_time=datetime.now(UTC),
            broker_fill_id=broker_order_id,
        )
        self._fills[broker_order_id] = [fill]
        return broker_order_id

    def query_fills(self, broker_order_id: str) -> list[Fill]:
        """Return fills for a given broker order ID."""
        return self._fills.get(broker_order_id, [])

    def query_positions(self) -> list[dict]:
        """Positions tracked by PositionTracker, not broker."""
        return []

    def logout(self) -> None:
        """No-op for paper trading."""
        pass
