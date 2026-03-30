"""Abstract broker adapter interface."""

from abc import ABC, abstractmethod

from poseidon.orders.schemas import Fill, Order


class BrokerAdapter(ABC):
    """Abstract broker interface -- swappable between paper and live."""

    @abstractmethod
    def login(self) -> bool:
        """Authenticate with broker. Returns True on success."""
        ...

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Submit order. Returns broker_order_id string."""
        ...

    @abstractmethod
    def query_fills(self, broker_order_id: str) -> list[Fill]:
        """Query fills for a given broker order ID."""
        ...

    @abstractmethod
    def query_positions(self) -> list[dict]:
        """Query current broker positions. Returns list of position dicts."""
        ...

    @abstractmethod
    def logout(self) -> None:
        """Disconnect from broker."""
        ...
