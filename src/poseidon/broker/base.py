"""Abstract broker adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from poseidon.decision_loop.manifest import content_sha256
from poseidon.orders.schemas import DURABLE_PROTECTIVE_ORIGINS, Fill, Order


class BrokerCapabilityError(ValueError):
    """The adapter cannot safely execute a durable decision order."""


@dataclass(frozen=True)
class BrokerCapabilities:
    """Explicit restart-safe broker capabilities."""

    stable_client_reference: bool = False
    find_order_by_client_reference: bool = False
    query_order: bool = False
    query_fills: bool = False
    query_account_snapshot: bool = False

    @property
    def supports_reconciliation(self) -> bool:
        return not self.missing_contract

    @property
    def missing_contract(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "stable_client_reference",
                "find_order_by_client_reference",
                "query_order",
                "query_fills",
                "query_account_snapshot",
            )
            if not getattr(self, name)
        )


@dataclass(frozen=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    client_order_ref: str
    account_scope: str
    account_generation: str
    market: str
    symbol: str
    instrument: str
    action: str
    side: str
    order_type: str
    quantity: float
    price: float | None
    status: str
    accepted_at: datetime
    state_version: int

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.market, self.symbol, self.instrument, self.side)


@dataclass(frozen=True)
class BrokerFillSnapshot:
    broker_order_id: str
    broker_fill_id: str
    account_scope: str
    account_generation: str
    market: str
    symbol: str
    instrument: str
    side: str
    fill_price: float
    fill_quantity: float
    fill_time: datetime
    state_version: int

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.market, self.symbol, self.instrument, self.side)


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    market: str
    symbol: str
    instrument: str
    side: str
    quantity: float

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.market, self.symbol, self.instrument, self.side)


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    account_scope: str
    account_generation: str
    currency: str
    cash: float
    state_version: int
    as_of: datetime
    positions: tuple[BrokerPositionSnapshot, ...]


class BrokerAdapter(ABC):
    """Abstract broker interface -- swappable between paper and live."""

    capabilities = BrokerCapabilities()

    @property
    def supports_reconciliation(self) -> bool:
        return self.capabilities.supports_reconciliation

    def place_decision_order(self, order: Order, *, client_order_ref: str) -> BrokerOrderSnapshot:
        """Capability-gated decision submission; legacy callers use ``place_order``."""
        self._require_decision_submission(order, client_order_ref)
        return self.place_order(order, client_order_ref=client_order_ref)

    def _require_decision_submission(self, order: Order, client_order_ref: str) -> None:
        if not self.supports_reconciliation:
            missing = ", ".join(self.capabilities.missing_contract)
            raise BrokerCapabilityError(f"missing reconciliation contract: {missing}")
        if order.broker_mode != "paper":
            raise BrokerCapabilityError("decision execution requires broker_mode=paper")
        if order.order_origin not in {"decision", *DURABLE_PROTECTIVE_ORIGINS}:
            raise BrokerCapabilityError("decision execution requires order_origin=decision or exact protective origin")
        if not client_order_ref or order.client_order_ref != client_order_ref:
            raise BrokerCapabilityError("decision execution requires the stored client reference")
        if not order.account_scope or not order.account_generation or not order.instrument:
            raise BrokerCapabilityError("decision execution requires complete account and instrument identity")
        legacy_protective = (
            order.order_origin in DURABLE_PROTECTIVE_ORIGINS
            and isinstance(order.protective_context_json, dict)
            and order.protective_context_json.get("legacy_exception") is True
        )
        if order.execution_key is None or (order.decision_id is None) != legacy_protective:
            raise BrokerCapabilityError("decision execution requires durable decision identity")
        if order.order_origin in DURABLE_PROTECTIVE_ORIGINS:
            frozen = (order.intent_json or {}).get("frozen_intent", {})
            if frozen.get("action") not in {"reduce", "exit"} or not order.protective_context_json:
                raise BrokerCapabilityError("protective execution requires reduction-only durable context")
            if legacy_protective:
                base_context = {
                    "account_scope",
                    "account_generation",
                    "identity",
                    "origin",
                    "trigger_generation",
                    "source_lot_ids",
                    "source_decision_ids",
                    "source_holding_ids",
                    "source_holding_quantities",
                    "source_holding_risk",
                    "legacy_context_sha256",
                    "legacy_exception",
                    "dedupe_sha256",
                }
                generation_context = {"predecessor_order_ids", "remaining_source_quantities"}
                if set(order.protective_context_json) not in (base_context, base_context | generation_context):
                    raise BrokerCapabilityError("legacy protective execution requires complete durable context")
        try:
            valid_intent = bool(order.intent_json) and order.intent_sha256 == content_sha256(order.intent_json)
        except ValueError as error:
            raise BrokerCapabilityError("decision execution requires finite frozen intent economics") from error
        if not valid_intent:
            raise BrokerCapabilityError("decision execution requires a valid frozen intent fingerprint")

    @abstractmethod
    def login(self) -> bool:
        """Authenticate with broker. Returns True on success."""
        ...

    @abstractmethod
    def place_order(
        self,
        order: Order,
        *,
        client_order_ref: str | None = None,
    ) -> str | BrokerOrderSnapshot:
        """Submit a legacy order or a capability-gated decision order."""
        ...

    @abstractmethod
    def query_fills(
        self,
        broker_order_id: str,
        *,
        account_scope: str | None = None,
        account_generation: str | None = None,
    ) -> list[Fill] | list[BrokerFillSnapshot]:
        """Query legacy fills or normalized fills for an account generation."""
        ...

    @abstractmethod
    def query_positions(self) -> list[dict]:
        """Query current broker positions. Returns list of position dicts."""
        ...

    @abstractmethod
    def logout(self) -> None:
        """Disconnect from broker."""
        ...

    def find_order_by_client_ref(
        self,
        client_order_ref: str,
        *,
        account_scope: str,
        account_generation: str,
    ) -> BrokerOrderSnapshot | None:
        raise BrokerCapabilityError("adapter cannot find orders by client reference")

    def query_order(
        self,
        broker_order_id: str,
        *,
        account_scope: str,
        account_generation: str,
    ) -> BrokerOrderSnapshot | None:
        raise BrokerCapabilityError("adapter cannot query normalized orders")

    def query_account_snapshot(self, account_scope: str, account_generation: str) -> BrokerAccountSnapshot:
        raise BrokerCapabilityError("adapter cannot query normalized account snapshots")
