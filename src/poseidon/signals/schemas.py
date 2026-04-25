"""Signal schema — standardized output from all strategies.

Both ModelStrategy and RuleStrategy produce Signal objects.
The risk engine evaluates them equally regardless of source.
"""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class SignalAction(StrEnum):
    """Valid signal actions."""

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


class InstrumentType(StrEnum):
    """Supported instrument types."""

    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"


class OrderType(StrEnum):
    """Order execution type."""

    MARKET = "market"
    LIMIT = "limit"


class SignalStatus(StrEnum):
    """Signal risk check status."""

    PASSED = "passed"
    REJECTED = "rejected"
    PENDING = "pending"


class Signal(BaseModel):
    """Standardized trading signal produced by strategies."""

    # Identity
    id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID | None = None
    model_id: UUID | None = None  # None for rule-based strategies

    # Instrument
    symbol: str
    market: str
    instrument: InstrumentType = InstrumentType.SPOT

    # Signal content
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    quantity_pct: float | None = Field(None, ge=0.0, le=1.0)

    # Time
    signal_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    interval: str = "1d"

    # Instrument-specific params (JSONB in DB)
    params: dict = Field(default_factory=dict)

    # Limit order fields (Phase 49 -- all optional, backward-compatible)
    order_type: OrderType | None = None
    order_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    # Risk
    status: SignalStatus = SignalStatus.PENDING
    reject_reason: str | None = None

    # Metadata
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def validate_limit_order_fields(self) -> "Signal":
        """When order_type is LIMIT, order_price is required."""
        if self.order_type == OrderType.LIMIT and self.order_price is None:
            raise ValueError("order_price is required when order_type is LIMIT")
        return self
