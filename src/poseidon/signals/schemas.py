"""Signal schema — standardized output from all strategies.

Both ModelStrategy and RuleStrategy produce Signal objects.
The risk engine evaluates them equally regardless of source.
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SignalAction(str, Enum):
    """Valid signal actions."""

    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


class InstrumentType(str, Enum):
    """Supported instrument types."""

    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"


class SignalStatus(str, Enum):
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
    signal_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: datetime | None = None
    interval: str = "1d"

    # Instrument-specific params (JSONB in DB)
    params: dict = Field(default_factory=dict)

    # Risk
    status: SignalStatus = SignalStatus.PENDING
    reject_reason: str | None = None

    # Metadata
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}
