"""The durable protective-exit context stored on an order (``OrderRecord.protective_context_json``).

One typed definition of the context's shape, parsed by the broker boundary, the paper adapter, and protective
execution. Checks against other rows (order identity, dedupe digest, source lots and holdings) stay with each caller.

Two variants share the base fields: an ordinary context closes decision lots (``source_lot_ids`` and
``source_decision_ids``), a legacy context (``legacy_exception: true``) closes pre-loop holdings
(``source_holding_*`` and ``legacy_context_sha256``, empty lot and decision ids). A successor order of either
variant adds ``predecessor_order_ids`` (exactly one) and ``remaining_source_quantities`` (per source id).
"""

import math
import uuid
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

GENERATION_FIELDS = frozenset({"predecessor_order_ids", "remaining_source_quantities"})
LEGACY_FIELDS = frozenset(
    {
        "legacy_exception",
        "source_holding_ids",
        "source_holding_quantities",
        "source_holding_risk",
        "legacy_context_sha256",
    }
)
_GENERATION_ERROR = "protective_generation"


class ProtectiveContextError(ValueError):
    """The stored context is not a protective context; ``generation`` marks a bad successor link."""

    def __init__(self, message: str, *, generation: bool) -> None:
        super().__init__(message)
        self.generation = generation


def _uuid_text(value: str) -> str:
    uuid.UUID(value)
    return value


def _canonical_uuid_text(value: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise ValueError("UUID text is not canonical")
    return value


def _finite(value: int | float) -> int | float:
    if not math.isfinite(value):
        raise ValueError("number is not finite")
    return value


def _positive(value: int | float) -> int | float:
    if value <= 0:
        raise ValueError("number is not positive")
    return value


def _non_negative(value: int | float) -> int | float:
    if value < 0:
        raise ValueError("number is negative")
    return value


def _trimmed_text(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("text is empty or padded")
    return value


UUIDText = Annotated[StrictStr, AfterValidator(_uuid_text)]
CanonicalUUIDText = Annotated[StrictStr, AfterValidator(_canonical_uuid_text)]
PositiveNumber = Annotated[StrictInt | StrictFloat, AfterValidator(_finite), AfterValidator(_positive)]
NonNegativeNumber = Annotated[StrictInt | StrictFloat, AfterValidator(_finite), AfterValidator(_non_negative)]


class _Frozen(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ProtectiveIdentity(_Frozen):
    market: StrictStr
    symbol: StrictStr
    instrument: StrictStr
    side: StrictStr


class LegacySourceRisk(_Frozen):
    entry_price: PositiveNumber
    stop_loss_pct: PositiveNumber


class ProtectiveContext(_Frozen):
    account_scope: StrictStr
    account_generation: StrictStr
    identity: ProtectiveIdentity
    origin: StrictStr
    trigger_generation: Annotated[StrictStr, AfterValidator(_trimmed_text)]
    source_lot_ids: list[UUIDText]
    source_decision_ids: list[UUIDText]
    dedupe_sha256: StrictStr
    legacy_exception: Literal[True] | None = None
    source_holding_ids: list[CanonicalUUIDText] | None = None
    source_holding_quantities: dict[StrictStr, PositiveNumber] | None = None
    source_holding_risk: dict[StrictStr, LegacySourceRisk] | None = None
    legacy_context_sha256: StrictStr | None = None
    predecessor_order_ids: list[CanonicalUUIDText] | None = None
    remaining_source_quantities: dict[StrictStr, NonNegativeNumber] | None = None

    @model_validator(mode="before")
    @classmethod
    def require_whole_variants(cls, value):
        """Optional groups are all present or all absent, never null; a context is a JSON object."""
        if not isinstance(value, dict):
            raise PydanticCustomError("protective_context", "protective context must be an object")
        keys = set(value)
        if keys & GENERATION_FIELDS and GENERATION_FIELDS - keys:
            raise PydanticCustomError(_GENERATION_ERROR, "successor context needs both generation fields")
        if keys & LEGACY_FIELDS and LEGACY_FIELDS - keys:
            raise PydanticCustomError("protective_context", "legacy context needs every legacy field")
        if any(value[key] is None for key in GENERATION_FIELDS & keys):
            raise PydanticCustomError(_GENERATION_ERROR, "successor context fields are never null")
        if any(value[key] is None for key in LEGACY_FIELDS & keys):
            raise PydanticCustomError("protective_context", "legacy context fields are never null")
        return value

    @model_validator(mode="after")
    def require_consistent_sources(self):
        if self.legacy:
            holding_ids = self.source_holding_ids
            if (
                not holding_ids
                or holding_ids != sorted(holding_ids)
                or len(set(holding_ids)) != len(holding_ids)
                or set(self.source_holding_quantities) != set(holding_ids)
                or set(self.source_holding_risk) != set(holding_ids)
                or self.source_lot_ids
                or self.source_decision_ids
            ):
                raise PydanticCustomError("protective_context", "legacy sources are not one sorted holding set")
        elif (
            not self.source_lot_ids
            or len({uuid.UUID(value) for value in self.source_lot_ids}) != len(self.source_lot_ids)
            or not self.source_decision_ids
            or self.source_decision_ids != sorted(self.source_decision_ids)
        ):
            raise PydanticCustomError("protective_context", "ordinary sources need unique lots and sorted decisions")
        if self.predecessor_order_ids is not None:
            remaining = self.remaining_source_quantities
            if (
                len(self.predecessor_order_ids) != 1
                or set(remaining) != set(self.source_ids)
                or sum(remaining.values()) <= 0
            ):
                raise PydanticCustomError(
                    _GENERATION_ERROR, "successor context needs one predecessor and its remainder"
                )
        return self

    @property
    def legacy(self) -> bool:
        return self.legacy_exception is True

    @property
    def successor(self) -> bool:
        return self.predecessor_order_ids is not None

    @property
    def source_ids(self) -> list[str]:
        """The ids the order closes: legacy holdings or decision lots."""
        return self.source_holding_ids if self.legacy else self.source_lot_ids

    @classmethod
    def parse(cls, value) -> "ProtectiveContext":
        """Validate stored context JSON; every failure is a ``ProtectiveContextError``."""
        try:
            return cls.model_validate(value)
        except ValidationError as error:
            generation = any(
                item["type"] == _GENERATION_ERROR or (item["loc"] and item["loc"][0] in GENERATION_FIELDS)
                for item in error.errors()
            )
            raise ProtectiveContextError(str(error), generation=generation) from error
