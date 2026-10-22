"""DSL JSON schema — Pydantic models for rule-based strategy definitions."""

from typing import Literal

from pydantic import BaseModel, Field


class RuleEntry(BaseModel):
    """A single rule: condition tree + action to take if condition is met."""

    condition: dict  # Recursive tree — validated by executor at eval time
    action: str  # "long", "short", "close"
    quantity_pct: float | None = Field(None, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class RuleConfig(BaseModel):
    """Top-level DSL document for a rule-based strategy (DSL v1).

    ``dsl_version`` only accepts 1 so a v2 document can never be silently
    parsed as v1; it is excluded from ``model_dump()`` to keep v1 dumps
    byte-identical to what they were before versioning existed.
    """

    dsl_version: Literal[1] = Field(1, exclude=True)
    name: str
    description: str = ""
    symbol: str
    market: str
    interval: str = "1d"
    rules: list[RuleEntry] = Field(..., min_length=1)
