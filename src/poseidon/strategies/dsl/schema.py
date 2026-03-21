"""DSL JSON schema — Pydantic models for rule-based strategy definitions."""

from pydantic import BaseModel, Field


class RuleEntry(BaseModel):
    """A single rule: condition tree + action to take if condition is met."""

    condition: dict  # Recursive tree — validated by executor at eval time
    action: str  # "long", "short", "close"
    quantity_pct: float | None = Field(None, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class RuleConfig(BaseModel):
    """Top-level DSL document for a rule-based strategy."""

    name: str
    description: str = ""
    symbol: str
    market: str
    interval: str = "1d"
    rules: list[RuleEntry] = Field(..., min_length=1)
