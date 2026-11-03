"""Decision policy models: hard limits, protective exit, release gate, freshness, reconciliation, intents."""

import uuid
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]


PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


PositiveStrictFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


NonNegativeStrictFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class HardLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_gross_exposure: PositiveFiniteFloat
    max_position_weight: PositiveFiniteFloat

    @field_validator("max_gross_exposure", "max_position_weight", mode="before")
    @classmethod
    def reject_boolean_limits(cls, value):
        if isinstance(value, bool):
            raise ValueError("hard limits must be numeric")
        return value


class ProtectiveExit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_without_approval: StrictBool


class ReleaseGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_mature_samples: PositiveStrictInt
    requires_human_release: StrictBool


class FreshnessRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_age_seconds: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class PerpInstrumentRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity_step: PositiveStrictFiniteFloat
    contract_multiplier: PositiveStrictFiniteFloat
    margin_semantics: Annotated[str, Field(min_length=1)]
    funding_semantics: Annotated[str, Field(min_length=1)]

    @field_validator("margin_semantics", "funding_semantics")
    @classmethod
    def require_trimmed_semantics(cls, value):
        if value != value.strip():
            raise ValueError("semantics must be non-empty trimmed text")
        return value


class ReconciliationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_generation: Annotated[str, Field(min_length=1)]
    opening_cash: NonNegativeStrictFiniteFloat
    currency: Annotated[str, Field(min_length=3, max_length=10, pattern=r"^[A-Z][A-Z0-9]+$")]
    max_reconciliation_age_seconds: PositiveStrictInt
    cash_tolerance: NonNegativeStrictFiniteFloat
    position_tolerance: NonNegativeStrictFiniteFloat
    fill_tolerance: NonNegativeStrictFiniteFloat
    tw_stock_quantity_rounding: Literal["whole_share_floor"]
    perp_instrument_rules: dict[str, PerpInstrumentRule]

    @field_validator("account_generation")
    @classmethod
    def require_trimmed_generation(cls, value):
        if value != value.strip():
            raise ValueError("account_generation must be non-empty trimmed text")
        return value

    @field_validator("perp_instrument_rules")
    @classmethod
    def require_canonical_instrument_keys(cls, rules):
        if any(not instrument.strip() or instrument != instrument.strip() for instrument in rules):
            raise ValueError("perp instrument keys must be non-empty trimmed text")
        return rules


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: Annotated[str, Field(min_length=1)]
    mode: Literal["human", "assisted", "automatic"]
    market: Literal["tw_stock", "crypto_perp"]
    account_scope: Annotated[str, Field(min_length=1)]
    universe_id: Annotated[str, Field(min_length=1)]
    decision_ttl_seconds: PositiveStrictInt
    required_data: Annotated[dict[str, FreshnessRule], Field(min_length=1)]
    hard_limits: HardLimits
    approval_roles: Annotated[list[str], Field(min_length=1)]
    protective_exit: ProtectiveExit
    release_gate: ReleaseGate
    reconciliation: ReconciliationPolicy | None = None

    @field_validator("policy_id", "account_scope", "universe_id")
    @classmethod
    def require_trimmed_text(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError("must be non-empty trimmed text")
        return value

    @field_validator("required_data")
    @classmethod
    def require_data_kinds(cls, rules):
        if any(not kind.strip() or kind != kind.strip() for kind in rules):
            raise ValueError("required data kinds must be non-empty trimmed text")
        return rules

    @field_validator("approval_roles")
    @classmethod
    def require_portfolio_manager(cls, roles):
        if len(set(roles)) != len(roles) or any(not role.strip() or role != role.strip() for role in roles):
            raise ValueError("approval roles must be unique non-empty text")
        if "portfolio_manager" not in roles:
            raise ValueError("portfolio_manager approval is required")
        return roles

    @model_validator(mode="after")
    def require_perp_rules(self):
        if self.market == "crypto_perp" and (
            self.reconciliation is None or not self.reconciliation.perp_instrument_rules
        ):
            raise ValueError("crypto_perp requires perp_instrument_rules")
        return self


class ExecutableIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_snapshot_id: uuid.UUID
    symbol: Annotated[str, Field(min_length=1)]
    market: Literal["tw_stock", "crypto_perp"]
    instrument: Annotated[str, Field(min_length=1)]
    action: Literal["enter", "add", "reduce", "exit"]
    side: Literal["long", "short"]
    target_weight: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    order_type: Literal["market"] = "market"

    @field_validator("symbol", "instrument")
    @classmethod
    def require_trimmed_identity(cls, value):
        if value != value.strip():
            raise ValueError("identity must be non-empty trimmed text")
        return value
