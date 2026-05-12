"""Capabilities API -- component capability matrix endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel as PydanticBaseModel

from poseidon.capabilities.registry import get_all_capabilities

router = APIRouter()


class ComponentCapabilityResponse(PydanticBaseModel):
    """Single component capability entry."""

    name: str
    component_type: str
    supports_backtest: bool
    supports_live: bool
    bias_risk: list[str]
    stateful: bool


class CapabilitiesResponse(PydanticBaseModel):
    """Full capabilities matrix."""

    components: list[ComponentCapabilityResponse]
    total: int


@router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities(
    live_safe: bool | None = Query(None, description="Filter to live-safe components only"),
    component_type: str | None = Query(
        None, description="Filter by type: feature|strategy|model|rule|portfolio_strategy"
    ),
) -> CapabilitiesResponse:
    """Return capability flags matrix for all registered components.

    Returns a flat matrix grouped by component_type.
    Supports optional query params for filtering.
    """
    caps = get_all_capabilities()

    results = []
    for cap in caps:
        # filter by live_safe if specified
        if live_safe is True and not cap.supports_live:
            continue
        if live_safe is False and cap.supports_live:
            continue
        # Filter by component_type if specified
        if component_type and cap.component_type != component_type:
            continue
        results.append(
            ComponentCapabilityResponse(
                name=cap.name,
                component_type=cap.component_type,
                supports_backtest=cap.supports_backtest,
                supports_live=cap.supports_live,
                bias_risk=cap.bias_risk,
                stateful=cap.stateful,
            )
        )

    return CapabilitiesResponse(components=results, total=len(results))
