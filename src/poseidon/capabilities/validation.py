"""Pipeline enforcement -- reject live-unsafe, warn on bias_risk."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_live_components(components: list[Any]) -> None:
    """Reject any component with supports_live=False. Per D-04.

    Called by the Celery task or pipeline that assembles live components.
    Raises ValueError immediately on first rejection.
    """
    for comp in components:
        if not getattr(comp, "supports_live", False):
            name = getattr(comp, "name", type(comp).__name__)
            raise ValueError(
                f"Component '{name}' does not support live mode "
                f"(supports_live=False). Remove it or mark supports_live=True."
            )


def validate_backtest_components(components: list[Any]) -> None:
    """Reject any component with supports_backtest=False. Per D-06."""
    for comp in components:
        if not getattr(comp, "supports_backtest", True):
            name = getattr(comp, "name", type(comp).__name__)
            raise ValueError(f"Component '{name}' does not support backtest mode (supports_backtest=False).")


def warn_bias_risks(components: list[Any]) -> None:
    """Log warnings for components with non-empty bias_risk. Per D-05, D-10."""
    for comp in components:
        risks = getattr(comp, "bias_risk", [])
        if risks:
            name = getattr(comp, "name", type(comp).__name__)
            logger.warning(
                "BIAS WARNING: Component '%s' has bias risks: %s. Backtest results may be unreliable.",
                name,
                ", ".join(risks),
            )
