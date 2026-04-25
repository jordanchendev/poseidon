"""Per-order risk checks for the order management subsystem.

Three checks per ORDER-02:
1. Single stock weight <= position_limit_pct
2. Total exposure after order <= max_exposure (buy orders only)
3. Stop-loss must be configured for buy orders
"""

import logging

from poseidon.orders.schemas import Order, RiskCheckResult
from poseidon.strategies.portfolio.schemas import Holding

logger = logging.getLogger(__name__)


class OrderRiskChecker:
    """Per-order risk checks. Each order checked independently (isolation).

    Three checks (per ORDER-02):
    1. Single stock weight <= position_limit_pct
    2. Total exposure after order <= max_exposure
    3. Stop-loss must be configured for buy orders
    """

    def __init__(
        self,
        position_limit_pct: float = 0.15,
        max_exposure: float = 1.0,
        stop_loss_pct: float | None = 0.10,
        market: str | None = None,
    ):
        self.position_limit_pct = position_limit_pct
        self.max_exposure = max_exposure
        self.stop_loss_pct = stop_loss_pct
        self.market = market

    def check(
        self,
        order: Order,
        current_holdings: dict[str, Holding],
        total_nav: float,
    ) -> RiskCheckResult:
        """Check a single order against risk limits. Returns pass/reject."""

        # Check 1: Single stock position limit
        if order.target_weight > self.position_limit_pct:
            reason = (
                f"position_limit: {order.symbol} weight {order.target_weight:.2%} "
                f"exceeds limit {self.position_limit_pct:.2%}"
            )
            logger.warning("Risk rejected: %s", reason)
            return RiskCheckResult(passed=False, reason=reason)

        # Check 2: Total exposure limit (only for buy orders)
        if order.action == "buy":
            holdings = current_holdings.values()
            if self.market:
                holdings = [h for h in holdings if getattr(h, "market", None) == self.market]
            current_exposure = sum(h.weight for h in holdings)
            projected = current_exposure + order.target_weight
            if projected > self.max_exposure:
                reason = f"max_exposure: projected {projected:.2%} exceeds limit {self.max_exposure:.2%}"
                logger.warning("Risk rejected: %s", reason)
                return RiskCheckResult(passed=False, reason=reason)

        # Check 3: Stop-loss configured for buy orders
        if order.action == "buy" and self.stop_loss_pct is None:
            reason = "stop_loss: no stop-loss configured for buy orders"
            logger.warning("Risk rejected: %s", reason)
            return RiskCheckResult(passed=False, reason=reason)

        return RiskCheckResult(passed=True)
