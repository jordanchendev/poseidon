"""Portfolio rebalancer -- computes differential orders.

Compares target holdings (from strategy) vs current holdings (from position tracker)
and produces a list of RebalanceOrder (buy new / sell removed / adjust changed).

This is a pure function with no side effects or DB dependency.
"""

import logging

from poseidon.strategies.portfolio.schemas import Holding, RebalanceOrder, TargetPosition

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """Compares target portfolio vs current holdings, produces differential orders.

    The rebalancer works with weights only (0.0-1.0).
    Share quantity calculation is deferred to Phase 23 (OrderManager).
    """

    def __init__(self, adjust_threshold: float = 0.01):
        """Initialize rebalancer.

        Args:
            adjust_threshold: Minimum weight delta to generate an adjust order.
                              Avoids tiny rebalance trades. Default 1%.
        """
        self.adjust_threshold = adjust_threshold

    def rebalance(
        self,
        targets: list[TargetPosition],
        current: dict[str, Holding],
    ) -> list[RebalanceOrder]:
        """Compare target portfolio vs current holdings.

        Args:
            targets: Target positions from PortfolioStrategy.select_stocks()
            current: Current holdings from PositionTracker.current_holdings()

        Returns:
            List of RebalanceOrder:
            - action="buy": symbol in targets but not in current
            - action="sell": symbol in current but not in targets
            - action="adjust": symbol in both but weight delta > adjust_threshold
        """
        orders: list[RebalanceOrder] = []
        target_map = {t.symbol: t for t in targets}
        current_symbols = set(current.keys())
        target_symbols = set(target_map.keys())

        # New positions to buy
        for sym in sorted(target_symbols - current_symbols):
            orders.append(
                RebalanceOrder(
                    symbol=sym,
                    action="buy",
                    target_weight=target_map[sym].weight,
                    current_weight=0.0,
                    delta_weight=target_map[sym].weight,
                    side=target_map[sym].side,
                )
            )

        # Positions to sell (no longer in target)
        for sym in sorted(current_symbols - target_symbols):
            orders.append(
                RebalanceOrder(
                    symbol=sym,
                    action="sell",
                    target_weight=0.0,
                    current_weight=current[sym].weight,
                    delta_weight=-current[sym].weight,
                    side=current[sym].side,
                )
            )

        # Positions to adjust (weight changed beyond threshold)
        for sym in sorted(current_symbols & target_symbols):
            target = target_map[sym]
            cur = current[sym]

            # Side flip detection: close old + open new
            if cur.side != target.side:
                orders.append(
                    RebalanceOrder(
                        symbol=sym,
                        action="sell",
                        target_weight=0.0,
                        current_weight=cur.weight,
                        delta_weight=-cur.weight,
                        side=cur.side,
                    )
                )
                orders.append(
                    RebalanceOrder(
                        symbol=sym,
                        action="buy",
                        target_weight=target.weight,
                        current_weight=0.0,
                        delta_weight=target.weight,
                        side=target.side,
                    )
                )
                continue

            # Normal adjust (same side)
            delta = target.weight - cur.weight
            if abs(delta) > self.adjust_threshold:
                orders.append(
                    RebalanceOrder(
                        symbol=sym,
                        action="adjust",
                        target_weight=target.weight,
                        current_weight=cur.weight,
                        delta_weight=delta,
                        side=target.side,
                    )
                )

        logger.info(
            "Rebalance: %d buy, %d sell, %d adjust",
            sum(1 for o in orders if o.action == "buy"),
            sum(1 for o in orders if o.action == "sell"),
            sum(1 for o in orders if o.action == "adjust"),
        )
        return orders
