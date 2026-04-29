"""OrderManager -- orchestrates rebalance orders through risk checks to broker execution.

Flow: RebalanceOrder -> weight_to_shares -> risk check -> broker.place_order -> persist -> update positions.
Per-order isolation: rejection of one order does not block others.
"""

import logging
import uuid
from datetime import UTC, datetime

from poseidon.broker.base import BrokerAdapter
from poseidon.broker.config import BrokerConfig
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.orders.risk_checker import OrderRiskChecker
from poseidon.orders.schemas import Fill, Order, OrderResult
from poseidon.orders.state_machine import OrderStatus, transition_order
from poseidon.risk.reject_reason import build_reject_reason
from poseidon.strategies.portfolio.schemas import RebalanceOrder

logger = logging.getLogger(__name__)


def weight_to_shares(
    weight: float,
    total_nav: float,
    price: float,
    lot_size: int = 1000,
    fractional: bool = False,
) -> float:
    """Convert portfolio weight to share quantity.

    When fractional=True (perps): return exact notional/price (no rounding).
    When fractional=False (TW stock): round DOWN to nearest lot_size.

    Returns 0 if resulting shares < lot_size (skip the order) in non-fractional mode.
    """
    if price <= 0:
        return 0.0
    notional = weight * total_nav
    raw_shares = notional / price
    if fractional:
        return raw_shares
    lots = int(raw_shares // lot_size)
    return float(lots * lot_size)


class OrderManager:
    """Orchestrates the rebalance -> risk check -> broker dispatch -> persist pipeline.

    Constructor args:
        broker: BrokerAdapter (paper or live)
        risk_checker: OrderRiskChecker instance
        position_tracker: PositionTracker for updating holdings after fills
        session_factory: callable returning SQLAlchemy Session
        config: BrokerConfig with mode, total_nav, lot_size
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        risk_checker: OrderRiskChecker,
        position_tracker,  # PositionTracker (avoid circular import)
        session_factory,
        config: BrokerConfig,
        leverage_limits: dict[str, int] | None = None,
    ):
        self._broker = broker
        self._risk_checker = risk_checker
        self._tracker = position_tracker
        self._session_factory = session_factory
        self._config = config
        self._leverage_limits = leverage_limits or {}

    def execute_rebalance(
        self,
        rebalance_orders: list[RebalanceOrder],
        strategy_name: str,
        prices: dict[str, float],
        market: str = "tw_stock",
        *,
        signal_ids: dict[str, uuid.UUID] | None = None,
    ) -> list[OrderResult]:
        """Process rebalance orders end-to-end.

        Args:
            rebalance_orders: Differential orders from PortfolioRebalancer
            strategy_name: Strategy that produced the orders
            prices: {symbol: current_price} for weight-to-shares conversion
            market: Market identifier (default "tw_stock")
            signal_ids: Optional {symbol: signal_uuid} mapping. When provided,
                each Order gets ``signal_id`` populated from this dict so the
                order is auditably traceable to the upstream PASSED signal
                (Phase 89-01 D-04, F8 wiring fix). Symbols missing from the
                dict (or when signal_ids is None) yield Order.signal_id=None,
                which is the correct semantic for protective close-outs and
                portfolio-level rebalances that aren't signal-driven.

        Returns:
            List of OrderResult (one per processed RebalanceOrder, excludes qty==0 skips)
        """
        results: list[OrderResult] = []
        processed_rorders: list[RebalanceOrder] = []  # parallel to results (qty==0 skips excluded)
        current_holdings = self._tracker.current_holdings()

        for rorder in rebalance_orders:
            # Skip "adjust" -- convert to buy/sell based on delta
            action = rorder.action
            if action == "adjust":
                action = "buy" if rorder.delta_weight > 0 else "sell"

            # Get price for weight-to-shares
            price = prices.get(rorder.symbol, 0.0)
            abs_weight = abs(rorder.delta_weight) if action == "sell" else rorder.target_weight

            # Convert weight to shares
            quantity = weight_to_shares(
                weight=abs_weight,
                total_nav=self._config.paper_initial_nav,
                price=price,
                lot_size=self._config.lot_size,
            )

            if quantity == 0:
                logger.info("Skipping %s %s: quantity rounds to 0 shares", action, rorder.symbol)
                continue

            # Phase 89-01 (D-04): attach signal_id when this rebalance order
            # was triggered by a PASSED signal. Falls back to None for
            # non-signal-driven flows (protective close, portfolio rebalance).
            sid = signal_ids.get(rorder.symbol) if signal_ids else None

            # Create Order object
            order = Order(
                symbol=rorder.symbol,
                market=market,
                action=action,
                order_type="market",
                target_weight=rorder.target_weight,
                quantity=quantity,
                strategy_name=strategy_name,
                broker_mode=self._config.mode,
                side=rorder.side,
                signal_id=sid,
            )

            # Per-order risk check (isolation: rejection of one does not block others)
            check = self._risk_checker.check(order, current_holdings, self._config.paper_initial_nav)
            if not check.passed:
                order.status = OrderStatus.REJECTED
                # TRUTH-03 (D-13/D-14): structured 4-key reject_reason via factory.
                order.reject_reason = build_reject_reason(
                    check_name=check.check_name or "unknown",
                    rule=check.reason,
                    shortfall=check.shortfall,
                )
                self._persist_order(order)
                processed_rorders.append(rorder)
                results.append(OrderResult(order=order, fills=[], success=False))
                logger.warning("Order rejected: %s %s -- %s", action, rorder.symbol, check.reason)
                continue

            # Leverage enforcement (PRSK-03, D-07)
            if market == "crypto_perp" and self._leverage_limits:
                max_lev = self._leverage_limits.get(order.symbol)
                if max_lev is not None:
                    # Read current leverage from broker adapter
                    actual_leverage = getattr(self._broker, "_leverage_per_symbol", {}).get(
                        order.symbol,
                        getattr(self._broker, "_default_leverage", 1),
                    )
                    if actual_leverage > max_lev:
                        order.status = OrderStatus.REJECTED
                        rule_text = f"leverage_limit: {order.symbol} leverage {actual_leverage}x exceeds max {max_lev}x"
                        # TRUTH-03 (D-13/D-14): structured payload with leverage shortfall.
                        order.reject_reason = build_reject_reason(
                            check_name="leverage_limit",
                            rule=rule_text,
                            shortfall={
                                "symbol": order.symbol,
                                "actual_leverage": actual_leverage,
                                "max_leverage": max_lev,
                            },
                        )
                        self._persist_order(order)
                        processed_rorders.append(rorder)
                        results.append(OrderResult(order=order, fills=[], success=False))
                        logger.warning("Order rejected: %s", rule_text)
                        continue

            # Dispatch to broker
            try:
                broker_id = self._broker.place_order(order)
                order.broker_order_id = broker_id
                order.status = transition_order(order.status, OrderStatus.SUBMITTED)

                # Query fills (paper mode fills immediately)
                fills = self._broker.query_fills(broker_id)
                if fills:
                    order.status = transition_order(order.status, OrderStatus.FILLED)
            except Exception as e:
                order.status = OrderStatus.REJECTED
                # TRUTH-03 (D-13/D-14): wrap broker exceptions in structured payload.
                order.reject_reason = build_reject_reason(
                    check_name="broker_error",
                    rule=f"broker_error: {type(e).__name__}",
                    details=str(e),
                )
                fills = []
                logger.error("Broker error for %s %s: %s", action, rorder.symbol, e)

            # Persist order + fills to DB
            self._persist_order(order, fills)
            processed_rorders.append(rorder)
            results.append(OrderResult(order=order, fills=fills, success=order.status == OrderStatus.FILLED))

        # Update PositionTracker with filled orders (per-order to support mixed sides)
        # IMPORTANT: zip against processed_rorders (not rebalance_orders) because qty==0
        # orders are skipped via continue without appending to results. Using
        # rebalance_orders here would misalign rorder<->result pairs.
        filled_count = 0
        for rorder, result in zip(processed_rorders, results, strict=False):
            if result.success:
                total_shares = sum(f.fill_quantity for f in result.fills)
                avg_price = (
                    sum(f.fill_price * f.fill_quantity for f in result.fills) / total_shares
                    if total_shares > 0
                    else 0.0
                )
                self._tracker.apply_orders(
                    [rorder],
                    strategy_name,
                    market,
                    fill_info={rorder.symbol: (total_shares, avg_price)},
                    side=rorder.side,
                )
                filled_count += 1
        if filled_count:
            logger.info("Updated positions: %d filled orders", filled_count)

        # Create cooldown protection locks for filled orders (D-14)
        for _rorder, result in zip(processed_rorders, results, strict=False):
            if result.success:
                try:
                    from datetime import timedelta

                    from poseidon.protections.cooldown import CooldownProtection

                    cooldown = CooldownProtection()
                    expires = datetime.now(UTC) + timedelta(hours=cooldown.cooldown_hours)
                    session = self._session_factory()
                    try:
                        cooldown.create_lock(
                            symbol=result.order.symbol,
                            market=market,
                            reason=f"Cooldown after {result.order.action} execution",
                            expires_at=expires,
                            db=session,
                        )
                    finally:
                        session.close()
                except Exception as e:
                    logger.warning("Failed to create cooldown lock for %s: %s", result.order.symbol, e)

        return results

    def _persist_order(self, order: Order, fills: list[Fill] | None = None) -> None:
        """Write OrderRecord + OrderFillRecords to DB."""
        session = self._session_factory()
        try:
            record = OrderRecord(
                strategy_name=order.strategy_name,
                symbol=order.symbol,
                market=order.market,
                action=order.action,
                order_type=order.order_type,
                target_weight=order.target_weight,
                quantity=order.quantity,
                price=order.price,
                status=order.status.value if isinstance(order.status, OrderStatus) else order.status,
                broker_order_id=order.broker_order_id,
                broker_mode=order.broker_mode,
                reject_reason=order.reject_reason,
                side=order.side,
                # Phase 89-01 (D-04, F8 wiring fix): persist FK to upstream signal.
                signal_id=order.signal_id,
            )
            session.add(record)
            session.flush()  # get the DB-generated UUID

            if fills:
                for fill in fills:
                    fill_record = OrderFillRecord(
                        order_id=record.id,
                        fill_price=fill.fill_price,
                        fill_quantity=fill.fill_quantity,
                        fill_time=fill.fill_time,
                        broker_fill_id=fill.broker_fill_id,
                    )
                    session.add(fill_record)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
