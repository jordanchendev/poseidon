"""Paper trading adapter for perpetual contracts with margin tracking.

Simulates perpetual futures trading: fills at DB close price, tracks
margin per position, calculates liquidation prices, settles funding
rates, and monitors margin ratio. Never calls any real exchange API.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from poseidon.broker.base import BrokerAdapter
from poseidon.orders.schemas import Fill, Order

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Liquidation price helper
# ---------------------------------------------------------------------------

def calc_liquidation_price(
    entry_price: float,
    leverage: int,
    side: str,
    maint_margin_rate: float = 0.005,
) -> float:
    """Calculate estimated liquidation price for a perpetual position.

    Long:  liq = entry_price * (1 - 1/leverage + maint_margin_rate)
    Short: liq = entry_price * (1 + 1/leverage - maint_margin_rate)

    Uses Binance default maintenance margin rate of 0.5% for BTC/ETH.
    """
    if side == "long":
        return entry_price * (1 - 1 / leverage + maint_margin_rate)
    else:
        return entry_price * (1 + 1 / leverage - maint_margin_rate)


# ---------------------------------------------------------------------------
# Internal position state
# ---------------------------------------------------------------------------

@dataclass
class PerpPosition:
    """In-memory state for a single perpetual contract position."""

    symbol: str
    side: str  # "long" | "short"
    entry_price: float
    quantity: float  # fractional (e.g. 0.001 BTC)
    leverage: int
    margin: float  # initial margin = notional / leverage
    liquidation_price: float
    unrealized_pnl: float = 0.0
    cumulative_funding: float = 0.0


# ---------------------------------------------------------------------------
# PerpPaperAdapter
# ---------------------------------------------------------------------------

class PerpPaperAdapter(BrokerAdapter):
    """Simulates perpetual contract trading with margin tracking.

    Fills orders at the latest OHLCV close price from the database
    (filtering for perpetual instrument type). Tracks margin, leverage,
    liquidation price, unrealized PnL, and funding settlement per position.

    Perp-specific methods (set_leverage, settle_funding, etc.) live on
    this subclass only -- they are NOT part of the BrokerAdapter ABC.
    """

    def __init__(
        self,
        session_factory,
        leverage: int = 3,
        maint_margin_rate: float = 0.005,
    ):
        self._session_factory = session_factory
        self._default_leverage = leverage
        self._maint_margin_rate = maint_margin_rate
        self._positions: dict[str, PerpPosition] = {}  # keyed by symbol
        self._fills: dict[str, list[Fill]] = {}
        self._leverage_per_symbol: dict[str, int] = {}
        self._margin_mode_per_symbol: dict[str, str] = {}
        logger.info(
            "PerpPaperAdapter created (default_leverage=%d, maint_margin_rate=%.4f)",
            leverage,
            maint_margin_rate,
        )

    # ------------------------------------------------------------------
    # BrokerAdapter ABC implementation
    # ------------------------------------------------------------------

    def login(self) -> bool:
        """No-op for paper trading."""
        return True

    def place_order(self, order: Order) -> str:
        """Fill order at the latest perpetual mark price from Thalassa.

        Creates or updates a PerpPosition with margin tracking and
        liquidation price calculation. Closing orders (sell-when-long,
        buy-when-short) remove the position.

        Raises ValueError if no perpetual price data exists for the symbol.
        """
        from poseidon.data.remote_repository import RemoteDataRepository

        repo = RemoteDataRepository.from_settings()
        latest_price = repo.read_latest_price(order.symbol)
        if latest_price is None:
            raise ValueError(f"No perpetual price data for {order.symbol}")

        fill_price = float(latest_price)
        session = self._session_factory()
        try:
            leverage = self._leverage_per_symbol.get(
                order.symbol, self._default_leverage
            )

            # Determine if this is a closing order
            existing = self._positions.get(order.symbol)
            is_closing = existing is not None and (
                (existing.side == "long" and order.action == "sell")
                or (existing.side == "short" and order.action == "buy")
            )

            if is_closing:
                # Close position
                logger.info(
                    "Closing %s position on %s at %.4f",
                    existing.side,
                    order.symbol,
                    fill_price,
                )
                del self._positions[order.symbol]
            else:
                # Open or add to position
                notional = fill_price * order.quantity
                margin = notional / leverage
                side = order.side
                liq_price = calc_liquidation_price(
                    fill_price, leverage, side, self._maint_margin_rate
                )

                if existing is not None and existing.side == side:
                    # Average into existing position
                    total_qty = existing.quantity + order.quantity
                    avg_price = (
                        existing.entry_price * existing.quantity
                        + fill_price * order.quantity
                    ) / total_qty
                    new_liq = calc_liquidation_price(
                        avg_price, leverage, side, self._maint_margin_rate
                    )
                    existing.entry_price = avg_price
                    existing.quantity = total_qty
                    existing.margin = avg_price * total_qty / leverage
                    existing.liquidation_price = new_liq
                    logger.info(
                        "Added to %s position on %s: qty=%.6f avg=%.4f",
                        side,
                        order.symbol,
                        total_qty,
                        avg_price,
                    )
                else:
                    # New position
                    self._positions[order.symbol] = PerpPosition(
                        symbol=order.symbol,
                        side=side,
                        entry_price=fill_price,
                        quantity=order.quantity,
                        leverage=leverage,
                        margin=margin,
                        liquidation_price=liq_price,
                    )
                    logger.info(
                        "Opened %s position on %s: qty=%.6f price=%.4f "
                        "margin=%.4f liq=%.4f",
                        side,
                        order.symbol,
                        order.quantity,
                        fill_price,
                        margin,
                        liq_price,
                    )

            # Create fill record
            broker_order_id = f"PERP-PAPER-{uuid.uuid4().hex[:12]}"
            fill = Fill(
                order_id=order.id,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                fill_time=datetime.now(timezone.utc),
                broker_fill_id=broker_order_id,
            )
            self._fills[broker_order_id] = [fill]
            return broker_order_id
        finally:
            session.close()

    def query_fills(self, broker_order_id: str) -> list[Fill]:
        """Return fills for a given broker order ID."""
        return self._fills.get(broker_order_id, [])

    def query_positions(self) -> list[dict]:
        """Return all open perpetual positions as dicts."""
        positions: list[dict] = []
        for pos in self._positions.values():
            mark_price = pos.entry_price  # default if not updated
            margin_ratio = (
                (pos.margin + pos.unrealized_pnl)
                / (mark_price * pos.quantity)
                if pos.quantity > 0
                else 0.0
            )
            positions.append(
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entryPrice": pos.entry_price,
                    "leverage": pos.leverage,
                    "margin": pos.margin,
                    "liquidationPrice": pos.liquidation_price,
                    "unrealizedPnl": pos.unrealized_pnl,
                    "marginRatio": margin_ratio,
                    "cumulativeFunding": pos.cumulative_funding,
                }
            )
        return positions

    def logout(self) -> None:
        """No-op for paper trading."""
        pass

    # ------------------------------------------------------------------
    # Perp-specific methods (NOT on BrokerAdapter ABC, per D-08)
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol (stored for next order)."""
        self._leverage_per_symbol[symbol] = leverage
        logger.info("Paper leverage set to %dx for %s", leverage, symbol)

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> None:
        """Record margin mode setting (paper always uses isolated)."""
        self._margin_mode_per_symbol[symbol] = mode
        logger.info("Paper margin mode set to '%s' for %s", mode, symbol)

    def update_mark_prices(self, prices: dict[str, float]) -> None:
        """Update unrealized PnL and margin ratio using mark prices.

        Args:
            prices: Mapping of symbol to current mark price.
        """
        for symbol, mark_price in prices.items():
            pos = self._positions.get(symbol)
            if pos is None:
                continue
            direction = 1 if pos.side == "long" else -1
            pos.unrealized_pnl = (
                (mark_price - pos.entry_price) * pos.quantity * direction
            )
            logger.debug(
                "Mark price update %s: mark=%.4f pnl=%.4f",
                symbol,
                mark_price,
                pos.unrealized_pnl,
            )

    def settle_funding(self, symbol: str, funding_rate: float) -> float:
        """Settle a funding payment for a position.

        Longs pay funding (negative), shorts receive (positive) when
        funding_rate > 0. Returns the funding amount applied.

        Args:
            symbol: The perpetual contract symbol.
            funding_rate: The funding rate for this period.

        Returns:
            The funding amount (positive = received, negative = paid).
        """
        pos = self._positions.get(symbol)
        if pos is None:
            logger.warning("No position for %s to settle funding", symbol)
            return 0.0

        payment = pos.quantity * pos.entry_price * funding_rate
        direction = -1 if pos.side == "long" else 1
        amount = payment * direction
        pos.cumulative_funding += amount
        pos.margin += amount  # funding affects margin balance
        logger.info(
            "Funding settled %s: rate=%.6f amount=%.4f cumulative=%.4f",
            symbol,
            funding_rate,
            amount,
            pos.cumulative_funding,
        )
        return amount

    def check_liquidation(self, symbol: str, mark_price: float) -> bool:
        """Check if a position should be liquidated at the given mark price.

        Returns True if mark_price has crossed the liquidation price.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return False

        if pos.side == "long":
            liquidated = mark_price <= pos.liquidation_price
        else:
            liquidated = mark_price >= pos.liquidation_price

        if liquidated:
            logger.warning(
                "LIQUIDATION triggered for %s %s at mark=%.4f (liq=%.4f)",
                pos.side,
                symbol,
                mark_price,
                pos.liquidation_price,
            )
        return liquidated

    def get_margin_info(self, symbol: str) -> dict | None:
        """Return margin details for a position, or None if no position."""
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        mark_price = pos.entry_price  # best available without update
        margin_ratio = (
            (pos.margin + pos.unrealized_pnl)
            / (mark_price * pos.quantity)
            if pos.quantity > 0
            else 0.0
        )
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "margin": pos.margin,
            "leverage": pos.leverage,
            "liquidationPrice": pos.liquidation_price,
            "unrealizedPnl": pos.unrealized_pnl,
            "marginRatio": margin_ratio,
            "cumulativeFunding": pos.cumulative_funding,
        }
