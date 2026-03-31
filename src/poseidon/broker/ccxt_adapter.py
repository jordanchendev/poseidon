"""CCXT broker adapter for Binance perpetual futures.

Uses ccxt library to interact with Binance USDM perpetual contracts.
CCXT is lazily imported so the module loads on machines without ccxt installed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from poseidon.broker.base import BrokerAdapter
from poseidon.orders.schemas import Fill, Order

logger = logging.getLogger(__name__)


class CCXTBrokerAdapter(BrokerAdapter):
    """Binance perpetual futures via CCXT unified API.

    Creates an independent ccxt.binance instance configured for
    USDM perpetual swaps (defaultType='swap'). Supports isolated
    margin mode and per-symbol leverage control.

    Perp-specific methods (set_leverage, set_margin_mode, etc.) live
    on this subclass only -- they are NOT part of the BrokerAdapter ABC.
    """

    def __init__(
        self,
        api_key: str,
        secret: str,
        leverage: int = 3,
        testnet: bool = False,
    ):
        import ccxt

        self._exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        if testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("CCXT adapter initialized in testnet/sandbox mode")

        self._leverage = leverage
        self._fills: dict[str, list[Fill]] = {}
        logger.info(
            "CCXTBrokerAdapter created (leverage=%d, testnet=%s)",
            leverage,
            testnet,
        )

    # ------------------------------------------------------------------
    # BrokerAdapter ABC implementation
    # ------------------------------------------------------------------

    def login(self) -> bool:
        """Load exchange markets. Returns True on success."""
        try:
            self._exchange.load_markets()
            logger.info("CCXT markets loaded (%d symbols)", len(self._exchange.markets))
            return True
        except Exception:
            logger.exception("Failed to load CCXT markets")
            return False

    def place_order(self, order: Order) -> str:
        """Submit a market/limit order to Binance perps.

        Maps ``order.action`` ('buy'/'sell') to CCXT side and uses
        ``order.quantity`` (float) as the contract amount.
        """
        side = "buy" if order.action == "buy" else "sell"
        logger.info(
            "Placing %s %s order: %s qty=%.6f",
            side,
            order.order_type,
            order.symbol,
            order.quantity,
        )

        result = self._exchange.create_order(
            symbol=order.symbol,
            type=order.order_type,
            side=side,
            amount=order.quantity,
            price=order.price,
        )
        broker_order_id = str(result["id"])
        logger.info("Order placed, broker_order_id=%s", broker_order_id)

        # Immediately fetch order details for fill information
        try:
            fetched = self._exchange.fetch_order(broker_order_id, order.symbol)
            fills: list[Fill] = []
            if fetched.get("filled", 0) > 0:
                fills.append(
                    Fill(
                        order_id=order.id,
                        fill_price=float(fetched.get("average", fetched.get("price", 0))),
                        fill_quantity=float(fetched["filled"]),
                        fill_time=datetime.now(timezone.utc),
                        broker_fill_id=f"{broker_order_id}-fill",
                    )
                )
            self._fills[broker_order_id] = fills
        except Exception:
            logger.warning("Could not fetch fills for order %s", broker_order_id)
            self._fills[broker_order_id] = []

        return broker_order_id

    def query_fills(self, broker_order_id: str) -> list[Fill]:
        """Return cached fills for a given broker order ID."""
        return self._fills.get(broker_order_id, [])

    def query_positions(self) -> list[dict]:
        """Fetch open perpetual positions from Binance.

        Returns a list of dicts with unified position fields.
        """
        raw_positions = self._exchange.fetch_positions()
        positions: list[dict] = []
        for pos in raw_positions:
            contracts = float(pos.get("contracts", 0))
            if contracts == 0:
                continue
            positions.append(
                {
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side"),
                    "contracts": contracts,
                    "entryPrice": float(pos.get("entryPrice", 0)),
                    "markPrice": float(pos.get("markPrice", 0)),
                    "unrealizedPnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": int(pos.get("leverage", 1)),
                    "liquidationPrice": float(pos.get("liquidationPrice", 0)),
                    "marginRatio": float(pos.get("percentage", 0)),
                }
            )
        return positions

    def logout(self) -> None:
        """No-op -- CCXT has no persistent connection to close."""
        logger.debug("CCXTBrokerAdapter logout (no-op)")

    # ------------------------------------------------------------------
    # Perp-specific methods (NOT on BrokerAdapter ABC, per D-08)
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol on Binance."""
        self._exchange.set_leverage(leverage, symbol=symbol)
        logger.info("Leverage set to %dx for %s", leverage, symbol)

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> None:
        """Set margin mode (isolated/cross) for a symbol.

        Binance throws an error if the margin mode already matches
        the current setting -- we catch and log that gracefully.
        """
        try:
            self._exchange.set_margin_mode(mode, symbol=symbol)
            logger.info("Margin mode set to '%s' for %s", mode, symbol)
        except Exception as exc:
            # Binance returns error if mode is already set
            msg = str(exc).lower()
            if "already" in msg or "no need" in msg:
                logger.debug("Margin mode already '%s' for %s", mode, symbol)
            else:
                raise

    def query_funding_history(
        self, symbol: str, since: int | None = None
    ) -> list[dict]:
        """Fetch funding rate payment history for a symbol."""
        return self._exchange.fetch_funding_history(symbol, since=since)

    def get_margin_info(self, symbol: str) -> dict:
        """Get margin details for a single symbol position."""
        pos = self._exchange.fetch_position(symbol)
        return {
            "symbol": pos.get("symbol"),
            "margin": float(pos.get("collateral", 0)),
            "leverage": int(pos.get("leverage", 1)),
            "marginRatio": float(pos.get("percentage", 0)),
            "liquidationPrice": float(pos.get("liquidationPrice", 0)),
            "maintenanceMargin": float(pos.get("maintenanceMargin", 0)),
        }
