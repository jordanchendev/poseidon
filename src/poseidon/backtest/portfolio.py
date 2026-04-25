"""Backtest portfolio simulator with fill execution, fee/slippage, and equity tracking.

BacktestPortfolio is the core of the backtest engine's simulation layer.
It tracks cash, positions, completed trades, and an equity curve (per bar).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.pending_orders import FillEvent
from poseidon.signals.schemas import Signal, SignalAction


class SizingMode(StrEnum):
    """Position sizing mode for backtests."""

    FIXED_PCT = "fixed_pct"
    FIXED_NOTIONAL = "fixed_notional"
    VOL_TARGET = "vol_target"
    FIXED_RISK = "fixed_risk"


@dataclass(frozen=True)
class SizingConfig:
    """Configuration for position sizing in backtests.

    Attributes:
        mode: Sizing strategy to use.
        notional_pct: Fraction of initial capital per trade (fixed_notional mode).
        quantity_pct: Fraction of current cash per trade (fixed_pct mode).
        target_vol: Annualized target volatility (vol_target mode).
        vol_lookback: Lookback bars for realized volatility (vol_target mode).
        max_position_pct: Hard cap on position size as fraction of capital (vol_target mode).
        risk_pct: Fraction of equity risked per trade (fixed_risk mode).
        max_notional_pct: Hard cap on position notional as fraction of equity (fixed_risk mode).
    """

    mode: SizingMode = SizingMode.FIXED_NOTIONAL
    notional_pct: float = 0.1
    quantity_pct: float = 0.5
    target_vol: float = 0.15
    vol_lookback: int = 20
    max_position_pct: float = 0.3
    risk_pct: float = 0.01  # 1% of equity risked per trade (FIXED_RISK only)
    max_notional_pct: float = 0.3  # Hard cap as fraction of equity (FIXED_RISK only)


@dataclass
class TradeRecord:
    """A completed or in-progress trade record."""

    symbol: str
    action: str
    entry_time: datetime
    entry_price: float
    quantity: float
    fees: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float | None = None
    metadata: dict = field(default_factory=dict)


class BacktestPortfolio:
    """Virtual portfolio for backtesting with fee/slippage simulation.

    Tracks cash, positions, equity curve (per bar), and completed trades.
    """

    def __init__(
        self,
        initial_capital: float,
        cost_model: CostModel,
        sizing_config: SizingConfig | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_model = cost_model
        self.sizing_config = sizing_config or SizingConfig()
        self.cash: float = initial_capital
        self.positions: dict[str, dict] = {}  # key -> {side, quantity, entry_price, entry_time}
        self.equity_curve: list[tuple[datetime, float, float]] = []  # (time, equity, drawdown)
        self.trades: list[TradeRecord] = []
        self._peak_equity: float = initial_capital

    @property
    def equity(self) -> float:
        """Current portfolio equity (cash + mark-to-market positions).

        Without a current price, returns cash + positions at entry price.
        Use record_equity_point() for mark-to-market valuation.
        """
        position_value = sum(pos["quantity"] * pos["entry_price"] for pos in self.positions.values())
        return self.cash + position_value

    def execute_fill(self, signal: Signal, bar: pd.Series) -> TradeRecord | None:
        """Execute a simulated fill with fees and slippage.

        For LONG: opens a new position, deducts cash.
        For CLOSE: closes an existing position, adds cash, computes PnL.

        Args:
            signal: The trading signal to execute.
            bar: Current OHLCV bar with at least a 'close' column.

        Returns:
            TradeRecord for the fill, or None if fill cannot be executed.
        """
        price = float(bar["close"])
        key = f"{signal.market}:{signal.symbol}"

        if signal.action == SignalAction.LONG:
            return self._execute_long(signal, price, key)
        elif signal.action == SignalAction.CLOSE:
            return self._execute_close(signal, price, key)
        elif signal.action == SignalAction.SHORT:
            return self._execute_short(signal, price, key)
        return None

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to fill price.

        Buy side: price goes up (worse fill). Sell side: price goes down.
        Supports both percentage-based and tick-based slippage.
        Tick-based slippage is approximated as ticks * (price * 0.001) for
        markets without explicit tick size tables.
        """
        slippage = 0.0
        if self.cost_model.slippage_pct > 0:
            slippage += price * self.cost_model.slippage_pct
        if self.cost_model.slippage_ticks > 0:
            tick_size = getattr(self.cost_model, "tick_size", 0.0)
            tick_value = tick_size if tick_size > 0 else price * 0.001
            slippage += self.cost_model.slippage_ticks * tick_value

        if is_buy:
            price += slippage
        else:
            price -= slippage
        return price

    def _sizing_base(self) -> float:
        """Return the capital base for quantity calculation.

        fixed_pct uses current cash (compounding). All other modes use
        initial capital to avoid compounding distortion.
        """
        if self.sizing_config.mode == SizingMode.FIXED_PCT:
            return self.cash
        return self.initial_capital

    def _is_crypto_market(self, signal: Signal) -> bool:
        """Check if signal is for a crypto market (fractional quantities allowed)."""
        return signal.market in ("crypto_spot", "crypto_perp")

    def _execute_long(self, signal: Signal, price: float, key: str) -> TradeRecord | None:
        """Open a long position."""
        fill_price = self._apply_slippage(price, is_buy=True)
        quantity_pct = signal.quantity_pct or 0.1
        raw_quantity = self._sizing_base() * quantity_pct / fill_price
        quantity = raw_quantity if self._is_crypto_market(signal) else int(raw_quantity)
        if quantity <= 0:
            return None

        trade_value = fill_price * quantity
        fees = trade_value * self.cost_model.buy_commission_rate
        total_cost = trade_value + fees

        if total_cost > self.cash:
            return None

        self.cash -= total_cost
        self.positions[key] = {
            "side": "long",
            "quantity": quantity,
            "entry_price": fill_price,
            "entry_time": signal.signal_time,
            "entry_fees": fees,
            "stop_loss_price": signal.stop_loss_price,
            "take_profit_price": signal.take_profit_price,
        }

        trade = TradeRecord(
            symbol=signal.symbol,
            action="long",
            entry_time=signal.signal_time,
            entry_price=fill_price,
            quantity=quantity,
            fees=fees,
        )
        self.trades.append(trade)
        return trade

    def _execute_short(self, signal: Signal, price: float, key: str) -> TradeRecord | None:
        """Open a short position."""
        fill_price = self._apply_slippage(price, is_buy=False)
        quantity_pct = signal.quantity_pct or 0.1
        raw_quantity = self._sizing_base() * quantity_pct / fill_price
        quantity = raw_quantity if self._is_crypto_market(signal) else int(raw_quantity)
        if quantity <= 0:
            return None

        trade_value = fill_price * quantity
        fees = trade_value * self.cost_model.sell_commission_rate

        self.cash += trade_value - fees  # short: receive proceeds
        self.positions[key] = {
            "side": "short",
            "quantity": quantity,
            "entry_price": fill_price,
            "entry_time": signal.signal_time,
            "entry_fees": fees,
            "stop_loss_price": signal.stop_loss_price,
            "take_profit_price": signal.take_profit_price,
        }

        trade = TradeRecord(
            symbol=signal.symbol,
            action="short",
            entry_time=signal.signal_time,
            entry_price=fill_price,
            quantity=quantity,
            fees=fees,
        )
        self.trades.append(trade)
        return trade

    def _execute_close(self, signal: Signal, price: float, key: str) -> TradeRecord | None:
        """Close an existing position.

        Slippage direction depends on position side:
        - Long close = selling → slippage moves price down (is_buy=False)
        - Short close = buying to cover → slippage moves price up (is_buy=True)

        PnL includes both entry and exit fees for accurate round-trip cost.
        """
        if key not in self.positions:
            return None

        pos = self.positions.pop(key)
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]
        entry_fees = pos.get("entry_fees", 0.0)

        # Slippage direction: closing long = sell, closing short = buy
        is_buy = pos["side"] == "short"
        fill_price = self._apply_slippage(price, is_buy=is_buy)

        trade_value = fill_price * quantity
        exit_fees = trade_value * (self.cost_model.sell_commission_rate + self.cost_model.tax_rate)
        total_fees = entry_fees + exit_fees

        point_value = getattr(self.cost_model, "point_value", 1.0)
        if pos["side"] == "long":
            pnl = (fill_price - entry_price) * quantity * point_value - total_fees
            self.cash += trade_value - exit_fees
        else:
            # Short close: buy to cover
            pnl = (entry_price - fill_price) * quantity * point_value - total_fees
            self.cash -= trade_value + exit_fees

        trade = TradeRecord(
            symbol=signal.symbol,
            action="close",
            entry_time=pos["entry_time"],
            exit_time=signal.signal_time,
            entry_price=entry_price,
            exit_price=fill_price,
            quantity=quantity,
            fees=total_fees,
            pnl=pnl,
        )
        self.trades.append(trade)
        return trade

    def execute_sl_tp_exit(
        self,
        key: str,
        exit_price: float,
        exit_type: str,
        bar: pd.Series,
    ) -> TradeRecord | None:
        """Close a position via SL or TP trigger at the exact trigger price.

        No slippage is applied -- the trigger price IS the fill price (D-13).
        Exit fees use sell_commission_rate + tax_rate (same as _execute_close).

        Args:
            key: Position key (e.g., "crypto_perp:BTCUSDT").
            exit_price: The SL or TP trigger price (used directly as fill price).
            exit_type: "sl" or "tp" for trade record annotation.
            bar: Current OHLCV bar (used for exit time).

        Returns:
            TradeRecord for the exit, or None if position not found.
        """
        if key not in self.positions:
            return None

        pos = self.positions.pop(key)
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]
        entry_fees = pos.get("entry_fees", 0.0)

        trade_value = exit_price * quantity
        exit_fees = trade_value * (self.cost_model.sell_commission_rate + self.cost_model.tax_rate)
        total_fees = entry_fees + exit_fees

        point_value = getattr(self.cost_model, "point_value", 1.0)
        if pos["side"] == "long":
            pnl = (exit_price - entry_price) * quantity * point_value - total_fees
            self.cash += trade_value - exit_fees
        else:
            # Short close: buy to cover
            pnl = (entry_price - exit_price) * quantity * point_value - total_fees
            self.cash -= trade_value + exit_fees

        trade = TradeRecord(
            symbol=key.split(":")[-1] if ":" in key else key,
            action=f"close_{exit_type}",
            entry_time=pos["entry_time"],
            exit_time=bar.name,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            fees=total_fees,
            pnl=pnl,
        )
        self.trades.append(trade)
        return trade

    def execute_limit_fill(
        self,
        fill_event: FillEvent,
        bar: pd.Series,
    ) -> TradeRecord | None:
        """Execute a limit order fill using maker fee rate with no slippage.

        Fill price comes from the FillEvent (= order's limit price).
        Uses buy_commission_rate (maker rate 0.0002 for crypto_perp) for BOTH
        buy and sell sides, since limit orders are always maker fills.

        Args:
            fill_event: FillEvent from PendingOrderBook.check_fills().
            bar: Current OHLCV bar (used for time context).

        Returns:
            TradeRecord for the fill, or None if fill cannot be executed.
        """
        signal = fill_event.signal
        fill_price = fill_event.fill_price  # No slippage (D-21)
        key = f"{signal.market}:{signal.symbol}"
        fee_rate = self.cost_model.buy_commission_rate  # Maker rate (D-20)

        if signal.action == SignalAction.LONG:
            quantity_pct = signal.quantity_pct or 0.1
            raw_quantity = self._sizing_base() * quantity_pct / fill_price
            quantity = raw_quantity if self._is_crypto_market(signal) else int(raw_quantity)
            if quantity <= 0:
                return None

            trade_value = fill_price * quantity
            fees = trade_value * fee_rate
            total_cost = trade_value + fees

            if total_cost > self.cash:
                return None

            self.cash -= total_cost
            self.positions[key] = {
                "side": "long",
                "quantity": quantity,
                "entry_price": fill_price,
                "entry_time": signal.signal_time,
                "entry_fees": fees,
                "stop_loss_price": signal.stop_loss_price,
                "take_profit_price": signal.take_profit_price,
            }

            trade = TradeRecord(
                symbol=signal.symbol,
                action="long",
                entry_time=signal.signal_time,
                entry_price=fill_price,
                quantity=quantity,
                fees=fees,
            )
            self.trades.append(trade)
            return trade

        elif signal.action == SignalAction.SHORT:
            quantity_pct = signal.quantity_pct or 0.1
            raw_quantity = self._sizing_base() * quantity_pct / fill_price
            quantity = raw_quantity if self._is_crypto_market(signal) else int(raw_quantity)
            if quantity <= 0:
                return None

            trade_value = fill_price * quantity
            fees = trade_value * fee_rate

            self.cash += trade_value - fees  # short: receive proceeds
            self.positions[key] = {
                "side": "short",
                "quantity": quantity,
                "entry_price": fill_price,
                "entry_time": signal.signal_time,
                "entry_fees": fees,
                "stop_loss_price": signal.stop_loss_price,
                "take_profit_price": signal.take_profit_price,
            }

            trade = TradeRecord(
                symbol=signal.symbol,
                action="short",
                entry_time=signal.signal_time,
                entry_price=fill_price,
                quantity=quantity,
                fees=fees,
            )
            self.trades.append(trade)
            return trade

        # CLOSE action is not expected from limit fills
        return None

    def record_equity_point(self, time: datetime, current_price: float) -> None:
        """Record mark-to-market equity and drawdown at a given time.

        Args:
            time: Timestamp for this equity point.
            current_price: Current market price for mark-to-market valuation.

        Short positions are valued as (entry_price - current_price) * quantity,
        reflecting profit when price drops and loss when price rises (D-12).
        """
        point_value = getattr(self.cost_model, "point_value", 1.0)
        position_value = 0.0
        for pos in self.positions.values():
            entry = pos["entry_price"]
            qty = pos["quantity"]
            if pos.get("side") == "short":
                unrealized = (entry - current_price) * qty * point_value
            else:
                unrealized = (current_price - entry) * qty * point_value
            # Position notional at entry (cash already deducted this amount)
            position_value += entry * qty + unrealized
        current_equity = self.cash + position_value

        self._peak_equity = max(self._peak_equity, current_equity)
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0.0

        self.equity_curve.append((time, current_equity, drawdown))
