"""Backtest portfolio simulator with fill execution, fee/slippage, and equity tracking.

BacktestPortfolio is the core of the backtest engine's simulation layer.
It tracks cash, positions, completed trades, and an equity curve (per bar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.signals.schemas import Signal, SignalAction


class SizingMode(str, Enum):
    """Position sizing mode for backtests."""

    FIXED_PCT = "fixed_pct"
    FIXED_NOTIONAL = "fixed_notional"
    VOL_TARGET = "vol_target"


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
    """

    mode: SizingMode = SizingMode.FIXED_NOTIONAL
    notional_pct: float = 0.1
    quantity_pct: float = 0.5
    target_vol: float = 0.15
    vol_lookback: int = 20
    max_position_pct: float = 0.3


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
        position_value = sum(
            pos["quantity"] * pos["entry_price"] for pos in self.positions.values()
        )
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
            # Approximate tick value as 0.1% of price (conservative estimate)
            tick_value = price * 0.001
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

    def _execute_long(self, signal: Signal, price: float, key: str) -> TradeRecord | None:
        """Open a long position."""
        fill_price = self._apply_slippage(price, is_buy=True)
        quantity_pct = signal.quantity_pct or 0.1
        quantity = int(self._sizing_base() * quantity_pct / fill_price)
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
        quantity = int(self._sizing_base() * quantity_pct / fill_price)
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

        if pos["side"] == "long":
            pnl = (fill_price - entry_price) * quantity - total_fees
            self.cash += trade_value - exit_fees
        else:
            # Short close: buy to cover
            pnl = (entry_price - fill_price) * quantity - total_fees
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

    def record_equity_point(self, time: datetime, current_price: float) -> None:
        """Record mark-to-market equity and drawdown at a given time.

        Args:
            time: Timestamp for this equity point.
            current_price: Current market price for mark-to-market valuation.
        """
        position_value = sum(
            pos["quantity"] * current_price for pos in self.positions.values()
        )
        current_equity = self.cash + position_value

        self._peak_equity = max(self._peak_equity, current_equity)
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0.0

        self.equity_curve.append((time, current_equity, drawdown))
