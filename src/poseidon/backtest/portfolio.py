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
    FIXED_CONTRACTS = "fixed_contracts"


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
        contracts: Fixed number of contracts per entry (fixed_contracts mode;
            futures research where notional-based sizing would drift with price).
    """

    mode: SizingMode = SizingMode.FIXED_NOTIONAL
    notional_pct: float = 0.1
    quantity_pct: float = 0.5
    target_vol: float = 0.15
    vol_lookback: int = 20
    max_position_pct: float = 0.3
    risk_pct: float = 0.01  # 1% of equity risked per trade (FIXED_RISK only)
    max_notional_pct: float = 0.3  # Hard cap as fraction of equity (FIXED_RISK only)
    contracts: int = 1  # FIXED_CONTRACTS only


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

    Two cash models, picked by ``cost_model.point_value``:

    - ``point_value == 1.0`` (stock, ETF, crypto spot/perp): fully funded.
      Entry moves ``price * quantity`` through cash, exit moves it back.
    - ``point_value != 1.0`` (contract-multiplier futures such as TX=200,
      MTX=50): margined. Notional (``price * quantity * point_value``) never
      moves through cash; entry and exit only charge fees, and the exit
      settles ``price_diff * quantity * point_value`` in cash. Mark-to-market
      equity is ``cash + unrealized``, so the equity change over a round trip
      equals ``TradeRecord.pnl``. Fees and tax are charged on the full
      notional in both models. Tax is charged on the exit leg only unless
      ``cost_model.tax_on_entry``; ``cost_model.commission_per_contract`` adds
      a fixed fee per unit on every leg (both default off).
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
    def point_value(self) -> float:
        """Currency value of one price point per unit (1.0 for stock/crypto)."""
        return float(getattr(self.cost_model, "point_value", 1.0))

    @property
    def is_margined(self) -> bool:
        """True for contract-multiplier instruments whose notional is not funded in cash."""
        return self.point_value != 1.0

    def _notional(self, price: float, quantity: float) -> float:
        """Currency notional of ``quantity`` units at ``price`` (fee/tax base)."""
        return price * quantity * self.point_value

    def _entry_fees(self, notional: float, quantity: float, rate: float) -> float:
        """Entry-leg fees: commission rate, optional entry tax, optional per-contract fee."""
        fees = notional * rate
        if getattr(self.cost_model, "tax_on_entry", False):
            fees += notional * self.cost_model.tax_rate
        return fees + abs(quantity) * getattr(self.cost_model, "commission_per_contract", 0.0)

    def _position_value(self, pos: dict, price: float) -> float:
        """Value of an open position added on top of cash for equity.

        Fully funded: entry notional (already deducted from cash) plus
        unrealized PnL. Margined: unrealized PnL only.
        """
        entry = pos["entry_price"]
        qty = pos["quantity"]
        if pos.get("side") == "short":
            unrealized = (entry - price) * qty * self.point_value
        else:
            unrealized = (price - entry) * qty * self.point_value
        if self.is_margined:
            return unrealized
        return entry * qty + unrealized

    @property
    def equity(self) -> float:
        """Current portfolio equity (cash + mark-to-market positions).

        Without a current price, returns cash + positions at entry price.
        Use record_equity_point() for mark-to-market valuation.
        """
        position_value = sum(self._position_value(pos, pos["entry_price"]) for pos in self.positions.values())
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

    def _resolve_quantity(self, signal: Signal, fill_price: float) -> float:
        """Quantity for an entry: fixed contracts, or notional-derived (int for non-crypto).

        ``quantity_pct`` is a fraction of the sizing base in currency notional, so a unit
        costs ``fill_price * point_value`` (one TX contract at 20,000 points is 4M TWD).
        """
        if self.sizing_config.mode == SizingMode.FIXED_CONTRACTS:
            return float(self.sizing_config.contracts)
        quantity_pct = signal.quantity_pct or 0.1
        raw_quantity = self._sizing_base() * quantity_pct / (fill_price * self.point_value)
        return raw_quantity if self._is_crypto_market(signal) else int(raw_quantity)

    def _execute_long(self, signal: Signal, price: float, key: str) -> TradeRecord | None:
        """Open a long position."""
        fill_price = self._apply_slippage(price, is_buy=True)
        quantity = self._resolve_quantity(signal, fill_price)
        if quantity <= 0:
            return None

        notional = self._notional(fill_price, quantity)
        fees = self._entry_fees(notional, quantity, self.cost_model.buy_commission_rate)
        total_cost = fees if self.is_margined else notional + fees

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
        quantity = self._resolve_quantity(signal, fill_price)
        if quantity <= 0:
            return None

        notional = self._notional(fill_price, quantity)
        fees = self._entry_fees(notional, quantity, self.cost_model.sell_commission_rate)

        # Fully funded short receives the proceeds; margined only pays fees.
        self.cash += (0.0 if self.is_margined else notional) - fees
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

        # Slippage direction: closing long = sell, closing short = buy
        is_buy = pos["side"] == "short"
        fill_price = self._apply_slippage(price, is_buy=is_buy)

        total_fees, pnl = self._settle_close(pos, fill_price)

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

    def _settle_close(self, pos: dict, exit_price: float) -> tuple[float, float]:
        """Move cash for closing ``pos`` at ``exit_price``.

        Exit fees are (sell commission + tax) on the exit notional plus any
        per-contract commission.

        Returns:
            (total_fees, pnl) where pnl is net of entry and exit fees.
        """
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]
        notional = self._notional(exit_price, quantity)
        exit_fees = notional * (self.cost_model.sell_commission_rate + self.cost_model.tax_rate)
        exit_fees += abs(quantity) * getattr(self.cost_model, "commission_per_contract", 0.0)
        total_fees = pos.get("entry_fees", 0.0) + exit_fees

        if pos["side"] == "long":
            gross = (exit_price - entry_price) * quantity * self.point_value
        else:
            gross = (entry_price - exit_price) * quantity * self.point_value
        pnl = gross - total_fees

        if self.is_margined:
            self.cash += gross - exit_fees
        elif pos["side"] == "long":
            self.cash += notional - exit_fees
        else:
            # Short close: buy to cover
            self.cash -= notional + exit_fees
        return total_fees, pnl

    def execute_sl_tp_exit(
        self,
        key: str,
        exit_price: float,
        exit_type: str,
        bar: pd.Series,
    ) -> TradeRecord | None:
        """Close a position via SL or TP trigger at the exact trigger price.

        No slippage is applied -- the trigger price IS the fill price.
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

        total_fees, pnl = self._settle_close(pos, exit_price)

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
        fill_price = fill_event.fill_price  # No slippage
        key = f"{signal.market}:{signal.symbol}"
        fee_rate = self.cost_model.buy_commission_rate  # Maker rate

        if signal.action == SignalAction.LONG:
            quantity = self._resolve_quantity(signal, fill_price)
            if quantity <= 0:
                return None

            notional = self._notional(fill_price, quantity)
            fees = self._entry_fees(notional, quantity, fee_rate)
            total_cost = fees if self.is_margined else notional + fees

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
            quantity = self._resolve_quantity(signal, fill_price)
            if quantity <= 0:
                return None

            notional = self._notional(fill_price, quantity)
            fees = self._entry_fees(notional, quantity, fee_rate)

            # Fully funded short receives the proceeds; margined only pays fees.
            self.cash += (0.0 if self.is_margined else notional) - fees
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

        Unrealized PnL is ``price_diff * quantity * point_value`` (short:
        entry - current). See the class docstring for the two cash models.
        """
        position_value = sum(self._position_value(pos, current_price) for pos in self.positions.values())
        current_equity = self.cash + position_value

        self._peak_equity = max(self._peak_equity, current_equity)
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0.0

        self.equity_curve.append((time, current_equity, drawdown))
