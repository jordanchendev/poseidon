"""StructuralReversalStrategy -- 2-3 condition limit order reversal.

Implements a structural reversal thesis using IC-validated features:
  - Primary: oiwap_distance_168 (IC=-0.115) crosses threshold
  - Confirmation: cascade_direction (IC=0.034) agrees
  - Optional: cvd_change_20 (IC=-0.027) confirms volume delta

Limit price derived from OIWAP +/- (ATR_multiplier * ATR_14).
ATR trailing stop, time-based expiry, 20% MaxDD enforcement.

Designed for 4H crypto_perp (BTCUSDT/ETHUSDT) with pessimistic fill model.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel

from poseidon.signals.schemas import InstrumentType, OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy

logger = logging.getLogger(__name__)


class StructuralReversalConfig(BaseModel):
    """Configuration for StructuralReversalStrategy.

    All parameters match CONTEXT.md decisions and v15_decision_gate.yaml.
    """

    strategy: str = "structural_reversal"
    name: str = "Structural Reversal 4H"
    market: str = "crypto_perp"
    interval: str = "4h"
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]

    # Entry conditions
    oiwap_threshold: float = 3.0  # D-07: oiwap_distance_168 threshold
    use_cvd_filter: bool = False  # D-09: toggle for 3-condition variant

    # Limit price derivation (D-12, D-13)
    atr_multiplier: float = 1.0  # optimizable: 0.5-2.0

    # Exit parameters (D-16, D-17, D-18)
    stop_atr_multiplier: float = 3.0  # optimizable: 2.0-5.0
    max_holding_bars: int = 40  # ~7 days at 4H
    order_expiry_bars: int = 6  # 24h at 4H

    # Risk (D-25)
    max_drawdown: float = 0.20  # 20% portfolio-level hard stop


@register_strategy
class StructuralReversalStrategy(BaseStrategy):
    """2-3 condition limit order reversal strategy using structural price levels.

    Detects mean-reversion setups when price deviates significantly from
    OI-weighted average cost basis (OIWAP), confirmed by micro-structure
    cascade signals. Places limit orders at OIWAP +/- ATR offset.

    Stateful: tracks position for trailing stop and time expiry.
    MaxDD enforcement via set_equity() / _check_drawdown().
    """

    name = "structural_reversal"
    strategy_type = StrategyType.RULE
    market = "crypto_perp"
    interval = "4h"
    supports_backtest = True
    supports_live = False
    stateful = True

    def __init__(
        self,
        config: StructuralReversalConfig,
        symbol: str = "BTCUSDT",
        strategy_id: UUID | None = None,
    ):
        self.config = config
        self.name = config.name
        self.symbol = symbol
        self.market = config.market
        self.interval = config.interval
        self.strategy_id = strategy_id or uuid4()

        # Position tracking state
        self._in_position: bool = False
        self._position_side: str | None = None  # "long" or "short"
        self._entry_price: float | None = None
        self._bars_in_position: int = 0
        self._current_stop_loss: float | None = None
        self._position_high_watermark: float | None = None  # for long trailing
        self._position_low_watermark: float | None = None  # for short trailing

        # MaxDD tracking (D-25)
        self._peak_equity: float | None = None
        self._current_equity: float | None = None

        # ATR at fill time (for on_fill stop calculation)
        self._fill_atr: float | None = None

    def reset(self) -> None:
        """Reset all stateful fields for WFE window isolation.

        CRITICAL: WalkForwardAnalyzer calls reset() between IS and OOS windows
        (walk_forward.py line 256). Missing fields cause AttributeError.
        """
        self._in_position = False
        self._position_side = None
        self._entry_price = None
        self._bars_in_position = 0
        self._current_stop_loss = None
        self._position_high_watermark = None
        self._position_low_watermark = None
        self._peak_equity = None
        self._current_equity = None
        self._fill_atr = None

    def set_equity(self, equity: float) -> None:
        """Update equity tracking for MaxDD enforcement (D-25).

        Called by backtest script before each evaluate().
        """
        self._current_equity = equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def _check_drawdown(self) -> bool:
        """Returns True if current drawdown exceeds max_drawdown threshold.

        D-25: 20% portfolio-level drawdown hard stop.
        D-26: When breached, suppress new entries.
        """
        if self._peak_equity is None or self._current_equity is None:
            return False
        if self._peak_equity <= 0:
            return False
        dd = 1.0 - (self._current_equity / self._peak_equity)
        return dd > self.config.max_drawdown

    def _check_entry(self, row: pd.Series) -> tuple[str | None, float | None]:
        """Check entry conditions. Returns (direction, limit_price) or (None, None).

        D-07: Primary -- oiwap_distance_168 crosses threshold
        D-08: Confirmation -- cascade_direction agrees
        D-09: Optional -- cvd_change_20 confirms (if use_cvd_filter=True)
        D-12: Limit price = OIWAP +/- (ATR_multiplier * ATR_14)
        """
        oiwap_dist = row.get("oiwap_distance_168")
        cascade_dir = row.get("cascade_direction")
        cvd_change = row.get("cvd_change_20")
        atr = row.get("atr_14")
        oiwap = row.get("oiwap_168")

        if pd.isna(oiwap_dist) or pd.isna(cascade_dir) or pd.isna(atr) or pd.isna(oiwap):
            return None, None

        # D-07: Primary -- oiwap_distance below negative threshold = bullish
        if oiwap_dist < -self.config.oiwap_threshold:
            direction = "long"
        elif oiwap_dist > self.config.oiwap_threshold:
            direction = "short"
        else:
            return None, None

        # D-08: Confirmation -- cascade_direction must agree with direction
        if direction == "long" and cascade_dir <= 0:
            return None, None
        if direction == "short" and cascade_dir >= 0:
            return None, None

        # D-09: Optional CVD filter (Variation 2)
        if self.config.use_cvd_filter:
            if pd.isna(cvd_change):
                return None, None
            # IC is negative: negative cvd_change = bullish, positive = bearish
            if direction == "long" and cvd_change >= 0:
                return None, None
            if direction == "short" and cvd_change <= 0:
                return None, None

        # D-12: Limit price = OIWAP +/- (ATR_multiplier * ATR_14)
        oiwap_val = float(oiwap)
        atr_val = float(atr)
        if direction == "long":
            limit_price = oiwap_val - (self.config.atr_multiplier * atr_val)
        else:
            limit_price = oiwap_val + (self.config.atr_multiplier * atr_val)

        return direction, limit_price

    def _compute_trailing_stop(self, row: pd.Series) -> float | None:
        """Returns new SL if tighter than current, else None. SL only tightens.

        D-16: ATR trailing stop from high/low watermark.
        D-17: stop_atr_multiplier controls distance.
        """
        if self._entry_price is None or self._current_stop_loss is None:
            return None

        atr = float(row.get("atr_14", 0.0))
        if pd.isna(atr) or atr == 0:
            return None

        close = float(row.get("close", 0.0))

        if self._position_side == "long":
            # Track high watermark
            if self._position_high_watermark is None:
                self._position_high_watermark = close
            else:
                self._position_high_watermark = max(self._position_high_watermark, close)
            new_sl = self._position_high_watermark - self.config.stop_atr_multiplier * atr
            # Only update if tighter (higher for longs)
            if new_sl > self._current_stop_loss:
                return new_sl
        else:  # short
            if self._position_low_watermark is None:
                self._position_low_watermark = close
            else:
                self._position_low_watermark = min(self._position_low_watermark, close)
            new_sl = self._position_low_watermark + self.config.stop_atr_multiplier * atr
            # Only update if tighter (lower for shorts)
            if new_sl < self._current_stop_loss:
                return new_sl

        return None

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate strategy against feature data and produce signals.

        Structure:
        1. If in position: trailing stop update + time-based expiry
        2. If not in position: MaxDD check + entry conditions + limit order
        """
        if features.empty:
            return []

        row = features.iloc[-1]
        signal_time = (
            features.index[-1]
            if isinstance(features.index[-1], datetime)
            else datetime.now(timezone.utc)
        )

        # === IN POSITION: trailing stop update + time-based expiry ===
        if self._in_position:
            self._bars_in_position += 1

            # D-18: time-based expiry
            if self._bars_in_position >= self.config.max_holding_bars:
                signal = Signal(
                    strategy_id=self.strategy_id,
                    symbol=self.symbol,
                    market=self.market,
                    instrument=InstrumentType.PERPETUAL,
                    action=SignalAction.CLOSE,
                    confidence=1.0,
                    order_type=OrderType.MARKET,
                    interval=self.interval,
                    signal_time=signal_time,
                    metadata={"exit_reason": "max_holding_bars"},
                )
                self.on_close()
                return [signal]

            # D-16: ATR trailing stop update
            new_sl = self._compute_trailing_stop(row)
            if new_sl is not None and new_sl != self._current_stop_loss:
                self._current_stop_loss = new_sl
                return [
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        market=self.market,
                        instrument=InstrumentType.PERPETUAL,
                        action=SignalAction.HOLD,
                        confidence=0.0,
                        interval=self.interval,
                        signal_time=signal_time,
                        metadata={"updated_stop_loss": new_sl},
                    )
                ]
            return []

        # === NOT IN POSITION: check entry conditions ===

        # D-26: MaxDD gate -- suppress new entries when drawdown exceeds threshold
        if self._check_drawdown():
            return []

        direction, limit_price = self._check_entry(row)
        if direction is None or limit_price is None:
            return []

        # D-12: stop loss at limit_price +/- stop_atr_multiplier * ATR
        atr = float(row.get("atr_14", 0.0))
        if direction == "long":
            stop_loss = limit_price - self.config.stop_atr_multiplier * atr
        else:
            stop_loss = limit_price + self.config.stop_atr_multiplier * atr

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=InstrumentType.PERPETUAL,
            action=SignalAction.LONG if direction == "long" else SignalAction.SHORT,
            confidence=0.5,  # fixed for rule-based strategies
            order_type=OrderType.LIMIT,
            order_price=limit_price,
            stop_loss_price=stop_loss,
            interval=self.interval,
            signal_time=signal_time,
        )

        return [signal]

    def validate_config(self) -> bool:
        """Validate strategy configuration.

        Returns True if valid, raises ValueError otherwise.
        """
        if not self.symbol:
            raise ValueError("StructuralReversalStrategy requires a symbol")
        if not self.market:
            raise ValueError("StructuralReversalStrategy requires a market")
        if self.config.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")
        if self.config.stop_atr_multiplier <= 0:
            raise ValueError("stop_atr_multiplier must be positive")
        return True

    def on_fill(self, fill_price: float, side: str, atr: float = 0.0) -> None:
        """Called by backtest script when PendingOrderBook confirms a fill.

        Sets position state and initializes trailing stop.
        """
        self._in_position = True
        self._position_side = side
        self._entry_price = fill_price
        self._bars_in_position = 0
        self._fill_atr = atr

        # Initial stop loss
        if side == "long":
            self._current_stop_loss = fill_price - self.config.stop_atr_multiplier * atr
            self._position_high_watermark = fill_price
            self._position_low_watermark = None
        else:
            self._current_stop_loss = fill_price + self.config.stop_atr_multiplier * atr
            self._position_low_watermark = fill_price
            self._position_high_watermark = None

    def on_close(self) -> None:
        """Reset position state after exit."""
        self._in_position = False
        self._position_side = None
        self._entry_price = None
        self._bars_in_position = 0
        self._current_stop_loss = None
        self._position_high_watermark = None
        self._position_low_watermark = None
        self._fill_atr = None
