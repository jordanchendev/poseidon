"""LiquiditySweepStrategy -- three-stage maker ambush detection.

Implements the liquidity sweep strategy:
  Stage 1: Zone identification -- OIBuildup + SwingHigh/SwingLow + OIWAP cost-basis
  Stage 2: Sweep detection -- mandatory boolean gates + confirmatory weighted scoring
  Stage 3: Ambush placement -- Fibonacci extension entry with volatility-adaptive distance

Supports direction modes (long_only, short_only, bidirectional) and
stateful position tracking with cooldown after exit.

All numerical thresholds are config parameters (not hardcoded) so Phase 52
Optuna can search them.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class LiquiditySweepStrategy(BaseStrategy):
    """Three-stage maker ambush strategy for liquidity sweep events.

    Detects stop-loss cluster zones via OI accumulation + swing levels,
    confirms sweep events via wick/breakout/OI-drop gates + weighted scoring,
    and places limit orders at Fibonacci extension ambush points with
    volatility-adaptive distance.
    """

    name = "liquidity_sweep_strategy"
    strategy_type = StrategyType.RULE
    supports_backtest = True
    supports_live = False
    stateful = True

    def __init__(
        self,
        *,
        config: dict,
        instrument: InstrumentType = InstrumentType.PERPETUAL,
        strategy_id: UUID | None = None,
    ):
        # Top-level config
        self.name = config.get("name", "liquidity_sweep_strategy")
        self.symbol = config.get("symbol", "")
        self.market = config.get("market", "")
        self.interval = config.get("interval", "1h")
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()

        # Detection config (D-04, D-08, D-09)
        detection = config.get("detection", {})
        self._lookback_bars: int = detection.get("lookback_bars", 24)
        self._wick_ratio_min: float = detection.get("wick_ratio_min", 0.15)
        self._breakout_distance_min: float = detection.get("breakout_distance_min", 0.1)
        self._oi_buildup_min: float = detection.get("oi_buildup_min", 1.0)
        self._confirmation_threshold: float = detection.get("confirmation_threshold", 0.5)
        self._w_oi_drop: float = detection.get("w_oi_drop", 0.4)
        self._w_volume: float = detection.get("w_volume", 0.3)
        self._w_funding: float = detection.get("w_funding", 0.3)

        # Entry config (D-12, D-13)
        entry = config.get("entry", {})
        self._fib_level: float = entry.get("fib_level", 0.618)
        raw_multipliers = entry.get("atr_multipliers", {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0})
        # Ensure keys are int (JSON may deserialize as str)
        self._atr_multipliers: dict[int, float] = {
            int(k): float(v) for k, v in raw_multipliers.items()
        }

        # Exit config (D-20, D-21)
        exit_cfg = config.get("exit", {})
        self._cooldown_bars: int = exit_cfg.get("cooldown_bars", 4)
        self._max_holding_bars: int | None = exit_cfg.get("max_holding_bars", None)

        # Trailing stop config (D-07, D-08, D-09)
        trailing = config.get("trailing", {})
        self._trailing_activation_r: float = trailing.get("activation_r", 1.0)
        self._trail_atr_multiplier: float = trailing.get("atr_multiplier", 2.0)

        # Direction mode (D-16)
        self._direction_mode: str = config.get("direction_mode", "bidirectional")

        # Position state (D-18, D-20)
        self._position_direction: str | None = None
        self._bars_since_exit: int = 999  # starts high so first trade isn't blocked
        self._bars_in_position: int = 0

        # Trailing stop state (D-07)
        self._trailing_active: bool = False
        self._position_high_watermark: float | None = None
        self._position_low_watermark: float | None = None
        self._entry_price: float | None = None
        self._initial_stop_loss: float | None = None
        self._current_stop_loss: float | None = None

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        """Declare all required features for this strategy (D-03)."""
        return [
            ("swing_high", {"period": self._lookback_bars}),
            ("swing_low", {"period": self._lookback_bars}),
            ("oi_buildup", {"period": self._lookback_bars}),
            ("oi_change", {"period": 20}),
            ("oi_cost_basis", {"period": 168}),
            ("breakout_distance", {"period": self._lookback_bars, "atr_period": 14}),
            ("fib_extension", {"period": self._lookback_bars}),
            ("wick_ratio", {}),
            ("range_expansion", {"period": 14}),
            ("vol_regime", {}),
            ("atr", {"period": 14}),
            ("volume_ratio", {"period": 20}),
            ("funding_rate_daily", {}),
        ]

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate strategy against feature data and produce signals.

        Three-stage detection pipeline:
        1. Identify zones (OI buildup + swing levels)
        2. Detect sweep events (mandatory gates + confirmatory scoring)
        3. Calculate ambush entry (Fibonacci extension + vol-adaptive distance)
        """
        if features.empty or len(features) < self._lookback_bars:
            return []

        row = features.iloc[-1]
        signal_time = (
            features.index[-1]
            if isinstance(features.index[-1], datetime)
            else datetime.now(timezone.utc)
        )

        signals: list[Signal] = []

        # Increment cooldown counter
        self._bars_since_exit += 1

        # === IN POSITION: check trailing stop + exits ===
        if self._position_direction is not None:
            self._bars_in_position += 1

            # Max holding bars exit (D-21c)
            if (
                self._max_holding_bars is not None
                and self._bars_in_position > self._max_holding_bars
            ):
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        market=self.market,
                        instrument=self.instrument,
                        action=SignalAction.CLOSE,
                        confidence=1.0,
                        interval=self.interval,
                        signal_time=signal_time,
                        metadata={"exit_reason": "max_holding_bars"},
                    )
                )
                self._position_direction = None
                self._bars_since_exit = 0
                self._bars_in_position = 0
                return signals

            # Check trailing stop (ADV-01)
            close = float(row.get("close", 0.0))
            new_sl = self._check_trailing_stop(row, close)
            if new_sl is not None:
                return [
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        market=self.market,
                        instrument=self.instrument,
                        action=SignalAction.HOLD,
                        confidence=0.0,
                        interval=self.interval,
                        signal_time=signal_time,
                        metadata={"updated_stop_loss": new_sl},
                    )
                ]

            # SL/TP handled by BacktestRunner Phase A -- no strategy-side exit
            return []

        # === NOT IN POSITION: check cooldown then entries ===
        if self._bars_since_exit <= self._cooldown_bars:
            return []

        # Stage 1: Identify zones (D-05, D-06, D-07)
        zones = self._identify_zones(row)

        # Stage 2 & 3: Detect sweeps and calculate entries
        sweep_candidates: list[tuple[str, float, float]] = (
            []
        )  # (direction, score, zone_level)

        if self._direction_mode != "short_only" and "downward" in zones:
            triggered, score, zone_level = self._detect_sweep(row, "downward", zones)
            if triggered:
                sweep_candidates.append(("long", score, zone_level))

        if self._direction_mode != "long_only" and "upward" in zones:
            triggered, score, zone_level = self._detect_sweep(row, "upward", zones)
            if triggered:
                sweep_candidates.append(("short", score, zone_level))

        if not sweep_candidates:
            return []

        # If both directions trigger, prefer higher confirmation score (D-13 note)
        sweep_candidates.sort(key=lambda x: x[1], reverse=True)
        direction, score, zone_level = sweep_candidates[0]

        # Stage 3: Calculate entry price (D-12, D-13, D-14)
        entry_price, stop_loss, take_profit = self._calculate_entry_price(
            row, zone_level, direction
        )

        # Determine vol regime for metadata
        vol_regime = (
            int(row["vol_regime"])
            if not pd.isna(row.get("vol_regime", float("nan")))
            else 1
        )
        vol_multiplier = self._atr_multipliers.get(vol_regime, 1.0)

        # Emit signal (D-14, D-15)
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=SignalAction.LONG if direction == "long" else SignalAction.SHORT,
            confidence=min(score / self._confirmation_threshold, 1.0),
            order_type=OrderType.LIMIT,
            order_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            interval=self.interval,
            signal_time=signal_time,
            metadata={
                "sweep_type": "downward" if direction == "long" else "upward",
                "zone_level": zone_level,
                "confirmation_score": score,
                "vol_regime": vol_regime,
                "atr_multiplier": vol_multiplier,
            },
        )
        signals.append(signal)

        # Update position state (D-18) and trailing stop entry context
        self._position_direction = direction
        self._bars_in_position = 0
        self._entry_price = entry_price
        self._initial_stop_loss = stop_loss
        self._current_stop_loss = stop_loss
        self._trailing_active = False
        if direction == "long":
            self._position_high_watermark = entry_price
            self._position_low_watermark = None
        else:
            self._position_low_watermark = entry_price
            self._position_high_watermark = None

        logger.debug(
            "Sweep detected: %s at zone %.2f, score=%.3f, entry=%.2f, sl=%.2f, tp=%.2f",
            direction,
            zone_level,
            score,
            entry_price,
            stop_loss,
            take_profit,
        )

        return signals

    def _check_trailing_stop(self, row: pd.Series, close: float) -> float | None:
        """Check and update trailing stop if applicable (ADV-01, D-07/D-08/D-09).

        Trailing stop activates when unrealized profit reaches
        trailing_activation_r * 1R (where 1R = |entry - initial_stop|).
        Once active, SL trails via ATR-based distance from high/low watermark.
        SL only tightens, never loosens (D-08).

        Args:
            row: Current bar feature values.
            close: Current bar close price.

        Returns:
            New SL value if changed, None otherwise.
        """
        if self._entry_price is None or self._initial_stop_loss is None:
            return None
        if self._current_stop_loss is None:
            return None

        one_r = abs(self._entry_price - self._initial_stop_loss)
        if one_r == 0:
            return None

        atr = row.get("atr_14", 0.0)
        if pd.isna(atr):
            atr = 0.0
        atr = float(atr)

        old_sl = self._current_stop_loss

        if self._position_direction == "long":
            # Track high watermark
            if self._position_high_watermark is None:
                self._position_high_watermark = close
            else:
                self._position_high_watermark = max(self._position_high_watermark, close)

            unrealized = close - self._entry_price
            if unrealized >= one_r * self._trailing_activation_r:
                self._trailing_active = True

            if self._trailing_active and atr > 0:
                trail_sl = self._position_high_watermark - atr * self._trail_atr_multiplier
                # Only tighten (D-08): for long, SL can only move up
                self._current_stop_loss = max(trail_sl, self._current_stop_loss)

        elif self._position_direction == "short":
            # Track low watermark
            if self._position_low_watermark is None:
                self._position_low_watermark = close
            else:
                self._position_low_watermark = min(self._position_low_watermark, close)

            unrealized = self._entry_price - close
            if unrealized >= one_r * self._trailing_activation_r:
                self._trailing_active = True

            if self._trailing_active and atr > 0:
                trail_sl = self._position_low_watermark + atr * self._trail_atr_multiplier
                # Only tighten (D-08): for short, SL can only move down
                self._current_stop_loss = min(trail_sl, self._current_stop_loss)

        # Return new SL only if it changed
        if self._current_stop_loss != old_sl:
            return self._current_stop_loss
        return None

    def _identify_zones(self, row: pd.Series) -> dict:
        """Stage 1: Identify stop-loss cluster zones (D-05, D-06, D-07).

        Returns dict with keys 'downward' and/or 'upward', each containing
        {'level': float, 'oiwap_dist': float}.
        """
        swing_high = row.get(f"swing_high_{self._lookback_bars}", float("nan"))
        swing_low = row.get(f"swing_low_{self._lookback_bars}", float("nan"))
        oi_buildup = row.get(f"oi_buildup_{self._lookback_bars}", float("nan"))
        oiwap_distance = row.get("oiwap_distance_168", 0.0)

        # Handle NaN for oiwap_distance
        if pd.isna(oiwap_distance):
            oiwap_distance = 0.0

        zones: dict = {}

        # OI buildup must exceed threshold for zone to be active (D-06)
        if pd.isna(oi_buildup) or oi_buildup <= self._oi_buildup_min:
            return zones

        if not pd.isna(swing_low):
            zones["downward"] = {"level": float(swing_low), "oiwap_dist": oiwap_distance}
        if not pd.isna(swing_high):
            zones["upward"] = {"level": float(swing_high), "oiwap_dist": oiwap_distance}

        return zones

    def _detect_sweep(
        self, row: pd.Series, direction: str, zones: dict
    ) -> tuple[bool, float, float]:
        """Stage 2: Detect sweep event via mandatory gates + confirmatory scoring.

        Args:
            row: Current bar feature values.
            direction: 'downward' or 'upward'.
            zones: Output from _identify_zones().

        Returns:
            (triggered, score, zone_level) -- triggered is True if sweep confirmed.
        """
        if direction not in zones:
            return False, 0.0, 0.0

        zone_level = zones[direction]["level"]
        close = row.get("close", float("nan"))

        if pd.isna(close):
            return False, 0.0, 0.0

        close = float(close)

        # === Mandatory boolean gates (D-08) ===
        if direction == "downward":
            wick_ratio = row.get("wick_ratio_lower", float("nan"))
            breakout_dist = row.get(f"breakout_down_{self._lookback_bars}", float("nan"))
            # Reversal: close recovered above swing_low
            reversal = close > zone_level
        else:  # upward
            wick_ratio = row.get("wick_ratio_upper", float("nan"))
            breakout_dist = row.get(f"breakout_up_{self._lookback_bars}", float("nan"))
            # Reversal: close recovered below swing_high
            reversal = close < zone_level

        # NaN in any mandatory gate value -> fail
        if pd.isna(wick_ratio) or pd.isna(breakout_dist):
            return False, 0.0, 0.0

        wick_ratio = float(wick_ratio)
        breakout_dist = float(breakout_dist)

        # Gate checks
        if wick_ratio <= self._wick_ratio_min:
            return False, 0.0, 0.0
        if breakout_dist <= self._breakout_distance_min:
            return False, 0.0, 0.0
        if not reversal:
            return False, 0.0, 0.0

        # === Confirmatory weighted scoring (D-09, D-10) ===
        oi_zscore = row.get("oi_change_zscore_20", 0.0)
        if pd.isna(oi_zscore):
            oi_zscore = 0.0
        oi_drop_score = max(0.0, -float(oi_zscore))  # negative z-score -> positive score

        range_exp = row.get("range_expansion_14", 0.0)
        if pd.isna(range_exp):
            range_exp = 0.0
        volume_score = max(0.0, float(range_exp) - 1.0)  # above 1 = expansion

        funding = row.get("funding_rate_daily", 0.0)
        if pd.isna(funding):
            funding = 0.0
        funding = float(funding)

        if direction == "downward":
            # Positive funding = crowded long = supports downward sweep thesis
            funding_alignment = 1.0 if funding > 0 else 0.0
        else:
            # Negative funding = crowded short = supports upward sweep thesis
            funding_alignment = 1.0 if funding < 0 else 0.0

        score = (
            oi_drop_score * self._w_oi_drop
            + volume_score * self._w_volume
            + funding_alignment * self._w_funding
        )

        if score < self._confirmation_threshold:
            return False, 0.0, 0.0

        return True, score, zone_level

    def _calculate_entry_price(
        self, row: pd.Series, sweep_level: float, direction: str
    ) -> tuple[float, float, float]:
        """Stage 3: Calculate ambush entry, stop-loss, and take-profit prices.

        Uses Fibonacci extension from sweep level with volatility-adaptive
        distance (D-12, D-13, D-14).

        Returns:
            (entry_price, stop_loss_price, take_profit_price)
        """
        atr = row.get("atr_14", float("nan"))
        if pd.isna(atr):
            atr = 0.0
        atr = float(atr)

        vol_regime_raw = row.get("vol_regime", float("nan"))
        vol_regime = int(vol_regime_raw) if not pd.isna(vol_regime_raw) else 1
        vol_multiplier = self._atr_multipliers.get(vol_regime, 1.0)

        fib_distance = atr * self._fib_level * vol_multiplier

        oiwap = row.get("oiwap_168", float("nan"))

        if direction == "long":
            entry_price = sweep_level - fib_distance
            stop_loss = sweep_level - fib_distance * 2.0  # Beyond sweep, invalidation
            fib_target = sweep_level + fib_distance
            if not pd.isna(oiwap) and float(oiwap) > entry_price:
                take_profit = min(float(oiwap), fib_target)
            else:
                take_profit = fib_target
        else:  # short
            entry_price = sweep_level + fib_distance
            stop_loss = sweep_level + fib_distance * 2.0
            fib_target = sweep_level - fib_distance
            if not pd.isna(oiwap) and float(oiwap) < entry_price:
                take_profit = max(float(oiwap), fib_target)
            else:
                take_profit = fib_target

        return entry_price, stop_loss, take_profit

    def validate_config(self) -> bool:
        """Validate strategy configuration (D-02).

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not self.market:
            raise ValueError("market must be non-empty")
        if self._wick_ratio_min <= 0:
            raise ValueError("detection.wick_ratio_min must be > 0")
        if self._confirmation_threshold <= 0:
            raise ValueError("detection.confirmation_threshold must be > 0")
        if self._fib_level <= 0 or self._fib_level >= 2.0:
            raise ValueError("entry.fib_level must be > 0 and < 2.0")
        if self._direction_mode not in ("long_only", "short_only", "bidirectional"):
            raise ValueError(
                f"direction_mode must be one of: long_only, short_only, bidirectional; "
                f"got '{self._direction_mode}'"
            )
        return True

    def reset(self) -> None:
        """Reset position state and trailing stop state (D-18, D-20, ADV-01)."""
        self._position_direction = None
        self._bars_since_exit = 999
        self._bars_in_position = 0
        # Trailing stop state (D-07)
        self._trailing_active = False
        self._position_high_watermark = None
        self._position_low_watermark = None
        self._entry_price = None
        self._initial_stop_loss = None
        self._current_stop_loss = None
