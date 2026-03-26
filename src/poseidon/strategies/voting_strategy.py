"""VotingStrategy -- multi-signal voting with ATR trailing stop.

Wraps N child conditions (sub-signals). Emits LONG when >= min_votes
conditions are true, SHORT when >= bear_min_votes bear conditions are true.
Tracks position state for ATR trailing stop exits, RSI mean-reversion exits,
signal flip exits, and cooldown mechanism.

Position sizing note (D-10):
    VotingStrategy sets quantity_pct=0.08 on long entry signals and
    bear_position_pct on short entry signals as strategy-level sizing intent.
    However, BacktestRunner._compute_sizing() controls final sizing based
    on its SizingConfig:
      - FIXED_PCT mode: uses signal.quantity_pct directly
      - FIXED_NOTIONAL mode: overrides with SizingConfig.notional_pct
    Per D-10, callers MUST instantiate BacktestRunner with
    SizingConfig(mode=FIXED_NOTIONAL, notional_pct=0.08) for correct
    fixed 8% position sizing behavior.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.dsl.executor import evaluate_condition

logger = logging.getLogger(__name__)


VOTING_FEATURE_SPECS: list[tuple[str, dict]] = [
    ("ema", {"period": 7}),
    ("ema", {"period": 26}),
    ("rsi", {"period": 8}),
    ("macd", {"fast_period": 14, "slow_period": 23, "signal_period": 9}),
    ("bollinger", {"period": 20, "num_std": 2.0}),
    ("atr", {"period": 14}),
    ("cum_return", {"period": 6}),
    ("cum_return", {"period": 12}),
    ("returns", {}),
]


class VotingStrategy(BaseStrategy):
    """Multi-signal voting strategy with ATR trailing stop exit.

    Supports bidirectional trading: LONG via bull sub_signals,
    SHORT via bear_sub_signals. Exit mechanisms include ATR trailing stop,
    RSI mean-reversion, signal flip, and 2-bar cooldown.
    """

    strategy_type = StrategyType.VOTING

    def __init__(
        self,
        *,
        config: dict,
        atr_multiplier: float = 5.5,
        atr_period: int = 14,
        instrument: InstrumentType = InstrumentType.SPOT,
        strategy_id: UUID | None = None,
    ):
        self.name = config.get("name", "voting_strategy")
        self.symbol = config.get("symbol", "")
        self.market = config.get("market", "")
        self.interval = config.get("interval", "1d")
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()

        # Bull (long) signals
        self._sub_signals: list[dict] = config.get("sub_signals", [])
        self._min_votes: int = config.get("min_votes", 4)
        self._position_pct: float = config.get("position_pct", 0.08)

        # Bear (short) signals
        self._bear_sub_signals: list[dict] = config.get("bear_sub_signals", [])
        self._bear_min_votes: int = config.get("bear_min_votes", 4)
        self._bear_position_pct: float = config.get("bear_position_pct", 0.06)

        # ATR parameters
        self._atr_multiplier = atr_multiplier
        self._atr_period = atr_period

        # Position state (replaces _in_position bool)
        self._position_direction: str | None = None  # "long", "short", or None
        self._position_high_watermark: float | None = None  # for long trailing stop
        self._position_low_watermark: float | None = None   # for short trailing stop

        # Cooldown state
        self._bars_since_exit: int = 999  # starts high so first trade isn't blocked
        self._last_exit_direction: str | None = None

        # Deduplicate sub-signal warnings
        self._warned_signals: set[str] = set()

    @property
    def _in_position(self) -> bool:
        """Backward compatibility: True if in any position."""
        return self._position_direction is not None

    @_in_position.setter
    def _in_position(self, value: bool) -> None:
        """Backward compatibility setter."""
        if not value:
            self._position_direction = None

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate voting conditions and exits against feature data."""
        if features.empty:
            return []

        row_idx = len(features) - 1
        close = float(features.iloc[row_idx]["close"])
        signals: list[Signal] = []

        signal_time = (
            features.index[row_idx]
            if isinstance(features.index[row_idx], datetime)
            else datetime.now(timezone.utc)
        )

        # Increment cooldown counter
        self._bars_since_exit += 1

        # === EXIT CHECKS (only if in position) ===
        if self._position_direction is not None:
            exit_signal = self._check_exits(features, row_idx, close, signal_time)
            if exit_signal:
                signals.append(exit_signal)
                return signals

        # === ENTRY CHECKS (only if not in position) ===
        if self._position_direction is None:
            # Count bull votes
            bull_votes = self._count_votes(self._sub_signals, features, row_idx)
            bear_votes = (
                self._count_votes(self._bear_sub_signals, features, row_idx)
                if self._bear_sub_signals
                else 0
            )

            # Check cooldown: cannot re-enter same direction within 2 bars of exit
            bull_blocked = (
                self._last_exit_direction == "long" and self._bars_since_exit <= 2
            )
            bear_blocked = (
                self._last_exit_direction == "short" and self._bars_since_exit <= 2
            )

            if bull_votes >= self._min_votes and not bull_blocked:
                confidence = (
                    bull_votes / len(self._sub_signals) if self._sub_signals else 0.0
                )
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        market=self.market,
                        instrument=self.instrument,
                        action=SignalAction.LONG,
                        confidence=confidence,
                        quantity_pct=self._position_pct,
                        interval=self.interval,
                        signal_time=signal_time,
                        metadata={
                            "vote_count": bull_votes,
                            "min_votes": self._min_votes,
                        },
                    )
                )
                self._position_direction = "long"
                self._position_high_watermark = close
            elif (
                bear_votes >= self._bear_min_votes
                and not bear_blocked
                and self._bear_sub_signals
            ):
                confidence = (
                    bear_votes / len(self._bear_sub_signals)
                    if self._bear_sub_signals
                    else 0.0
                )
                signals.append(
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        market=self.market,
                        instrument=self.instrument,
                        action=SignalAction.SHORT,
                        confidence=confidence,
                        quantity_pct=self._bear_position_pct,
                        interval=self.interval,
                        signal_time=signal_time,
                        metadata={
                            "vote_count": bear_votes,
                            "bear_min_votes": self._bear_min_votes,
                        },
                    )
                )
                self._position_direction = "short"
                self._position_low_watermark = close

        logger.debug(
            "VotingStrategy '%s': position=%s | in_position=%s",
            self.name,
            self._position_direction,
            self._in_position,
        )
        return signals

    def _check_exits(
        self,
        features: pd.DataFrame,
        row_idx: int,
        close: float,
        signal_time: datetime,
    ) -> Signal | None:
        """Check exit conditions in priority order: ATR > RSI > signal flip."""
        atr_col = f"atr_{self._atr_period}"
        atr_val = float(features.iloc[row_idx][atr_col])

        # Priority 1: ATR trailing stop (D-04)
        if self._position_direction == "long":
            self._position_high_watermark = max(
                self._position_high_watermark, close  # type: ignore[arg-type]
            )
            stop_level = self._position_high_watermark - self._atr_multiplier * atr_val
            if close < stop_level:
                return self._emit_exit(
                    "atr_trailing_stop",
                    signal_time,
                    hwm=self._position_high_watermark,
                    stop_level=stop_level,
                )
        elif self._position_direction == "short":
            self._position_low_watermark = min(
                self._position_low_watermark, close  # type: ignore[arg-type]
            )
            stop_level = self._position_low_watermark + self._atr_multiplier * atr_val
            if close > stop_level:
                return self._emit_exit(
                    "atr_trailing_stop",
                    signal_time,
                    lwm=self._position_low_watermark,
                    stop_level=stop_level,
                )

        # Priority 2: RSI mean-reversion exit (D-01)
        rsi_period = self._get_rsi_period()
        rsi_col = f"rsi_{rsi_period}"
        if rsi_col in features.columns:
            rsi_val = float(features.iloc[row_idx][rsi_col])
            if self._position_direction == "long" and rsi_val > 69:
                return self._emit_exit("rsi_exit", signal_time, rsi=rsi_val)
            elif self._position_direction == "short" and rsi_val < 31:
                return self._emit_exit("rsi_exit", signal_time, rsi=rsi_val)

        # Priority 3: Signal flip (D-02)
        if self._position_direction == "long" and self._bear_sub_signals:
            bear_votes = self._count_votes(
                self._bear_sub_signals, features, row_idx
            )
            if bear_votes >= self._bear_min_votes:
                return self._emit_exit(
                    "signal_flip", signal_time, opposing_votes=bear_votes
                )
        elif self._position_direction == "short":
            bull_votes = self._count_votes(self._sub_signals, features, row_idx)
            if bull_votes >= self._min_votes:
                return self._emit_exit(
                    "signal_flip", signal_time, opposing_votes=bull_votes
                )

        return None

    def _emit_exit(
        self, reason: str, signal_time: datetime, **metadata: object
    ) -> Signal:
        """Create CLOSE signal and reset position state."""
        self._last_exit_direction = self._position_direction
        self._bars_since_exit = 0
        self._position_direction = None
        self._position_high_watermark = None
        self._position_low_watermark = None
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=SignalAction.CLOSE,
            confidence=1.0,
            quantity_pct=None,
            interval=self.interval,
            signal_time=signal_time,
            metadata={"reason": reason, **metadata},
        )

    def _count_votes(
        self, conditions: list[dict], features: pd.DataFrame, row_idx: int
    ) -> int:
        """Count how many conditions evaluate to True."""
        count = 0
        for cond in conditions:
            try:
                if evaluate_condition(cond, features, row_idx):
                    count += 1
            except (ValueError, KeyError) as e:
                err_key = str(e)
                if err_key not in self._warned_signals:
                    self._warned_signals.add(err_key)
                    logger.warning(
                        "VotingStrategy '%s' sub-signal error: %s", self.name, e
                    )
        return count

    def _get_rsi_period(self) -> int:
        """Extract RSI period from sub_signals config."""
        for cond in self._sub_signals + self._bear_sub_signals:
            if cond.get("indicator") == "rsi":
                return cond.get("params", {}).get("period", 8)
        return 8  # default

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        """Build feature specs dynamically from sub_signals config.

        Ensures FeatureEngine computes exactly the features this strategy
        needs, regardless of what periods Optuna chose.
        """
        specs: list[tuple[str, dict]] = []
        seen: set[str] = set()

        def _add(name: str, params: dict) -> None:
            key = f"{name}:{sorted(params.items())}"
            if key not in seen:
                seen.add(key)
                specs.append((name, params))

        # Process both bull and bear sub_signals
        for cond in self._sub_signals + self._bear_sub_signals:
            cond_type = cond.get("type", "")
            indicator = cond.get("indicator", "")
            params = cond.get("params", {})

            if cond_type == "indicator_above" and indicator == "cum_return":
                _add("cum_return", {"period": params.get("period", 6)})
            elif cond_type == "indicator_above" and indicator == "rsi":
                _add("rsi", {"period": params.get("period", 8)})
            elif cond_type == "indicator_above" and indicator == "macd_histogram":
                _add(
                    "macd",
                    {
                        "fast_period": params.get("fast_period", 14),
                        "slow_period": params.get("slow_period", 23),
                        "signal_period": params.get("signal_period", 9),
                    },
                )
            elif cond_type == "indicator_comparison":
                period_a = params.get("period_a")
                period_b = params.get("period_b")
                ind_a = cond.get("indicator_a", "ema")
                ind_b = cond.get("indicator_b", "ema")
                if period_a is not None:
                    _add(ind_a, {"period": period_a})
                if period_b is not None:
                    _add(ind_b, {"period": period_b})
            elif cond_type == "bollinger_width_percentile":
                _add(
                    "bollinger",
                    {"period": params.get("period", 20), "num_std": 2.0},
                )

        # Always need ATR for trailing stop and returns
        _add("atr", {"period": self._atr_period})
        _add("returns", {})

        return specs

    def validate_config(self) -> bool:
        """Validate that the strategy configuration is complete and correct."""
        if not self._sub_signals:
            raise ValueError(
                "VotingStrategy requires at least one sub-signal condition"
            )
        if not self.symbol:
            raise ValueError("VotingStrategy requires a symbol")
        if not self.market:
            raise ValueError("VotingStrategy requires a market")
        if self._min_votes < 1:
            raise ValueError("min_votes must be >= 1")
        if self._min_votes > len(self._sub_signals):
            raise ValueError(
                f"min_votes ({self._min_votes}) > number of sub-signals "
                f"({len(self._sub_signals)})"
            )
        # Validate bear_sub_signals if provided
        if self._bear_sub_signals:
            if self._bear_min_votes < 1:
                raise ValueError("bear_min_votes must be >= 1")
            if self._bear_min_votes > len(self._bear_sub_signals):
                raise ValueError(
                    f"bear_min_votes ({self._bear_min_votes}) > number of "
                    f"bear_sub_signals ({len(self._bear_sub_signals)})"
                )
        return True

    def reset(self) -> None:
        """Reset all position and trailing stop state between backtest runs."""
        self._position_direction = None
        self._position_high_watermark = None
        self._position_low_watermark = None
        self._bars_since_exit = 999
        self._last_exit_direction = None
        self._warned_signals.clear()
