"""VotingStrategy -- multi-signal voting with ATR trailing stop.

Wraps N child conditions (sub-signals). Emits LONG when >= min_votes
conditions are true. Tracks position state for ATR trailing stop exits.

Position sizing note (D-10):
    VotingStrategy sets quantity_pct=0.08 on entry signals as strategy-level
    sizing intent. However, BacktestRunner._compute_sizing() controls final
    sizing based on its SizingConfig:
      - FIXED_PCT mode: uses signal.quantity_pct (0.08) directly
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
    """Multi-signal voting strategy with ATR trailing stop exit."""

    strategy_type = StrategyType.VOTING

    def __init__(
        self,
        *,
        config: dict,
        atr_multiplier: float = 2.0,
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

        self._sub_signals: list[dict] = config.get("sub_signals", [])
        self._min_votes: int = config.get("min_votes", 4)
        self._position_pct: float = config.get("position_pct", 0.08)
        self._atr_multiplier = atr_multiplier
        self._atr_period = atr_period

        # Trailing stop state
        self._in_position: bool = False
        self._position_high_watermark: float | None = None

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate voting conditions and trailing stop against feature data."""
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

        # 1. Check trailing stop FIRST (if in position)
        if self._in_position:
            self._position_high_watermark = max(self._position_high_watermark, close)
            atr_col = f"atr_{self._atr_period}"
            atr_val = float(features.iloc[row_idx][atr_col])
            stop_level = self._position_high_watermark - self._atr_multiplier * atr_val

            if close < stop_level:
                signals.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=self.symbol,
                    market=self.market,
                    instrument=self.instrument,
                    action=SignalAction.CLOSE,
                    confidence=1.0,
                    quantity_pct=None,
                    interval=self.interval,
                    signal_time=signal_time,
                    metadata={
                        "reason": "atr_trailing_stop",
                        "hwm": self._position_high_watermark,
                        "stop_level": stop_level,
                    },
                ))
                self._in_position = False
                self._position_high_watermark = None
                return signals

        # 2. Count votes from sub-signals
        vote_count = 0
        for cond in self._sub_signals:
            try:
                if evaluate_condition(cond, features, row_idx):
                    vote_count += 1
            except (ValueError, KeyError) as e:
                logger.warning("VotingStrategy '%s' sub-signal error: %s", self.name, e)

        # 3. Emit LONG if threshold met and not already in position
        if vote_count >= self._min_votes and not self._in_position:
            confidence = vote_count / len(self._sub_signals) if self._sub_signals else 0.0
            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                market=self.market,
                instrument=self.instrument,
                action=SignalAction.LONG,
                confidence=confidence,
                quantity_pct=self._position_pct,
                interval=self.interval,
                signal_time=signal_time,
                metadata={"vote_count": vote_count, "min_votes": self._min_votes},
            ))
            self._in_position = True
            self._position_high_watermark = close

        logger.info(
            "VotingStrategy '%s': %d/%d votes (need %d) | in_position=%s",
            self.name, vote_count, len(self._sub_signals), self._min_votes, self._in_position,
        )
        return signals

    def validate_config(self) -> bool:
        """Validate that the strategy configuration is complete and correct."""
        if not self._sub_signals:
            raise ValueError("VotingStrategy requires at least one sub-signal condition")
        if not self.symbol:
            raise ValueError("VotingStrategy requires a symbol")
        if not self.market:
            raise ValueError("VotingStrategy requires a market")
        if self._min_votes < 1:
            raise ValueError("min_votes must be >= 1")
        if self._min_votes > len(self._sub_signals):
            raise ValueError(
                f"min_votes ({self._min_votes}) > number of sub-signals ({len(self._sub_signals)})"
            )
        return True

    def reset(self) -> None:
        """Reset trailing stop state between backtest runs."""
        self._in_position = False
        self._position_high_watermark = None
