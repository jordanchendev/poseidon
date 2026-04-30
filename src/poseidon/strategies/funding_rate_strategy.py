"""FundingRateStrategy — contrarian on extreme perp funding rate.

Thesis: when daily funding rate is at extremes, perp positioning is over-crowded
and tends to mean-revert. Going against the funding direction captures both the
funding payments and the price reversion.

Rules:
    funding_rate > entry_high  -> SHORT (longs over-crowded; collect funding)
    funding_rate < entry_low   -> LONG  (shorts over-crowded; collect funding)
    |funding_rate| < exit_band -> CLOSE (positioning normalised)
    bars_in_position >= hold_max_bars -> CLOSE (time-stop)

Requires the input feature DataFrame to carry a `funding_rate_daily` column
(pre-merged on date). Single-instrument, perp-only.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy


@register_strategy
class FundingRateStrategy(BaseStrategy):
    """Contrarian funding-rate strategy on a single perp symbol."""

    name = "funding_rate_contrarian"
    strategy_type = StrategyType.RULE
    supports_backtest = True
    supports_live = False
    stateful = True

    def __init__(
        self,
        *,
        symbol: str,
        market: str = "crypto_perp",
        interval: str = "1d",
        entry_high: float = 0.0006,
        entry_low: float = -0.0001,
        exit_band: float = 0.0002,
        position_pct: float = 0.5,
        hold_max_bars: int = 10,
        instrument: InstrumentType = InstrumentType.PERPETUAL,
        strategy_id: UUID | None = None,
    ):
        self.symbol = symbol
        self.market = market
        self.interval = interval
        self.entry_high = entry_high
        self.entry_low = entry_low
        self.exit_band = exit_band
        self.position_pct = position_pct
        self.hold_max_bars = hold_max_bars
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()
        # State (stateful=True; reset per backtest run)
        self._position: str | None = None
        self._bars_in_position = 0

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        if features.empty or "funding_rate_daily" not in features.columns:
            return []
        last = features.iloc[-1]
        funding = last.get("funding_rate_daily")
        if funding is None or pd.isna(funding):
            return []

        bar_time = features.index[-1]
        signal_time = bar_time if isinstance(bar_time, datetime) else datetime.now(UTC)

        signals: list[Signal] = []

        if self._position is not None:
            self._bars_in_position += 1
            time_stop = self._bars_in_position >= self.hold_max_bars
            band_exit = abs(funding) < self.exit_band
            flip_to_long = self._position == "short" and funding < self.entry_low
            flip_to_short = self._position == "long" and funding > self.entry_high

            if time_stop or band_exit or flip_to_long or flip_to_short:
                signals.append(self._mk_signal(SignalAction.CLOSE, signal_time, 1.0))
                self._position = None
                self._bars_in_position = 0
                if not (flip_to_long or flip_to_short):
                    return signals
                # else fall through to entry on the same bar

        if self._position is None:
            if funding > self.entry_high:
                signals.append(self._mk_signal(SignalAction.SHORT, signal_time, self.position_pct))
                self._position = "short"
                self._bars_in_position = 0
            elif funding < self.entry_low:
                signals.append(self._mk_signal(SignalAction.LONG, signal_time, self.position_pct))
                self._position = "long"
                self._bars_in_position = 0

        return signals

    def _mk_signal(self, action: SignalAction, signal_time: datetime, qty_pct: float) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=action,
            confidence=0.7,
            quantity_pct=qty_pct,
            interval=self.interval,
            signal_time=signal_time,
        )

    def validate_config(self) -> bool:
        if not self.symbol:
            raise ValueError("symbol required")
        if self.entry_high <= 0:
            raise ValueError("entry_high must be > 0")
        if self.entry_low >= 0:
            raise ValueError("entry_low must be < 0")
        if self.exit_band <= 0:
            raise ValueError("exit_band must be > 0")
        return True
