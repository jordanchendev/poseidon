"""RangeReversionStrategy -- mechanical "Range 50% Reversion" (Kane PO3 sweep).

Port of the four-step discretionary method to Taiwan index futures (TX 5m):

  1. Dealing range on the daily chart (prior N completed trading days). The
     bias is bearish when the window low is more recent than the window high.
  2. Multi-timeframe sweep: the running 1h / 4h / trading-day bar must have
     pushed beyond the previous bar's extreme (configurable which are
     mandatory and how many are needed). The sweep stays "armed" for
     ``sweep_valid_bars`` while the extreme keeps being tracked.
  3. Confirmation and trigger: optional SMT divergence against a companion
     (TX vs TE) plus a close back through the reversal gap (bullish FVG formed
     after the sweep started) or back below the swept 1h level. Entry is a
     market order on that bar's close; stop above the sweep extreme, target at
     the 50% of the leg (sweep extreme vs. the lowest low of the last
     ``leg_lookback_bars``). Reward/risk must clear ``min_rr``.
  4. Management: when the current 1h bar turns against the entry direction
     the stop moves to breakeven (plus a fee buffer); the runner handles
     SL/TP intra-bar; anything still open at ``flat_at`` is closed; after a
     winning trade the strategy stops for the day.

Everything subjective in the original (which range to draw, "signs of a
bottom", scoring) is replaced by explicit config parameters so the edge can
be measured and searched rather than copied from screenshots.

Stateful: tracks its own position mirror and replicates the runner's SL/TP
decision at the start of each bar so it knows when the runner closed it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.registry import register_strategy

logger = logging.getLogger(__name__)


def _hhmm_to_minute(value: str | int) -> int:
    if isinstance(value, int):
        return value
    hh, mm = value.split(":")
    return int(hh) * 60 + int(mm)


def _f(row: pd.Series, col: str, default: float = float("nan")) -> float:
    v = row.get(col, default)
    if v is None or pd.isna(v):
        return default
    return float(v)


@register_strategy
class RangeReversionStrategy(BaseStrategy):
    """Range 50% reversion with multi-timeframe sweep entry (see module docstring)."""

    name = "range_50_reversion"
    strategy_type = StrategyType.RULE
    supports_backtest = True
    supports_live = False
    stateful = True

    def __init__(
        self,
        *,
        config: dict,
        instrument: InstrumentType = InstrumentType.FUTURES,
        strategy_id: UUID | None = None,
    ):
        self.name = config.get("name", "range_50_reversion")
        self.symbol = config.get("symbol", "TX")
        self.market = config.get("market", "tw_futures")
        self.interval = config.get("interval", "5m")
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()
        self._direction_mode: str = config.get("direction_mode", "bidirectional")

        session = config.get("session", {})
        self._tz: str = session.get("tz", "Asia/Taipei")
        self._day_rollover_minute: int = _hhmm_to_minute(session.get("day_rollover", "14:30"))
        self._entry_start: int = _hhmm_to_minute(session.get("entry_start", "09:00"))
        self._entry_end: int = _hhmm_to_minute(session.get("entry_end", "11:15"))
        self._flat_at: int = _hhmm_to_minute(session.get("flat_at", "13:40"))

        rng = config.get("range", {})
        self._lookback_days: int = int(rng.get("lookback_days", 20))
        self._bias_mode: str = rng.get("bias_mode", "range")  # range | off

        sweep = config.get("sweep", {})
        self._require_h1: bool = bool(sweep.get("require_h1", True))
        self._require_h4: bool = bool(sweep.get("require_h4", True))
        self._require_d1: bool = bool(sweep.get("require_d1", False))
        self._min_sweeps: int = int(sweep.get("min_sweeps", 2))
        self._sweep_valid_bars: int = int(sweep.get("sweep_valid_bars", 12))

        confirm = config.get("confirm", {})
        self._smt_mode: str = confirm.get("smt_mode", "off")  # off | optional | required
        self._companion_symbol: str = confirm.get("companion_symbol", "TE")
        self._companion_market: str = confirm.get("companion_market", self.market)
        self._entry_trigger: str = confirm.get("entry_trigger", "fvg_or_level")  # fvg | level | fvg_or_level
        self._fvg_max_age_bars: int = int(confirm.get("fvg_max_age_bars", 24))

        risk = config.get("risk", {})
        self._stop_buffer_points: float = float(risk.get("stop_buffer_points", 5.0))
        self._leg_lookback_bars: int = int(risk.get("leg_lookback_bars", 48))
        self._min_rr: float = float(risk.get("min_rr", 1.5))
        self._max_stop_points: float = float(risk.get("max_stop_points", 150.0))

        manage = config.get("manage", {})
        self._breakeven_on_h1_flip: bool = bool(manage.get("breakeven_on_h1_flip", True))
        self._breakeven_offset_points: float = float(manage.get("breakeven_offset_points", 2.0))
        self._max_trades_per_day: int = int(manage.get("max_trades_per_day", 2))
        self._stop_after_win: bool = bool(manage.get("stop_after_win", True))

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._day: float | None = None
        self._trades_today: int = 0
        self._won_today: bool = False
        self._bar_count: int = 0
        # armed sweep (per direction)
        self._sweep: dict[str, dict] = {}
        # open position mirror
        self._side: str | None = None
        self._entry_price: float | None = None
        self._sl: float | None = None
        self._tp: float | None = None
        self._breakeven_done: bool = False
        self._entry_bar: int = -1

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        specs: list[tuple[str, dict]] = [
            ("session_levels", {"tz": self._tz, "day_rollover_minute": self._day_rollover_minute}),
            (
                "dealing_range",
                {
                    "lookback_days": self._lookback_days,
                    "tz": self._tz,
                    "day_rollover_minute": self._day_rollover_minute,
                },
            ),
            ("fair_value_gap", {"max_age_bars": self._fvg_max_age_bars}),
            ("atr", {"period": 14}),
        ]
        if self._smt_mode != "off":
            specs.append(
                (
                    "smt_divergence",
                    {
                        "companion_symbol": self._companion_symbol,
                        "companion_market": self._companion_market,
                        "tz": self._tz,
                        "day_rollover_minute": self._day_rollover_minute,
                    },
                )
            )
        return specs

    def validate_config(self) -> bool:
        if not self.symbol or not self.market:
            raise ValueError("symbol and market must be non-empty")
        if self._direction_mode not in ("long_only", "short_only", "bidirectional"):
            raise ValueError(f"direction_mode invalid: {self._direction_mode!r}")
        if self._bias_mode not in ("range", "off"):
            raise ValueError(f"range.bias_mode invalid: {self._bias_mode!r}")
        if self._smt_mode not in ("off", "optional", "required"):
            raise ValueError(f"confirm.smt_mode invalid: {self._smt_mode!r}")
        if self._entry_trigger not in ("fvg", "level", "fvg_or_level"):
            raise ValueError(f"confirm.entry_trigger invalid: {self._entry_trigger!r}")
        if not self._entry_start < self._entry_end <= self._flat_at:
            raise ValueError("session must satisfy entry_start < entry_end <= flat_at")
        if self._min_rr <= 0:
            raise ValueError("risk.min_rr must be > 0")
        if self._lookback_days < 2:
            raise ValueError("range.lookback_days must be >= 2")
        if self._leg_lookback_bars < 2:
            raise ValueError("risk.leg_lookback_bars must be >= 2")
        return True

    # --------------------------------------------------------------- helpers
    def _signal(self, action: SignalAction, signal_time: datetime, **kwargs) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=action,
            interval=self.interval,
            signal_time=signal_time,
            **kwargs,
        )

    def _sync_runner_exit(self, row: pd.Series) -> None:
        """Mirror BacktestRunner Phase A: SL/TP on this bar's high/low, SL first."""
        if self._side is None:
            return
        high = _f(row, "high")
        low = _f(row, "low")
        if self._side == "short":
            sl_hit = self._sl is not None and high >= self._sl
            tp_hit = self._tp is not None and low <= self._tp
        else:
            sl_hit = self._sl is not None and low <= self._sl
            tp_hit = self._tp is not None and high >= self._tp
        if sl_hit or tp_hit:
            self._close_position(won=(not sl_hit) and tp_hit)

    def _close_position(self, won: bool) -> None:
        self._side = None
        self._entry_price = None
        self._sl = None
        self._tp = None
        self._breakeven_done = False
        self._trades_today += 1
        if won:
            self._won_today = True

    def _bias_allows(self, row: pd.Series, side: str) -> bool:
        if self._direction_mode == "long_only" and side == "short":
            return False
        if self._direction_mode == "short_only" and side == "long":
            return False
        if self._bias_mode == "off":
            return True
        bias = _f(row, f"range_bias_{self._lookback_days}")
        if pd.isna(bias):
            return False
        return (bias < 0) if side == "short" else (bias > 0)

    def _sweep_flags(self, row: pd.Series, side: str) -> tuple[bool, dict[str, bool]]:
        if side == "short":
            flags = {
                "h1": _f(row, "h1_cur_high") > _f(row, "h1_prev_high", float("inf")),
                "h4": _f(row, "h4_cur_high") > _f(row, "h4_prev_high", float("inf")),
                "d1": _f(row, "d1_cur_high") > _f(row, "d1_prev_high", float("inf")),
            }
        else:
            flags = {
                "h1": _f(row, "h1_cur_low") < _f(row, "h1_prev_low", float("-inf")),
                "h4": _f(row, "h4_cur_low") < _f(row, "h4_prev_low", float("-inf")),
                "d1": _f(row, "d1_cur_low") < _f(row, "d1_prev_low", float("-inf")),
            }
        required = {"h1": self._require_h1, "h4": self._require_h4, "d1": self._require_d1}
        if any(required[k] and not flags[k] for k in flags):
            return False, flags
        return sum(flags.values()) >= self._min_sweeps, flags

    def _update_armed_sweeps(self, row: pd.Series) -> None:
        """Arm / refresh / expire the sweep state for each direction."""
        high = _f(row, "high")
        low = _f(row, "low")
        for side in ("short", "long"):
            ok, flags = self._sweep_flags(row, side)
            state = self._sweep.get(side)
            if ok:
                if state is None:
                    self._sweep[side] = {
                        "start_bar": self._bar_count,
                        "last_bar": self._bar_count,
                        # running extreme of the current 1h bar (includes the sweep spike even
                        # if evaluation starts mid-hour)
                        "extreme": _f(row, "h1_cur_high", high) if side == "short" else _f(row, "h1_cur_low", low),
                        "level": _f(row, "h1_prev_high") if side == "short" else _f(row, "h1_prev_low"),
                        "flags": flags,
                        "smt": False,
                    }
                else:
                    state["last_bar"] = self._bar_count
                    state["extreme"] = max(state["extreme"], high) if side == "short" else min(state["extreme"], low)
                    # the swept 1h level moves when a later hour sweeps again
                    state["level"] = _f(row, "h1_prev_high") if side == "short" else _f(row, "h1_prev_low")
                    state["flags"] = flags
            elif state is not None:
                # keep tracking the extreme while armed (price may still poke higher)
                state["extreme"] = max(state["extreme"], high) if side == "short" else min(state["extreme"], low)
                if self._bar_count - state["last_bar"] > self._sweep_valid_bars:
                    del self._sweep[side]
            if side in self._sweep and self._smt_mode != "off":
                col = "smt_bear_div" if side == "short" else "smt_bull_div"
                if _f(row, col, 0.0) >= 1.0:
                    self._sweep[side]["smt"] = True

    def _trigger(self, row: pd.Series, side: str, state: dict) -> str | None:
        close = _f(row, "close")
        bars_since = self._bar_count - state["start_bar"]
        if side == "short":
            fvg_age = _f(row, "fvg_bull_age", -1.0)
            fvg_ok = 0 <= fvg_age <= bars_since and close < _f(row, "fvg_bull_low", float("inf"))
            level_ok = close < state["level"]
        else:
            fvg_age = _f(row, "fvg_bear_age", -1.0)
            fvg_ok = 0 <= fvg_age <= bars_since and close > _f(row, "fvg_bear_high", float("-inf"))
            level_ok = close > state["level"]
        if self._entry_trigger == "fvg" and fvg_ok:
            return "fvg"
        if self._entry_trigger == "level" and level_ok:
            return "level"
        if self._entry_trigger == "fvg_or_level":
            if fvg_ok:
                return "fvg"
            if level_ok:
                return "level"
        return None

    def _plan_trade(
        self, features: pd.DataFrame, row: pd.Series, side: str, state: dict
    ) -> tuple[float, float, float] | None:
        close = _f(row, "close")
        leg = features.tail(self._leg_lookback_bars)
        if side == "short":
            stop = state["extreme"] + self._stop_buffer_points
            leg_low = float(leg["low"].min())
            target = (state["extreme"] + leg_low) / 2.0
            risk = stop - close
            reward = close - target
        else:
            stop = state["extreme"] - self._stop_buffer_points
            leg_high = float(leg["high"].max())
            target = (state["extreme"] + leg_high) / 2.0
            risk = close - stop
            reward = target - close
        if risk <= 0 or reward <= 0 or risk > self._max_stop_points:
            return None
        rr = reward / risk
        if rr < self._min_rr:
            return None
        return stop, target, rr

    # -------------------------------------------------------------- evaluate
    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        if features.empty:
            return []
        row = features.iloc[-1]
        ts = features.index[-1]
        signal_time = ts if isinstance(ts, datetime) else datetime.now(UTC)
        self._bar_count = len(features)

        day = _f(row, "trading_day")
        minute = _f(row, "tpe_minute")
        if pd.isna(day) or pd.isna(minute):
            return []
        if day != self._day:
            self._day = day
            self._trades_today = 0
            self._won_today = False
            self._sweep = {}

        # Runner already applied SL/TP on this bar before calling us.
        self._sync_runner_exit(row)

        if self._side is not None:
            return self._manage(row, minute, signal_time)

        self._update_armed_sweeps(row)

        if self._trades_today >= self._max_trades_per_day:
            return []
        if self._stop_after_win and self._won_today:
            return []
        if not (self._entry_start <= minute <= self._entry_end):
            return []

        for side in ("short", "long"):
            state = self._sweep.get(side)
            if state is None or not self._bias_allows(row, side):
                continue
            if self._smt_mode == "required" and not state["smt"]:
                continue
            trig = self._trigger(row, side, state)
            if trig is None:
                continue
            plan = self._plan_trade(features, row, side, state)
            if plan is None:
                continue
            stop, target, rr = plan
            close = _f(row, "close")
            self._side = side
            self._entry_price = close
            self._sl = stop
            self._tp = target
            self._breakeven_done = False
            self._entry_bar = self._bar_count
            del self._sweep[side]
            logger.debug("range50 %s @%.1f sl=%.1f tp=%.1f rr=%.2f trig=%s", side, close, stop, target, rr, trig)
            return [
                self._signal(
                    SignalAction.SHORT if side == "short" else SignalAction.LONG,
                    signal_time,
                    confidence=min(1.0, rr / 3.0),
                    order_type=OrderType.MARKET,
                    stop_loss_price=stop,
                    take_profit_price=target,
                    metadata={
                        "trigger": trig,
                        "rr": rr,
                        "sweep_extreme": state["extreme"],
                        "sweep_flags": {k: bool(v) for k, v in state["flags"].items()},
                        "smt": bool(state["smt"]),
                        "range_eq": _f(row, f"range_eq_{self._lookback_days}"),
                    },
                )
            ]
        return []

    def _manage(self, row: pd.Series, minute: float, signal_time: datetime) -> list[Signal]:
        if minute >= self._flat_at:
            self._close_position(won=False)
            return [self._signal(SignalAction.CLOSE, signal_time, confidence=1.0, metadata={"exit_reason": "flat_at"})]

        if not self._breakeven_on_h1_flip or self._breakeven_done or self._entry_price is None:
            return []
        close = _f(row, "close")
        h1_open = _f(row, "h1_cur_open")
        if pd.isna(h1_open):
            return []
        if self._side == "short":
            flipped = close < h1_open
            new_sl = self._entry_price - self._breakeven_offset_points
            tighter = self._sl is None or new_sl < self._sl
        else:
            flipped = close > h1_open
            new_sl = self._entry_price + self._breakeven_offset_points
            tighter = self._sl is None or new_sl > self._sl
        if not (flipped and tighter):
            return []
        self._sl = new_sl
        self._breakeven_done = True
        return [
            self._signal(
                SignalAction.HOLD,
                signal_time,
                confidence=0.0,
                metadata={"updated_stop_loss": new_sl, "reason": "breakeven_h1_flip"},
            )
        ]
