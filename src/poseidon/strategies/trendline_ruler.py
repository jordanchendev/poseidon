"""TrendlineRulerStrategy -- mechanical "one line, one ruler" (Marci / Little Rizzy).

Port of a discretionary NQ 5m intraday method to Taiwan index futures (TX 5m).
Short side described; the long side is the exact mirror (one sign variable).

  1. Daily direction filter (``htf_bias``): ``htf.mode="sma_slope"`` only
     shorts when the completed daily trend is down; ``"block_counter"`` also
     allows a neutral bias (0) and only blocks an up bias; ``"off"`` disables it.
  2. Bollinger Bands 20/2 on the 5m close. The pattern must appear at or above
     the middle band (``tls_zone_ok``), the entry bar must close below the
     middle band and still inside the lower band (no chasing). Only the first
     ``max_pattern_index`` patterns of a down-leg are traded.
  3. Pattern: a descending trendline through two consecutive pivot highs. The
     ruler is the distance from the pattern's lowest low up to the line above
     it; target = lowest low - ruler (``trendline_ruler`` feature).
  4. Entry: a bar whose high touches the line (within ``touch_atr_frac`` x
     ATR) but closes below it -> market SHORT at the close. A close through
     the line breaks the pattern and exits any open position. Simultaneous
     valid long and short patterns cancel each other.
  5. Stop (``risk.stop_mode``): ``"touch_bar_high"`` = touch-bar high +
     ``stop_buffer_points``; ``"line"`` = the line's value at the entry bar +
     buffer; ``"p2"`` = the line's second pivot high + buffer. If
     ``min_stop_atr`` > 0 and the stop is closer than ``min_stop_atr`` x ATR,
     it is widened to exactly that distance. Then reward/risk >= ``min_rr`` and
     stop distance <= ``max_stop_points`` (both on the final stop).
  6. Exit (``manage.exit_mode``): ``"target_and_break"`` = all-out at the
     target (no scaling) via the runner's SL/TP or on a close through the
     line; ``"line_trail"`` additionally emits a HOLD each bar in position with
     ``updated_stop_loss`` = this bar's projected line + buffer, so the
     runner's intra-bar stop follows the line (tighten-only, never loosens);
     ``"line_trail_no_tp"`` is the same without a take-profit. In every mode
     the position is also closed at the next ``flat_at`` clock time or if the
     trading day rolls.

Session times are Asia/Taipei. ``allowed_windows`` are [start, end] local
clock ranges (a range with start > end wraps past midnight for the night
session). ``flat_at`` is a list of local clock times; an open position is
closed on the first bar at or after the next such time following its entry.

Stateful: tracks its own position mirror and replicates the runner's SL/TP
decision at the start of each bar so it knows when the runner closed it.
``reject_counts`` records why candidate touches were not traded.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, OrderType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.range_reversion import _f, _hhmm_to_minute
from poseidon.strategies.registry import register_strategy

logger = logging.getLogger(__name__)

# side -> (feature prefix, opposite prefix, sign): sign=+1 short, -1 long so that
# "above the line" for a short is sign*(price - line) > 0.
_SIDES = {"short": ("tls", "tll", 1.0), "long": ("tll", "tls", -1.0)}


def _in_window(minute: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= minute <= end
    return minute >= start or minute <= end


@register_strategy
class TrendlineRulerStrategy(BaseStrategy):
    """Trendline touch with ruler target (see module docstring)."""

    name = "trendline_ruler"
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
        self.name = config.get("name", "trendline_ruler")
        self.symbol = config.get("symbol", "TX")
        self.market = config.get("market", "tw_futures")
        self.interval = config.get("interval", "5m")
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()
        self._direction_mode: str = config.get("direction_mode", "bidirectional")

        session = config.get("session", {})
        self._tz: str = session.get("tz", "Asia/Taipei")
        self._day_rollover_minute: int = _hhmm_to_minute(session.get("day_rollover", "14:30"))
        self._windows: list[tuple[int, int]] = [
            (_hhmm_to_minute(a), _hhmm_to_minute(b)) for a, b in session.get("allowed_windows", [["09:15", "13:15"]])
        ]
        flat_at = session.get("flat_at", ["13:40"])
        self._flat_at: list[int] = [_hhmm_to_minute(v) for v in ([flat_at] if isinstance(flat_at, str) else flat_at)]

        htf = config.get("htf", {})
        self._htf_mode: str = htf.get("mode", "sma_slope")
        self._htf_sma_period: int = int(htf.get("sma_period", 20))
        self._htf_slope_lookback: int = int(htf.get("slope_lookback", 5))

        bands = config.get("bands", {})
        self._bb_period: int = int(bands.get("bb_period", 20))
        self._bb_std: float = float(bands.get("bb_std", 2.0))
        self._atr_period: int = int(bands.get("atr_period", 14))

        pattern = config.get("pattern", {})
        self._pivot_left: int = int(pattern.get("pivot_left", 3))
        self._pivot_right: int = int(pattern.get("pivot_right", 3))
        self._max_line_age_bars: int = int(pattern.get("max_line_age_bars", 60))
        self._max_pattern_index: int = int(pattern.get("max_pattern_index", 2))
        self._leg_reset: str = pattern.get("leg_reset", "bb_upper")
        self._setup_zone: str = pattern.get("setup_zone", "mid_or_beyond")
        self._max_entries_per_line: int = int(pattern.get("max_entries_per_line", 1))
        self._skip_conflicting: bool = bool(pattern.get("skip_conflicting", True))

        entry = config.get("entry", {})
        self._touch_atr_frac: float = float(entry.get("touch_atr_frac", 0.25))
        self._require_mid_close: bool = bool(entry.get("require_mid_close", True))
        self._no_chase: bool = bool(entry.get("no_chase", True))

        risk = config.get("risk", {})
        self._stop_buffer_points: float = float(risk.get("stop_buffer_points", 5.0))
        self._min_rr: float = float(risk.get("min_rr", 1.0))
        self._max_stop_points: float = float(risk.get("max_stop_points", 150.0))
        self._stop_mode: str = risk.get("stop_mode", "touch_bar_high")
        self._min_stop_atr: float = float(risk.get("min_stop_atr", 0.0))

        manage = config.get("manage", {})
        self._max_trades_per_day: int = int(manage.get("max_trades_per_day", 3))
        self._exit_mode: str = manage.get("exit_mode", "target_and_break")

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._day: float | None = None
        self._trades_today: int = 0
        self._bar_count: int = 0
        self._entries_per_line: dict[str, tuple[float, int]] = {}  # side -> (line id, entries)
        self.reject_counts: Counter[str] = Counter()
        self.exit_counts: Counter[str] = Counter()
        self.entry_rr: list[float] = []  # planned reward/risk of each entry (driver reporting)
        self.entry_stop_points: list[float] = []  # planned stop distance of each entry
        self.entry_target_points: list[float] = []  # planned target distance of each entry
        # open position mirror
        self._side: str | None = None
        self._entry_price: float | None = None
        self._sl: float | None = None
        self._tp: float | None = None
        self._line_id: float = -1.0
        self._line_at_entry: float = 0.0
        self._line_slope: float = 0.0
        self._entry_bar: int = -1
        self._flat_deadline: pd.Timestamp | None = None

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        return [
            ("session_clock", {"tz": self._tz, "day_rollover_minute": self._day_rollover_minute}),
            ("bollinger", {"period": self._bb_period, "num_std": self._bb_std}),
            ("atr", {"period": self._atr_period}),
            (
                "htf_bias",
                {
                    "htf_sma_period": self._htf_sma_period,
                    "htf_slope_lookback": self._htf_slope_lookback,
                    # block_counter reads the same sma_slope bias, it only filters differently
                    "htf_mode": "off" if self._htf_mode == "off" else "sma_slope",
                    "tz": self._tz,
                    "day_rollover_minute": self._day_rollover_minute,
                },
            ),
            (
                "trendline_ruler",
                {
                    "pivot_left": self._pivot_left,
                    "pivot_right": self._pivot_right,
                    "max_line_age_bars": self._max_line_age_bars,
                    "leg_reset": self._leg_reset,
                    "setup_zone": self._setup_zone,
                    "bb_period": self._bb_period,
                    "bb_std": self._bb_std,
                },
            ),
        ]

    def validate_config(self) -> bool:
        if not self.symbol or not self.market:
            raise ValueError("symbol and market must be non-empty")
        if self._direction_mode not in ("long_only", "short_only", "bidirectional"):
            raise ValueError(f"direction_mode invalid: {self._direction_mode!r}")
        if self._htf_mode not in ("sma_slope", "block_counter", "off"):
            raise ValueError(f"htf.mode invalid: {self._htf_mode!r}")
        if self._stop_mode not in ("touch_bar_high", "line", "p2"):
            raise ValueError(f"risk.stop_mode invalid: {self._stop_mode!r}")
        if self._min_stop_atr < 0:
            raise ValueError("risk.min_stop_atr must be >= 0")
        if self._exit_mode not in ("target_and_break", "line_trail", "line_trail_no_tp"):
            raise ValueError(f"manage.exit_mode invalid: {self._exit_mode!r}")
        if self._leg_reset not in ("bb_upper", "off"):
            raise ValueError(f"pattern.leg_reset invalid: {self._leg_reset!r}")
        if self._setup_zone not in ("mid_or_beyond", "off"):
            raise ValueError(f"pattern.setup_zone invalid: {self._setup_zone!r}")
        if not self._windows or not self._flat_at:
            raise ValueError("session.allowed_windows and session.flat_at must be non-empty")
        if self._min_rr <= 0:
            raise ValueError("risk.min_rr must be > 0")
        if self._pivot_left < 1 or self._pivot_right < 1:
            raise ValueError("pattern.pivot_left/pivot_right must be >= 1")
        if self._max_pattern_index < 1 or self._max_entries_per_line < 1:
            raise ValueError("pattern.max_pattern_index and max_entries_per_line must be >= 1")
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
            self.exit_counts["sl" if sl_hit else "tp"] += 1
            self._clear_position()

    def _clear_position(self) -> None:
        self._side = None
        self._entry_price = None
        self._sl = None
        self._tp = None
        self._line_id = -1.0
        self._flat_deadline = None

    def _local_time(self, ts: datetime) -> pd.Timestamp:
        stamp = pd.Timestamp(ts)
        return (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp).tz_convert(self._tz)

    def _next_flat(self, local_ts: pd.Timestamp) -> pd.Timestamp:
        """First ``flat_at`` clock time strictly after ``local_ts``."""
        day = local_ts.normalize()
        candidates = []
        for minute in self._flat_at:
            at = day + pd.Timedelta(minutes=minute)
            candidates.append(at if at > local_ts else at + pd.Timedelta(days=1))
        return min(candidates)

    def _close(self, signal_time: datetime, reason: str) -> list[Signal]:
        self.exit_counts[reason] += 1
        self._clear_position()
        return [self._signal(SignalAction.CLOSE, signal_time, confidence=1.0, metadata={"exit_reason": reason})]

    def _valid_setup(self, row: pd.Series, prefix: str) -> bool:
        return (
            _f(row, f"{prefix}_active", 0.0) == 1.0
            and (self._setup_zone == "off" or _f(row, f"{prefix}_zone_ok", 0.0) == 1.0)
            and _f(row, f"{prefix}_pattern_index", 0.0) <= self._max_pattern_index
        )

    def _reject_reason(self, row: pd.Series, side: str, minute: int) -> str | None:
        """First failing entry condition for a touch candidate (None = tradeable)."""
        prefix, opposite, sign = _SIDES[side]
        close = _f(row, "close")
        line = _f(row, f"{prefix}_line")
        if not any(_in_window(minute, a, b) for a, b in self._windows):
            return "window"
        if self._trades_today >= self._max_trades_per_day:
            return "max_trades_per_day"
        if self._direction_mode == ("long_only" if side == "short" else "short_only"):
            return "direction_mode"
        bias = _f(row, "htf_bias", 0.0)
        if self._htf_mode == "sma_slope" and bias != -sign:
            return "htf_bias"
        if self._htf_mode == "block_counter" and bias == sign:
            return "htf_bias"
        if _f(row, f"{prefix}_pattern_index", 0.0) > self._max_pattern_index:
            return "pattern_index"
        if self._setup_zone != "off" and _f(row, f"{prefix}_zone_ok", 0.0) != 1.0:
            return "setup_zone"
        line_id, entries = self._entries_per_line.get(side, (-1.0, 0))
        if line_id == _f(row, f"{prefix}_id") and entries >= self._max_entries_per_line:
            return "line_already_traded"
        if not sign * (close - line) < 0:
            return "close_not_inside_line"
        mid = _f(row, f"bb_middle_{self._bb_period}")
        if self._require_mid_close and not sign * (close - mid) < 0:
            return "mid_close"
        outer = _f(row, f"bb_{'lower' if side == 'short' else 'upper'}_{self._bb_period}")
        if self._no_chase and not sign * (close - outer) > 0:
            return "chase"
        if self._skip_conflicting and self._valid_setup(row, opposite):
            return "conflicting_setup"
        return None

    def _stop_anchor(self, row: pd.Series, side: str) -> float:
        """Price the stop is placed beyond (before the buffer), per ``stop_mode``."""
        prefix = _SIDES[side][0]
        if self._stop_mode == "line":
            return _f(row, f"{prefix}_line")
        if self._stop_mode == "p2":
            return _f(row, f"{prefix}_p2")
        return _f(row, "high") if side == "short" else _f(row, "low")

    def _plan(self, row: pd.Series, side: str) -> tuple[float, float, float] | str:
        """(stop, target, rr) or the rejection reason."""
        prefix, _, sign = _SIDES[side]
        close = _f(row, "close")
        stop = self._stop_anchor(row, side) + sign * self._stop_buffer_points
        target = _f(row, f"{prefix}_target")
        risk = sign * (stop - close)
        if risk <= 0:
            return "stop_inside"
        min_risk = self._min_stop_atr * _f(row, f"atr_{self._atr_period}", 0.0)
        if risk < min_risk:
            risk = min_risk
            stop = close + sign * risk
        reward = sign * (close - target)
        if _f(row, f"{prefix}_ruler", 0.0) <= 0 or reward <= 0:
            return "no_reward"
        if risk > self._max_stop_points:
            return "max_stop"
        if reward < self._min_rr * risk:
            return "rr"
        return stop, target, reward / risk

    # -------------------------------------------------------------- evaluate
    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        if features.empty:
            return []
        row = features.iloc[-1]
        ts = features.index[-1]
        signal_time = ts if isinstance(ts, datetime) else datetime.now(UTC)
        self._bar_count = len(features)
        day = _f(row, "trading_day")
        minute_f = _f(row, "tpe_minute")
        if pd.isna(day) or pd.isna(minute_f):
            return []
        minute = int(minute_f)

        # Runner already applied SL/TP on this bar before calling us.
        self._sync_runner_exit(row)

        day_changed = day != self._day
        if day_changed:
            self._day = day
            self._trades_today = 0

        if self._side is not None:
            return self._manage(row, signal_time, day_changed)

        signals: list[Signal] = []
        for side, (prefix, _, sign) in _SIDES.items():
            if _f(row, f"{prefix}_active", 0.0) != 1.0:
                continue
            line = _f(row, f"{prefix}_line")
            touch_tol = self._touch_atr_frac * _f(row, f"atr_{self._atr_period}", 0.0)
            touch_price = _f(row, "high") if side == "short" else _f(row, "low")
            if not sign * (touch_price - line) >= -touch_tol:
                continue
            reason = self._reject_reason(row, side, minute)
            plan = self._plan(row, side) if reason is None else reason
            if isinstance(plan, str):
                self.reject_counts[plan] += 1
                continue
            stop, target, rr = plan
            close = _f(row, "close")
            take_profit = None if self._exit_mode == "line_trail_no_tp" else target
            line_id = _f(row, f"{prefix}_id")
            prev_id, entries = self._entries_per_line.get(side, (-1.0, 0))
            self._entries_per_line[side] = (line_id, entries + 1 if prev_id == line_id else 1)
            self._side = side
            self._entry_price = close
            self._sl = stop
            self._tp = take_profit
            self._line_id = line_id
            self._line_at_entry = line
            self._line_slope = _f(row, f"{prefix}_slope", 0.0)
            self._entry_bar = self._bar_count
            self._flat_deadline = self._next_flat(self._local_time(signal_time))
            self._trades_today += 1
            self.reject_counts["entered"] += 1
            self.entry_rr.append(rr)
            self.entry_stop_points.append(sign * (stop - close))
            self.entry_target_points.append(sign * (close - target))
            logger.debug("trendline %s @%.1f sl=%.1f tp=%.1f rr=%.2f", side, close, stop, target, rr)
            signals.append(
                self._signal(
                    SignalAction.SHORT if side == "short" else SignalAction.LONG,
                    signal_time,
                    confidence=min(1.0, rr / 3.0),
                    order_type=OrderType.MARKET,
                    stop_loss_price=stop,
                    take_profit_price=take_profit,
                    metadata={
                        "line_id": int(line_id),
                        "pattern_index": int(_f(row, f"{prefix}_pattern_index", 0.0)),
                        "ruler": _f(row, f"{prefix}_ruler"),
                        "rr": rr,
                        "stop_points": sign * (stop - close),
                        "target_points": sign * (close - target),
                        "touch_tol": touch_tol,
                    },
                )
            )
            break
        return signals

    def _manage(self, row: pd.Series, signal_time: datetime, day_changed: bool) -> list[Signal]:
        if day_changed:
            return self._close(signal_time, "day_change")
        prefix, _, sign = _SIDES[self._side or "short"]
        line_now = self._line_at_entry + self._line_slope * (self._bar_count - self._entry_bar)
        broken_flag = _f(row, f"{prefix}_broken", 0.0) == 1.0 and _f(row, f"{prefix}_id") == self._line_id
        if broken_flag or sign * (_f(row, "close") - line_now) > 0:
            return self._close(signal_time, "line_break")
        if self._flat_deadline is not None and self._local_time(signal_time) >= self._flat_deadline:
            return self._close(signal_time, "flat")
        if self._exit_mode == "target_and_break":
            return []
        # Trail: the runner applies this stop from the next bar on; tighten only.
        trail = line_now + sign * self._stop_buffer_points
        if self._sl is None or sign * (trail - self._sl) < 0:
            self._sl = trail
        return [self._signal(SignalAction.HOLD, signal_time, confidence=1.0, metadata={"updated_stop_loss": self._sl})]
