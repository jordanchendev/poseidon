"""RuleStrategyV2 — position-aware DSL v2 strategy for TX/MTX intraday.

State machine (one position at most, driven by what the runner reports):

- flat: evaluate entry rules in list order; the first rule that matches and
  yields a valid stop/target emits LONG/SHORT with ``stop_loss_price`` /
  ``take_profit_price`` set. Gates: in-session bar, per-session trade cap,
  no re-entry after a stop, not inside the flat-before-session-end window.
- in position: automatic exits of the rule's merged risk block first
  (session change / scheduled session end when ``close_on_session_change``
  is on (default), ``flat_before_session_end_minutes``, ``max_hold_bars``),
  then ``close`` rules whose ``exit_side`` matches, then a tighten-only
  trailing stop sent as HOLD ``metadata.updated_stop_loss``.

Trade counts and stop-outs come from the position/closed-trade the runner
passes to :meth:`on_bar_context` every bar, never from emitted signals (an
entry the risk engine rejects never counts).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.dsl.v2.evaluator import (
    DEFAULT_BAR_LABEL,
    SESSION_ID,
    SESSION_KIND,
    SESSION_MIN_CLOSE,
    SESSION_MIN_OPEN,
    SESSION_TOD,
    ColumnCache,
    CompiledCondition,
    atr_column,
    derive_feature_specs,
    operand_column,
    operand_value,
)
from poseidon.strategies.dsl.v2.schema import (
    AtrStop,
    AtrTarget,
    AtrTrailing,
    EntryRule,
    LevelStop,
    LevelTarget,
    PointsStop,
    PointsTarget,
    RiskBlock,
    RMultipleTarget,
    StrategyV2,
)
from poseidon.strategies.dsl.v2.validate import ValidatedV2, merge_risk, validate_v2, walk_model

logger = logging.getLogger(__name__)


@dataclass
class _EntryPlan:
    """What the strategy intended when it emitted an entry signal."""

    rule_index: int
    session_id: float
    row: int
    stop_loss: float | None
    risk_per_unit: float | None  # R = |close - SL| at the signal bar


@dataclass
class _ActiveTrade:
    entry_time: Any
    side: str
    rule_index: int
    session_id: float
    entry_row: int
    risk_per_unit: float | None
    stop_loss: float | None
    entry_price: float = math.nan
    best_price: float = math.nan  # highest high (long) / lowest low (short) since entry
    trailing_active: bool = False


class RuleStrategyV2(BaseStrategy):
    """DSL v2 rule strategy (``dsl_version: 2``)."""

    name = "rule_strategy_v2"
    strategy_type = StrategyType.RULE
    position_aware = True
    stateful = True

    def __init__(
        self,
        *,
        config: dict | StrategyV2 | ValidatedV2,
        instrument: InstrumentType = InstrumentType.FUTURES,
        strategy_id: UUID | None = None,
        param_values: dict[str, Any] | None = None,
        bar_label: str = DEFAULT_BAR_LABEL,
    ):
        if isinstance(config, ValidatedV2):
            validated = config
        elif isinstance(config, StrategyV2):
            validated = ValidatedV2(model=config, template=config.model_dump(mode="json", exclude_none=True))
        else:
            validated = validate_v2(config, param_values=param_values)
        self.validated = validated
        self.config: StrategyV2 = validated.model
        self.name = self.config.name
        self.symbol = self.config.symbol
        self.market = self.config.market
        self.interval = self.config.interval
        self.instrument = instrument
        self.strategy_id = strategy_id or uuid4()
        self.bar_label = bar_label

        self._entries = [
            (i, rule, CompiledCondition(rule.condition), merge_risk(self.config.risk, rule.risk))
            for i, rule in enumerate(self.config.rules)
            if isinstance(rule, EntryRule)
        ]
        self._closes = [
            (rule, CompiledCondition(rule.condition)) for rule in self.config.rules if rule.action == "close"
        ]
        self._risk_by_rule = {i: risk for i, _rule, _cond, risk in self._entries}
        columns = [
            "open",
            "high",
            "low",
            "close",
            SESSION_ID,
            SESSION_KIND,
            SESSION_TOD,
            SESSION_MIN_OPEN,
            SESSION_MIN_CLOSE,
        ]
        for _i, _rule, cond, _risk in self._entries:
            columns += cond.columns
        for _rule, cond in self._closes:
            columns += cond.columns
        for _path, node in walk_model(self.config):
            if isinstance(node, LevelStop | LevelTarget):
                col = operand_column(node.level)
                if col is not None:
                    columns.append(col)
            atr_period = getattr(node, "atr_period", None)
            if isinstance(atr_period, int) and not isinstance(node, dict):
                columns.append(atr_column(atr_period))
        self._columns = list(dict.fromkeys(columns))
        self.reset()

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        """Clear all per-run state (the runner calls this at the start of every run)."""
        self._cache = ColumnCache(self._columns)
        self._position: dict | None = None
        self._last_closed = None
        self._bar_index: int | None = None
        self._plans: dict[Any, _EntryPlan] = {}
        self._counted: set[Any] = set()
        self._processed_closes: set[tuple[Any, Any]] = set()
        self._session_trades: dict[float, int] = {}
        self._stopped_sessions: set[float] = set()
        self._active: _ActiveTrade | None = None

    def get_feature_specs(self) -> list[tuple[str, dict]]:
        return derive_feature_specs(self.config, bar_label=self.bar_label)

    def on_bar_context(self, *, position: dict | None, last_closed_trade: Any = None, bar_index: int | None = None):
        """Runner hook: the position and the latest closed trade after Phase A/B of this bar."""
        self._position = position
        self._last_closed = last_closed_trade
        self._bar_index = bar_index

    def validate_config(self) -> bool:
        return True

    # ------------------------------------------------------------------ bookkeeping

    def _session_of(self, row: int) -> float:
        return self._cache.get(SESSION_ID, row)

    def _count_entry(self, entry_time: Any, fallback_session: float) -> _EntryPlan | None:
        plan = self._plans.pop(entry_time, None)
        if entry_time in self._counted:
            return plan
        self._counted.add(entry_time)
        session = plan.session_id if plan is not None else fallback_session
        if not math.isnan(session):
            self._session_trades[session] = self._session_trades.get(session, 0) + 1
        return plan

    def _reconcile(self, row: int) -> None:
        """Fold the runner-reported position changes into the strategy state."""
        current_session = self._session_of(row)
        closed = self._last_closed
        if closed is not None:
            key = (closed.entry_time, closed.exit_time)
            if key not in self._processed_closes:
                self._processed_closes.add(key)
                self._count_entry(closed.entry_time, current_session)
                if closed.action == "close_sl":
                    session = current_session
                    if math.isnan(session) and self._active is not None:
                        session = self._active.session_id
                    self._stopped_sessions.add(session)
                if self._active is not None and self._active.entry_time == closed.entry_time:
                    self._active = None

        pos = self._position
        if pos is None:
            self._active = None
            return
        if self._active is None or self._active.entry_time != pos["entry_time"]:
            plan = self._count_entry(pos["entry_time"], current_session)
            self._active = _ActiveTrade(
                entry_time=pos["entry_time"],
                side=pos["side"],
                rule_index=plan.rule_index if plan is not None else -1,
                session_id=plan.session_id if plan is not None else current_session,
                entry_row=plan.row if plan is not None else row,
                risk_per_unit=plan.risk_per_unit if plan is not None else None,
                stop_loss=pos.get("stop_loss_price"),
                entry_price=float(pos["entry_price"]),
                best_price=float(pos["entry_price"]),
            )
        else:
            self._active.stop_loss = pos.get("stop_loss_price")
        stale = [ts for ts, plan in self._plans.items() if plan.row < row - 2]
        for ts in stale:
            del self._plans[ts]

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        if features.empty:
            return []
        row = self._cache.sync(features)
        self._reconcile(row)
        if self._position is not None:
            return self._manage_position(row)
        return self._try_entries(row)

    def _signal(self, action: SignalAction, row: int, **extra) -> Signal:
        ts = self._cache.index[row]
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            market=self.market,
            instrument=self.instrument,
            action=action,
            confidence=1.0,
            interval=self.interval,
            signal_time=ts if isinstance(ts, datetime) else pd.Timestamp(ts).to_pydatetime(),
            **extra,
        )

    def _entry_blocked(self, risk: RiskBlock, session: float, row: int) -> bool:
        cap = risk.max_trades_per_session
        if cap is not None and self._session_trades.get(session, 0) >= cap:
            return True
        if risk.no_reentry_after_stop and session in self._stopped_sessions:
            return True
        min_close = self._cache.get(SESSION_MIN_CLOSE, row)
        if risk.close_on_session_change is not False and min_close <= 0:
            return True  # the position would be closed on this very bar's session end
        flat_m = risk.flat_before_session_end_minutes
        return flat_m is not None and min_close <= flat_m

    def _try_entries(self, row: int) -> list[Signal]:
        session = self._session_of(row)
        if math.isnan(session):
            return []
        for index, rule, cond, risk in self._entries:
            if self._entry_blocked(risk, session, row):
                continue
            if not cond.evaluate(self._cache, row):
                continue
            levels = self._entry_levels(rule.action, risk, row)
            if levels is None:
                continue
            stop, target, r_unit = levels
            ts = self._cache.index[row]
            self._plans[ts] = _EntryPlan(index, session, row, stop, r_unit)
            action = SignalAction.LONG if rule.action == "long" else SignalAction.SHORT
            return [
                self._signal(
                    action,
                    row,
                    stop_loss_price=stop,
                    take_profit_price=target,
                    metadata={"dsl_version": 2, "rule_id": rule.id},
                )
            ]
        return []

    def _entry_levels(
        self, side: str, risk: RiskBlock, row: int
    ) -> tuple[float | None, float | None, float | None] | None:
        """(stop, target, R) at the signal bar's close, or None to skip the entry."""
        close = self._cache.get("close", row)
        sign = 1.0 if side == "long" else -1.0
        stop = None
        sl = risk.stop_loss
        if isinstance(sl, PointsStop):
            stop = close - sign * sl.value
        elif isinstance(sl, AtrStop):
            stop = close - sign * sl.value * self._cache.get(atr_column(sl.atr_period), row)
        elif isinstance(sl, LevelStop):
            stop = operand_value(sl.level, self._cache, row) - sign * sl.offset_points
        if stop is not None and (math.isnan(stop) or sign * (close - stop) <= 0):
            logger.info(
                "RuleStrategyV2 %s: skip %s entry, stop %.2f not beyond close %.2f", self.name, side, stop, close
            )
            return None
        r_unit = abs(close - stop) if stop is not None else None

        target = None
        tp = risk.take_profit
        if isinstance(tp, PointsTarget):
            target = close + sign * tp.value
        elif isinstance(tp, AtrTarget):
            target = close + sign * tp.value * self._cache.get(atr_column(tp.atr_period), row)
        elif isinstance(tp, RMultipleTarget):
            target = close + sign * tp.value * r_unit
        elif isinstance(tp, LevelTarget):
            target = operand_value(tp.level, self._cache, row)
        if target is not None and (math.isnan(target) or sign * (target - close) <= 0):
            logger.info(
                "RuleStrategyV2 %s: skip %s entry, target %.2f not beyond close %.2f", self.name, side, target, close
            )
            return None
        return stop, target, r_unit

    def _manage_position(self, row: int) -> list[Signal]:
        """Automatic exits first, then close rules, then a trailing-stop update."""
        side = self._position["side"]
        active = self._active
        risk = self._risk_by_rule.get(active.rule_index) if active is not None else None
        if risk is None:
            risk = merge_risk(self.config.risk, None)

        reason = self._automatic_exit(risk, active, row)
        if reason is not None:
            return [self._signal(SignalAction.CLOSE, row, metadata={"dsl_version": 2, "exit": reason})]

        for rule, cond in self._closes:
            if rule.exit_side in (side, "any") and cond.evaluate(self._cache, row):
                return [self._signal(SignalAction.CLOSE, row, metadata={"dsl_version": 2, "rule_id": rule.id})]

        if risk.trailing is not None and active is not None:
            new_stop = self._trail(risk, active, side, row)
            if new_stop is not None:
                active.stop_loss = new_stop
                return [
                    self._signal(SignalAction.HOLD, row, metadata={"dsl_version": 2, "updated_stop_loss": new_stop})
                ]
        return []

    def _automatic_exit(self, risk: RiskBlock, active: _ActiveTrade | None, row: int) -> str | None:
        session = self._session_of(row)
        min_close = self._cache.get(SESSION_MIN_CLOSE, row)
        close_on_change = risk.close_on_session_change is not False
        if close_on_change and active is not None:
            # Never carry a position across a session boundary: leave on the bar
            # that closes at the scheduled end, or on the first bar that is not
            # in the entry session (half days, missing end bars, out-of-session).
            if math.isnan(session) or session != active.session_id:
                return "session_change"
            if min_close <= 0:
                return "session_end"
        flat_m = risk.flat_before_session_end_minutes
        if flat_m is not None and not math.isnan(min_close) and min_close <= flat_m:
            return "flat_before_session_end"
        if risk.max_hold_bars is not None and active is not None and row - active.entry_row >= risk.max_hold_bars:
            return "max_hold_bars"
        return None

    def _trail(self, risk: RiskBlock, active: _ActiveTrade, side: str, row: int) -> float | None:
        """Tighten-only trailing stop from the best price since entry; None when unchanged."""
        trailing = risk.trailing
        close = self._cache.get("close", row)
        if side == "long":
            active.best_price = max(active.best_price, self._cache.get("high", row))
        else:
            active.best_price = min(active.best_price, self._cache.get("low", row))

        if not active.trailing_active:
            threshold = trailing.activate_after_r
            if threshold is None:
                active.trailing_active = True
            elif active.risk_per_unit is not None:
                excursion = (
                    active.best_price - active.entry_price if side == "long" else active.entry_price - active.best_price
                )
                active.trailing_active = excursion >= threshold * active.risk_per_unit
            if not active.trailing_active:
                return None

        if isinstance(trailing, AtrTrailing):
            distance = trailing.value * self._cache.get(atr_column(trailing.atr_period), row)
        else:
            distance = float(trailing.value)
        if math.isnan(distance) or math.isnan(close):
            return None
        current = active.stop_loss
        if side == "long":
            candidate = min(active.best_price - distance, close)
            if current is None or candidate > current:
                return candidate
        else:
            candidate = max(active.best_price + distance, close)
            if current is None or candidate < current:
                return candidate
        return None


def build_rule_strategy(config: dict, *, strategy_id: UUID | None = None) -> BaseStrategy:
    """Strategy for a stored ``rule`` config: v1 ``RuleStrategy`` or, for ``dsl_version: 2``, ``RuleStrategyV2``."""
    from poseidon.strategies.dsl.versions import dsl_version_of
    from poseidon.strategies.rule_strategy import RuleStrategy

    if isinstance(config, dict) and dsl_version_of(config) == 2:
        return RuleStrategyV2(config=config, strategy_id=strategy_id)
    return RuleStrategy(config=config, strategy_id=strategy_id)
