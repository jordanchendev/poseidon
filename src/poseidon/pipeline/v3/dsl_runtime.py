"""``direction.dsl_rules`` at run time: a ``RuleStrategyV2`` driven by the target-position engine.

The v1 runner (``backtest.runner.BacktestRunner`` + ``BacktestPortfolio``) is the reference; this
adapter reproduces what that runner shows the strategy, so a DSL v2 document trades the same bars
the same way on either engine (design 2026-10-24 §3.4, step 7):

* features are computed once on the whole bar frame the run is given and sliced to the decision
  bar (the v1 convention; causality rests on the DSL v2 truncation tests), and the DSL is not
  evaluated before every feature column is non-NaN (``BacktestRunner._find_warmup_end``);
* each bar the strategy gets ``on_bar_context`` with the position after protective exits, built
  from the engine's fills (entry time and price of the opening fill, the resting stop and
  target) and the last closed trade (``close_sl`` / ``close_tp`` / ``close``), never from its own
  signals. Roll legs are not trades: the position continues on the next month;
* LONG / SHORT / CLOSE become direction +1 / -1 / 0 and no signal keeps the current side; an
  entry's stop and target and a HOLD's ``updated_stop_loss`` become ``Protect`` orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from poseidon.pipeline.engine.view import Fill, Protect
from poseidon.signals.schemas import SignalAction

if TYPE_CHECKING:
    from poseidon.pipeline.v3.runtime import SleeveView
    from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2

CLOSE_ACTION = {"stop": "close_sl", "take_profit": "close_tp"}  # any other closing fill: "close"
SIDE = {SignalAction.LONG: 1, SignalAction.SHORT: -1, SignalAction.CLOSE: 0}


@dataclass(frozen=True)
class ClosedTrade:
    """What ``RuleStrategyV2._reconcile`` reads of the runner's last closed ``TradeRecord``."""

    entry_time: Any
    exit_time: Any
    action: str


@dataclass(frozen=True)
class _Open:
    entry_time: pd.Timestamp
    side: str
    entry_price: float


class DslRulesState:
    """Per-run state of one ``direction.dsl_rules`` sleeve; :meth:`bind` before the first bar."""

    def __init__(self) -> None:
        self.rules: RuleStrategyV2 | None = None
        self.features: pd.DataFrame | None = None
        self.warmup_end = 0
        self.units = 0
        self.open: _Open | None = None
        self.last_closed: ClosedTrade | None = None
        self.pending_protect: Protect | None = None

    def bind(self, rules: RuleStrategyV2, bars: pd.DataFrame) -> None:
        """Drive ``rules`` over ``bars``: the exact frame the engine is given."""

        from poseidon.backtest.runner import BacktestRunner
        from poseidon.data.feature_engine import FeatureEngine, is_nonprice_spec

        specs = rules.get_feature_specs()
        if any(is_nonprice_spec(name) for name, _ in specs):
            raise NotImplementedError("direction.dsl_rules on the target engine computes price features only")
        features = FeatureEngine().compute_from_df(bars, specs)
        added = [column for column in features.columns if column not in set(bars.columns)]
        self.warmup_end = BacktestRunner._find_warmup_end(features, added)
        rules.reset()
        self.rules = rules
        self.features = features

    def _fold(self, fills: tuple[Fill, ...], alias: str) -> None:
        for fill in fills:
            if fill.instrument != alias or fill.reason == "roll":
                continue
            before, after = self.units, self.units + fill.units
            if before and (after == 0 or (after > 0) != (before > 0)):
                action = CLOSE_ACTION.get(fill.reason, "close")
                self.last_closed = ClosedTrade(self.open.entry_time, fill.date, action)
                self.open = None
            if after and (before == 0 or (after > 0) != (before > 0)):
                self.open = _Open(fill.date, "long" if after > 0 else "short", fill.price)
            self.units = after

    def step(self, view: SleeveView) -> int:
        if self.rules is None:
            raise RuntimeError("direction.dsl_rules state is not bound to a strategy and bars")
        book, alias = view.book, view.alias
        self._fold(book.fills_today, alias)
        held = book.positions[alias]
        if held != self.units:
            raise RuntimeError(f"dsl_rules ledger holds {self.units} units, the engine {held}")
        row = len(view.series("self:close").values) - 1
        if self.features.index[row] != view.date:
            raise RuntimeError(f"bar {view.date} is not feature row {row} ({self.features.index[row]})")
        self.pending_protect = None
        side = (held > 0) - (held < 0)
        if row < self.warmup_end:
            return side

        resting = book.protective.get(alias)
        position = None
        if held:
            position = {
                "side": self.open.side,
                "quantity": float(abs(held)),
                "entry_price": self.open.entry_price,
                "entry_time": self.open.entry_time,
                "stop_loss_price": resting.stop if resting else None,
                "take_profit_price": resting.take_profit if resting else None,
            }
        self.rules.on_bar_context(position=position, last_closed_trade=self.last_closed, bar_index=row)
        direction = side
        for signal in self.rules.evaluate(self.features.iloc[: row + 1]):
            if signal.action == SignalAction.HOLD:
                stop = (signal.metadata or {}).get("updated_stop_loss")
                if stop is not None and held:
                    self.pending_protect = Protect(stop, resting.take_profit if resting else None)
                continue
            direction = SIDE[signal.action]
            self.pending_protect = Protect(signal.stop_loss_price, signal.take_profit_price) if direction else None
        return direction
