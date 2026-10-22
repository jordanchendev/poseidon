"""Gap-aware stop fills (owner decision): opt-in runner flag, stops fill at the worse open."""

from __future__ import annotations

import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.optimizer import BayesianOptimizer, GridSearchOptimizer
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.walk_forward import WalkForwardAnalyzer
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType


class _EnterOnce(BaseStrategy):
    name = "enter_once"
    strategy_type = StrategyType.RULE
    symbol = "TX"
    market = "tw_futures"
    interval = "5m"

    def __init__(self, action: SignalAction, stop: float | None, target: float | None):
        self.action, self.stop, self.target = action, stop, target

    def evaluate(self, features):
        if len(features) != 1:
            return []
        return [
            Signal(
                symbol="TX",
                market="tw_futures",
                action=self.action,
                confidence=1.0,
                signal_time=features.index[0],
                stop_loss_price=self.stop,
                take_profit_price=self.target,
            )
        ]

    def validate_config(self):
        return True


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-09-07 01:00", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def _exit(action, stop, target, rows, gap: bool):
    runner = BacktestRunner(
        strategy=_EnterOnce(action, stop, target),
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(),
        cost_model=COST_MODELS["tw_futures"],
        gap_aware_stops=gap,
    )
    result = runner.run(_bars(rows), feature_specs=[])
    (trip,) = [t for t in result.trades if t["exit_time"] is not None]
    return trip["action"], trip["exit_price"]


LONG_GAP = [(23000, 23002, 22998, 23000), (22900, 22920, 22880, 22910)]
SHORT_GAP = [(23000, 23002, 22998, 23000), (23100, 23120, 23090, 23110)]


def test_long_stop_gapped_through_fills_at_open_only_when_enabled():
    assert _exit(SignalAction.LONG, 22950, None, LONG_GAP, gap=False) == ("close_sl", 22950)
    assert _exit(SignalAction.LONG, 22950, None, LONG_GAP, gap=True) == ("close_sl", 22900)


def test_short_stop_gapped_through_fills_at_open_only_when_enabled():
    assert _exit(SignalAction.SHORT, 23050, None, SHORT_GAP, gap=False) == ("close_sl", 23050)
    assert _exit(SignalAction.SHORT, 23050, None, SHORT_GAP, gap=True) == ("close_sl", 23100)


def test_stop_touched_intrabar_still_fills_at_stop():
    rows = [(23000, 23002, 22998, 23000), (22990, 22995, 22940, 22960)]
    assert _exit(SignalAction.LONG, 22950, None, rows, gap=True) == ("close_sl", 22950)


@pytest.mark.parametrize("gap", [False, True])
def test_take_profit_never_gets_the_better_open(gap):
    rows = [(23000, 23002, 22998, 23000), (23200, 23210, 23190, 23205)]
    assert _exit(SignalAction.LONG, 22950, 23100, rows, gap=gap) == ("close_tp", 23100)


def test_flag_passes_through_optimizers_and_walk_forward():
    args = (FeatureEngine(), RiskEngine(), COST_MODELS["tw_futures"])
    assert GridSearchOptimizer(*args, gap_aware_stops=True)._runner_extra == {"gap_aware_stops": True}
    assert BayesianOptimizer(*args, gap_aware_stops=True)._runner_extra == {"gap_aware_stops": True}
    assert WalkForwardAnalyzer(*args, gap_aware_stops=True)._runner_extra == {"gap_aware_stops": True}
    assert BayesianOptimizer(*args)._runner_extra == {}
