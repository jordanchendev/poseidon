"""Runner position hook + RuleStrategyV2 state machine (DSL v2 slice 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.data.feature_engine import FeatureEngine
from poseidon.signals.schemas import Signal, SignalAction, SignalStatus
from poseidon.strategies.base import BaseStrategy, StrategyType
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2
from tests.dsl_v2_support import bars_from_close, make_runner, random_walk_bars, round_trips, run_v2, thalassa_stamps

ALWAYS = {"type": "compare", "a": {"price": "close"}, "op": ">", "b": {"const": 0}}
NEVER = {"type": "compare", "a": {"price": "close"}, "op": "<", "b": {"const": 0}}
DAYS = ["2026-09-07", "2026-09-08", "2026-09-09"]


def _doc(rules: list[dict], risk: dict | None = None) -> dict:
    doc = {
        "dsl_version": 2,
        "name": "test_v2",
        "symbol": "TX",
        "market": "tw_futures",
        "interval": "5m",
        "rules": rules,
    }
    if risk is not None:
        doc["risk"] = risk
    return doc


class _Recorder(BaseStrategy):
    name = "recorder"
    strategy_type = StrategyType.RULE
    symbol = "TX"
    market = "tw_futures"
    interval = "5m"
    position_aware = True

    def __init__(self):
        self.contexts: list[tuple] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def on_bar_context(self, *, position, last_closed_trade, bar_index):
        if position is not None:
            position["quantity"] = -999  # must not leak into the runner's portfolio
        self.contexts.append((bar_index, position, last_closed_trade))

    def evaluate(self, features):
        i = len(features) - 1
        action = {3: SignalAction.LONG, 6: SignalAction.CLOSE}.get(i)
        if action is None:
            return []
        return [Signal(symbol="TX", market="tw_futures", action=action, confidence=1.0, signal_time=features.index[i])]

    def validate_config(self):
        return True


def test_runner_feeds_position_copies_and_last_closed_trade():
    idx = thalassa_stamps(["2026-09-07"], night=False)[:10]
    ohlcv = bars_from_close(idx, np.arange(10) + 23000.0)
    strategy = _Recorder()
    runner = make_runner(strategy)
    result = runner.run(ohlcv, feature_specs=[])
    assert result.status == "completed"
    assert strategy.resets == 1
    by_bar = {i: (pos, closed) for i, pos, closed in strategy.contexts}
    assert sorted(by_bar) == list(range(10))
    assert by_bar[3] == (None, None)
    pos4, _ = by_bar[4]
    assert pos4["side"] == "long" and pos4["quantity"] == -999
    assert by_bar[7][0] is None and by_bar[7][1].exit_time == ohlcv.index[6]
    assert round_trips(result.trades)[0]["quantity"] > 0  # the copy mutation did not leak


def test_non_position_aware_strategy_gets_no_hook():
    calls = []

    class Plain(_Recorder):
        position_aware = False

        def on_bar_context(self, **kw):
            calls.append(kw)

    make_runner(Plain()).run(bars_from_close(thalassa_stamps(["2026-09-07"], night=False)[:8], np.ones(8) * 23000))
    assert calls == []


def test_v2_holds_one_position_where_v1_overwrites():
    ohlcv = random_walk_bars(DAYS, seed=1)
    result, _, _ = run_v2(
        _doc([{"id": "go", "action": "long", "condition": ALWAYS}], {"close_on_session_change": False}), ohlcv
    )
    assert len(result.trades) == 1 and result.trades[0]["action"] == "long"
    v1 = RuleStrategy(
        config={
            "name": "v1",
            "symbol": "TX",
            "market": "tw_futures",
            "interval": "5m",
            "rules": [{"condition": {"type": "feature_above", "column": "close", "threshold": 0}, "action": "long"}],
        }
    )
    v1_result = make_runner(v1).run(ohlcv, feature_specs=[])
    assert v1_result.trade_count > 100  # v1 re-enters every bar, overwriting the position


def test_first_matching_entry_rule_wins():
    ohlcv = random_walk_bars(DAYS, seed=2)
    rules = [
        {"id": "first", "action": "short", "condition": ALWAYS},
        {"id": "second", "action": "long", "condition": ALWAYS},
    ]
    result, _, _ = run_v2(_doc(rules, {"close_on_session_change": False}), ohlcv)
    assert [t["action"] for t in result.trades] == ["short"]


def test_close_rules_respect_exit_side():
    ohlcv = random_walk_bars(["2026-09-07"], seed=3, night=False)
    keep = _doc(
        [
            {"id": "go", "action": "long", "condition": ALWAYS},
            {"id": "x", "action": "close", "exit_side": "short", "condition": ALWAYS},
        ],
        {"close_on_session_change": False},
    )
    result, _, _ = run_v2(keep, ohlcv)
    assert len(round_trips(result.trades)) == 0
    flip = _doc(
        [
            {"id": "go", "action": "long", "condition": ALWAYS},
            {"id": "x", "action": "close", "exit_side": "any", "condition": ALWAYS},
        ]
    )
    result, _, _ = run_v2(flip, ohlcv)
    trips = round_trips(result.trades)
    assert len(trips) >= 25  # enter, close next bar, re-enter...
    assert all(t["action"] == "close" for t in trips)


class _RejectFirstEntry:
    def __init__(self):
        self.rejected = 0

    def evaluate(self, signal, _portfolio):
        if signal.action != SignalAction.CLOSE and self.rejected == 0:
            self.rejected += 1
            signal.status = SignalStatus.REJECTED
            return signal
        signal.status = SignalStatus.PASSED
        return signal


def test_trade_cap_counts_runner_fills_not_emitted_signals():
    ohlcv = random_walk_bars(["2026-09-07", "2026-09-08"], seed=4, night=False)
    doc = _doc(
        [
            {"id": "go", "action": "long", "condition": ALWAYS},
            {"id": "x", "action": "close", "exit_side": "long", "condition": ALWAYS},
        ],
        {"max_trades_per_session": 1},
    )
    risk = _RejectFirstEntry()
    result, _, _ = run_v2(doc, ohlcv, risk_engine=risk)
    assert risk.rejected == 1
    trips = round_trips(result.trades)
    sessions = {pd.Timestamp(t["entry_time"]).tz_convert("Asia/Taipei").date() for t in trips}
    assert len(trips) == 2 and len(sessions) == 2  # one per session, the rejected attempt not counted


def test_trade_cap_counts_long_and_short_together():
    ohlcv = random_walk_bars(["2026-09-07"], seed=5, night=False)
    doc = _doc(
        [
            {"id": "s", "action": "short", "condition": ALWAYS},
            {"id": "l", "action": "long", "condition": ALWAYS},
            {"id": "x", "action": "close", "exit_side": "any", "condition": ALWAYS},
        ],
        {"max_trades_per_session": 3},
    )
    result, _, _ = run_v2(doc, ohlcv)
    assert len(round_trips(result.trades)) == 3


def _falling_day(dates: list[str]) -> pd.DataFrame:
    idx = thalassa_stamps(dates, night=False)
    close = 23000.0 - 3.0 * np.arange(len(idx))
    return bars_from_close(idx, close)


def test_no_reentry_after_stop_blocks_rest_of_session_only():
    ohlcv = _falling_day(["2026-09-07", "2026-09-08"])
    rules = [{"id": "go", "action": "long", "condition": ALWAYS}]
    stop = {"stop_loss": {"type": "points", "value": 5}}
    blocked, _, _ = run_v2(_doc(rules, {**stop, "no_reentry_after_stop": True}), ohlcv)
    trips = round_trips(blocked.trades)
    assert [t["action"] for t in trips] == ["close_sl", "close_sl"]
    assert len({pd.Timestamp(t["entry_time"]).tz_convert("Asia/Taipei").date() for t in trips}) == 2
    free, _, _ = run_v2(_doc(rules, stop), ohlcv)
    assert len(round_trips(free.trades)) > 20


def test_stop_on_wrong_side_skips_entry():
    ohlcv = random_walk_bars(["2026-09-07"], seed=6, night=False)
    long_bad = _doc(
        [
            {
                "id": "l",
                "action": "long",
                "condition": ALWAYS,
                "risk": {"stop_loss": {"type": "level", "level": {"session": "high"}}},
            }
        ]
    )
    assert run_v2(long_bad, ohlcv)[0].trade_count == 0
    short_bad = _doc(
        [
            {
                "id": "s",
                "action": "short",
                "condition": ALWAYS,
                "risk": {"stop_loss": {"type": "level", "level": {"session": "low"}}},
            }
        ]
    )
    assert run_v2(short_bad, ohlcv)[0].trade_count == 0


def _signal_at(doc: dict, ohlcv: pd.DataFrame, row: int) -> Signal:
    strategy = RuleStrategyV2(config=doc)
    features = FeatureEngine().compute_from_df(ohlcv, strategy.get_feature_specs())
    strategy.on_bar_context(position=None, last_closed_trade=None, bar_index=row)
    (signal,) = strategy.evaluate(features.iloc[: row + 1])
    return signal


def test_initial_stop_and_r_multiple_target_on_signal():
    ohlcv = random_walk_bars(["2026-09-07"], seed=7, night=False)
    close = float(ohlcv["close"].iloc[10])
    long_doc = _doc(
        [{"id": "l", "action": "long", "condition": ALWAYS}],
        {"stop_loss": {"type": "points", "value": 10}, "take_profit": {"type": "r_multiple", "value": 2}},
    )
    sig = _signal_at(long_doc, ohlcv, 10)
    assert sig.action == SignalAction.LONG
    assert (sig.stop_loss_price, sig.take_profit_price) == (close - 10, close + 20)
    assert sig.metadata["rule_id"] == "l"
    short_doc = _doc(
        [
            {
                "id": "s",
                "action": "short",
                "condition": ALWAYS,
                "risk": {"stop_loss": {"type": "points", "value": 8}, "take_profit": {"type": "points", "value": 30}},
            }
        ],
    )
    sig = _signal_at(short_doc, ohlcv, 10)
    assert (sig.stop_loss_price, sig.take_profit_price) == (close + 8, close - 30)


def test_reset_clears_state_between_runs():
    ohlcv = random_walk_bars(["2026-09-07"], seed=8, night=False)
    strategy = RuleStrategyV2(
        config=_doc(
            [
                {"id": "go", "action": "long", "condition": ALWAYS},
                {"id": "x", "action": "close", "exit_side": "any", "condition": ALWAYS},
            ],
            {"max_trades_per_session": 2},
        )
    )
    runner = make_runner(strategy)
    first = runner.run(ohlcv)
    second = make_runner(strategy).run(ohlcv)
    assert first.trades == second.trades and len(round_trips(first.trades)) == 2
