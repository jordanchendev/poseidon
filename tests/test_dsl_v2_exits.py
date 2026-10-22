"""DSL v2 automatic exits (slice 7): session boundaries, max hold, trailing stops."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.session import compute_tx_session
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2
from tests.dsl_v2_support import (
    bars_from_close,
    make_runner,
    random_walk_bars,
    real_tx_fixture,
    round_trips,
    run_v2,
    thalassa_stamps,
)

ALWAYS = {"type": "compare", "a": {"price": "close"}, "op": ">", "b": {"const": 0}}
DAYS = ["2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"]


def _doc(action: str, risk: dict | None = None) -> dict:
    doc = {
        "dsl_version": 2,
        "name": "exit_test",
        "symbol": "TX",
        "market": "tw_futures",
        "interval": "5m",
        "rules": [{"id": "go", "action": action, "condition": ALWAYS}],
    }
    if risk:
        doc["risk"] = risk
    return doc


def _session_end_label(session_id: float) -> pd.Timestamp:
    opendate = pd.Timestamp(str(int(session_id) // 10))
    if int(session_id) % 10 == 1:
        return (opendate + pd.Timedelta("13:45:00")).tz_localize("Asia/Taipei")
    return (opendate + pd.Timedelta(days=1, hours=5)).tz_localize("Asia/Taipei")


def _assert_no_session_crossing(ohlcv: pd.DataFrame, trades: list[dict], strict: bool) -> None:
    sessions = compute_tx_session(ohlcv)
    trips = round_trips(trades)
    assert trips, "strategy should trade"
    for t in trips:
        entry = pd.Timestamp(t["entry_time"])
        exit_ = pd.Timestamp(t["exit_time"])
        entry_sid = sessions.loc[entry, "s_id"]
        assert not np.isnan(entry_sid)
        assert exit_.tz_convert("Asia/Taipei") <= _session_end_label(entry_sid), t
        if strict:
            held = sessions.loc[(sessions.index > entry) & (sessions.index <= exit_), "s_id"]
            assert (held == entry_sid).all(), t


def _no_stub(df: pd.DataFrame) -> pd.DataFrame:
    hhmm = df.index.tz_convert("Asia/Taipei").strftime("%H:%M")
    return df[~np.isin(hhmm, ["13:45", "05:00"])]


@pytest.mark.parametrize("action", ["long", "short"])
@pytest.mark.parametrize(
    "risk",
    [
        None,
        {"flat_before_session_end_minutes": 10},
        {"close_on_session_change": False, "flat_before_session_end_minutes": 0},
    ],
    ids=["default", "flat10", "flat0_only"],
)
@pytest.mark.parametrize("data", ["synthetic", "synthetic_no_stub", "real_tx"])
def test_no_position_crosses_1345_or_0500(action, risk, data):
    ohlcv = {
        "synthetic": lambda: random_walk_bars(DAYS, seed=11),
        "synthetic_no_stub": lambda: _no_stub(random_walk_bars(DAYS, seed=12)),
        "real_tx": real_tx_fixture,
    }[data]()
    result, _, _ = run_v2(_doc(action, risk), ohlcv)
    _assert_no_session_crossing(ohlcv, result.trades, strict=True)
    sessions = compute_tx_session(ohlcv)
    exits = [pd.Timestamp(t["exit_time"]) for t in round_trips(result.trades)]
    expected_min_close = 10 if risk and risk.get("flat_before_session_end_minutes") == 10 else 0
    exit_min_close = sessions.loc[exits, "s_min_close"]
    if data != "real_tx":  # complete synthetic sessions: exit exactly on the scheduled bar
        assert (exit_min_close == expected_min_close).all()


def test_session_change_fallback_on_half_day():
    """A session whose late bars are missing (half day) is left on the next session's first bar."""
    full = random_walk_bars(["2026-09-07", "2026-09-08"], seed=13, night=False)
    hhmm = full.index.tz_convert("Asia/Taipei")
    half = full[~((hhmm.day == 7) & (hhmm.strftime("%H:%M") > "11:55"))]
    result, _, _ = run_v2(_doc("long"), half)
    first = round_trips(result.trades)[0]
    exit_local = pd.Timestamp(first["exit_time"]).tz_convert("Asia/Taipei")
    assert exit_local == pd.Timestamp("2026-09-08 08:45", tz="Asia/Taipei")


def test_max_hold_bars():
    ohlcv = random_walk_bars(["2026-09-07", "2026-09-08"], seed=14, night=False)
    result, _, _ = run_v2(_doc("long", {"max_hold_bars": 3, "close_on_session_change": False}), ohlcv)
    pos = {ts: i for i, ts in enumerate(ohlcv.index)}
    holds = [pos[pd.Timestamp(t["exit_time"])] - pos[pd.Timestamp(t["entry_time"])] for t in round_trips(result.trades)]
    assert len(holds) > 20 and set(holds) == {3}


class _StopRecorder(RuleStrategyV2):
    def on_bar_context(self, *, position, last_closed_trade=None, bar_index=None):
        if position is not None:
            self.stops.setdefault(position["entry_time"], []).append(position["stop_loss_price"])
        super().on_bar_context(position=position, last_closed_trade=last_closed_trade, bar_index=bar_index)

    def reset(self):
        super().reset()
        self.stops = {}


def _trend_bars(up_bars: int, down_bars: int, step: float, sign: float = 1.0) -> pd.DataFrame:
    idx = thalassa_stamps(["2026-09-07"], night=False)[: up_bars + down_bars + 1]
    moves = np.concatenate([[0.0], np.full(up_bars, step), np.full(down_bars, -step)]) * sign
    return bars_from_close(idx, 23000.0 + np.cumsum(moves))


def _run_recorder(doc: dict, ohlcv: pd.DataFrame):
    strategy = _StopRecorder(config=doc)
    result = make_runner(strategy).run(ohlcv)
    assert result.status == "completed", result.error_message
    return result, strategy


def test_trailing_stop_is_monotone_and_exits_long():
    ohlcv = _trend_bars(30, 20, 3.0)
    result, strategy = _run_recorder(
        _doc("long", {"trailing": {"type": "points", "value": 10}, "max_trades_per_session": 1}), ohlcv
    )
    (stops,) = strategy.stops.values()
    seen = [s for s in stops if s is not None]
    assert len(set(seen)) > 10
    assert all(b >= a for a, b in itertools.pairwise(seen))
    assert round_trips(result.trades)[0]["action"] == "close_sl"


def test_trailing_stop_is_monotone_and_exits_short():
    ohlcv = _trend_bars(30, 20, 3.0, sign=-1.0)
    result, strategy = _run_recorder(
        _doc("short", {"trailing": {"type": "points", "value": 10}, "max_trades_per_session": 1}), ohlcv
    )
    (stops,) = strategy.stops.values()
    seen = [s for s in stops if s is not None]
    assert len(set(seen)) > 10
    assert all(b <= a for a, b in itertools.pairwise(seen))
    assert round_trips(result.trades)[0]["action"] == "close_sl"


@pytest.mark.parametrize("action", ["long", "short"])
def test_atr_trailing_monotone_on_random_walk(action):
    ohlcv = random_walk_bars(DAYS, seed=15)
    doc = _doc(action, {"stop_loss": {"type": "points", "value": 30}, "trailing": {"type": "atr_mult", "value": 1.5}})
    _result, strategy = _run_recorder(doc, ohlcv)
    assert len(strategy.stops) > 5
    for stops in strategy.stops.values():
        seq = [s for s in stops if s is not None]
        pairs = list(itertools.pairwise(seq))
        assert all((b >= a) if action == "long" else (b <= a) for a, b in pairs)


def test_trailing_waits_for_activate_after_r():
    ohlcv = _trend_bars(30, 20, 3.0)
    doc = _doc(
        "long",
        {
            "stop_loss": {"type": "points", "value": 10},
            "trailing": {"type": "points", "value": 5, "activate_after_r": 2},
        },
    )
    result, strategy = _run_recorder(doc, ohlcv)
    (entry_time, stops) = next(iter(strategy.stops.items()))
    initial = stops[0]
    first_move = next(i for i, s in enumerate(stops) if s != initial)
    entry_price = round_trips(result.trades)[0]["entry_price"]
    rows = ohlcv.index.get_loc(pd.Timestamp(entry_time))
    # The stop moves on the bar after the one where the best high reached entry + 2R (R = 10).
    best_high_before_move = ohlcv["high"].iloc[rows + 1 : rows + first_move + 1].max()
    assert best_high_before_move - entry_price >= 20
    best_high_earlier = ohlcv["high"].iloc[rows + 1 : rows + first_move].max()
    assert best_high_earlier - entry_price < 20
