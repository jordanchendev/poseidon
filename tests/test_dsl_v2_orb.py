"""ORB example end to end (DSL v2 slice 9): hand-reconciled trades + strategy-level no-look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.data.features.session import compute_tx_session
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from tests.dsl_v2_support import random_walk_bars, real_tx_fixture, round_trips, run_v2

COST = COST_MODELS["tw_futures"]


def _reference_orb(ohlcv: pd.DataFrame) -> list[dict]:
    """Independent re-implementation of tx_orb30_day on raw bars (no DSL code)."""
    s = compute_tx_session(ohlcv)
    df = ohlcv.join(s)
    trades = []
    for sid, day in df[df["s_kind"] == 1].groupby("s_id"):
        if day["s_min_open"].iloc[0] > 5:
            continue  # data starts mid-session: no opening range
        window = day[day["s_min_open"] <= 30]
        if window["s_min_open"].max() < 30:
            continue
        or_high, or_low = window["high"].max(), window["low"].min()
        after = day[day["s_min_open"] >= 30]
        entry = None
        for k in range(1, len(after)):
            prev, bar = after.iloc[k - 1], after.iloc[k]
            in_window = 9 * 60 + 15 <= bar["s_tod"] <= 13 * 60 + 35 and bar["s_min_close"] > 10
            if not in_window:
                continue
            if prev["close"] <= or_high < bar["close"]:
                entry = (k, "long", or_low)
            elif prev["close"] >= or_low > bar["close"]:
                entry = (k, "short", or_high)
            if entry:
                break
        if entry is None:
            continue
        k, side, stop = entry
        bar = after.iloc[k]
        close = bar["close"]
        sign = 1 if side == "long" else -1
        if sign * (close - stop) <= 0:
            continue
        target = close + sign * 2 * abs(close - stop)
        fill = close + sign * COST.tick_size * COST.slippage_ticks
        exit_time = exit_price = kind = None
        for j in range(k + 1, len(after)):
            b = after.iloc[j]
            hit_sl = b["low"] <= stop if side == "long" else b["high"] >= stop
            hit_tp = b["high"] >= target if side == "long" else b["low"] <= target
            if hit_sl or hit_tp:
                exit_time, exit_price, kind = (
                    b.name,
                    (stop if hit_sl else target),
                    ("close_sl" if hit_sl else "close_tp"),
                )
                break
            if b["s_min_close"] <= 10:
                exit_time, exit_price, kind = b.name, b["close"] - sign * COST.tick_size * COST.slippage_ticks, "close"
                break
        if exit_time is None:
            continue
        trades.append(
            {
                "side": side,
                "entry_time": bar.name.isoformat(),
                "entry_price": round(fill, 6),
                "exit_time": exit_time.isoformat(),
                "exit_price": round(exit_price, 6),
                "action": kind,
                "session": sid,
            }
        )
    return trades


def _engine_trades(result) -> list[dict]:
    entries = {t["entry_time"]: t["action"] for t in result.trades if t["exit_time"] is None}
    return [
        {
            "side": entries[t["entry_time"]],
            "entry_time": t["entry_time"],
            "entry_price": round(t["entry_price"], 6),
            "exit_time": t["exit_time"],
            "exit_price": round(t["exit_price"], 6),
            "action": t["action"],
        }
        for t in round_trips(result.trades)
    ]


@pytest.mark.parametrize("data", ["real_tx", "synthetic"])
def test_orb_matches_hand_reconciliation(data):
    ohlcv = (
        real_tx_fixture()
        if data == "real_tx"
        else random_walk_bars([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-03-02", "2026-04-30")], seed=21)
    )
    result, _, _ = run_v2(tx_orb30_day(), ohlcv)
    reference = _reference_orb(ohlcv)
    engine = _engine_trades(result)
    assert len(reference) >= 5
    assert engine == [{k: v for k, v in t.items() if k != "session"} for t in reference]
    # PnL is the futures round trip: points * 200 per contract minus fees on both legs.
    for t in round_trips(result.trades):
        sign = 1 if any(e["entry_time"] == t["entry_time"] and e["action"] == "long" for e in result.trades) else -1
        gross = sign * (t["exit_price"] - t["entry_price"]) * t["quantity"] * COST.point_value
        assert t["pnl"] == pytest.approx(gross - t["fees"])
    # One trade per day session at most, never held past 13:35.
    sessions = compute_tx_session(ohlcv)
    exits = [pd.Timestamp(t["exit_time"]) for t in round_trips(result.trades)]
    assert (sessions.loc[exits, "s_min_close"] >= 10).all()
    assert (sessions.loc[exits, "s_kind"] == 1).all()


def _records_until(trades: list[dict], cut: pd.Timestamp) -> list[dict]:
    keep = []
    for t in trades:
        stamp = pd.Timestamp(t["exit_time"] or t["entry_time"])
        if stamp <= cut:
            keep.append(t)
    return keep


@pytest.mark.parametrize("data", ["real_tx", "synthetic"])
def test_strategy_level_no_look_ahead(data):
    """Backtesting bars[:k+1] reproduces every trade record of the full run up to bar k."""
    ohlcv = (
        real_tx_fixture()
        if data == "real_tx"
        else random_walk_bars([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-03-02", "2026-03-13")], seed=22)
    )
    doc = tx_orb30_day()
    doc["risk"]["trailing"] = {"type": "atr_mult", "value": 1.5, "activate_after_r": 0.5}
    doc["rules"][0]["condition"]["all"].append(
        {
            "type": "compare",
            "a": {"session": "vwap"},
            "op": ">",
            "b": {"session": "prev_close"},
            "offset": {"points": 0},
        }
    )
    full, _, _ = run_v2(doc, ohlcv)
    stamps = sorted({pd.Timestamp(t["exit_time"] or t["entry_time"]) for t in full.trades})
    assert len(stamps) >= 6
    rng = np.random.default_rng(0)
    cuts = set(stamps)
    cuts |= {s - pd.Timedelta(minutes=5) for s in stamps}
    cuts |= set(ohlcv.index[rng.choice(np.arange(400, len(ohlcv)), size=10, replace=False)])
    for cut in sorted(cuts):
        k = ohlcv.index.searchsorted(cut, side="right")
        truncated, _, _ = run_v2(doc, ohlcv.iloc[:k])
        assert truncated.trades == _records_until(full.trades, cut), cut
