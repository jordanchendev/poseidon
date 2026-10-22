"""DSL v2 performance (slice 8): one year of TX 5m bars in under 30 s; cache == rebuild."""

from __future__ import annotations

import time

import pandas as pd

from poseidon.strategies.dsl.v2 import evaluator
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from tests.dsl_v2_support import random_walk_bars, real_tx_fixture, run_v2


def _busy_orb() -> dict:
    doc = tx_orb30_day()
    doc["rules"][0]["condition"]["all"].append(
        {"type": "compare", "a": {"price": "close"}, "op": ">", "b": {"ind": "ema", "period": 50}}
    )
    doc["rules"][1]["condition"]["all"].append(
        {"type": "compare", "a": {"price": "close"}, "op": "<", "b": {"session": "vwap"}}
    )
    doc["risk"]["trailing"] = {"type": "atr_mult", "value": 2}
    return doc


def test_one_year_of_5m_bars_under_30_seconds():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-01", "2025-12-31")]
    ohlcv = random_walk_bars(days, seed=1)
    assert len(ohlcv) > 55_000
    start = time.perf_counter()
    result, _, _ = run_v2(_busy_orb(), ohlcv)
    elapsed = time.perf_counter() - start
    assert result.trade_count > 100
    assert elapsed < 30.0, f"{len(ohlcv)} bars took {elapsed:.1f}s"


class _AlwaysRebuild(evaluator.ColumnCache):
    def sync(self, frame):
        self._len = 0  # force the full rebuild path every bar
        return super().sync(frame)


def test_incremental_cache_matches_full_rebuild(monkeypatch):
    ohlcv = real_tx_fixture()
    fast, _, _ = run_v2(_busy_orb(), ohlcv)
    monkeypatch.setattr("poseidon.strategies.rule_strategy_v2.ColumnCache", _AlwaysRebuild)
    slow, _, _ = run_v2(_busy_orb(), ohlcv)
    assert fast.trades == slow.trades and fast.trade_count > 0
