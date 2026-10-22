"""DSL v2 evaluator + feature spec derivation (slice 5)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from poseidon.data.feature_engine import FeatureEngine
from poseidon.strategies.dsl.v2 import validate_v2
from poseidon.strategies.dsl.v2.evaluator import ColumnCache, CompiledCondition, derive_feature_specs
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from poseidon.strategies.dsl.v2.schema import Condition

_COND = TypeAdapter(Condition)


def _frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {"s_id": [20260901 * 10 + 1.0] * n, "s_kind": [1.0] * n, "s_tod": [600.0] * n, "s_min_open": [75.0] * n}
    base.update(cols)
    idx = pd.date_range("2026-09-01 01:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(base, index=idx)


def _fires(cond: dict, frame: pd.DataFrame, row: int | None = None) -> bool:
    compiled = CompiledCondition(_COND.validate_python(cond))
    cache = ColumnCache(compiled.columns)
    last = cache.sync(frame)
    return compiled.evaluate(cache, last if row is None else row)


CROSS_UP = {"type": "cross", "a": {"price": "close"}, "direction": "above", "b": {"ind": "ema", "period": 20}}


def test_cross_above_and_below():
    f = _frame(close=[99.0, 101.0], ema_20=[100.0, 100.0])
    assert _fires(CROSS_UP, f)
    assert not _fires({**CROSS_UP, "direction": "below"}, f)
    g = _frame(close=[101.0, 99.0], ema_20=[100.0, 100.0])
    assert _fires({**CROSS_UP, "direction": "below"}, g)
    # Touching (prev equal) counts as "from below"; staying above does not cross.
    assert _fires(CROSS_UP, _frame(close=[100.0, 100.5], ema_20=[100.0, 100.0]))
    assert not _fires(CROSS_UP, _frame(close=[100.5, 101.0], ema_20=[100.0, 100.0]))


def test_cross_points_and_atr_offsets():
    f = _frame(close=[99.0, 103.0], ema_20=[100.0, 100.0], atr_14=[4.0, 4.0])
    assert _fires({**CROSS_UP, "offset": {"points": 2}}, f)
    assert not _fires({**CROSS_UP, "offset": {"points": 3}}, f)
    assert not _fires({**CROSS_UP, "offset": {"atr_mult": 0.75}}, f)  # 100 + 3 = 103, needs > 103
    assert _fires({**CROSS_UP, "offset": {"atr_mult": 0.5}}, f)
    below = _frame(close=[101.0, 97.0], ema_20=[100.0, 100.0])
    assert _fires({**CROSS_UP, "direction": "below", "offset": {"points": 2}}, below)
    assert not _fires({**CROSS_UP, "direction": "below", "offset": {"points": 3}}, below)


def test_compare_ops_and_offset_direction():
    f = _frame(close=[105.0], s_vwap=[100.0])
    base = {"type": "compare", "a": {"price": "close"}, "b": {"session": "vwap"}}
    assert _fires({**base, "op": ">"}, f)
    assert _fires({**base, "op": ">=", "offset": {"points": 5}}, f)
    assert not _fires({**base, "op": ">", "offset": {"points": 5}}, f)
    assert not _fires({**base, "op": "<"}, f)
    g = _frame(close=[95.0], s_vwap=[100.0])
    assert _fires({**base, "op": "<=", "offset": {"points": 5}}, g)
    assert not _fires({**base, "op": "<", "offset": {"points": 5}}, g)
    assert _fires({"type": "compare", "a": {"price": "close"}, "op": ">", "b": {"const": 104.5}}, f)


def test_nan_operand_blocks_whole_rule_even_under_any_and_none():
    f = _frame(close=[99.0, 101.0], ema_20=[100.0, 100.0], s_or_high_30=[np.nan, np.nan])
    nan_leaf = {"type": "compare", "a": {"price": "close"}, "op": ">", "b": {"session": "or_high", "minutes": 30}}
    assert _fires({"any": [CROSS_UP]}, f)
    assert not _fires({"any": [CROSS_UP, nan_leaf]}, f)
    assert not _fires({"none": [nan_leaf]}, f)
    # Cross needs the previous bar too.
    assert not _fires(CROSS_UP, _frame(close=[101.0], ema_20=[100.0]))
    assert not _fires(CROSS_UP, _frame(close=[99.0, 101.0], ema_20=[np.nan, 100.0]))


def test_out_of_session_bar_never_fires():
    f = _frame(close=[99.0, 101.0], ema_20=[100.0, 100.0])
    f.loc[f.index[-1], "s_id"] = np.nan
    assert not _fires(CROSS_UP, f)


def test_time_window_and_session_elapsed():
    tw = {"type": "time_window", "session": "day", "start": "09:15", "end": "13:35"}
    assert _fires(tw, _frame(close=[1.0], s_tod=[9 * 60 + 15.0]))
    assert _fires(tw, _frame(close=[1.0], s_tod=[13 * 60 + 35.0]))
    assert not _fires(tw, _frame(close=[1.0], s_tod=[9 * 60 + 10.0]))
    assert not _fires(tw, _frame(close=[1.0], s_tod=[600.0], s_kind=[2.0]))
    wrap = {"type": "time_window", "session": "night", "start": "21:00", "end": "02:00"}
    night = {"s_kind": [2.0]}
    assert _fires(wrap, _frame(close=[1.0], s_tod=[23 * 60.0], **night))
    assert _fires(wrap, _frame(close=[1.0], s_tod=[0.0], **night))
    assert _fires(wrap, _frame(close=[1.0], s_tod=[120.0], **night))
    assert not _fires(wrap, _frame(close=[1.0], s_tod=[20 * 60.0], **night))
    assert not _fires(wrap, _frame(close=[1.0], s_tod=[180.0], **night))
    anyw = {"type": "time_window", "session": "any", "start": "09:00", "end": "10:00"}
    assert _fires(anyw, _frame(close=[1.0], s_tod=[570.0], s_kind=[1.0]))
    el = {"type": "session_elapsed", "min_minutes": 30, "max_minutes": 90}
    assert _fires(el, _frame(close=[1.0], s_min_open=[30.0]))
    assert _fires(el, _frame(close=[1.0], s_min_open=[90.0]))
    assert not _fires(el, _frame(close=[1.0], s_min_open=[95.0]))
    assert not _fires({"type": "session_elapsed", "min_minutes": 100}, _frame(close=[1.0], s_min_open=[95.0]))


def test_column_cache_appends_and_rebuilds_without_reading_ahead():
    f = _frame(close=[float(i) for i in range(40)])
    cache = ColumnCache(["close"])
    assert cache.sync(f.iloc[:5]) == 4
    for k in range(6, 41):
        assert cache.sync(f.iloc[:k]) == k - 1
        assert cache.get("close", k - 1) == k - 1
        assert math.isnan(cache.get("close", k))  # nothing past the frame
    # A different frame (live re-evaluation) rebuilds.
    g = _frame(close=[7.0, 8.0])
    assert cache.sync(g) == 1 and cache.get("close", 0) == 7.0
    with pytest.raises(KeyError):
        ColumnCache(["nope"]).sync(g)


def test_derive_feature_specs_orb():
    spec = validate_v2(tx_orb30_day()).model
    assert derive_feature_specs(spec) == [
        ("tx_session", {"interval_minutes": 5, "bar_label": "open"}),
        ("opening_range", {"minutes": 30, "interval_minutes": 5, "bar_label": "open"}),
    ]


def test_derived_specs_produce_every_column_the_rules_read():
    doc = tx_orb30_day()
    doc["rules"][0]["condition"]["all"].append(
        {
            "type": "compare",
            "a": {"ind": "rsi", "period": 14},
            "op": "<",
            "b": {"session": "prev_day_high"},
            "offset": {"atr_mult": 1, "atr_period": 10},
        }
    )
    doc["risk"]["trailing"] = {"type": "atr_mult", "value": 2, "atr_period": 21}
    doc["rules"][1]["risk"]["stop_loss"] = {"type": "atr_mult", "value": 1.5, "atr_period": 30}
    spec = validate_v2(doc).model
    specs = derive_feature_specs(spec)
    names = [n for n, _ in specs]
    assert names.count("atr") == 3 and "rsi" in names and "session_levels" in names
    idx = pd.date_range("2026-09-01 00:45", periods=300, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    close = 23000 + np.cumsum(rng.standard_normal(300))
    ohlcv = pd.DataFrame({"open": close, "high": close + 2, "low": close - 2, "close": close, "volume": 5.0}, index=idx)
    features = FeatureEngine().compute_from_df(ohlcv, specs)
    for rule in spec.rules:
        assert set(CompiledCondition(rule.condition).columns) <= set(features.columns)
    assert {"atr_21", "atr_30", "atr_10"} <= set(features.columns)
