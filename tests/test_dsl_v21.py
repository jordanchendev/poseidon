"""DSL v2.1: smoothed indicator, rolling channel and b_mult (schema, features, evaluator, strategy)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.features.derived import compute_rolling_channel, compute_smoothed_indicator
from poseidon.data.features.technical import ATR, RSI
from poseidon.strategies.dsl.v2 import DslValidationError, dsl_v2_json_schema, evaluator, validate_v2
from poseidon.strategies.dsl.v2.evaluator import ColumnCache, CompiledCondition, derive_feature_specs
from poseidon.strategies.dsl.v2.examples import tx_atr_squeeze_breakout, tx_orb30_day
from poseidon.strategies.dsl.v2.schema import Condition
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2
from tests.dsl_v2_support import random_walk_bars, real_tx_fixture, round_trips, run_v2

_COND = TypeAdapter(Condition)

ATR14 = {"ind": "atr", "period": 14}
ATR14_SMA50 = {"ind": "atr", "period": 14, "smooth": {"type": "sma", "period": 50}}
HH20 = {"channel": "highest", "source": "high", "bars": 20}
LL20 = {"channel": "lowest", "source": "low", "bars": 20}
SQUEEZE = {"type": "compare", "a": ATR14, "op": "<", "b": ATR14_SMA50, "b_mult": 0.7}
BREAKOUT = {"type": "cross", "a": {"price": "close"}, "direction": "above", "b": HH20}


def _with_leaf(leaf: dict) -> dict:
    doc = tx_orb30_day()
    doc["rules"][0]["condition"]["all"].append(leaf)
    return doc


# --------------------------------------------------------------------------- schema + validator

LEAF_PATH = "rules.0.condition.all.2"

INVALID: list[tuple[str, dict, str]] = [
    ("channel_bars_low", {**BREAKOUT, "b": {**HH20, "bars": 1}}, f"{LEAF_PATH}.b.bars"),
    ("channel_bars_high", {**BREAKOUT, "b": {**HH20, "bars": 401}}, f"{LEAF_PATH}.b.bars"),
    ("channel_bars_float", {**BREAKOUT, "b": {**HH20, "bars": 20.0}}, f"{LEAF_PATH}.b.bars"),
    ("channel_kind", {**BREAKOUT, "b": {**HH20, "channel": "max"}}, f"{LEAF_PATH}.b.channel"),
    ("channel_source", {**BREAKOUT, "b": {**HH20, "source": "vwap"}}, f"{LEAF_PATH}.b.source"),
    ("channel_extra_key", {**BREAKOUT, "b": {**HH20, "period": 20}}, f"{LEAF_PATH}.b.period"),
    ("channel_missing_bars", {**BREAKOUT, "b": {"channel": "highest", "source": "high"}}, f"{LEAF_PATH}.b.bars"),
    (
        "cross_channel_including_current_bar",
        {**BREAKOUT, "b": {**HH20, "exclude_current": False}},
        f"{LEAF_PATH}.b.exclude_current",
    ),
    (
        "cross_channel_including_current_bar_as_a",
        {"type": "cross", "a": {**LL20, "exclude_current": False}, "direction": "above", "b": {"price": "close"}},
        f"{LEAF_PATH}.a.exclude_current",
    ),
    (
        "smooth_period_low",
        {**SQUEEZE, "b": {**ATR14, "smooth": {"type": "sma", "period": 1}}},
        f"{LEAF_PATH}.b.smooth.period",
    ),
    (
        "smooth_period_high",
        {**SQUEEZE, "b": {**ATR14, "smooth": {"type": "ema", "period": 401}}},
        f"{LEAF_PATH}.b.smooth.period",
    ),
    ("smooth_type", {**SQUEEZE, "b": {**ATR14, "smooth": {"type": "wma", "period": 50}}}, f"{LEAF_PATH}.b.smooth.type"),
    (
        "smooth_extra_key",
        {**SQUEEZE, "b": {**ATR14, "smooth": {"type": "sma", "period": 50, "of": "atr"}}},
        f"{LEAF_PATH}.b.smooth.of",
    ),
    (
        "smooth_base_period_still_bounded",
        {**SQUEEZE, "b": {"ind": "atr", "period": 101, "smooth": {"type": "sma", "period": 50}}},
        f"{LEAF_PATH}.b",
    ),
    ("b_mult_low", {**SQUEEZE, "b_mult": 0.05}, f"{LEAF_PATH}.b_mult"),
    ("b_mult_high", {**SQUEEZE, "b_mult": 11}, f"{LEAF_PATH}.b_mult"),
    ("b_mult_string", {**SQUEEZE, "b_mult": "0.7"}, f"{LEAF_PATH}.b_mult"),
    ("b_mult_on_const", {**SQUEEZE, "b": {"const": 40}}, f"{LEAF_PATH}.b_mult"),
    (
        "compare_high_le_own_channel_is_always_true",
        {"type": "compare", "a": {"price": "high"}, "op": "<=", "b": {**HH20, "exclude_current": False}},
        LEAF_PATH,
    ),
    (
        "compare_close_gt_own_channel_is_never_true",
        {"type": "compare", "a": {"price": "close"}, "op": ">", "b": {**HH20, "exclude_current": False}},
        LEAF_PATH,
    ),
    (
        "compare_flipped_lowest_channel",
        {"type": "compare", "a": {**LL20, "exclude_current": False}, "op": "<=", "b": {"price": "close"}},
        LEAF_PATH,
    ),
]


@pytest.mark.parametrize(("case", "leaf", "path"), INVALID, ids=[c[0] for c in INVALID])
def test_invalid_v21_leaf_rejected_at_path(case, leaf, path):
    with pytest.raises(DslValidationError) as info:
        validate_v2(_with_leaf(leaf))
    assert path in [e["path"] for e in info.value.errors], (case, info.value.errors)


@pytest.mark.parametrize(
    "leaf",
    [
        SQUEEZE,
        BREAKOUT,
        {**BREAKOUT, "b": {**LL20, "source": "close"}, "b_mult": 1.0, "offset": {"atr_mult": 0.5}},
        # "close made a new 20-bar closing high this bar" is meaningful with the current bar included.
        {
            "type": "compare",
            "a": {"price": "close"},
            "op": ">=",
            "b": {**HH20, "source": "close", "exclude_current": False},
        },
        # within 5 points of the running high is not constant either.
        {
            "type": "compare",
            "a": {"price": "close"},
            "op": ">=",
            "b": {**HH20, "exclude_current": False},
            "offset": {"points": 5},
        },
        {
            "type": "compare",
            "a": {"ind": "rsi", "period": 14},
            "op": ">",
            "b": {"ind": "rsi", "period": 14, "smooth": {"type": "ema", "period": 9}},
        },
    ],
)
def test_valid_v21_leaves(leaf):
    assert validate_v2(_with_leaf(leaf)).model


def test_example_validates_and_defaults_are_explicit():
    model = validate_v2(tx_atr_squeeze_breakout()).model
    cross = model.rules[0].condition.all[1]
    assert cross.b.exclude_current is True and cross.b_mult is None
    assert model.rules[0].condition.all[0].b.smooth.period == 50


def test_v21_fields_take_params():
    doc = tx_atr_squeeze_breakout()
    doc["params"] = {
        "ratio": {"type": "float", "low": 0.6, "high": 0.9, "step": 0.1, "default": 0.8},
        "bars": {"type": "int", "low": 10, "high": 40, "step": 10, "default": 20},
        "smooth": {"type": "int", "low": 30, "high": 90, "step": 30, "default": 60},
    }
    for rule in doc["rules"]:
        squeeze, cross = rule["condition"]["all"]
        squeeze["b_mult"] = {"$param": "ratio"}
        squeeze["b"]["smooth"]["period"] = {"$param": "smooth"}
        cross["b"]["bars"] = {"$param": "bars"}
    validated = validate_v2(doc)
    assert validated.values == {"ratio": 0.8, "bars": 20, "smooth": 60}
    trial = validate_v2(doc, param_values={"ratio": 0.6, "bars": 40, "smooth": 30}).model
    assert trial.rules[0].condition.all[0].b_mult == 0.6
    assert trial.rules[1].condition.all[1].b.bars == 40
    doc["params"]["bars"]["high"] = 500  # grid reaches past the 400-bar bound
    with pytest.raises(DslValidationError) as info:
        validate_v2(doc)
    assert info.value.errors[0]["path"] == "params.bars"


def test_v2_documents_dump_unchanged():
    """v2 documents get no new keys in their dumps (exclude_none), so stored configs stay identical."""
    doc = tx_orb30_day()
    dumped = validate_v2(doc).model.model_dump(mode="json", exclude_none=True)
    assert "b_mult" not in json.dumps(dumped) and "smooth" not in json.dumps(dumped)


def test_json_schema_exports_v21():
    schema = dsl_v2_json_schema()
    defs = schema["$defs"]
    assert defs["ChannelOperand"]["additionalProperties"] is False
    assert set(defs["ChannelOperand"]["properties"]) == {"channel", "source", "bars", "exclude_current"}
    assert defs["ChannelOperand"]["properties"]["bars"]["minimum"] == 2
    assert defs["ChannelOperand"]["properties"]["bars"]["maximum"] == 400
    assert set(defs["SmoothSpec"]["properties"]) == {"type", "period"}
    assert "smooth" in defs["IndicatorOperand"]["properties"]
    for cond in ("CrossCondition", "CompareCondition"):
        b_mult = json.dumps(defs[cond]["properties"]["b_mult"])
        assert '"minimum": 0.1' in b_mult and '"maximum": 10' in b_mult


# --------------------------------------------------------------------------- features


def _bars(highs, lows=None, closes=None) -> pd.DataFrame:
    highs = np.asarray(highs, dtype=float)
    lows = highs - 2 if lows is None else np.asarray(lows, dtype=float)
    closes = (highs + lows) / 2 if closes is None else np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-09-01 00:45", periods=len(highs), freq="5min", tz="UTC")
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1.0}, index=idx)


def test_rolling_channel_values():
    df = _bars([10, 12, 11, 15, 9, 8])
    prev = compute_rolling_channel(df, "highest", "high", 3)
    incl = compute_rolling_channel(df, "highest", "high", 3, exclude_current=False)
    assert prev.name == "ch_highest_high_3_prev" and incl.name == "ch_highest_high_3_incl"
    assert prev.iloc[:3].isna().all() and list(prev.iloc[3:]) == [12, 15, 15]
    assert incl.iloc[:2].isna().all() and list(incl.iloc[2:]) == [12, 15, 15, 15]
    low = compute_rolling_channel(df, "lowest", "low", 2)
    assert list(low.iloc[2:]) == [8, 9, 9, 7]  # lows 8, 10, 9, 13, 7, 6


def test_smoothed_indicator_values():
    rng = np.random.default_rng(3)
    close = 23000 + np.cumsum(rng.standard_normal(200) * 5)
    df = _bars(close + 3, close - 3, close)
    atr = ATR().compute(df, period=14)
    sma = compute_smoothed_indicator(df, "atr", 14, "sma", 50)
    assert sma.name == "atr_14_sma_50"
    # NaN until 50 valid ATR values exist (ATR(14) itself is NaN for the first 13 bars).
    assert sma.iloc[: 13 + 49].isna().all() and sma.iloc[13 + 49 :].notna().all()
    assert sma.iloc[150] == pytest.approx(atr.iloc[101:151].mean())
    ema = compute_smoothed_indicator(df, "rsi", 14, "ema", 9)
    rsi = RSI().compute(df, period=14)
    assert ema.first_valid_index() == df.index[13 + 8]  # RSI(14) valid from bar 13, then 9 values
    expected = rsi.ewm(span=9, adjust=False, min_periods=9).mean()
    pd.testing.assert_series_equal(ema, expected, check_names=False)


V21_SPECS = [
    ("rolling_channel", {"channel": c, "source": s, "bars": n, "exclude_current": x})
    for c in ("highest", "lowest")
    for s in ("high", "low", "close")
    for n, x in ((2, True), (20, True), (20, False))
] + [
    ("ind_smooth", {"ind": i, "period": p, "smooth": t, "smooth_period": q})
    for i, p in (("atr", 14), ("rsi", 14), ("ema", 20), ("sma", 10))
    for t in ("sma", "ema")
    for q in (2, 50)
]


def _synthetic_with_gaps(seed: int = 9) -> pd.DataFrame:
    df = random_walk_bars([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-03-02", "2026-03-06")], seed=seed)
    keep = np.random.default_rng(seed).random(len(df)) > 0.07
    return df[keep]


def _assert_v21_no_look_ahead(df: pd.DataFrame) -> None:
    engine = FeatureEngine()
    full = engine.compute_from_df(df, V21_SPECS)
    new_cols = [c for c in full.columns if c not in df.columns]
    assert len(new_cols) == len(V21_SPECS)
    assert full[new_cols].notna().any().all()
    for k in range(len(df)):
        trunc = engine.compute_from_df(df.iloc[: k + 1], V21_SPECS)
        a = trunc[new_cols].iloc[-1].to_numpy(dtype=float)
        b = full[new_cols].iloc[k].to_numpy(dtype=float)
        mismatch = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        assert not mismatch.any(), (k, [c for c, m in zip(new_cols, mismatch, strict=True) if m])


def test_v21_columns_no_look_ahead_synthetic():
    _assert_v21_no_look_ahead(_synthetic_with_gaps())


def test_v21_columns_no_look_ahead_real_tx_two_weeks():
    _assert_v21_no_look_ahead(real_tx_fixture())


def test_exclude_current_channel_ignores_the_current_bar():
    df = _bars([10, 11, 12, 13])
    spiked = df.copy()
    spiked.iloc[-1, spiked.columns.get_loc("high")] = 1000.0
    a = compute_rolling_channel(df, "highest", "high", 2)
    b = compute_rolling_channel(spiked, "highest", "high", 2)
    assert a.iloc[-1] == b.iloc[-1] == 12


# --------------------------------------------------------------------------- evaluator + specs


def _frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {"s_id": [20260901 * 10 + 1.0] * n, "s_kind": [1.0] * n, "s_tod": [600.0] * n, "s_min_open": [75.0] * n}
    base.update(cols)
    idx = pd.date_range("2026-09-01 01:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(base, index=idx)


def _fires(cond: dict, frame: pd.DataFrame) -> bool:
    compiled = CompiledCondition(_COND.validate_python(cond))
    cache = ColumnCache(compiled.columns)
    return compiled.evaluate(cache, cache.sync(frame))


def test_compare_b_mult_is_applied_before_the_offset():
    f = _frame(atr_14=[13.0], atr_14_sma_50=[20.0])
    assert _fires(SQUEEZE, f)  # 13 < 0.7 * 20 = 14
    assert not _fires({**SQUEEZE, "b_mult": 0.6}, f)  # 13 < 12 is false
    # The offset is applied after the multiplier (subtracted for "<"): 13 < 14 - 1 is false, 13 < 14 - 0.5 true.
    assert not _fires({**SQUEEZE, "offset": {"points": 1.0}}, f)
    assert _fires({**SQUEEZE, "offset": {"points": 0.5}}, f)
    assert _fires({**SQUEEZE, "op": ">", "b_mult": 0.5, "offset": {"points": 2}}, f)  # 13 > 10 + 2


def test_cross_channel_and_b_mult():
    f = _frame(close=[99.0, 101.0], ch_highest_high_20_prev=[100.0, 100.0])
    assert _fires(BREAKOUT, f)
    assert not _fires({**BREAKOUT, "b_mult": 1.02}, f)
    assert _fires({**BREAKOUT, "b_mult": 0.995}, _frame(close=[99.0, 99.6], ch_highest_high_20_prev=[100.0, 100.0]))
    # a NaN channel (warm-up) blocks the rule
    assert not _fires(BREAKOUT, _frame(close=[99.0, 101.0], ch_highest_high_20_prev=[np.nan, 100.0]))


def test_derive_feature_specs_v21():
    spec = validate_v2(tx_atr_squeeze_breakout()).model
    assert derive_feature_specs(spec) == [
        ("tx_session", {"interval_minutes": 5, "bar_label": "open"}),
        ("atr", {"period": 14}),
        ("ind_smooth", {"ind": "atr", "period": 14, "smooth": "sma", "smooth_period": 50}),
        ("rolling_channel", {"channel": "highest", "source": "high", "bars": 20, "exclude_current": True}),
        ("rolling_channel", {"channel": "lowest", "source": "low", "bars": 20, "exclude_current": True}),
    ]


def test_derived_specs_cover_every_column_including_channel_stops():
    doc = tx_atr_squeeze_breakout()
    doc["rules"][0]["risk"] = {
        "stop_loss": {"type": "level", "level": {"channel": "lowest", "source": "low", "bars": 10}, "offset_points": 2}
    }
    doc["rules"][1]["risk"] = {
        "take_profit": {
            "type": "level",
            "level": {"channel": "lowest", "source": "close", "bars": 60, "exclude_current": False},
        }
    }
    strategy = RuleStrategyV2(config=doc)
    features = FeatureEngine().compute_from_df(real_tx_fixture().iloc[:400], strategy.get_feature_specs())
    assert set(strategy._columns) <= set(features.columns)
    assert {"ch_lowest_low_10_prev", "ch_lowest_close_60_incl", "atr_14_sma_50"} <= set(strategy._columns)


# --------------------------------------------------------------------------- strategy (numpy cache path)


def test_squeeze_breakout_backtests_on_the_real_two_week_fixture():
    ohlcv = real_tx_fixture()
    result, _, strategy = run_v2(tx_atr_squeeze_breakout(), ohlcv)
    trips = round_trips(result.trades)
    assert len(trips) >= 1
    assert {"atr_14_sma_50", "ch_highest_high_20_prev", "ch_lowest_low_20_prev"} <= set(strategy._columns)
    # Every entry happened on a bar where the contraction + breakout really held (recomputed from raw bars).
    atr = ATR().compute(ohlcv, period=14)
    atr_sma = atr.rolling(50, min_periods=50).mean()
    hh = ohlcv["high"].rolling(20, min_periods=20).max().shift(1)
    ll = ohlcv["low"].rolling(20, min_periods=20).min().shift(1)
    close = ohlcv["close"]
    entries = [t for t in result.trades if t["exit_time"] is None]
    for t in entries:
        k = ohlcv.index.get_loc(pd.Timestamp(t["entry_time"]))
        assert atr.iloc[k] < 0.8 * atr_sma.iloc[k]
        if t["action"] == "long":
            assert close.iloc[k - 1] <= hh.iloc[k - 1] and close.iloc[k] > hh.iloc[k]
        else:
            assert close.iloc[k - 1] >= ll.iloc[k - 1] and close.iloc[k] < ll.iloc[k]


class _AlwaysRebuild(evaluator.ColumnCache):
    def sync(self, frame):
        self._len = 0
        return super().sync(frame)


def test_incremental_cache_matches_full_rebuild_v21(monkeypatch):
    ohlcv = real_tx_fixture()
    fast, _, _ = run_v2(tx_atr_squeeze_breakout(), ohlcv)
    monkeypatch.setattr("poseidon.strategies.rule_strategy_v2.ColumnCache", _AlwaysRebuild)
    slow, _, _ = run_v2(tx_atr_squeeze_breakout(), ohlcv)
    assert fast.trades == slow.trades and fast.trade_count > 0


def _records_until(trades: list[dict], cut: pd.Timestamp) -> list[dict[str, Any]]:
    return [t for t in trades if pd.Timestamp(t["exit_time"] or t["entry_time"]) <= cut]


@pytest.mark.parametrize("data", ["real_tx", "synthetic"])
def test_strategy_level_no_look_ahead_v21(data):
    ohlcv = (
        real_tx_fixture()
        if data == "real_tx"
        else random_walk_bars([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-03-02", "2026-03-13")], seed=23)
    )
    doc = tx_atr_squeeze_breakout()
    doc["rules"][0]["risk"] = {
        "stop_loss": {"type": "level", "level": {"channel": "lowest", "source": "low", "bars": 10}}
    }
    full, _, _ = run_v2(doc, ohlcv)
    stamps = sorted({pd.Timestamp(t["exit_time"] or t["entry_time"]) for t in full.trades})
    assert len(stamps) >= 2
    rng = np.random.default_rng(1)
    cuts = set(stamps) | {s - pd.Timedelta(minutes=5) for s in stamps}
    cuts |= set(ohlcv.index[rng.choice(np.arange(200, len(ohlcv)), size=6, replace=False)])
    for cut in sorted(cuts):
        k = ohlcv.index.searchsorted(cut, side="right")
        truncated, _, _ = run_v2(doc, ohlcv.iloc[:k])
        assert truncated.trades == _records_until(full.trades, cut), cut
