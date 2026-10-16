"""Tests for RangeReversionStrategy (range 50% reversion, PO3 sweep entry)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import OrderType, SignalAction
from poseidon.strategies.range_reversion import RangeReversionStrategy
from poseidon.strategies.registry import get_strategy

TPE = "Asia/Taipei"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "src" / "poseidon" / "strategies" / "configs" / "range50_tx_5m.json"


def base_config(**overrides) -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = value
    return cfg


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TPE).tz_convert("UTC")


def make_day(date: str, base: float, seed: int, step: float = 2.0) -> pd.DataFrame:
    """One TX day session 08:45..13:40 of gentle random-walk 5m bars."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{date} 08:45", periods=60, freq="5min", tz=TPE).tz_convert("UTC")
    close = base + np.cumsum(rng.normal(0, step, 60))
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, step / 2, 60))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, step / 2, 60))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx)


def build_scenario(lookback_days: int = 3) -> tuple[pd.DataFrame, dict]:
    """Downtrending days then a setup day with an engineered 1h+4h sweep and reclaim.

    Setup day (2026-03-09, Monday):
      08:45-08:55 stub hour: high 20050 (h1 prev for the 09:00 bar); it is also
      part of the 08:00 4h clock bar, whose previous (04:00) bar does not exist
      on this synthetic frame -> we add a small 07:55 bar so h4_prev exists.
      09:00-09:25: drifts around 20000.
      09:30: spike high 20120 (sweeps h1 prev 20050 and h4 prev 20080), close 20090
      09:35: closes 20040 (< swept level 20050, < spike) -> SHORT trigger "level"
      then price falls to the target so the runner's TP fires.
    """
    dates = ["2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]  # Tue..Fri, falling bases
    bases = [20600, 20500, 20400, 20300]
    days = [make_day(d, b, i) for i, (d, b) in enumerate(zip(dates, bases, strict=True))]

    idx = pd.date_range("2026-03-09 07:55", periods=1, freq="5min", tz=TPE).tz_convert("UTC")
    pre = pd.DataFrame({"open": 20020, "high": 20030, "low": 20010, "close": 20020, "volume": 1.0}, index=idx)

    rows = []
    t = pd.Timestamp("2026-03-09 08:45", tz=TPE)
    while t <= pd.Timestamp("2026-03-09 13:40", tz=TPE):
        rows.append(t)
        t += pd.Timedelta(minutes=5)
    sidx = pd.DatetimeIndex(rows).tz_convert("UTC")
    # 08:50..09:25 sit at 19900 (the leg low), 09:30 spikes to 20070 (> h1 prev 20050, > h4 prev 20030),
    # 09:35 closes back below 20050 -> SHORT at 20040, stop 20075, target (20070+19890)/2 = 19980.
    setup = pd.DataFrame(
        {"open": 19900.0, "high": 19910.0, "low": 19890.0, "close": 19900.0, "volume": 100.0}, index=sidx
    )
    setup.loc[local("2026-03-09 08:45"), ["open", "high", "low", "close"]] = [20020, 20050, 20010, 20020]
    setup.loc[local("2026-03-09 09:30"), ["open", "high", "low", "close"]] = [19905, 20070, 19900, 20060]
    setup.loc[local("2026-03-09 09:35"), ["open", "high", "low", "close"]] = [20060, 20065, 20035, 20040]
    # after entry the 09:00 hour opened at 19900, so closes above it keep the hour green (no breakeven);
    # the 10:00 hour opens at 20015 and its first bar closes below -> breakeven; 10:05 hits the target.
    for ts, (o, h, lo, c) in {
        "09:40": (20040, 20045, 20030, 20035),
        "09:45": (20035, 20040, 20020, 20025),
        "09:50": (20025, 20030, 20015, 20020),
        "09:55": (20020, 20025, 20010, 20015),
        "10:00": (20015, 20018, 19990, 19995),
        "10:05": (19995, 20000, 19960, 19965),
        "10:10": (19965, 19970, 19900, 19910),
    }.items():
        setup.loc[local(f"2026-03-09 {ts}"), ["open", "high", "low", "close"]] = [o, h, lo, c]
    rest = setup.index > local("2026-03-09 10:10")
    setup.loc[rest, ["open", "high", "low", "close"]] = [19910, 19915, 19905, 19910]

    df = pd.concat([*days, pre, setup]).sort_index()
    cfg = base_config(**{"range.lookback_days": lookback_days, "sweep.require_h4": True, "risk.min_rr": 1.0})
    return df, cfg


def compute(df: pd.DataFrame, strategy: RangeReversionStrategy) -> pd.DataFrame:
    return FeatureEngine().compute_from_df(df, strategy.get_feature_specs())


class TestConfig:
    def test_registered_and_defaults(self):
        assert get_strategy("range_50_reversion") is RangeReversionStrategy
        s = RangeReversionStrategy(config=base_config())
        assert s.validate_config()
        assert s.symbol == "TX" and s.market == "tw_futures" and s.interval == "5m"
        names = [n for n, _ in s.get_feature_specs()]
        assert names == ["session_levels", "dealing_range", "fair_value_gap", "atr"]

    def test_smt_adds_companion_spec(self):
        s = RangeReversionStrategy(config=base_config(**{"confirm.smt_mode": "required"}))
        spec = dict(s.get_feature_specs())["smt_divergence"]
        assert spec["companion_symbol"] == "TE" and spec["companion_market"] == "tw_futures"

    @pytest.mark.parametrize(
        "override",
        [
            {"direction_mode": "sideways"},
            {"confirm.smt_mode": "maybe"},
            {"session.entry_end": "08:00"},
            {"risk.min_rr": 0},
        ],
    )
    def test_invalid(self, override):
        with pytest.raises(ValueError):
            RangeReversionStrategy(config=base_config(**override)).validate_config()


class TestSignals:
    def test_short_entry_after_sweep_and_reclaim(self):
        df, cfg = build_scenario()
        strategy = RangeReversionStrategy(config=cfg)
        feats = compute(df, strategy)
        i_trig = feats.index.get_loc(local("2026-03-09 09:35"))
        # no signal on the sweep bar itself (close still above the swept level)
        assert strategy.evaluate(feats.iloc[:i_trig]) == []
        sigs = strategy.evaluate(feats.iloc[: i_trig + 1])
        assert len(sigs) == 1
        sig = sigs[0]
        assert sig.action == SignalAction.SHORT and sig.order_type == OrderType.MARKET
        assert sig.stop_loss_price == pytest.approx(20070 + cfg["risk"]["stop_buffer_points"])
        assert sig.take_profit_price == pytest.approx((20070 + 19890) / 2)
        assert sig.metadata["trigger"] == "level"
        assert sig.metadata["sweep_flags"]["h1"] and sig.metadata["sweep_flags"]["h4"]
        assert sig.metadata["rr"] >= cfg["risk"]["min_rr"]

    def test_long_blocked_by_bearish_bias_and_short_only_mode(self):
        df, cfg = build_scenario()
        feats = compute(df, RangeReversionStrategy(config=cfg))
        i_trig = feats.index.get_loc(local("2026-03-09 09:35"))
        assert feats.iloc[i_trig]["range_bias_3"] == -1.0
        long_only = RangeReversionStrategy(config={**cfg, "direction_mode": "long_only"})
        assert long_only.evaluate(feats.iloc[: i_trig + 1]) == []

    def test_outside_entry_window_no_signal(self):
        df, cfg = build_scenario()
        cfg["session"]["entry_start"] = "10:30"
        cfg["session"]["entry_end"] = "11:15"
        strategy = RangeReversionStrategy(config=cfg)
        feats = compute(df, strategy)
        i_trig = feats.index.get_loc(local("2026-03-09 09:35"))
        assert strategy.evaluate(feats.iloc[: i_trig + 1]) == []

    def test_breakeven_hold_then_tp_sync(self):
        df, cfg = build_scenario()
        strategy = RangeReversionStrategy(config=cfg)
        feats = compute(df, strategy)
        i_trig = feats.index.get_loc(local("2026-03-09 09:35"))
        entry = strategy.evaluate(feats.iloc[: i_trig + 1])[0]
        holds = []
        for i in range(i_trig + 1, feats.index.get_loc(local("2026-03-09 10:00")) + 1):
            holds += strategy.evaluate(feats.iloc[: i + 1])
        # 09:40-09:55 bars close above the 09:00 hour open (20000) -> no flip;
        # 10:00 bar closes below its own hour open (20015) -> breakeven HOLD
        assert [h.action for h in holds] == [SignalAction.HOLD]
        be = holds[0].metadata["updated_stop_loss"]
        assert entry.action == SignalAction.SHORT
        assert be == pytest.approx(20040 - cfg["manage"]["breakeven_offset_points"])
        # 10:05 bar trades through the target -> strategy mirrors the runner's TP exit and stops for the day
        i_tp = feats.index.get_loc(local("2026-03-09 10:05"))
        assert strategy._side == "short"
        strategy.evaluate(feats.iloc[: i_tp + 1])
        assert strategy._side is None and strategy._won_today is True
        # nothing more that day even if another setup appeared
        assert strategy.evaluate(feats.iloc[: i_tp + 2]) == []

    def test_flat_at_closes_open_position(self):
        df, cfg = build_scenario()
        # make the target unreachable so the position is still open at 13:40
        cfg["risk"]["leg_lookback_bars"] = 2
        cfg["risk"]["min_rr"] = 0.01
        strategy = RangeReversionStrategy(config=cfg)
        feats = compute(df, strategy)
        i_trig = feats.index.get_loc(local("2026-03-09 09:35"))
        sig = strategy.evaluate(feats.iloc[: i_trig + 1])
        assert sig and sig[0].action == SignalAction.SHORT
        strategy._tp = 0.0  # force "never hit" to exercise the time exit
        out = []
        for i in range(i_trig + 1, len(feats)):
            out += strategy.evaluate(feats.iloc[: i + 1])
        closes = [s for s in out if s.action == SignalAction.CLOSE]
        assert len(closes) == 1
        assert closes[0].signal_time == local("2026-03-09 13:40")


class TestE2E:
    def test_runner_fixed_contracts_round_trip(self):
        df, cfg = build_scenario()
        strategy = RangeReversionStrategy(config=cfg)
        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=COST_MODELS["tw_futures"],
            initial_capital=1_000_000.0,
            sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=2),
        )
        result = runner.run(df, feature_specs=strategy.get_feature_specs())
        assert result.status == "completed", result.error_message
        actions = [t["action"] for t in result.trades]
        assert actions[:2] == ["short", "close_tp"], actions
        assert result.trades[0]["quantity"] == 2
        assert result.trades[1]["pnl"] > 0
        assert result.trades[1]["exit_time"].startswith("2026-03-09T02:05")  # 10:05 Taipei in UTC
