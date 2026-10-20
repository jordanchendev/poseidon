"""Tests for TrendlineRulerStrategy (trendline touch entry, ruler target, management)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import OrderType, SignalAction
from poseidon.strategies.registry import get_strategy
from poseidon.strategies.trendline_ruler import TrendlineRulerStrategy
from tests.test_trendline_ruler_features import KNOWN, TOUCH, frame, line_at, mirror, short_bars

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "poseidon" / "strategies" / "configs" / "trendline_ruler_tx_5m.json"
)
# Bar 32 (TOUCH) is 11:40 Taipei: high 19993, close 19920 under the line (19995).
# stop = 19993 + 5 = 19998, target = 19750 -> risk 78, reward 170, rr ~2.18.
STOP, TARGET, ENTRY = 19998.0, 19750.0, 19920.0
RR = (ENTRY - TARGET) / (STOP - ENTRY)


def config(**overrides) -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["pattern"]["pivot_left"] = 2
    cfg["pattern"]["pivot_right"] = 2
    cfg["htf"]["mode"] = "off"
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = value
    return cfg


def features_for(strategy: TrendlineRulerStrategy, bars: list) -> pd.DataFrame:
    return FeatureEngine().compute_from_df(frame(bars), strategy.get_feature_specs())


def run_bars(strategy: TrendlineRulerStrategy, feats: pd.DataFrame, upto: int) -> list:
    """Evaluate bar by bar like the runner; return (bar, signal) pairs."""
    out = []
    for i in range(upto + 1):
        out += [(i, s) for s in strategy.evaluate(feats.iloc[: i + 1])]
    return out


class TestConfig:
    def test_registered_and_specs(self):
        assert get_strategy("trendline_ruler") is TrendlineRulerStrategy
        s = TrendlineRulerStrategy(config=json.loads(CONFIG_PATH.read_text()))
        assert s.validate_config()
        assert (s.symbol, s.market, s.interval) == ("TX", "tw_futures", "5m")
        names = [n for n, _ in s.get_feature_specs()]
        assert names == ["session_clock", "bollinger", "atr", "htf_bias", "trendline_ruler"]

    @pytest.mark.parametrize(
        "override",
        [
            {"direction_mode": "sideways"},
            {"htf.mode": "weekly"},
            {"risk.min_rr": 0},
            {"session.flat_at": []},
            {"risk.stop_mode": "swing"},
            {"risk.min_stop_atr": -1.0},
            {"manage.exit_mode": "trail"},
        ],
    )
    def test_invalid(self, override):
        with pytest.raises(ValueError):
            TrendlineRulerStrategy(config=config(**override)).validate_config()


class TestShortEntry:
    def test_touch_emits_short_with_stop_target_rr(self):
        s = TrendlineRulerStrategy(config=config())
        feats = features_for(s, short_bars())
        sigs = run_bars(s, feats, TOUCH)
        assert len(sigs) == 1
        bar, sig = sigs[0]
        assert bar == TOUCH
        assert sig.action == SignalAction.SHORT and sig.order_type == OrderType.MARKET
        assert sig.stop_loss_price == pytest.approx(STOP)
        assert sig.take_profit_price == pytest.approx(TARGET)
        assert sig.metadata["rr"] == pytest.approx(RR)
        assert sig.metadata["ruler"] == pytest.approx(130.0)
        assert sig.metadata["line_id"] == 0 and sig.metadata["pattern_index"] == 1
        assert sig.metadata["touch_tol"] == pytest.approx(0.25 * feats["atr_14"].iloc[TOUCH])
        assert s.reject_counts["entered"] == 1

    def test_close_through_line_no_entry(self):
        s = TrendlineRulerStrategy(config=config())
        feats = features_for(s, short_bars({TOUCH: (20001.0, 19950.0, 20000.0)}))
        assert feats["tls_broken"].iloc[TOUCH] == 1.0
        assert run_bars(s, feats, TOUCH + 3) == []

    def test_rr_below_min_rejected_and_counted(self):
        s = TrendlineRulerStrategy(config=config(**{"risk.min_rr": 3.0}))
        feats = features_for(s, short_bars())
        assert run_bars(s, feats, TOUCH) == []
        assert s.reject_counts["rr"] == 1 and s.reject_counts["entered"] == 0

    def test_max_stop_rejected(self):
        s = TrendlineRulerStrategy(config=config(**{"risk.max_stop_points": 50.0}))
        assert run_bars(s, features_for(s, short_bars()), TOUCH) == []
        assert s.reject_counts["max_stop"] == 1

    def test_outside_window_no_entry(self):
        s = TrendlineRulerStrategy(config=config(**{"session.allowed_windows": [["12:00", "13:15"]]}))
        assert run_bars(s, features_for(s, short_bars()), TOUCH) == []
        assert s.reject_counts["window"] == 1

    def test_htf_bias_must_point_down(self):
        cfg = config(**{"htf.mode": "sma_slope"})
        s = TrendlineRulerStrategy(config=cfg)
        feats = features_for(s, short_bars())
        feats["htf_bias"] = 1.0
        assert run_bars(s, feats, TOUCH) == []
        assert s.reject_counts["htf_bias"] == 1
        s2 = TrendlineRulerStrategy(config=cfg)
        feats["htf_bias"] = -1.0
        assert [sig.action for _, sig in run_bars(s2, feats, TOUCH)] == [SignalAction.SHORT]

    def test_conflicting_long_setup_blocks_short(self):
        s = TrendlineRulerStrategy(config=config())
        feats = features_for(s, short_bars())
        feats.loc[feats.index[TOUCH], ["tll_active", "tll_zone_ok", "tll_pattern_index"]] = [1.0, 1.0, 1.0]
        assert run_bars(s, feats, TOUCH) == []
        assert s.reject_counts["conflicting_setup"] == 1
        # the same conflict is ignored when skip_conflicting is off
        s2 = TrendlineRulerStrategy(config=config(**{"pattern.skip_conflicting": False}))
        assert [sig.action for _, sig in run_bars(s2, feats, TOUCH)] == [SignalAction.SHORT]

    def test_late_pattern_in_leg_skipped(self):
        s = TrendlineRulerStrategy(config=config())
        feats = features_for(s, short_bars())
        feats["tls_pattern_index"] = 3.0
        assert run_bars(s, feats, TOUCH) == []
        assert s.reject_counts["pattern_index"] == 1


class TestManagement:
    def test_line_break_while_short_closes(self):
        s = TrendlineRulerStrategy(config=config())
        # drift bars close 19930: line(37) = 19920 -> first close above the line at bar 37
        feats = features_for(s, short_bars())
        sigs = run_bars(s, feats, KNOWN + 8)
        assert [(b, sig.action) for b, sig in sigs] == [(TOUCH, SignalAction.SHORT), (37, SignalAction.CLOSE)]
        assert line_at(37) < 19930.0 < line_at(36)
        assert sigs[1][1].metadata["exit_reason"] == "line_break"
        assert s._side is None and s.exit_counts["line_break"] == 1

    def test_flat_at_closes(self):
        s = TrendlineRulerStrategy(config=config(**{"session.flat_at": ["11:50"]}))
        feats = features_for(s, short_bars())
        sigs = run_bars(s, feats, TOUCH + 3)
        assert [(b, sig.action) for b, sig in sigs] == [(TOUCH, SignalAction.SHORT), (TOUCH + 2, SignalAction.CLOSE)]
        assert sigs[1][1].metadata["exit_reason"] == "flat"

    def test_runner_tp_mirrored_and_one_entry_per_line(self):
        s = TrendlineRulerStrategy(config=config())
        bars = short_bars({TOUCH + 1: (19940.0, 19740.0, 19760.0), TOUCH + 2: (19985.0, 19900.0, 19930.0)})
        feats = features_for(s, bars)
        sigs = run_bars(s, feats, TOUCH + 2)
        assert [sig.action for _, sig in sigs] == [SignalAction.SHORT]
        assert s._side is None and s.exit_counts["tp"] == 1


class TestLongMirror:
    def test_mirrored_touch_emits_long(self):
        s = TrendlineRulerStrategy(config=config())
        feats = features_for(s, mirror(short_bars()))
        sigs = run_bars(s, feats, TOUCH)
        assert len(sigs) == 1
        bar, sig = sigs[0]
        assert bar == TOUCH and sig.action == SignalAction.LONG
        assert sig.stop_loss_price == pytest.approx(40_000 - STOP)
        assert sig.take_profit_price == pytest.approx(40_000 - TARGET)
        assert sig.metadata["rr"] == pytest.approx(RR)

    def test_short_only_blocks_long(self):
        s = TrendlineRulerStrategy(config=config(direction_mode="short_only"))
        assert run_bars(s, features_for(s, mirror(short_bars())), TOUCH) == []
        assert s.reject_counts["direction_mode"] == 1


class TestStopModes:
    # line(TOUCH) = 19995, P2 high = 20040 (bar 29)
    @pytest.mark.parametrize(
        ("mode", "stop"), [("touch_bar_high", STOP), ("line", line_at(TOUCH) + 5.0), ("p2", 20040.0 + 5.0)]
    )
    def test_stop_price_per_mode(self, mode, stop):
        s = TrendlineRulerStrategy(config=config(**{"risk.stop_mode": mode}))
        feats = features_for(s, short_bars())
        assert feats["tls_p2"].iloc[TOUCH] == 20040.0
        [(_, sig)] = run_bars(s, feats, TOUCH)
        assert sig.stop_loss_price == pytest.approx(stop)
        assert sig.metadata["rr"] == pytest.approx((ENTRY - TARGET) / (stop - ENTRY))
        assert sig.metadata["stop_points"] == pytest.approx(stop - ENTRY)
        assert s.entry_stop_points == [pytest.approx(stop - ENTRY)]

    @pytest.mark.parametrize(("mode", "stop"), [("line", line_at(TOUCH) + 5.0), ("p2", 20045.0)])
    def test_long_mirror_stop(self, mode, stop):
        s = TrendlineRulerStrategy(config=config(**{"risk.stop_mode": mode}))
        feats = features_for(s, mirror(short_bars()))
        assert feats["tll_p2"].iloc[TOUCH] == 40_000 - 20040.0
        [(_, sig)] = run_bars(s, feats, TOUCH)
        assert sig.action == SignalAction.LONG
        assert sig.stop_loss_price == pytest.approx(40_000 - stop)

    def test_min_stop_atr_widens_and_lowers_rr(self):
        probe = TrendlineRulerStrategy(config=config())
        atr = features_for(probe, short_bars())["atr_14"].iloc[TOUCH]
        mult = 100.0 / atr  # widen the 78-point stop to exactly 100 points
        s = TrendlineRulerStrategy(config=config(**{"risk.min_stop_atr": mult}))
        [(_, sig)] = run_bars(s, features_for(s, short_bars()), TOUCH)
        assert sig.stop_loss_price == pytest.approx(ENTRY + 100.0)
        assert sig.metadata["rr"] == pytest.approx(170.0 / 100.0)
        # a min_stop_atr below the natural distance changes nothing
        s_small = TrendlineRulerStrategy(config=config(**{"risk.min_stop_atr": 50.0 / atr}))
        [(_, sig_small)] = run_bars(s_small, features_for(s_small, short_bars()), TOUCH)
        assert sig_small.stop_loss_price == pytest.approx(STOP)
        # RR and max_stop use the widened stop: 2.18 passes min_rr 2, 1.7 does not
        cfg = {"risk.min_rr": 2.0}
        assert len(run_bars(TrendlineRulerStrategy(config=config(**cfg)), features_for(s, short_bars()), TOUCH)) == 1
        s_rr = TrendlineRulerStrategy(config=config(**cfg, **{"risk.min_stop_atr": mult}))
        assert run_bars(s_rr, features_for(s_rr, short_bars()), TOUCH) == []
        assert s_rr.reject_counts["rr"] == 1
        s_max = TrendlineRulerStrategy(config=config(**{"risk.min_stop_atr": mult, "risk.max_stop_points": 90.0}))
        assert run_bars(s_max, features_for(s_max, short_bars()), TOUCH) == []
        assert s_max.reject_counts["max_stop"] == 1


class TestTrailing:
    def test_line_trail_holds_follow_line_then_stop_hits(self):
        s = TrendlineRulerStrategy(config=config(**{"manage.exit_mode": "line_trail"}))
        sigs = run_bars(s, features_for(s, short_bars()), KNOWN + 8)
        assert sigs[0][1].take_profit_price == pytest.approx(TARGET)
        holds = [(b, sig.metadata["updated_stop_loss"]) for b, sig in sigs if sig.action == SignalAction.HOLD]
        # line + 5 on bars 33..36; bar 37's high 19940 reaches the 19940 stop before the close-through check
        assert holds == [(b, pytest.approx(line_at(b) + 5.0)) for b in range(TOUCH + 1, 37)]
        assert [sig.action for _, sig in sigs if sig.action != SignalAction.HOLD] == [SignalAction.SHORT]
        assert s._side is None and s.exit_counts["sl"] == 1 and s.exit_counts["line_break"] == 0

    def test_line_trail_never_loosens(self):
        s = TrendlineRulerStrategy(config=config(**{"manage.exit_mode": "line_trail"}))
        feats = features_for(s, short_bars())
        run_bars(s, feats, TOUCH)
        s._sl = 19970.0  # tighter than line(33) + 5 = 19985
        stops = [sig.metadata["updated_stop_loss"] for i in (33, 34, 35) for sig in s.evaluate(feats.iloc[: i + 1])]
        assert stops == [19970.0, 19970.0, pytest.approx(19955.0)]

    def test_target_and_break_emits_no_hold(self):
        s = TrendlineRulerStrategy(config=config())
        sigs = run_bars(s, features_for(s, short_bars()), KNOWN + 8)
        assert all(sig.action != SignalAction.HOLD for _, sig in sigs)

    def test_line_trail_no_tp_ignores_target(self):
        s = TrendlineRulerStrategy(config=config(**{"manage.exit_mode": "line_trail_no_tp"}))
        feats = features_for(s, short_bars({TOUCH + 1: (19940.0, 19740.0, 19760.0)}))
        sigs = run_bars(s, feats, TOUCH + 1)
        assert sigs[0][1].action == SignalAction.SHORT and sigs[0][1].take_profit_price is None
        assert sigs[0][1].metadata["rr"] == pytest.approx(RR)  # RR filter still uses the ruler target
        assert s._side == "short" and s.exit_counts["tp"] == 0
        assert sigs[1][0] == TOUCH + 1 and sigs[1][1].action == SignalAction.HOLD

    def test_runner_applies_trailing_stop(self):
        s = TrendlineRulerStrategy(config=config(**{"manage.exit_mode": "line_trail"}))
        runner = BacktestRunner(
            strategy=s,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=COST_MODELS["tw_futures"],
            initial_capital=1_000_000.0,
            sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1),
        )
        result = runner.run(frame(short_bars()), feature_specs=s.get_feature_specs())
        assert result.status == "completed", result.error_message
        assert [t["action"] for t in result.trades] == ["short", "close_sl"]
        assert result.trades[1]["exit_price"] == pytest.approx(line_at(36) + 5.0)


class TestHtfBlockCounter:
    @pytest.mark.parametrize(("bias", "enters"), [(-1.0, True), (0.0, True), (1.0, False)])
    def test_short_blocked_only_by_up_bias(self, bias, enters):
        s = TrendlineRulerStrategy(config=config(**{"htf.mode": "block_counter"}))
        feats = features_for(s, short_bars())
        feats["htf_bias"] = bias
        assert len(run_bars(s, feats, TOUCH)) == (1 if enters else 0)
        assert s.reject_counts["htf_bias"] == (0 if enters else 1)

    @pytest.mark.parametrize(("bias", "enters"), [(1.0, True), (0.0, True), (-1.0, False)])
    def test_long_blocked_only_by_down_bias(self, bias, enters):
        s = TrendlineRulerStrategy(config=config(**{"htf.mode": "block_counter"}))
        feats = features_for(s, mirror(short_bars()))
        feats["htf_bias"] = bias
        assert len(run_bars(s, feats, TOUCH)) == (1 if enters else 0)

    def test_sma_slope_blocks_neutral(self):
        s = TrendlineRulerStrategy(config=config(**{"htf.mode": "sma_slope"}))
        feats = features_for(s, short_bars())
        feats["htf_bias"] = 0.0
        assert run_bars(s, feats, TOUCH) == []
        assert s.reject_counts["htf_bias"] == 1

    def test_feature_spec_uses_sma_slope_bias(self):
        s = TrendlineRulerStrategy(config=config(**{"htf.mode": "block_counter"}))
        assert dict(s.get_feature_specs())["htf_bias"]["htf_mode"] == "sma_slope"


class TestE2E:
    def test_runner_round_trip_take_profit(self):
        s = TrendlineRulerStrategy(config=config())
        df = frame(short_bars({TOUCH + 1: (19940.0, 19740.0, 19760.0)}))
        runner = BacktestRunner(
            strategy=s,
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=COST_MODELS["tw_futures"],
            initial_capital=1_000_000.0,
            sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1),
        )
        result = runner.run(df, feature_specs=s.get_feature_specs())
        assert result.status == "completed", result.error_message
        actions = [t["action"] for t in result.trades]
        assert actions == ["short", "close_tp"], actions
        assert result.trades[1]["exit_price"] == pytest.approx(TARGET)
        assert result.trades[1]["pnl"] > 0
