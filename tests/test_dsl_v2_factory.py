"""$param + optimizer step + DslV2StrategyFactory (DSL v2 slice 10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from poseidon.api.autoresearch import AutoResearchRequest
from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
from poseidon.backtest.optimizer import BayesianOptimizer
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.dsl.v2 import DslValidationError
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from tests.dsl_v2_support import random_walk_bars

PARAMS = {
    "or_min": {"type": "int", "low": 15, "high": 60, "step": 15, "default": 30},
    "tp_r": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25, "default": 2.0},
}


def _template() -> dict:
    doc = tx_orb30_day()
    doc["params"] = PARAMS
    for rule in doc["rules"]:
        rule["condition"]["all"][1]["b"]["minutes"] = {"$param": "or_min"}
        rule["risk"]["stop_loss"]["level"]["minutes"] = {"$param": "or_min"}
    doc["risk"]["take_profit"]["value"] = {"$param": "tp_r"}
    return doc


def _on_grid(value, spec) -> bool:
    k = (value - spec["low"]) / spec["step"]
    return abs(k - round(k)) < 1e-9 and spec["low"] <= value <= spec["high"]


def test_factory_defaults_and_substitution():
    factory = DslV2StrategyFactory(spec=_template())
    assert factory.param_bounds() == {"or_min": (15, 60, "int", 15), "tp_r": (1.0, 3.0, "float", 0.25)}
    default = factory.build()
    assert default.config.rules[0].condition.all[1].b.minutes == 30
    assert default.config.risk.take_profit.value == 2.0
    tuned = factory.build({"or_min": 45, "tp_r": 2.75})
    assert tuned.config.rules[1].risk.stop_loss.level.minutes == 45
    assert tuned.config.risk.take_profit.value == 2.75
    assert ("opening_range", {"minutes": 45, "interval_minutes": 5, "bar_label": "open"}) in tuned.get_feature_specs()
    with pytest.raises(DslValidationError):
        factory.build({"or_min": 40})
    with pytest.raises(DslValidationError):
        factory.build({"nope": 1})


def test_float_step_noise_is_snapped_to_the_grid():
    tuned = DslV2StrategyFactory(spec=_template()).build({"tp_r": 0.1 * 3 * 10 - 1.0 + 1e-12})  # ~2.0 with noise
    assert tuned.config.risk.take_profit.value == 2.0


def test_config_round_trip_is_resolved_and_revalidates():
    factory = DslV2StrategyFactory(spec=_template())
    strategy = factory.build({"or_min": 60, "tp_r": 1.5})
    cfg = DslV2StrategyFactory.to_config_dict(strategy)
    assert "params" not in cfg and "$param" not in str(cfg)
    again = DslV2StrategyFactory.from_config(cfg)
    assert again.config == strategy.config


def test_trial_factory_rejects_other_markets():
    factory = DslV2StrategyFactory(spec=_template())
    fn, bounds = factory.build_trial_factory(symbol="TX", market="tw_futures", interval="5m")
    assert fn({"or_min": 15, "tp_r": 1.0}).config.risk.take_profit.value == 1.0
    assert set(bounds) == {"or_min", "tp_r"}
    with pytest.raises(ValueError):
        factory.build_trial_factory(symbol="TX", market="tw_futures", interval="15m")


def test_optimizer_suggests_grid_values_only():
    factory = DslV2StrategyFactory(spec=_template())
    fn, bounds = factory.build_trial_factory(symbol="TX", market="tw_futures", interval="5m")
    ohlcv = random_walk_bars(["2026-09-07", "2026-09-08", "2026-09-09"], seed=31)
    optimizer = BayesianOptimizer(FeatureEngine(), RiskEngine(), COST_MODELS["tw_futures"])
    trials = optimizer.optimize(fn, ohlcv, bounds, n_trials=16, metric="total_return", seed=3)
    assert len(trials) == 16
    for trial in trials:
        assert _on_grid(trial.params["or_min"], PARAMS["or_min"]), trial.params
        assert _on_grid(trial.params["tp_r"], PARAMS["tp_r"]), trial.params
    assert len({t.params["or_min"] for t in trials}) > 1


def test_optimizer_three_tuple_stays_continuous():
    captured = []

    def factory(params):
        captured.append(params)
        return DslV2StrategyFactory(spec=_template()).build()

    optimizer = BayesianOptimizer(FeatureEngine(), RiskEngine(), COST_MODELS["tw_futures"])
    ohlcv = random_walk_bars(["2026-09-07"], seed=32, night=False)
    optimizer.optimize(factory, ohlcv, {"x": (0.0, 1.0, "float")}, n_trials=5, seed=1)
    off_grid = [p["x"] for p in captured if abs(p["x"] * 4 - round(p["x"] * 4)) > 1e-6]
    assert off_grid  # continuous suggestions, not snapped to any grid


def test_walk_forward_with_dsl_v2_factory():
    factory = DslV2StrategyFactory(spec=_template())
    days = [f"2026-09-{d:02d}" for d in (7, 8, 9, 10, 11, 14, 15, 16)]
    ohlcv = random_walk_bars(days, seed=33)
    analyzer = WalkForwardAnalyzer(
        FeatureEngine(),
        RiskEngine(),
        COST_MODELS["tw_futures"],
        sizing_config=SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1),  # a TX contract is ~4.6M TWD
        strategy_factory=factory,
    )
    cfg = WalkForwardConfig(train_days=700, validate_days=0, test_days=400, step_days=400, min_trades_per_oos=0)
    result = analyzer.analyze(factory.build({"or_min": 15, "tp_r": 2.0}), ohlcv, cfg)
    assert len(result.per_window) >= 2
    assert all(w.is_trade_count + w.oos_trade_count > 0 for w in result.per_window)


def test_autoresearch_request_accepts_and_validates_rule_v2():
    markets = [{"symbol": "TX", "market": "tw_futures", "interval": "5m"}]
    ok = AutoResearchRequest(markets=markets, strategy_type="rule_v2", base_config_json=_template())
    assert ok.strategy_type == "rule_v2"
    with pytest.raises(ValidationError, match="base_config_json"):
        AutoResearchRequest(markets=markets, strategy_type="rule_v2")
    bad = _template()
    bad["params"]["or_min"]["high"] = 150
    with pytest.raises(ValidationError, match=r"params\.or_min"):
        AutoResearchRequest(markets=markets, strategy_type="rule_v2", base_config_json=bad)
