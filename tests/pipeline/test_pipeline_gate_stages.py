"""Step 3b without a database: the six gates wired as stages on small synthetic data."""

from __future__ import annotations

import copy
import io
import itertools
import uuid
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline import artifacts
from poseidon.pipeline.family import HOLDOUT_EVALUATED, MemoryFamily
from poseidon.pipeline.gate_stages import (
    GATES_CONFIG_MISSING,
    build_gate_stages,
    daily_bars,
    walk_forward_windows,
)
from poseidon.pipeline.plan import (
    ALL_GATES_PASSED,
    ALL_STAGES_PASSED,
    CONFIG_VERSION,
    DEFAULT_STAGE_PLAN,
    GATE_STAGES,
    V1_STAGE_PLAN,
    StagePlanError,
    completion_reason,
    default_config,
    validate_stage_plan,
)
from poseidon.pipeline.stages import (
    STAGE_REGISTRY,
    BacktestStage,
    OptimizeStage,
    SpecValidateStage,
    StageContext,
    UpstreamResult,
)
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.trials import make_record
from poseidon.pipeline.window import MARKET_TZ
from tests.pipeline.conftest import SyntheticTxSource, tx_spec
from tests.pipeline.gate_support import (
    PlantedOrbSource,
    RandomEntryFactory,
    StageChain,
    random_entry_spec,
)

AS_OF = date(2026, 10, 22)


def small_config() -> dict:
    """Default v3 config with windows/thresholds shrunk to fit 20 synthetic business days."""

    config = default_config(stage_plan=None, as_of=AS_OF)
    config["gates"]["wfe"].update(windows=2, is_days=6, oos_days=3, max_trials_per_window=3)
    config["gates"]["warmup_days"] = 2
    config["thresholds"].update(
        regime_trend_sma_days=3,
        regime_atr_days=2,
        regime_vol_median_days=3,
        regime_min_trades_per_segment=1,
        regime_min_segments=1,
        holdout_min_trades=1,
        dsr_min_obs=5,
    )
    return config


def gate_registry(source=None, factory=None):
    return build_gate_stages(data_source=source, strategy_factory=factory)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """spec_validate -> backtest -> optimize, then every gate (whatever the verdicts)."""

    root = tmp_path_factory.mktemp("gates")
    source = SyntheticTxSource()
    chain = StageChain(root, tx_spec(), small_config(), family=None)
    chain.through([SpecValidateStage(), BacktestStage(source), OptimizeStage(source)])
    assert chain.outputs["optimize"].verdict == Verdict.PASS
    chain.ledger_after_optimize = len(chain.ledger.records)
    for stage in gate_registry(source).values():
        chain.run(stage)
    return chain


# ------------------------------------------------------------------ plan


def test_default_plan_is_the_gate_order_then_shadow_paper():
    config = default_config(stage_plan=None, as_of=AS_OF)
    assert config["version"] == CONFIG_VERSION == 5 and "gates" in config and "shadow_paper" in config
    assert config["stage_plan"] == list(DEFAULT_STAGE_PLAN)
    assert tuple(config["stage_plan"][4:11]) == GATE_STAGES
    assert GATE_STAGES == ("param_stability", "dsr", "cost_stress", "regime_split", "benchmark", "wfe", "holdout")
    assert V1_STAGE_PLAN[-2:] == ("shadow_paper", "awaiting_release")
    assert validate_stage_plan(list(DEFAULT_STAGE_PLAN), STAGE_REGISTRY) == list(DEFAULT_STAGE_PLAN)


@pytest.mark.parametrize(
    "plan",
    [
        ["spec_generate", "spec_validate", "backtest", "optimize", "holdout"],
        ["spec_generate", "spec_validate", "backtest", "optimize", "param_stability", "dsr", "cost_stress", "holdout"],
        ["spec_generate", "spec_validate", "backtest", "optimize", "wfe", "dsr"],  # out of order
        # every train gate of a version-4 plan, but no benchmark: a new run needs it before holdout
        [
            "spec_generate",
            "spec_validate",
            "backtest",
            "optimize",
            "param_stability",
            "dsr",
            "cost_stress",
            "regime_split",
            "wfe",
            "holdout",
        ],
    ],
)
def test_holdout_needs_every_train_gate_first(plan):
    with pytest.raises(StagePlanError):
        validate_stage_plan(plan, STAGE_REGISTRY)


def test_completion_reason():
    assert completion_reason(DEFAULT_STAGE_PLAN) == ALL_GATES_PASSED == "passed all validation gates"
    assert completion_reason(["spec_generate", "spec_validate", "backtest", "optimize"]) == ALL_STAGES_PASSED


@pytest.mark.parametrize("name", GATE_STAGES)
def test_runs_frozen_before_step3_are_unsupported_at_every_gate(name, tmp_path):
    config = small_config()
    config["version"] = 2
    config.pop("gates")
    stage = gate_registry(SyntheticTxSource())[name]
    report = {"best_spec": {"dsl_version": 2}, "best": {"params": {}}}
    upstream = {req: UpstreamResult(req, 1, "0" * 64, "1" * 64, report) for req in stage.requires}
    ctx = StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="x",
        input_spec_json=None,
        config=config,
        config_sha256=content_sha256(config),
        stage=name,
        attempt=1,
        upstream=upstream,
        artifacts_root=tmp_path,
    )
    result = stage.run(ctx)
    assert result.verdict == Verdict.UNSUPPORTED and result.reason == GATES_CONFIG_MISSING


# ------------------------------------------------------------------ report shape


@pytest.mark.parametrize("name", GATE_STAGES)
def test_every_gate_report_carries_the_summary_line_and_the_full_gate_report(chain, name):
    result = chain.outputs[name]
    assert result.verdict in (Verdict.PASS, Verdict.FAIL)
    report = result.report
    mark = "✅" if result.verdict == Verdict.PASS else "❌"
    assert report["summary"].startswith(f"{name} {mark} ")
    assert report["gate"] == name and report["verdict"] == result.verdict.value
    assert isinstance(report["gate_report"], dict) and isinstance(report["inputs"], dict)
    if result.verdict == Verdict.FAIL:
        assert result.reason == report["reason"]
    else:
        assert result.reason == report["summary"]
    artifact, _ = artifacts.build_artifact(report, [])
    assert artifact["report"]["summary"] == report["summary"]  # the path Kairos reads


def test_gates_use_the_optimize_best_spec(chain):
    best_sha = chain.outputs["optimize"].report["best_spec_sha256"]
    for name in ("param_stability", "dsr", "cost_stress", "regime_split", "benchmark", "holdout"):
        assert chain.outputs[name].report["inputs"]["best_spec_sha256"] == best_sha


# ------------------------------------------------------------------ individual gates


def test_param_stability_reads_the_optimize_neighbourhood(chain):
    optimize = chain.outputs["optimize"].report
    gate = chain.outputs["param_stability"].report["gate_report"]
    in_grid = [row for row in optimize["neighbours"] if row["in_grid"]]
    assert gate["evaluated"] == len(in_grid)
    assert gate["best_params"] == optimize["best"]["params"]


def test_param_stability_counts_a_failed_neighbour_as_not_profitable(tmp_path):
    stage = gate_registry()["param_stability"]
    optimize = {
        "best_spec": {"dsl_version": 2},
        "best_spec_sha256": "a" * 64,
        "best": {"trial_no": 1, "params": {"x": 2}, "metrics": {"total_pnl_twd": 1000.0}},
        "neighbours": [
            {
                "param": "x",
                "direction": -1,
                "in_grid": True,
                "params": {"x": 1},
                "state": "complete",
                "total_pnl_twd": 5.0,
            },
            {"param": "x", "direction": 1, "in_grid": True, "params": {"x": 3}, "state": "fail", "total_pnl_twd": None},
        ],
    }
    chain = StageChain(tmp_path, tx_spec(), small_config())
    chain.results["optimize"] = UpstreamResult("optimize", 1, "0" * 64, "1" * 64, optimize)
    result = chain.run(stage)
    gate = result.report["gate_report"]
    assert (gate["evaluated"], gate["profitable"], gate["profitable_share"]) == (2, 1, 0.5)
    assert result.verdict == Verdict.FAIL and result.report["gate_report"]["profitable_share_min"] == 0.6
    assert result.report["inputs"]["failed_neighbours_counted_unprofitable"] == [
        {"param": "x", "direction": 1, "trial_no": None}
    ]


def test_dsr_n_is_every_family_trial(chain):
    gate = chain.outputs["dsr"].report["gate_report"]
    inputs = chain.outputs["dsr"].report["inputs"]
    # dsr ran right after param_stability: N is the optimize ledger so far.
    assert gate["n_trials"] == inputs["n_trials_used"] == chain.ledger_after_optimize
    assert "before wfe" in inputs["order_note"]


def test_dsr_reads_the_family_view_when_there_is_one(tmp_path):
    from poseidon.pipeline.trials import MemoryTrialLedger

    stage = gate_registry()["dsr"]
    ledger = MemoryTrialLedger()
    for number in range(1, 41):
        ledger.append(make_record(number, {"x": number}, "complete", {"sharpe_daily": 0.01 * (number % 7)}, "w" * 64))
    ledger.append(make_record(41, {"x": 41}, "fail", None, "w" * 64))
    optimize = {
        "best_spec": {"dsl_version": 2},
        "best": {
            "trial_no": 3,
            "params": {},
            "metrics": {"sharpe_daily": 0.2, "n_obs": 500, "skew": 0.0, "kurtosis": 3.0},
        },
    }
    chain = StageChain(tmp_path, tx_spec(), small_config(), family=MemoryFamily(ledger))
    chain.results["optimize"] = UpstreamResult("optimize", 1, "0" * 64, "1" * 64, optimize)
    result = chain.run(stage)
    gate = result.report["gate_report"]
    assert gate["n_trials"] == 41 and gate["n_trials_with_sharpe"] == 40
    assert result.report["inputs"]["trial_states"] == {"complete": 40, "fail": 1}


def test_cost_stress_doubles_costs_and_slips_stop_fills(chain):
    report = chain.outputs["cost_stress"].report
    gate = report["gate_report"]
    assert gate["cost_multiplier"] == gate["stressed_run_multiplier"] == 2.0
    base = pd.read_csv(io.BytesIO(chain.outputs["optimize"].files["best_trades.csv"]))
    stressed = pd.read_csv(io.BytesIO(chain.outputs["cost_stress"].files["stressed_trades.csv"]))
    assert list(base["entry_time"]) == list(stressed["entry_time"])  # costs never move signals
    assert gate["stressed_costs"]["commission_twd"] == pytest.approx(2 * gate["base_costs"]["commission_twd"])
    sign = stressed["side"].map({"long": 1, "short": -1})
    # Entry legs: 2 ticks instead of 1 against the position.
    assert ((stressed["entry_price"] - base["entry_price"]) * sign == 1.0).all()
    stops = stressed["exit_reason"] == "stop_loss"
    assert stops.any()
    # Stop fills: 1 tick against the position in the stressed run only.
    assert ((base["exit_price"] - stressed["exit_price"])[stops] * sign[stops] == 1.0).all()
    assert gate["stressed_total_pnl_twd"] < gate["base_total_pnl_twd"]
    assert "stop-loss fill" in report["inputs"]["note"] and report["inputs"]["stop_slippage_ticks"] == 1.0


def test_regime_split_uses_the_optimize_trades_and_trading_day_bars(chain):
    report = chain.outputs["regime_split"].report
    assert report["inputs"]["trades_source"].startswith("optimize best_trades.csv")
    assert report["gate_report"]["trades"] == chain.outputs["optimize"].report["best"]["metrics"]["n_trades"]


def test_daily_bars_put_the_night_session_on_the_next_trading_day():
    stamps = (
        pd.DatetimeIndex(
            [
                "2026-10-22 08:45",  # Thu day
                "2026-10-22 13:45",
                "2026-10-22 15:00",  # Thu night -> Fri
                "2026-10-23 04:55",
                "2026-10-23 08:45",  # Fri day
                "2026-10-23 15:00",  # Fri night -> Mon
                "2026-10-24 04:55",
            ]
        )
        .tz_localize(MARKET_TZ)
        .tz_convert("UTC")
    )
    frame = pd.DataFrame(
        {"open": range(7), "high": range(10, 17), "low": range(7), "close": range(20, 27), "volume": 1.0},
        index=stamps,
        dtype=float,
    )
    daily = daily_bars(frame)
    assert list(daily.index) == [date(2026, 10, 22), date(2026, 10, 23), date(2026, 10, 26)]
    assert list(daily["open"]) == [0.0, 2.0, 5.0] and list(daily["close"]) == [21.0, 24.0, 26.0]
    assert list(daily["high"]) == [11.0, 14.0, 16.0]


def test_wfe_windows_are_rolling_and_anchored_at_the_train_end():
    until = datetime(2025, 6, 24, 0, 0, tzinfo=UTC)  # 08:00 Asia/Taipei
    windows = walk_forward_windows(until, windows=4, is_days=730, oos_days=182)
    assert windows[-1]["oos_end"] == until
    for before, after in itertools.pairwise(windows):
        assert before["oos_end"] == after["oos_start"]
    for window in windows:
        for key in ("is_start", "oos_start", "oos_end"):
            local = window[key].astimezone(MARKET_TZ)
            assert (local.hour, local.minute) == (8, 0)
        assert window["oos_start"] - window["is_start"] == timedelta(days=730)
        assert window["oos_end"] - window["oos_start"] == timedelta(days=182)


def test_wfe_reoptimises_per_window_and_records_every_trial(chain):
    report = chain.outputs["wfe"].report
    inputs = report["inputs"]
    records = chain.ledger.stage_records("wfe")
    assert inputs["n_trials_recorded"] == len(records) == sum(w["n_trials"] for w in inputs["windows"])
    assert [record.trial_no for record in records] == list(range(1, len(records) + 1))
    assert all(w["n_trials"] <= 3 for w in inputs["windows"]) and len(inputs["windows"]) == 2
    assert len({record.window_sha256 for record in records}) == 2  # one IS window hash per window
    assert len(report["gate_report"]["windows"]) == 2
    for window in inputs["windows"]:
        assert window["is_from_utc"] < window["oos_from_utc"] < window["oos_until_utc"]


def test_wfe_with_too_little_history_fails(tmp_path, chain):
    config = small_config()
    config["gates"]["wfe"]["is_days"] = 4000
    probe = StageChain(tmp_path, tx_spec(), config)
    probe.results.update({name: chain.results[name] for name in ("spec_validate", "optimize")})
    result = probe.run(gate_registry(SyntheticTxSource())["wfe"])
    assert result.verdict == Verdict.FAIL and result.report["reason_code"] == "insufficient_history"


def test_holdout_records_the_look_and_counts_other_runs(tmp_path, chain):
    other = uuid.uuid4()
    probe = StageChain(tmp_path, tx_spec(), small_config())
    family = MemoryFamily(probe.ledger, run_id=probe.run_id, holdout_runs=[other, other])
    probe.family = family
    probe.results.update(chain.results)
    result = probe.run(gate_registry(SyntheticTxSource())["holdout"])
    evaluations = result.report["inputs"]["family_holdout_evaluations"]
    assert (evaluations["other_runs"], evaluations["this_run_previous_evaluations"]) == (1, 0)
    assert [event["event_type"] for event in family.events] == [HOLDOUT_EVALUATED]
    assert family.events[0]["payload"]["family_runs_already_evaluated"] == 1
    split = pd.Timestamp(probe.config["data"]["holdout_start"])
    trades = pd.read_csv(io.BytesIO(result.files["holdout_trades.csv"]))
    assert len(trades) and (pd.to_datetime(trades["entry_time"], utc=True) >= split).all()
    # Criterion (b)'s T is the holdout backtest's trading-day count (its n_obs).
    gate = result.report["gate_report"]
    assert gate["holdout_n_obs"] == result.report["inputs"]["holdout_metrics"]["n_obs"] >= 2
    assert gate["criteria"]["se"]["n_obs"] == gate["holdout_n_obs"] and gate["criteria"]["se"]["enabled"]
    again = probe.run(gate_registry(SyntheticTxSource())["holdout"])
    assert again.report["inputs"]["family_holdout_evaluations"]["this_run_previous_evaluations"] == 1


def test_trade_from_makes_the_warmup_flat(tmp_path):
    from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
    from poseidon.pipeline.backtesting import run_backtest, trading_days

    bars = SyntheticTxSource().fetch(
        "TX", "tw_futures", "5m", datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )
    config = small_config()["backtest"]
    cut = bars.index[len(bars) // 2]
    full = run_backtest(DslV2StrategyFactory(spec=tx_spec()).build(), bars, config)
    late = run_backtest(DslV2StrategyFactory(spec=tx_spec()).build(), bars, config, trade_from=cut)
    assert full.metrics["n_trades"] > late.metrics["n_trades"] > 0
    assert all(pd.Timestamp(t["entry_time"]) >= cut for t in late.trades)
    assert late.daily.index.min() == trading_days(pd.DatetimeIndex([cut]))[0]
    assert late.metrics["n_obs"] == len(late.daily)


def test_stressed_cost_model_multiplies_every_cost():
    from poseidon.backtest.cost_model import get_cost_model
    from poseidon.pipeline.backtesting import stressed_cost_model

    base = get_cost_model("tw_futures_pipeline")
    model = stressed_cost_model("tw_futures_pipeline", 2.0)
    assert (model.commission_per_contract, model.tax_rate, model.slippage_ticks) == (
        2 * base.commission_per_contract,
        2 * base.tax_rate,
        2 * base.slippage_ticks,
    )
    assert (model.point_value, model.tick_size, model.tax_on_entry) == (base.point_value, base.tick_size, True)


# ------------------------------------------------------------------ control helpers


def test_random_entry_strategy_is_seeded_and_trades_both_sides():
    from poseidon.pipeline.backtesting import run_backtest

    bars = SyntheticTxSource(days_before=30).fetch(
        "TX", "tw_futures", "5m", datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )
    config = small_config()["backtest"]
    factory = RandomEntryFactory(spec=random_entry_spec())
    first = run_backtest(factory.build(), bars, config)
    second = run_backtest(factory.build(), bars, config)
    assert first.trades == second.trades and first.metrics["n_trades"] > 10
    assert first.metrics["n_long"] > 0 and first.metrics["n_short"] > 0
    entries = {pd.Timestamp(t["entry_time"]).tz_convert(MARKET_TZ).strftime("%H:%M") for t in first.trades}
    assert len(entries) > 5  # entry times are scattered, not one breakout bar


def test_planted_source_is_deterministic_and_plants_the_pattern():
    source = PlantedOrbSource()
    one = source.frame_for(date(2024, 1, 1), date(2024, 3, 29))
    two = PlantedOrbSource().frame_for(date(2024, 1, 1), date(2024, 3, 29))
    assert one.equals(two) and len(one) == 65 * 61
    assert source.planted_days > 50  # the 30-min range breaks on most days
    flat = PlantedOrbSource(edge=False)
    flat.frame_for(date(2024, 1, 1), date(2024, 3, 29))
    assert flat.planted_days == 0
    local = one.index.tz_convert(MARKET_TZ)
    assert local.min().strftime("%H:%M") == "08:45" and local.max().strftime("%H:%M") == "13:45"


def test_random_entry_spec_satisfies_the_param_policy():
    from poseidon.pipeline.spec_validation import DslV2Validator

    assert DslV2Validator().validate_spec(copy.deepcopy(random_entry_spec())).ok


def test_wfe_gate_reports_a_window_without_returns_as_invalid_input():
    from poseidon.pipeline.gates import wfe

    window = wfe.WindowMetrics(0, None, 0.1, 0.0, 10.0, 0, 3)
    result = wfe.evaluate(wfe.WfeInputs([window]), {"wfe_min": 0.5})
    assert result.reason_code == "invalid_input" and result.report["windows"][0]["wfe"] is None
