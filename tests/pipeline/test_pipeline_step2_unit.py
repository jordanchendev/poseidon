"""Step 2 without a database: DSL v2 validation, the holdout lock, real backtests, optimize."""

from __future__ import annotations

import copy
import io
import uuid
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline import artifacts
from poseidon.pipeline.data import OhlcvCache, load_bars
from poseidon.pipeline.plan import default_config
from poseidon.pipeline.spec_validation import V2_REQUIRED, DslV2Validator
from poseidon.pipeline.stages import (
    BacktestStage,
    OptimizeStage,
    SpecValidateStage,
    StageContext,
    StageStopped,
    UpstreamResult,
)
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.trials import MemoryTrialLedger
from poseidon.pipeline.window import MARKET_TZ, HoldoutLockedError, holdout_split, require_part
from poseidon.strategies.dsl.v2 import validate_v2
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from tests.pipeline.conftest import STEP2_PLAN, SyntheticTxSource, orb_template, tx_spec, tx_v1_spec

AS_OF = date(2026, 10, 22)


def _config(**optimize):
    config = default_config(stage_plan=STEP2_PLAN, as_of=AS_OF)
    config["optimize"].update(optimize)
    return config


class Pipeline:
    """Runs stages directly, wiring upstream results the way the dispatcher does."""

    def __init__(self, root, spec, *, source=None, config=None, ledger=None):
        self.root = root
        self.spec = spec
        self.source = source or SyntheticTxSource()
        self.config = config or _config()
        self.ledger = ledger if ledger is not None else MemoryTrialLedger()
        self.results: dict[str, UpstreamResult] = {}
        self.outputs: dict = {}

    def context(self, stage, **extra):
        ctx = StageContext(
            run_id=uuid.uuid4(),
            family_id=uuid.uuid4(),
            idea_text="TX ORB",
            input_spec_json=self.spec,
            config=self.config,
            config_sha256=content_sha256(self.config),
            stage=stage.name,
            attempt=1,
            upstream={name: self.results[name] for name in stage.requires},
            artifacts_root=self.root,
            trials=self.ledger,
            **extra,
        )
        ctx.input_sha256 = content_sha256(stage.input_payload(ctx))
        return ctx

    def run(self, stage, **extra):
        ctx = self.context(stage, **extra)
        result = stage.run(ctx)
        manifest = []
        if result.files:
            manifest = artifacts.write_files(self.root / "out" / stage.name / uuid.uuid4().hex, result.files)
        _artifact, sha = artifacts.build_artifact(result.report, manifest)
        self.results[stage.name] = UpstreamResult(stage.name, 1, ctx.input_sha256, sha, result.report)
        self.outputs[stage.name] = (result, sha)
        return result, sha

    def through(self, *stages):
        out = None
        for stage in stages:
            out = self.run(stage)
            if out[0].verdict != Verdict.PASS:
                break
        return out

    def seed(self, spec_report: dict):
        self.results["spec_validate"] = UpstreamResult("spec_validate", 1, "1" * 64, "2" * 64, spec_report)


def _validated(spec):
    pipeline = Pipeline(None, spec)
    pipeline.results["spec_generate"] = UpstreamResult(
        "spec_generate", 1, "0" * 64, "0" * 64, {"spec": spec, "spec_sha256": content_sha256(spec)}
    )
    return pipeline.run(SpecValidateStage())[0]


# ------------------------------------------------------------------ spec_validate


def test_v2_spec_passes_and_the_artifact_is_the_concrete_spec():
    result = _validated(tx_spec())
    assert result.verdict == Verdict.PASS, result.report
    report = result.report
    assert report["validator"] == "dsl_v2" and report["errors"] == [] and report["unsupported"] == []
    concrete = report["spec"]
    assert "params" not in concrete and "$param" not in str(concrete)
    assert concrete["rules"][0]["condition"]["all"][1]["b"]["minutes"] == 30
    assert concrete["risk"]["take_profit"]["value"] == 2.0
    assert report["template"] == tx_spec() and set(report["params"]) == {"or_min", "tp_r"}
    assert report["resolved_spec_sha256"] == content_sha256(concrete)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d.update(market="tw_stock"), "market"),
        (lambda d: d.update(symbol="MTX", market="tw_futures_mtx"), "must target tw_futures/TX"),
        (lambda d: d["params"]["or_min"].update(low=10, high=25), "params.or_min"),  # 10m OR is not on the grid
        (lambda d: d["rules"][0].update(action="buy"), "rules.0"),
        (lambda d: d.update(dsl_version=3), "dsl_version"),
    ],
)
def test_v2_validation_errors_fail(mutate, fragment):
    spec = tx_spec()
    mutate(spec)
    result = _validated(spec)
    assert result.verdict == Verdict.FAIL and result.report["unsupported"] == []
    assert any(fragment in error for error in result.report["errors"]), result.report["errors"]


def test_htf_is_unsupported():
    spec = tx_spec()
    spec["rules"][0]["condition"]["all"].append({"htf": "1h", "type": "compare"})
    result = _validated(spec)
    assert result.verdict == Verdict.UNSUPPORTED and "htf" in result.reason


@pytest.mark.parametrize("spec", [tx_v1_spec(), tx_v1_spec(dsl_version=1), {**tx_orb30_day(), "dsl_version": 1}])
def test_v1_specs_are_rejected_as_unsupported(spec):
    result = _validated(spec)
    assert result.verdict == Verdict.UNSUPPORTED and result.reason == V2_REQUIRED
    assert DslV2Validator().validate_spec(spec) == (False, (), (V2_REQUIRED,))


def test_an_interval_other_than_the_frozen_window_is_unsupported():
    spec = tx_spec(interval="15m")
    result = _validated(spec)
    assert result.verdict == Verdict.UNSUPPORTED and "5m" in result.reason


# ------------------------------------------------------------------ window / holdout


def test_default_config_freezes_the_window_and_the_holdout_split():
    config = default_config(stage_plan=None, as_of=AS_OF)
    data = config["data"]
    assert (data["start"], data["end"], data["interval"]) == ("2020-03-02", "2026-10-22", "5m")
    split = datetime.fromisoformat(data["holdout_start"])
    assert split == holdout_split("2020-03-02", "2026-10-22", 0.2)
    local = split.astimezone(MARKET_TZ)
    assert (local.hour, local.minute) == (8, 0)  # between the night close and the day open
    begin = datetime(2020, 3, 1, 16, tzinfo=UTC)
    end = datetime(2026, 10, 21, 16, tzinfo=UTC)
    assert abs((split - begin) / (end - begin) - 0.8) < 1 / 365
    assert config["backtest"] == {
        "market": "tw_futures",
        "symbol": "TX",
        "cost_model": "tw_futures_pipeline",
        "sizing": {"mode": "fixed_contracts", "contracts": 1},
        "gap_aware_stops": True,
        "initial_capital_twd": 1_000_000.0,
        "bar_label": "open",
        "sharpe": {"period": "trading_day", "annualisation_factor": 252},
    }


@pytest.mark.parametrize("stage", ["backtest", "optimize", "wfe", "cost_stress", "spec_validate"])
def test_holdout_bars_are_refused_to_every_stage_but_holdout(tmp_path, stage):
    cache = OhlcvCache(tmp_path, SyntheticTxSource())
    with pytest.raises(HoldoutLockedError):
        load_bars(stage, _config(), "holdout", cache)
    with pytest.raises(HoldoutLockedError):
        require_part(stage, "holdout")


def test_train_and_holdout_parts_split_at_the_frozen_timestamp(tmp_path):
    config = _config()
    cache = OhlcvCache(tmp_path, SyntheticTxSource())
    split = pd.Timestamp(config["data"]["holdout_start"])
    train = load_bars("backtest", config, "train", cache)
    holdout = load_bars("holdout", config, "holdout", cache)
    assert train.frame.index.max() < split <= holdout.frame.index.min()
    assert train.data_sha256 == holdout.data_sha256 and train.window_sha256 != holdout.window_sha256
    assert len(cache.source.calls) == 1  # one fetch per frozen window, then the parquet cache


def test_backtest_and_optimize_only_ever_see_train_bars(tmp_path, monkeypatch):
    from poseidon.pipeline import backtesting, optimization

    seen = []
    real = backtesting.run_backtest

    def spy(strategy, bars, config, **kwargs):
        seen.append(bars.index.max())
        return real(strategy, bars, config, **kwargs)

    monkeypatch.setattr(backtesting, "run_backtest", spy)
    monkeypatch.setattr(optimization, "run_backtest", spy)
    pipeline = Pipeline(tmp_path, tx_spec())
    pipeline.seed(_validated(tx_spec()).report)
    pipeline.through(BacktestStage(pipeline.source), OptimizeStage(pipeline.source))
    split = pd.Timestamp(pipeline.config["data"]["holdout_start"])
    assert len(seen) == 1 + pipeline.outputs["optimize"][0].report["n_trials"]
    assert all(stamp < split for stamp in seen)


def test_a_cache_file_that_no_longer_matches_its_manifest_is_refused(tmp_path):
    from poseidon.pipeline.data import DataUnavailableError

    config = _config()
    cache = OhlcvCache(tmp_path, SyntheticTxSource())
    first = load_bars("backtest", config, "train", cache)
    path = next((tmp_path / "cache" / "ohlcv").glob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(DataUnavailableError, match="manifest"):
        load_bars("backtest", config, "train", cache)
    assert first.data_sha256


# ------------------------------------------------------------------ backtest


def test_backtest_report_metrics_and_files(tmp_path):
    pipeline = Pipeline(tmp_path, tx_spec())
    pipeline.seed(_validated(tx_spec()).report)
    result, _ = pipeline.run(BacktestStage(pipeline.source))
    assert result.verdict == Verdict.PASS
    metrics = result.report["metrics"]
    for key in (
        "total_pnl_twd",
        "gross_pnl_twd",
        "fees_twd",
        "n_trades",
        "win_rate",
        "avg_trade_twd",
        "sharpe_daily",
        "sharpe_annualised",
        "max_drawdown_twd",
        "exit_reasons",
        "n_obs",
        "skew",
        "kurtosis",
    ):
        assert key in metrics, key
    assert metrics["n_trades"] > 0 and metrics["kurtosis_convention"] == "non_excess"
    assert metrics["gross_pnl_twd"] == pytest.approx(metrics["total_pnl_twd"] + metrics["fees_twd"])
    assert metrics["sharpe_annualised"] == pytest.approx(metrics["sharpe_daily"] * 252**0.5)
    assert metrics["kurtosis"] >= 1 + metrics["skew"] ** 2  # Pearson bound, so not excess kurtosis
    assert sum(metrics["exit_reasons"].values()) == metrics["n_trades"]
    # 1 contract, 50 TWD per side: commission is exactly 100 TWD a round trip.
    assert metrics["costs"]["commission_twd"] == pytest.approx(100.0 * metrics["n_trades"])
    trades = pd.read_csv(io.BytesIO(result.files["trades.csv"]))
    assert len(trades) == metrics["n_trades"] and trades["pnl_twd"].sum() == pytest.approx(metrics["total_pnl_twd"])
    assert set(trades["quantity"]) == {1.0}
    daily = pd.read_csv(io.BytesIO(result.files["daily_pnl.csv"]))
    assert len(daily) == metrics["n_obs"] and daily["trading_day"].is_unique
    report_data = result.report["data"]
    assert report_data["data_sha256"] and report_data["window_sha256"] and report_data["part"] == "train"


def test_same_input_gives_the_same_artifact_sha(tmp_path):
    shas = []
    for name in ("a", "b"):  # separate roots: the parquet cache itself is rebuilt
        pipeline = Pipeline(tmp_path / name, tx_spec())
        pipeline.seed(_validated(tx_spec()).report)
        pipeline.through(BacktestStage(pipeline.source), OptimizeStage(pipeline.source))
        shas.append(
            (
                pipeline.results["backtest"].artifact_sha256,
                pipeline.results["optimize"].artifact_sha256,
                pipeline.results["backtest"].report["data"]["data_sha256"],
            )
        )
        pipeline.run(BacktestStage(pipeline.source))  # again on the warm cache
        assert pipeline.results["backtest"].artifact_sha256 == shas[-1][0]
    assert shas[0] == shas[1]
    other = Pipeline(tmp_path / "c", tx_spec(), source=SyntheticTxSource(seed=8))
    other.seed(_validated(tx_spec()).report)
    other.run(BacktestStage(other.source))
    assert other.results["backtest"].artifact_sha256 != shas[0][0]


def test_stop_inside_a_backtest_raises_stage_stopped(tmp_path):
    pipeline = Pipeline(tmp_path, tx_spec())
    pipeline.seed(_validated(tx_spec()).report)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 50

    with pytest.raises(StageStopped):
        pipeline.run(BacktestStage(pipeline.source), should_stop=should_stop)


# ------------------------------------------------------------------ optimize


def _optimize(tmp_path, spec, *, ledger=None, **optimize):
    pipeline = Pipeline(tmp_path, spec, config=_config(**optimize), ledger=ledger)
    pipeline.seed(_validated(spec).report)
    pipeline.through(BacktestStage(pipeline.source), OptimizeStage(pipeline.source))
    return pipeline


def test_every_trial_is_in_the_ledger_and_the_best_spec_is_concrete(tmp_path):
    pipeline = _optimize(tmp_path, tx_spec())
    result, _ = pipeline.outputs["optimize"]
    assert result.verdict == Verdict.PASS, result.reason
    report = result.report
    records = pipeline.ledger.records
    assert report["n_trials"] == len(records) == 6  # the whole 2 x 3 grid, defaults first
    assert report["search"]["stopped_because"] in {"grid_exhausted", "grid_filled"}
    assert records[0].params == {"or_min": 30, "tp_r": 2.0}
    assert [record.trial_no for record in records] == list(range(1, 7))
    assert len({record.params_sha256 for record in records}) == 6
    window = pipeline.results["backtest"].report["data"]["window_sha256"]
    assert {record.window_sha256 for record in records} == {window}
    complete = [record for record in records if record.state == "complete"]
    best = max(complete, key=lambda record: record.sharpe)
    assert report["best"]["params"] == best.params and report["best"]["trial_no"] == best.trial_no
    best_spec = report["best_spec"]
    assert "$param" not in str(best_spec) and best_spec["risk"]["take_profit"]["value"] == best.params["tp_r"]
    validate_v2(best_spec)  # concrete and DSL-valid (no $param left, so not itself a pipeline input spec)
    # Trial 1 is the defaults: identical to the backtest stage's numbers.
    assert records[0].sharpe == pipeline.results["backtest"].report["metrics"]["sharpe_daily"]
    trials = result.files["trials.json"]
    assert b'"source":"search"' in trials and "best_trades.csv" in result.files


def test_failed_trials_are_recorded_too(tmp_path, monkeypatch):
    from poseidon.pipeline import optimization
    from poseidon.pipeline.backtesting import BacktestFailed

    real = optimization.run_backtest

    def flaky(strategy, bars, config, **kwargs):
        if strategy.config.risk.take_profit.value == 1.5:
            raise BacktestFailed("synthetic failure")
        return real(strategy, bars, config, **kwargs)

    monkeypatch.setattr(optimization, "run_backtest", flaky)
    pipeline = _optimize(tmp_path, tx_spec())
    report = pipeline.outputs["optimize"][0].report
    states = [record.state for record in pipeline.ledger.records]
    assert states.count("fail") == 2 and len(states) == report["n_trials"] == 6
    assert report["n_trials_by_state"]["fail"] == 2
    failed = [record for record in pipeline.ledger.records if record.state == "fail"]
    assert all(record.sharpe is None and record.params["tp_r"] == 1.5 for record in failed)


def test_missing_one_step_neighbours_of_the_best_are_evaluated(tmp_path):
    params = {
        "or_min": {"type": "int", "low": 15, "high": 60, "step": 15, "default": 30},
        "tp_r": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5, "default": 2.0},
    }
    pipeline = _optimize(tmp_path, orb_template(params), max_trials=3)
    report = pipeline.outputs["optimize"][0].report
    assert report["n_trials_by_source"]["search"] == 3
    best = report["best"]["params"]
    evaluated = {(r.params["or_min"], r.params["tp_r"]) for r in pipeline.ledger.records}
    neighbours = report["neighbours"]
    assert len(neighbours) == 4 and report["neighbourhood_settled"]
    for row in neighbours:
        if row["in_grid"]:
            assert (row["params"]["or_min"], row["params"]["tp_r"]) in evaluated and row["trial_no"]
            assert row["total_pnl_twd"] is not None or row["state"] != "complete"
            changed = [k for k in best if row["params"][k] != best[k]]
            assert changed == [row["param"]]
        else:
            assert row["trial_no"] is None
    assert report["n_trials"] == len(pipeline.ledger.records) == len(evaluated)
    assert report["n_trials_by_source"]["neighbour"] == report["n_trials"] - 3


def test_a_hand_written_spec_without_params_fails_spec_validate():
    result = _validated(tx_orb30_day())
    assert result.verdict == Verdict.FAIL and result.reason == "spec invalid: 1 error(s)"
    (error,) = result.report["errors"]
    assert error.startswith("params: the pipeline requires 1-4 tunable $params, the spec declares 0")
    assert "opening-range minutes" in error and "stop/target distances" in error and "thresholds" in error


def _five_params() -> dict:
    spec = orb_template()
    spec["params"]["sl_buf"] = {"type": "int", "low": 0, "high": 10, "step": 5, "default": 0}
    spec["params"]["flat_min"] = {"type": "int", "low": 5, "high": 15, "step": 5, "default": 10}
    spec["params"]["max_n"] = {"type": "int", "low": 1, "high": 2, "step": 1, "default": 1}
    for rule in spec["rules"]:
        rule["risk"]["stop_loss"]["offset_points"] = {"$param": "sl_buf"}
    spec["risk"]["flat_before_session_end_minutes"] = {"$param": "flat_min"}
    spec["risk"]["max_trades_per_session"] = {"$param": "max_n"}
    return spec


def test_more_than_four_params_fail_spec_validate():
    spec = _five_params()
    assert len(validate_v2(spec).params) == 5  # the DSL itself allows it; the pipeline does not
    result = _validated(spec)
    assert result.verdict == Verdict.FAIL
    assert result.report["errors"][0].startswith(
        "params: the pipeline requires 1-4 tunable $params, the spec declares 5"
    )


def test_optimize_without_params_still_records_the_base_backtest(tmp_path):
    """Defensive path only: spec_validate now rejects param-less specs before optimize."""

    spec = tx_orb30_day()
    resolved = DslV2Validator().resolve(spec)
    report = {"spec": resolved["spec"], "template": resolved["template"], "params": resolved["params"]}
    report["spec_sha256"] = content_sha256(spec)
    pipeline = Pipeline(tmp_path, spec)
    pipeline.seed(report)
    pipeline.through(BacktestStage(pipeline.source), OptimizeStage(pipeline.source))
    result = pipeline.outputs["optimize"][0]
    assert result.verdict == Verdict.PASS
    (record,) = pipeline.ledger.records
    metrics = pipeline.results["backtest"].report["metrics"]
    assert (record.trial_no, record.params, record.sharpe, record.n_obs) == (
        1,
        {},
        metrics["sharpe_daily"],
        metrics["n_obs"],
    )
    assert result.report["best_spec"] == report["spec"]
    assert result.report["neighbours"] == []


def test_optimize_stops_between_and_inside_trials(tmp_path):
    pipeline = Pipeline(tmp_path, tx_spec())
    pipeline.seed(_validated(tx_spec()).report)
    pipeline.run(BacktestStage(pipeline.source))
    flag = {"stop": False}
    ledger = pipeline.ledger

    class StopAfterFirst(MemoryTrialLedger):
        def append(self, record):
            ledger.append(record)
            flag["stop"] = True

    pipeline.ledger = StopAfterFirst()

    def should_stop():
        return flag["stop"]

    with pytest.raises(StageStopped):
        pipeline.run(OptimizeStage(pipeline.source), should_stop=should_stop)
    assert len(ledger.records) == 1


def test_optimize_report_fits_the_artifact_limit(tmp_path):
    pipeline = _optimize(tmp_path, tx_spec())
    artifact, _ = artifacts.build_artifact(pipeline.outputs["optimize"][0].report, [])
    assert artifact is not None


def test_template_is_not_mutated_by_the_stages(tmp_path):
    spec = tx_spec()
    before = copy.deepcopy(spec)
    _optimize(tmp_path, spec)
    assert spec == before


# ------------------------------------------------------------------ bar labels


def test_close_labelled_sessions_are_shifted_to_open_labels():
    from poseidon.pipeline.data import normalise, relabel_to_open
    from tests.dsl_v2_support import random_walk_bars

    days = ["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30", "2026-05-04"]
    opened = normalise(random_walk_bars(days, seed=3))
    local = opened.index.tz_convert(MARKET_TZ)
    hm = local.strftime("%H:%M")
    early = local < pd.Timestamp("2026-04-29 15:00", tz=MARKET_TZ)  # through the 04-29 day session
    extra = (hm == "13:45") | (hm == "05:00")  # Thalassa's auction bars exist only open-labelled
    old_core = opened[early & ~extra]
    closed = old_core.copy()
    closed.index = closed.index + pd.Timedelta(minutes=5)  # how the close-labelled history carries them
    mixed = pd.concat([closed, opened[~early]]).sort_index()
    relabelled, stats = relabel_to_open(mixed, 5)
    assert relabelled.equals(pd.concat([old_core, opened[~early]]).sort_index())
    assert stats["sessions_close_shifted"] == 5  # day+night 04-27, day+night 04-28, day 04-29
    assert stats["last_close_labelled_session"] == "2026-04-29 day"
    unchanged, stats = relabel_to_open(opened, 5)
    assert unchanged.equals(opened) and stats["sessions_close_shifted"] == 0


def test_cache_records_the_label_normalisation(tmp_path):
    bars = load_bars("backtest", _config(), "train", OhlcvCache(tmp_path, SyntheticTxSource()))
    assert bars.provenance()["bar_labels"]["convention"] == "open"
    assert bars.provenance()["bar_labels"]["sessions_close_shifted"] == 0


def test_the_wall_clock_budget_is_only_a_safety_stop(tmp_path):
    pipeline = _optimize(tmp_path, tx_spec(), budget_seconds=1e-6)
    report = pipeline.outputs["optimize"][0].report
    assert report["search"]["stopped_because"] == "budget" and report["search"]["budget_exhausted"]
    assert report["n_trials_by_source"]["search"] == 1  # the defaults, then the stop
    assert report["n_trials_by_source"]["neighbour"] >= 1  # the neighbourhood is still completed
