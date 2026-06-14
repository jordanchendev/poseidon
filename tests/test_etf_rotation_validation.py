from __future__ import annotations

import json
from pathlib import Path

from poseidon.research.etf_rotation.validation import (
    classify_start_regime,
    run_three_layer_validation,
    write_validation_outputs,
)


def _history(values: list[float]) -> list[dict[str, float | str]]:
    return [{"date": f"2020-{index + 1:02d}-29", "multiple": value} for index, value in enumerate(values)]


def _strategy_data() -> dict:
    return {
        "pairs": [{"key": "TEST", "name": "Core / Leveraged", "core": "CORE", "lev": "LEV"}],
        "strategies": [
            {
                "pair": "TEST",
                "choice": "core_buy_hold",
                "choiceName": "純核心持有",
                "rule": "不切換；全程持有 100% CORE。",
                "cagr": 0.08,
                "maxdd": -0.20,
                "switches": 0,
                "history": _history([1.00, 1.10, 1.20, 1.00, 0.90, 0.95, 1.05, 1.10, 1.08, 1.18, 1.25, 1.20]),
            },
            {
                "pair": "TEST",
                "choice": "leveraged_buy_hold",
                "choiceName": "純槓桿持有",
                "rule": "不切換；全程持有 100% LEV。",
                "cagr": 0.15,
                "maxdd": -0.45,
                "switches": 0,
                "history": _history([1.00, 1.22, 1.45, 0.95, 0.70, 0.82, 1.12, 1.32, 1.25, 1.55, 1.90, 1.70]),
            },
            {
                "pair": "TEST",
                "choice": "best_full_return",
                "choiceName": "歷史報酬最高",
                "rule": "平常持有 95% CORE / 5% LEV；CORE 下跌 10% 時，把 LEV 調到 100%。",
                "cagr": 0.16,
                "maxdd": -0.35,
                "switches": 3,
                "history": _history([1.00, 1.16, 1.32, 1.02, 0.86, 1.00, 1.20, 1.42, 1.38, 1.70, 2.05, 1.95]),
            },
        ],
    }


def test_classify_start_regime_uses_core_lookback_return_and_drawdown() -> None:
    core = [1.00, 1.10, 1.20, 1.00, 0.90, 0.95, 1.05, 1.10]

    assert classify_start_regime(core, start_index=2, lookback_months=2) == "bull"
    assert classify_start_regime(core, start_index=4, lookback_months=2) == "bear"
    assert classify_start_regime(core, start_index=7, lookback_months=2) == "sideways"
    assert classify_start_regime(core, start_index=1, lookback_months=2) == "unknown"


def test_three_layer_validation_summarizes_rolling_regime_and_monte_carlo() -> None:
    summary = run_three_layer_validation(
        _strategy_data(),
        pair="TEST",
        horizons_years=(0.25, 0.5),
        regime_horizon_years=0.25,
        monte_carlo_years=1,
        monte_carlo_paths=200,
        seed=42,
    )

    assert summary["pair"] == "TEST"
    assert summary["parameters"]["horizonsYears"] == [0.25, 0.5]

    rolling = summary["rolling"]
    best_3m = next(row for row in rolling if row["choice"] == "best_full_return" and row["horizonMonths"] == 3)
    assert best_3m["windows"] == 9
    assert best_3m["medianFinalMultiple"] > 1.0
    assert best_3m["winRateVsCore"] > 0.5
    assert best_3m["worstMaxDrawdown"] < 0

    regimes = summary["regimes"]
    labels = {row["regime"] for row in regimes}
    assert {"bull", "bear", "sideways"} <= labels
    bear_best = next(row for row in regimes if row["choice"] == "best_full_return" and row["regime"] == "bear")
    assert bear_best["windows"] >= 1

    monte_carlo = summary["monteCarlo"]
    leveraged = next(row for row in monte_carlo if row["choice"] == "leveraged_buy_hold")
    assert leveraged["paths"] == 200
    assert leveraged["p05FinalMultiple"] <= leveraged["p50FinalMultiple"] <= leveraged["p95FinalMultiple"]
    assert 0 <= leveraged["probabilityOfLoss"] <= 1

    repeat = run_three_layer_validation(
        _strategy_data(),
        pair="TEST",
        horizons_years=(0.25, 0.5),
        regime_horizon_years=0.25,
        monte_carlo_years=1,
        monte_carlo_paths=200,
        seed=42,
    )
    assert repeat["monteCarlo"] == monte_carlo


def test_write_validation_outputs_emits_json_and_layer_csvs(tmp_path: Path) -> None:
    summary = run_three_layer_validation(
        _strategy_data(),
        pair="TEST",
        horizons_years=(0.25,),
        regime_horizon_years=0.25,
        monte_carlo_years=1,
        monte_carlo_paths=50,
        seed=7,
    )

    outputs = write_validation_outputs(summary, tmp_path)

    assert set(outputs) == {"json", "rollingCsv", "regimeCsv", "monteCarloCsv"}
    assert Path(outputs["json"]).exists()
    assert json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))["pair"] == "TEST"
    assert Path(outputs["rollingCsv"]).read_text(encoding="utf-8").startswith("choice,choiceName")
    assert Path(outputs["regimeCsv"]).exists()
    assert Path(outputs["monteCarloCsv"]).exists()


def test_validation_script_wrapper_runs_selected_pair(tmp_path: Path) -> None:
    from scripts.etf_rotation_validate import run_etf_rotation_validation

    strategy_data = _strategy_data()
    root = tmp_path / "root"
    root.mkdir()
    strategy_json = root / "strategy-data.json"
    strategy_json.write_text(json.dumps(strategy_data), encoding="utf-8")

    summary = run_etf_rotation_validation(
        strategy_data=strategy_data,
        out_dir=tmp_path / "out",
        pairs=("TEST",),
        horizons_years=(0.25,),
        monte_carlo_paths=25,
        seed=11,
    )

    assert summary["pairs"] == ["TEST"]
    assert summary["outputs"]["TEST"]["json"].endswith("TEST_three_layer_validation.json")
    assert Path(summary["outputs"]["TEST"]["json"]).exists()
