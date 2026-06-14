from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from poseidon.research.etf_rotation.report import (
    build_strategy_calculator_html,
    load_embedded_strategy_data,
    verify_strategy_calculator_html,
)


def _write_minimal_rotation_research(root: Path) -> None:
    results = root / "results"
    data = root / "data"
    results.mkdir(parents=True)
    data.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "NASDAQ",
                "choice": "core_buy_hold",
                "stages": 0,
                "floor": 0.0,
                "max_lev": 0.0,
                "enter": "",
                "exit": "",
                "levels": "",
                "cagr": 0.10,
                "maxdd": -0.25,
                "switches": 0,
                "capital_twd": 1_000_000,
                "final_multiple": 2.0,
            },
            {
                "pair": "NASDAQ",
                "choice": "leveraged_buy_hold",
                "stages": 0,
                "floor": 1.0,
                "max_lev": 1.0,
                "enter": "",
                "exit": "",
                "levels": "",
                "cagr": 0.18,
                "maxdd": -0.70,
                "switches": 0,
                "capital_twd": 1_000_000,
                "final_multiple": 5.0,
            },
            {
                "pair": "NASDAQ",
                "choice": "best_full_return",
                "stages": 1,
                "floor": 0.05,
                "max_lev": 1.0,
                "enter": "0.10",
                "exit": "0.40",
                "levels": "1.0",
                "cagr": 0.20,
                "maxdd": -0.60,
                "switches": 3,
                "capital_twd": 1_000_000,
                "final_multiple": 6.0,
            },
            {
                "pair": "NASDAQ",
                "choice": "best_score",
                "stages": 1,
                "floor": 0.05,
                "max_lev": 1.0,
                "enter": "0.10",
                "exit": "0.40",
                "levels": "1.0",
                "cagr": 0.19,
                "maxdd": -0.58,
                "switches": 3,
                "capital_twd": 1_000_000,
                "final_multiple": 5.7,
            },
            {
                "pair": "NASDAQ",
                "choice": "current_mix_buy_hold",
                "stages": 0,
                "floor": 0.05,
                "max_lev": 0.05,
                "enter": "",
                "exit": "",
                "levels": "",
                "cagr": 0.11,
                "maxdd": -0.30,
                "switches": 0,
                "capital_twd": 1_000_000,
                "final_multiple": 2.2,
            },
        ]
    ).to_csv(results / "representative_choices.csv", index=False)

    pd.DataFrame(
        [
            {"date": "2020-01-31", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-02-28", "NASDAQ_core": 90.0, "NASDAQ_lev": 70.0},
            {"date": "2020-03-31", "NASDAQ_core": 95.0, "NASDAQ_lev": 85.0},
            {"date": "2020-04-30", "NASDAQ_core": 110.0, "NASDAQ_lev": 130.0},
        ]
    ).to_csv(data / "prices_extended.csv", index=False)

    (data / "price_report_extended.json").write_text(
        json.dumps(
            {
                "symbols": {
                    "QQQ": {"start": "2020-01-31"},
                    "TQQQ": {"start": "2020-01-31"},
                },
                "pairs": {
                    "NASDAQ": {
                        "start": "2020-01-31",
                        "end": "2020-04-30",
                        "rows": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    validation = root / "validation" / "NASDAQ"
    validation.mkdir(parents=True)
    (validation / "NASDAQ_three_layer_validation.json").write_text(
        json.dumps(
            {
                "pair": "NASDAQ",
                "parameters": {
                    "horizonsYears": [1, 3],
                    "regimeHorizonYears": 3,
                    "monteCarloYears": 10,
                    "monteCarloPaths": 25,
                    "seed": 42,
                },
                "rolling": [
                    {
                        "choice": "best_full_return",
                        "choiceName": "歷史報酬最高",
                        "horizonYears": 3,
                        "horizonMonths": 36,
                        "windows": 10,
                        "medianFinalMultiple": 1.4,
                        "p05FinalMultiple": 0.8,
                        "p95FinalMultiple": 2.1,
                        "winRateVsCore": 0.7,
                        "worstMaxDrawdown": -0.35,
                    }
                ],
                "regimes": [
                    {
                        "choice": "best_full_return",
                        "choiceName": "歷史報酬最高",
                        "regime": "bear",
                        "horizonYears": 3,
                        "horizonMonths": 36,
                        "windows": 4,
                        "medianFinalMultiple": 1.3,
                        "p05FinalMultiple": 0.75,
                        "p95FinalMultiple": 2.0,
                        "winRateVsCore": 0.75,
                        "worstMaxDrawdown": -0.4,
                    }
                ],
                "monteCarlo": [
                    {
                        "choice": "best_full_return",
                        "choiceName": "歷史報酬最高",
                        "years": 10,
                        "paths": 25,
                        "p05FinalMultiple": 0.6,
                        "p50FinalMultiple": 1.7,
                        "p95FinalMultiple": 4.5,
                        "probabilityOfLoss": 0.2,
                        "medianMaxDrawdown": -0.3,
                        "p05MaxDrawdown": -0.6,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_strategy_calculator_html_embeds_verified_poseidon_rotation_data(tmp_path: Path) -> None:
    root = tmp_path / "rotation-research"
    out = root / "strategy-calculator.html"
    _write_minimal_rotation_research(root)

    result = build_strategy_calculator_html(root=root, out=out)

    assert result == out
    assert out.exists()
    data, html = load_embedded_strategy_data(out)
    verify_strategy_calculator_html(out)

    assert data["pairs"][0]["key"] == "NASDAQ"
    assert {strategy["choice"] for strategy in data["strategies"]} >= {
        "core_buy_hold",
        "leveraged_buy_hold",
        "best_full_return",
    }
    assert "current_mix_buy_hold" not in {strategy["choice"] for strategy in data["strategies"]}
    assert "目前比例持有" not in html
    assert '<option value="history" selected>歷史回測</option>' in html
    assert 'id="scaleToggle"' in html
    assert 'id="validationPanel"' in html
    assert "三層驗證" in html
    assert "滾動進場" in html
    assert "市況分組" in html
    assert "蒙地卡羅" in html

    rules = [(strategy["pair"], strategy["rule"]) for strategy in data["strategies"]]
    assert len(rules) == len(set(rules))
    assert data["validation"]["NASDAQ"]["rolling"][0]["choiceName"] == "歷史報酬最高"


def test_etf_rotation_report_script_wrapper_builds_and_verifies_html(tmp_path: Path) -> None:
    from scripts.etf_rotation_report import run_etf_rotation_report

    root = tmp_path / "rotation-research"
    out_dir = tmp_path / "script-output"
    _write_minimal_rotation_research(root)

    summary = run_etf_rotation_report(root=root, out_dir=out_dir, verify=True)

    html_path = out_dir / "strategy-calculator.html"
    assert summary == {
        "root": str(root),
        "out": str(html_path),
        "verified": True,
    }
    assert html_path.exists()
    verify_strategy_calculator_html(html_path)
