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

    rules = [(strategy["pair"], strategy["rule"]) for strategy in data["strategies"]]
    assert len(rules) == len(set(rules))


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
