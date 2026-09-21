from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from poseidon.research.etf_rotation.representative import build_representative_choices


def _row(**overrides):
    row = {
        "mode": "buy_dip",
        "stages": 1,
        "floor": 0.0,
        "max_lev": 1.0,
        "enter": "0.10",
        "exit": "0.20",
        "levels": "1.0",
        "full_final": 2.0,
        "full_cagr": 0.10,
        "full_maxdd": -0.30,
        "full_sharpe": 1.0,
        "full_ulcer": 0.1,
        "full_switches": 4,
        "score": 1.0,
        "test_cagr": 0.08,
        "test_maxdd": -0.20,
    }
    row.update(overrides)
    return row


def test_representative_choices_keep_real_de_risk_rows_separate_from_buy_dip(tmp_path: Path) -> None:
    root = tmp_path / "rotation"
    results = root / "results"
    results.mkdir(parents=True)
    (results / "summary.json").write_text(
        json.dumps(
            {
                "pairs": {
                    "NASDAQ": {
                        "benchmark": {
                            "core": {"final": 1.5, "cagr": 0.05, "maxdd": -0.2},
                            "lev": {"final": 2.5, "cagr": 0.12, "maxdd": -0.7},
                            "current_mix": {"final": 1.6, "cagr": 0.06, "maxdd": -0.25},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            _row(mode="buy_dip", full_final=3.0, full_cagr=0.15, full_switches=6, score=1.2),
            _row(
                mode="de_risk_on_drawdown",
                enter="0.45",
                levels="0.20",
                full_final=10.0,
                full_cagr=0.40,
                full_switches=0,
                score=3.0,
            ),
            _row(
                mode="de_risk_on_drawdown",
                enter="0.15",
                levels="0.20",
                full_final=4.0,
                full_cagr=0.20,
                full_switches=4,
                score=1.8,
            ),
            _row(
                mode="de_risk_on_drawdown",
                stages=2,
                enter="0.10|0.20",
                exit="0.15|0.30",
                levels="0.40|0.20",
                full_final=3.5,
                full_cagr=0.18,
                full_maxdd=-0.38,
                full_switches=8,
                score=2.1,
            ),
            _row(
                mode="de_risk_on_drawdown",
                stages=3,
                enter="0.10|0.20|0.30",
                exit="0.15|0.30|0.40",
                levels="0.60|0.40|0.20",
                full_final=3.2,
                full_cagr=0.16,
                full_maxdd=-0.42,
                full_switches=12,
                score=2.2,
            ),
            _row(
                mode="de_risk_to_cash_on_drawdown",
                stages=1,
                enter="0.10",
                exit="0.30",
                levels="0.0",
                full_final=4.5,
                full_cagr=0.22,
                full_maxdd=-0.36,
                full_switches=5,
                score=2.4,
            ),
            _row(
                mode="leveraged_cash_band",
                stages=0,
                max_lev=0.6,
                enter="0.05",
                exit="",
                levels="",
                full_final=3.8,
                full_cagr=0.19,
                full_maxdd=-0.33,
                full_switches=18,
                score=2.3,
            ),
            _row(
                mode="ma_sma",
                stages=0,
                max_lev=1.0,
                enter="200",
                exit="",
                levels="",
                full_final=4.0,
                full_cagr=0.20,
                full_switches=9,
                score=2.0,
            ),
            _row(
                mode="ma_ema",
                stages=0,
                max_lev=1.0,
                enter="200",
                exit="",
                levels="",
                full_final=4.2,
                full_cagr=0.21,
                full_switches=11,
                score=2.1,
            ),
            _row(
                mode="ma_sma_band",
                stages=0,
                max_lev=1.0,
                enter="200|0.02",
                exit="",
                levels="",
                full_final=4.1,
                full_cagr=0.205,
                full_switches=5,
                score=2.05,
            ),
        ]
    ).to_csv(results / "NASDAQ_all_results.csv", index=False)

    choices = build_representative_choices(root, pairs=("NASDAQ",))["NASDAQ"]
    choice_names = {choice["choice"] for choice in choices}

    best_full_return = next(choice for choice in choices if choice["choice"] == "best_full_return")
    best_de_risk_return = next(choice for choice in choices if choice["choice"] == "best_de_risk_return")
    assert best_full_return["mode"] == "buy_dip"
    assert best_de_risk_return["mode"] == "de_risk_on_drawdown"
    assert best_de_risk_return["switches"] > 0
    assert {
        "best_de_risk_under_40dd",
        "best_de_risk_under_45dd",
        "best_de_risk_under_50dd",
        "best_de_risk_low_turnover_return",
        "best_de_risk_2_stage_score",
        "best_de_risk_3_stage_score",
        "best_de_risk_cash_return",
        "best_de_risk_cash_under_40dd",
        "best_leveraged_cash_band_return",
        "best_leveraged_cash_band_under_40dd",
        "ma_sma200",
        "ma_ema200",
        "ma_sma200_band_2",
    } <= choice_names
    assert not any(
        choice["mode"] == "de_risk_on_drawdown"
        and choice["switches"] == 0
        and not choice["choice"].endswith("buy_hold")
        for choice in choices
    )
