from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from poseidon.research.etf_rotation.choices import rule_text
from poseidon.research.etf_rotation.report import (
    build_strategy_calculator_html,
    load_embedded_strategy_data,
    strategy_curve,
    strategy_trace,
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
                "mode": "buy_dip",
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
                "mode": "buy_dip",
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
                "choice": "best_de_risk_score",
                "mode": "de_risk_on_drawdown",
                "stages": 1,
                "floor": 0.0,
                "max_lev": 1.0,
                "enter": "0.10",
                "exit": "0.20",
                "levels": "0.0",
                "cagr": 0.17,
                "maxdd": -0.45,
                "switches": 2,
                "capital_twd": 1_000_000,
                "final_multiple": 4.8,
            },
            {
                "pair": "NASDAQ",
                "choice": "current_mix_buy_hold",
                "mode": "buy_hold",
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
    assert 'id="strategyTablePanel"' in html
    assert 'id="allocationPanel"' in html
    assert 'id="selectedStrategySelect"' in html
    assert 'id="chartStrategyButton"' in html
    assert 'id="chartStrategyMenu"' in html
    assert 'id="chartStrategyOptions"' in html
    assert 'id="chartStrategyDefault"' in html
    assert 'id="chartStrategyAll"' in html
    assert 'id="chartStrategyBenchmarks"' in html
    assert "const defaultChartChoices" in html
    assert "function selectedChartChoices" in html
    assert "function renderChartStrategyOptions" in html
    assert "function updateChartStrategyButton" in html
    assert "const benchmark = benchmarkOrder.includes(strategy.choice);" in html
    assert "${benchmark ? ' disabled' : ''}" in html
    assert "${s.isBenchmark ? ' disabled' : ''}" in html
    assert "if (available.has(choice)) selected.add(choice);" in html
    assert "plot-toggle" in html
    assert 'id="chartLimit"' not in html
    assert '<div class="plotly-chart" id="equityChart"' in html
    assert '<div class="plotly-chart selected-strategy-chart" id="selectedStrategyChart"' in html
    assert 'id="bestValueLabel"' in html
    assert '<th class="num num-col">全期間 CAGR</th>' in html
    assert 'data-horizon-years="1"' in html
    assert 'data-horizon-years="10"' in html
    assert "function updateHorizonLabels" in html
    assert "historyMode ? `近 ${years} 年` : `推估 ${years} 年`" in html
    assert "Plotly.react(equityChart" in html
    assert "Plotly.react(selectedStrategyChart" in html
    assert "<b>%{x}</b><br>%{fullData.name}" not in html
    assert "%{fullData.name}: %{y:,.0f}<extra></extra>" in html
    assert "function plotMainChart" in html
    assert "function plotSelectedStrategyChart" in html
    assert "function allocationShapesForStrategy" in html
    assert "xref: 'x'" in html
    assert "yref: 'paper'" in html
    assert "function punctuateRulePart" in html
    assert "function formatNote" in html
    assert "function renderSelectedStrategySelect" in html
    assert "function renderAllocationDetails" in html
    assert "function allocationForIndex" in html
    assert "plotSelectedStrategyChart(strategy, pairInfo)" in html
    assert "selectedStrategySelect.addEventListener('change'" in html
    assert "bestValueLabel.textContent = historyMode ? '最高全歷史圖表終值' : '最高推估 10 年';" in html
    assert "三層驗證" in html
    assert "滾動進場" in html
    assert "市況分組" in html
    assert "蒙地卡羅" in html
    assert html.index('class="summary"') < html.index('id="validationPanel"') < html.index('id="strategyTablePanel"')

    rules = [(strategy["pair"], strategy["rule"]) for strategy in data["strategies"]]
    assert len(rules) == len(set(rules))
    assert data["validation"]["NASDAQ"]["rolling"][0]["choiceName"] == "歷史報酬最高"
    duplicate = next(strategy for strategy in data["strategies"] if strategy["choice"] == "best_full_return")
    assert duplicate["alsoMatched"] == ["綜合分數最佳"]
    reverse = next(strategy for strategy in data["strategies"] if strategy["choice"] == "best_de_risk_score")
    assert reverse["mode"] == "de_risk_on_drawdown"
    assert "平常持有 100% TQQQ" in reverse["rule"]
    assert "下跌 10%" in reverse["rule"]
    assert "降到 0% TQQQ" in reverse["rule"]
    assert duplicate["allocationSegments"]
    assert duplicate["switchEvents"]
    assert duplicate["holdingSummary"]["leveragedTradingDayRatio"] == pytest.approx(3.05 / 4)
    assert duplicate["holdingSummary"]["coreTradingDayRatio"] == pytest.approx(0.95 / 4)


def test_strategy_curve_supports_de_risk_mode() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2020-01-01", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-02", "NASDAQ_core": 90.0, "NASDAQ_lev": 80.0},
            {"date": "2020-01-03", "NASDAQ_core": 80.0, "NASDAQ_lev": 60.0},
            {"date": "2020-01-04", "NASDAQ_core": 88.0, "NASDAQ_lev": 80.0},
            {"date": "2020-01-05", "NASDAQ_core": 100.0, "NASDAQ_lev": 120.0},
        ]
    )
    row = pd.Series(
        {
            "choice": "best_de_risk_score",
            "mode": "de_risk_on_drawdown",
            "stages": 1,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "0.10",
            "exit": "0.20",
            "levels": "0.0",
        }
    )

    curve = strategy_curve("NASDAQ", row, frame)

    assert curve[-1] == pytest.approx(0.8888888889)


def test_rule_text_describes_ma_strategies() -> None:
    sma = pd.Series(
        {
            "choice": "ma_sma200",
            "mode": "ma_sma",
            "stages": 0,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "200",
            "exit": "",
            "levels": "",
        }
    )
    ema = sma.copy()
    ema["choice"] = "ma_ema200"
    ema["mode"] = "ma_ema"
    band = sma.copy()
    band["choice"] = "ma_sma200_band_2"
    band["mode"] = "ma_sma_band"
    band["enter"] = "200|0.02"

    assert rule_text("NASDAQ", sma) == "QQQ 收盤高於 SMA200 時，隔日持有 100% TQQQ；否則持有 100% QQQ。"
    assert rule_text("NASDAQ", ema) == "QQQ 收盤高於 EMA200 時，隔日持有 100% TQQQ；否則持有 100% QQQ。"
    assert rule_text("NASDAQ", band) == (
        "QQQ 收盤高於 SMA200 2% 時，隔日持有 100% TQQQ；QQQ 收盤低於 SMA200 2% 時，隔日回到 100% QQQ。"
    )


def test_strategy_trace_supports_ma_sma_mode() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2020-01-01", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-02", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-03", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-04", "NASDAQ_core": 110.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-05", "NASDAQ_core": 120.0, "NASDAQ_lev": 200.0},
        ]
    )
    row = pd.Series(
        {
            "choice": "ma_sma200",
            "mode": "ma_sma",
            "stages": 0,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "3",
            "exit": "",
            "levels": "",
        }
    )

    trace = strategy_trace("NASDAQ", row, frame)

    assert trace["equity"][-1] == pytest.approx(2.2)
    assert trace["switchEvents"][0]["reason"] == "QQQ 收盤高於 SMA3"
    assert trace["holdingSummary"]["leveragedTradingDayRatio"] == pytest.approx(2 / 5)


def test_strategy_curve_supports_de_risk_to_cash_mode() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2020-01-01", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-02", "NASDAQ_core": 90.0, "NASDAQ_lev": 80.0},
            {"date": "2020-01-03", "NASDAQ_core": 80.0, "NASDAQ_lev": 60.0},
            {"date": "2020-01-04", "NASDAQ_core": 88.0, "NASDAQ_lev": 80.0},
            {"date": "2020-01-05", "NASDAQ_core": 100.0, "NASDAQ_lev": 120.0},
        ]
    )
    row = pd.Series(
        {
            "choice": "best_de_risk_cash_score",
            "mode": "de_risk_to_cash_on_drawdown",
            "stages": 1,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "0.10",
            "exit": "0.20",
            "levels": "0.0",
        }
    )

    trace = strategy_trace("NASDAQ", row, frame)

    assert trace["equity"][-1] == pytest.approx(0.8)
    assert trace["allocationSegments"][1]["cashWeight"] == 1.0
    assert trace["holdingSummary"]["cashTradingDayRatio"] > 0


def test_strategy_trace_supports_leveraged_cash_band_rebalancing() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2020-01-01", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-02", "NASDAQ_core": 100.0, "NASDAQ_lev": 150.0},
            {"date": "2020-01-03", "NASDAQ_core": 100.0, "NASDAQ_lev": 150.0},
        ]
    )
    row = pd.Series(
        {
            "choice": "best_leveraged_cash_band_score",
            "mode": "leveraged_cash_band",
            "stages": 0,
            "floor": 0.0,
            "max_lev": 0.6,
            "enter": "0.05",
            "exit": "",
            "levels": "",
        }
    )

    trace = strategy_trace("NASDAQ", row, frame)

    assert trace["equity"][-1] == pytest.approx(1.3)
    assert trace["switchEvents"][0]["beforeLeveragedWeight"] == pytest.approx(0.692308)
    assert trace["switchEvents"][0]["afterLeveragedWeight"] == pytest.approx(0.6)
    assert trace["allocationSegments"][0]["cashWeight"] == 0.4
    assert trace["holdingSummary"]["cashTradingDayRatio"] == pytest.approx(0.4)


def test_strategy_trace_records_segments_events_and_weighted_holding_summary() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2020-01-01", "NASDAQ_core": 100.0, "NASDAQ_lev": 100.0},
            {"date": "2020-01-02", "NASDAQ_core": 90.0, "NASDAQ_lev": 80.0},
            {"date": "2020-01-03", "NASDAQ_core": 130.0, "NASDAQ_lev": 160.0},
        ]
    )
    row = pd.Series(
        {
            "choice": "best_full_return",
            "mode": "buy_dip",
            "stages": 1,
            "floor": 0.05,
            "max_lev": 1.0,
            "enter": "0.10",
            "exit": "0.40",
            "levels": "1.0",
        }
    )

    trace = strategy_trace("NASDAQ", row, frame)

    assert trace["equity"] == pytest.approx(strategy_curve("NASDAQ", row, frame))
    assert trace["allocationSegments"] == [
        {
            "start": "2020-01-01",
            "end": "2020-01-02",
            "tradingDays": 1,
            "coreWeight": 0.95,
            "leveragedWeight": 0.05,
        },
        {
            "start": "2020-01-02",
            "end": "2020-01-03",
            "tradingDays": 1,
            "coreWeight": 0.0,
            "leveragedWeight": 1.0,
        },
        {
            "start": "2020-01-03",
            "end": "2020-01-03",
            "tradingDays": 1,
            "coreWeight": 0.95,
            "leveragedWeight": 0.05,
        },
    ]
    assert trace["switchEvents"] == [
        {
            "date": "2020-01-02",
            "reason": "QQQ 從近期高點下跌 10%",
            "daysSincePrevious": None,
            "beforeCoreWeight": 0.95,
            "beforeLeveragedWeight": 0.05,
            "afterCoreWeight": 0.0,
            "afterLeveragedWeight": 1.0,
        },
        {
            "date": "2020-01-03",
            "reason": "QQQ 從低點反彈 40%",
            "daysSincePrevious": 1,
            "beforeCoreWeight": 0.0,
            "beforeLeveragedWeight": 1.0,
            "afterCoreWeight": 0.95,
            "afterLeveragedWeight": 0.05,
        },
    ]
    assert trace["holdingSummary"] == {
        "totalTradingDays": 3,
        "coreWeightedTradingDays": pytest.approx(1.9),
        "leveragedWeightedTradingDays": pytest.approx(1.1),
        "coreTradingDayRatio": pytest.approx(1.9 / 3),
        "leveragedTradingDayRatio": pytest.approx(1.1 / 3),
    }


def test_rule_text_separates_sentences_for_plain_text_readability() -> None:
    buy_dip = pd.Series(
        {
            "choice": "best_full_return",
            "mode": "buy_dip",
            "stages": 1,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "0.05",
            "exit": "0.35",
            "levels": "1.0",
        }
    )
    de_risk = pd.Series(
        {
            "choice": "best_de_risk_score",
            "mode": "de_risk_on_drawdown",
            "stages": 1,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "0.30",
            "exit": "0.10",
            "levels": "0.2",
        }
    )
    de_risk_to_cash = pd.Series(
        {
            "choice": "best_de_risk_cash_score",
            "mode": "de_risk_to_cash_on_drawdown",
            "stages": 1,
            "floor": 0.0,
            "max_lev": 1.0,
            "enter": "0.30",
            "exit": "0.10",
            "levels": "0.2",
        }
    )
    leveraged_cash_band = pd.Series(
        {
            "choice": "best_leveraged_cash_band_score",
            "mode": "leveraged_cash_band",
            "stages": 0,
            "floor": 0.0,
            "max_lev": 0.6,
            "enter": "0.05",
            "exit": "",
            "levels": "",
        }
    )

    assert "平常持有 100% 0050；0050 從近期高點下跌" in rule_text("TAIWAN50", buy_dip)
    assert "平常持有 100% 00631L；0050 從近期高點下跌" in rule_text("TAIWAN50", de_risk)
    assert "其餘持有現金" in rule_text("TAIWAN50", de_risk_to_cash)
    assert "平常持有 60% 00631L / 40% 現金" in rule_text("TAIWAN50", leveraged_cash_band)
    assert "權重高於 65% 或低於 55% 時，再平衡回 60% 00631L" in rule_text("TAIWAN50", leveraged_cash_band)


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
